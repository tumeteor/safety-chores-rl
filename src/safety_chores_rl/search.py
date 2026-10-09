"""Batched best-of-N, successive halving, and bounded open-loop beam search."""
from __future__ import annotations

from dataclasses import replace
import math
import statistics
import time
from typing import Callable

from .session import BudgetLedger, RolloutSession
from .types import CandidateEvaluation, Plan, RolloutBudget, SearchBudget, integer


def safe_completion(result):
    """An unresolved eventual success is unknown, not a zero-valued sample."""
    return float(result.safe_success) if result.valid and result.safe_success is not None else None


def reward_return(result):
    """Observed finite-horizon return, not an estimate of eventual completion."""
    return sum(result.rewards) if result.valid else None


def _evaluation(candidate, results, objective, max_cost):
    values = []
    for result in results:
        if not result.valid or (max_cost is not None and result.total_cost > max_cost):
            return CandidateEvaluation(candidate, None, None, len(results), tuple(results))
        value = objective(result)
        if value is None:
            return CandidateEvaluation(candidate, None, None, len(results), tuple(results))
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Objective must return a finite scalar or None")
        values.append(value)
    mean = statistics.fmean(values) if values else None
    error = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return CandidateEvaluation(candidate, mean, error, len(results), tuple(results))


def _rank(evaluations):
    return sorted((e for e in evaluations if e.score is not None), key=lambda e: (-e.score, e.candidate))


class BestOfN:
    """Exhaustive matched-sample evaluation, with stable lexicographic ties."""
    def __init__(self, session: RolloutSession, *, objective: Callable = safe_completion,
                 max_cost=None):
        if max_cost is not None and (not math.isfinite(max_cost) or max_cost < 0):
            raise ValueError("max_cost must be finite and nonnegative")
        self.session, self.objective, self.max_cost = session, objective, max_cost

    def plan(self, root, candidates, *, sample_seeds=(0, 1, 2, 3),
             rollout_budget: RolloutBudget, search_budget: SearchBudget = SearchBudget(),
             continuation=True, admissible=None):
        candidates = self._candidates(root, candidates, search_budget, admissible)
        seeds = self._seeds(sample_seeds)
        ledger = BudgetLedger(search_budget)
        if not candidates:
            return Plan(root.identity, None, "no_admissible_plan", (), ledger.report())
        results = self.session.rollout_batch(root, candidates, sample_seeds=seeds,
            continuation=continuation, budget=rollout_budget, ledger=ledger)
        groups = {candidate: [] for candidate in candidates}
        for result in results:
            groups[result.candidate].append(result)
        evaluations = tuple(_evaluation(c, groups[c], self.objective, self.max_cost) for c in candidates)
        ranked = _rank(evaluations)
        status = ("cancelled" if any(r.status == "cancelled" for r in results) else
                  "budget_exhausted" if ledger.exhausted else
                  "ok" if ranked else "insufficient_evidence")
        return Plan(root.identity, ranked[0].candidate if ranked else None, status,
                    evaluations, ledger.report())

    @staticmethod
    def _candidates(root, candidates, budget, admissible, *, preserve_order=False):
        unique = {}
        for index, candidate in enumerate(candidates):
            if index >= budget.max_candidates:
                raise ValueError("Candidate generator exceeded max_candidates")
            candidate = tuple(candidate)
            if not candidate:
                raise ValueError("Empty candidate")
            for action in candidate:
                integer(action, "action")
            if admissible is None or admissible(root.observation, candidate):
                unique.setdefault(candidate, None)
        return tuple(unique) if preserve_order else tuple(sorted(unique))

    @staticmethod
    def _seeds(seeds):
        seeds = tuple(seeds)
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("Use a nonempty list of distinct sample seeds")
        for seed in seeds:
            integer(seed, "sample_seed")
            if seed >= 2**32:
                raise ValueError("sample_seed must fit UInt32")
        return seeds


class FirstSuccess(BestOfN):
    """Stop after the first bounded worker wave containing a zero-cost success.

    Candidate order is caller priority. Retain the complete successful action
    sequence and per-step execution signatures; no optimality/risk claim is made.
    """
    def plan(self, root, candidates, *, sample_seeds=(0, 1, 2, 3),
             rollout_budget, search_budget=SearchBudget(), admissible=None):
        candidates = self._candidates(root, candidates, search_budget, admissible, preserve_order=True)
        seeds = self._seeds(sample_seeds)
        ledger = BudgetLedger(search_budget)
        groups = {c: [] for c in candidates}
        witness = None
        status = 'no_successful_trajectory' if candidates else 'no_admissible_plan'
        budget = replace(rollout_budget, record_trace=True, stop_on_cost=True)
        for seed in seeds:
            for offset in range(0, len(candidates), self.session.workers):
                wave = candidates[offset:offset+self.session.workers]
                results = self.session.rollout_batch(root, wave, sample_seeds=[seed],
                    budget=budget, ledger=ledger)
                for result in results:
                    groups[result.candidate].append(result)
                if any(not r.valid for r in results):
                    status = ('cancelled' if any(r.status == 'cancelled' for r in results) else
                              'budget_exhausted' if ledger.exhausted else 'invalid_evidence')
                    break
                witness = next((r for r in results if r.success is True and r.safe_success is True
                                and r.total_cost == 0 and len(r.trace) == len(r.actions)+1), None)
                if witness:
                    status = 'ok'
                    break
            if witness or status in ('cancelled', 'budget_exhausted', 'invalid_evidence'):
                break
        evaluations = tuple(_evaluation(c, groups[c], safe_completion, None) for c in candidates)
        return Plan(root.identity, witness.actions if witness else None, status, evaluations,
                    dict(ledger.report(), exhaustive_rollouts=len(candidates)*len(seeds)),
                    'first_success', witness)


class SuccessiveHalving(BestOfN):
    """Approximate racing: double samples and discard weak candidates each round.

    This trades ranking robustness for fewer simulations. It is not a confidence
    bound or an optimality guarantee.
    """
    def __init__(self, session, *, objective=safe_completion, max_cost=None,
                 initial_samples=1, reduction_factor=2):
        super().__init__(session, objective=objective, max_cost=max_cost)
        integer(initial_samples, "initial_samples", 1)
        integer(reduction_factor, "reduction_factor", 2)
        self.initial_samples, self.reduction_factor = initial_samples, reduction_factor

    def plan(self, root, candidates, *, sample_seeds=(0, 1, 2, 3, 4, 5, 6, 7),
             rollout_budget, search_budget=SearchBudget(), continuation=True, admissible=None):
        candidates = self._candidates(root, candidates, search_budget, admissible)
        seeds = self._seeds(sample_seeds)
        ledger = BudgetLedger(search_budget)
        active = list(candidates)
        groups = {candidate: [] for candidate in candidates}
        previous, target, rounds = 0, min(self.initial_samples, len(seeds)), 0
        incumbent = None
        cancelled = False
        while active:
            results = self.session.rollout_batch(root, active, sample_seeds=seeds[previous:target],
                continuation=continuation, budget=rollout_budget, ledger=ledger)
            for result in results:
                groups[result.candidate].append(result)
            ranked = _rank([_evaluation(c, groups[c], self.objective, self.max_cost) for c in active])
            rounds += 1
            if ranked:
                incumbent = ranked[0].candidate
            cancelled = any(r.status == "cancelled" for r in results)
            if not ranked or ledger.exhausted or cancelled or target == len(seeds):
                break
            active = [e.candidate for e in ranked[:max(1, math.ceil(len(ranked) / self.reduction_factor))]]
            previous, target = target, min(len(seeds), target * self.reduction_factor)
        evaluations = tuple(_evaluation(c, groups[c], self.objective, self.max_cost) for c in candidates)
        # If the incumbent later gained invalid evidence, it cannot remain recommended.
        valid = {e.candidate for e in evaluations if e.score is not None}
        if incumbent not in valid:
            incumbent = None
        status = ("cancelled" if cancelled else "budget_exhausted" if ledger.exhausted else
                  "no_admissible_plan" if not candidates else "ok" if incumbent else "insufficient_evidence")
        usage = dict(ledger.report(), rounds=rounds,
                     exhaustive_rollouts=len(candidates)*len(seeds), approximate_racing=True)
        return Plan(root.identity, incumbent, status, evaluations, usage, "successive_halving")


class BeamSearch:
    """Batch whole-prefix replays, deduplicate prefixes, retain a bounded frontier.

    expander receives only the public endpoint observation and action prefix.
    heuristic scores a prefix-only result; it never receives a live simulator.
    """
    def __init__(self, session, *, depth=3, width=4, branching_factor=8,
                 heuristic=reward_return, max_cost=0.0):
        for name, value in (("depth", depth), ("width", width), ("branching_factor", branching_factor)):
            integer(value, name, 1)
        if max_cost is not None and (not math.isfinite(max_cost) or max_cost < 0):
            raise ValueError("max_cost must be finite and nonnegative")
        self.session, self.depth, self.width = session, depth, width
        self.branching_factor, self.heuristic, self.max_cost = branching_factor, heuristic, max_cost

    def plan(self, root, expander, *, rollout_budget: RolloutBudget,
             search_budget=SearchBudget(), sample_seed=0, admissible=None):
        ledger = BudgetLedger(search_budget)
        # A one-shot iterable must be reusable at every expanded node.
        if not callable(expander):
            from itertools import islice
            expander = tuple(islice(expander, self.branching_factor))
        frontier = [((), root.observation)]
        history, terminals = [], []
        seen = set()
        ranked = []
        cancelled = False
        completed_depth = 0
        max_depth = min(self.depth, rollout_budget.max_steps, root.episode_limit-root.step)
        for depth in range(1, max_depth + 1):
            candidates = []
            for prefix, observation in frontier:
                options = expander(observation, prefix) if callable(expander) else expander
                for index, action in enumerate(options):
                    if index >= self.branching_factor:
                        break
                    integer(action, "action")
                    candidate = prefix + (action,)
                    if candidate in seen or (admissible and not admissible(observation, candidate)):
                        continue
                    if len(seen) >= search_budget.max_candidates:
                        ledger.exhausted = True
                        break
                    seen.add(candidate)
                    candidates.append(candidate)
            if not candidates or time.monotonic() >= ledger.deadline:
                if time.monotonic() >= ledger.deadline:
                    ledger.exhausted = True
                break
            # Prefix-only horizon avoids paying for unrelated terminal suffixes.
            budget = replace(rollout_budget, max_steps=depth)
            results = self.session.rollout_batch(root, sorted(candidates), sample_seeds=[sample_seed],
                continuation=False, budget=budget, ledger=ledger)
            evaluations = [_evaluation(r.candidate, [r], self.heuristic, self.max_cost) for r in results]
            history.extend(evaluations)
            ranked = _rank(evaluations)
            completed_depth = depth
            terminals.extend(e for e in ranked if e.results[0].terminated)
            frontier = [(e.candidate, e.results[0].endpoint) for e in ranked
                        if not e.results[0].terminated and
                        (not e.results[0].truncated or e.results[0].stop_reason == "prefix_complete")]
            frontier = frontier[:self.width]
            cancelled = any(r.status == "cancelled" for r in results)
            if cancelled or ledger.exhausted or not frontier:
                break
        finalists = _rank([*ranked, *terminals])
        status = ("cancelled" if cancelled else "budget_exhausted" if ledger.exhausted else
                  "ok" if finalists else "no_admissible_plan")
        usage = dict(ledger.report(), expanded_prefixes=len(seen), depth=completed_depth)
        return Plan(root.identity, finalists[0].candidate if finalists else None,
                    status, tuple(history), usage, "beam")


class RecedingHorizon:
    """Caller-driven execution: one action per call, explicit fallback on no plan."""
    def __init__(self, session, planner, *, fallback=None):
        self.session, self.planner, self.fallback = session, planner, fallback

    def step(self, episode, candidates, **options):
        root = self.session.capture_root(episode)
        try:
            plan = self.planner.plan(root, candidates, **options)
            if plan.action is None:
                if self.fallback is None:
                    return plan, None
                action = self.fallback(episode.observation, plan)
                if not episode.matches(root):
                    raise ValueError("Fallback mutated the live episode")
                transition = episode.step(action)
                return plan, transition
            return plan, self.session.execute_first(episode, plan)
        finally:
            if root.identity in self.session._roots:
                self.session.release_root(root)
