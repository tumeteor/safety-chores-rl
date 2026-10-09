"""Public rollout sessions, central budgets and bounded exact-request reuse."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
import threading
import time
import uuid

from .runtime import Episode
from .types import (API_VERSION, Root, RuntimeSpec, RolloutRequest, RolloutResult,
                    RolloutBudget, SearchBudget, canonical, integer, digest)
from .workers import WorkerPool


class StalePlanError(ValueError):
    pass


class TrajectoryMismatch(RuntimeError):
    """Execution stopped after an observed divergence; past actions cannot be undone."""
    pass


class BudgetLedger:
    """One coordinator reserves worst-case primitive work before dispatch."""
    def __init__(self, budget: SearchBudget):
        self.budget = budget
        self.started = time.monotonic()
        self.deadline = self.started + budget.wall_seconds
        self.rollouts = self.native_steps = self.cache_hits = 0
        self.exhausted = False

    def reserve(self, steps):
        if (time.monotonic() >= self.deadline or self.rollouts >= self.budget.max_rollouts
                or self.native_steps + steps > self.budget.max_native_steps):
            self.exhausted = True
            return False
        self.rollouts += 1
        self.native_steps += steps
        return True

    def reconcile(self, reserved, result):
        actual = result.native_steps if result.valid or result.status == "root_mismatch" else reserved
        if actual > reserved:
            raise RuntimeError("Backend exceeded reserved primitive budget")
        self.native_steps -= reserved - actual

    def report(self):
        return dict(rollouts=self.rollouts, native_steps=self.native_steps,
                    cache_hits=self.cache_hits, elapsed_seconds=time.monotonic() - self.started,
                    budget_exhausted=self.exhausted)


class RolloutSession:
    """Reusable process pool; roots and caches are local to this session.

    Calls are serialized per session. cancel() may be called by another thread;
    it interrupts the current batch, and later batches remain usable.
    """
    def __init__(self, spec: RuntimeSpec, *, workers=1, cache_size=4096):
        integer(workers, "workers", 1)
        integer(cache_size, "cache_size")
        self.spec, self.workers, self.cache_size = spec, workers, cache_size
        self._owner = uuid.uuid4().hex
        self._roots = {}
        self._cache = OrderedDict()
        self._pool = WorkerPool(spec, workers)
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._closed = False

    @contextmanager
    def _operation(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Session is busy; use another session for concurrent callers")
        try:
            if self._closed:
                raise RuntimeError("Session is closed")
            yield
        finally:
            self._lock.release()

    def capabilities(self):
        return dict(api_version=API_VERSION, runtime_id=self.spec.identity,
                    reconstruction="independent_reset_and_replay", exact_checkpoint=False,
                    workers=self.workers, cache_size=self.cache_size, process_isolation=True,
                    persistent_workers=True, bounded_search=True)

    def capture_root(self, episode: Episode):
        with self._operation():
            if episode.spec != self.spec or episode.done or episode.closed:
                raise ValueError("Root needs a live, nonterminal episode of this runtime")
            root = Root(self._owner, episode.episode_id, self.spec.identity, episode.task_id, episode.seed,
                        episode.max_steps, tuple(episode.actions), episode.cost,
                        canonical(episode.runtime.snapshot()), canonical(episode.observation))
            # Return the same registered immutable object for identical captures.
            return self._roots.setdefault(root.identity, root)

    def _check_root(self, root):
        if self._roots.get(root.identity) is not root:
            raise ValueError("Foreign, released, or unregistered root")

    def release_root(self, root):
        with self._operation():
            self._check_root(root)
            del self._roots[root.identity]
            self._cache = OrderedDict((k, v) for k, v in self._cache.items() if v.root_id != root.identity)

    def _request(self, root, candidate, seed, continuation, budget, validate_only=False, inspect_only=False):
        self._check_root(root)
        integer(seed, "sample_seed")
        if seed >= 2**32:
            raise ValueError("sample_seed must fit UInt32")
        candidate = tuple(candidate)
        for action in candidate:
            integer(action, "candidate action")
        if not validate_only and not inspect_only and not candidate:
            raise ValueError("Candidate prefix must not be empty")
        if len(candidate) > budget.max_steps or budget.max_steps > root.episode_limit - root.step:
            raise ValueError("Candidate/horizon exceeds the remaining episode budget")
        return RolloutRequest(root, candidate, seed, continuation, budget, validate_only, inspect_only)

    def _evaluate(self, requests, ledger, *, reuse=True):
        self._cancel.clear()
        output = [None] * len(requests)
        unique = OrderedDict()
        for index, request in enumerate(requests):
            unique.setdefault(request.key, []).append(index)
        pending = list(unique.items())
        cursor = 0
        while cursor < len(pending):
            wave, positions = [], []
            while cursor < len(pending) and len(wave) < self.workers:
                key, indexes = pending[cursor]
                cursor += 1
                request = requests[indexes[0]]
                result = None
                if self._cancel.is_set():
                    result = RolloutResult.failed(request, "cancelled", "cancelled")
                elif time.monotonic() >= ledger.deadline:
                    ledger.exhausted = True
                    result = RolloutResult.failed(request, "budget_exhausted", "wall_budget")
                elif reuse and key in self._cache:
                    result = replace(self._cache[key], cache_hit=True, native_steps=0)
                    self._cache.move_to_end(key)
                    ledger.cache_hits += len(indexes)
                elif wave and ledger.native_steps + request.reservation > ledger.budget.max_native_steps:
                    # Dispatch reserved work before rejecting work that may fit
                    # after early-terminal runs return unused primitive budget.
                    cursor -= 1
                    break
                elif ledger.reserve(request.reservation):
                    wave.append(request)
                    positions.append(indexes)
                else:
                    result = RolloutResult.failed(request, "budget_exhausted", "compute_budget")
                if result is not None:
                    for index in indexes:
                        output[index] = result
            if wave:
                results = self._pool.evaluate(wave, ledger.deadline, self._cancel)
                for request, result, indexes in zip(wave, results, positions):
                    ledger.reconcile(request.reservation, result)
                    if result.valid and reuse and self.cache_size:
                        self._cache[request.key] = result
                        while len(self._cache) > self.cache_size:
                            self._cache.popitem(last=False)
                    output[indexes[0]] = result
                    for index in indexes[1:]:
                        output[index] = replace(result, cache_hit=True, native_steps=0)
                        ledger.cache_hits += 1
        return tuple(output)

    def rollout_batch(self, root, candidates, *, sample_seeds=(0,), continuation=True,
                      budget: RolloutBudget, search_budget=None, ledger=None, reuse=True):
        with self._operation():
            candidates = tuple(tuple(c) for c in candidates)
            seeds = tuple(sample_seeds)
            if not candidates or not seeds:
                raise ValueError("Candidates and sample_seeds must be nonempty")
            ledger = ledger or BudgetLedger(search_budget or SearchBudget())
            if len(candidates) > ledger.budget.max_candidates:
                raise ValueError("Candidate count exceeds max_candidates")
            # Sample-major scheduling gives each candidate the same early budget.
            requests = [self._request(root, c, s, continuation, budget) for s in seeds for c in candidates]
            return self._evaluate(requests, ledger, reuse=reuse)

    def rollout(self, root, candidate, *, sample_seed=0, continuation=True,
                budget: RolloutBudget, search_budget=None, reuse=True):
        return self.rollout_batch(root, [candidate], sample_seeds=[sample_seed],
            continuation=continuation, budget=budget, search_budget=search_budget, reuse=reuse)[0]

    def validate_root(self, root, *, timeout_seconds=60.0):
        with self._operation():
            request = self._request(root, (), 0, False,
                RolloutBudget(1, timeout_seconds), validate_only=True)
            return self._evaluate([request], BudgetLedger(SearchBudget(
                max_rollouts=1, max_native_steps=max(1, root.step), wall_seconds=timeout_seconds)),
                reuse=False)[0]

    def inspect_candidates(self, root, *, timeout_seconds=60.0, ledger=None):
        """Probe a runtime's candidate generator in a worker, never on the live policy."""
        with self._operation():
            request = self._request(root, (), 0, False,
                RolloutBudget(1, timeout_seconds), inspect_only=True)
            ledger = ledger or BudgetLedger(SearchBudget(max_rollouts=1,
                max_native_steps=max(1, root.step), wall_seconds=timeout_seconds))
            return self._evaluate([request], ledger)[0]

    def execute_first(self, episode, plan):
        with self._operation():
            root = self._roots.get(plan.root_id)
            if root is None or not episode.matches(root):
                raise StalePlanError("Live episode changed since this plan was evaluated")
            if plan.action is None:
                raise ValueError("No action proposed; caller must choose an explicit fallback")
            transition = episode.step(plan.action)
            del self._roots[root.identity]
            self._cache = OrderedDict((k, v) for k, v in self._cache.items() if v.root_id != root.identity)
            return transition

    def cancel(self):
        self._cancel.set()

    def execute_trajectory(self, episode, plan, *, on_step=None):
        """Execute a zero-cost successful witness, checking every state/transition.

        A mismatch raises before any further action. This verifies observed
        agreement, not a guarantee that an unexpected event cannot cause harm.
        """
        with self._operation():
            root = self._roots.get(plan.root_id)
            witness = plan.witness
            if root is None or not episode.matches(root):
                raise StalePlanError('Live episode changed since this trajectory was sampled')
            if (plan.status != 'ok' or witness is None or not witness.valid
                    or witness.root_id != root.identity or witness.success is not True
                    or witness.safe_success is not True or witness.total_cost != 0
                    or not witness.actions or tuple(plan.candidate or ()) != witness.actions
                    or len(witness.trace) != len(witness.actions)+1
                    or len(witness.costs) != len(witness.actions)
                    or len(witness.rewards) != len(witness.actions)):
                raise ValueError('A complete traced zero-cost successful witness is required')
            transitions = []
            try:
                for index, action in enumerate(witness.actions):
                    if digest(episode.runtime.execution_signature()) != witness.trace[index]:
                        raise TrajectoryMismatch(f'Execution state differs before action {index}')
                    step = episode.step(action)
                    transitions.append(step)
                    matches = (digest(episode.runtime.execution_signature()) == witness.trace[index+1]
                               and step.cost == witness.costs[index] and step.reward == witness.rewards[index]
                               and episode.done == (index == len(witness.actions)-1))
                    if on_step is not None:
                        on_step(episode, step, index, matches)
                    # Recheck after the observer callback before any further move.
                    if (not matches or digest(episode.runtime.execution_signature()) != witness.trace[index+1]
                            or episode.cost != root.cost + sum(witness.costs[:index+1])):
                        raise TrajectoryMismatch(f'Execution diverged after action {index}')
                if not transitions[-1].terminated or transitions[-1].success is not True or episode.cost != 0:
                    raise TrajectoryMismatch('Execution did not confirm zero-cost task success')
                return tuple(transitions)
            finally:
                self._roots.pop(root.identity, None)
                self._cache = OrderedDict((k, v) for k, v in self._cache.items() if v.root_id != root.identity)

    def close(self):
        self.cancel()
        with self._lock:
            if not self._closed:
                self._closed = True
                self._pool.close()
                self._cache.clear()
                self._roots.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
