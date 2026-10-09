"""Runtime protocol and isolated replay execution; no simulator-specific imports."""
from __future__ import annotations

from dataclasses import replace
import importlib
import json
import time
import uuid
from typing import Protocol

from .types import Root, RuntimeSpec, Transition, RolloutRequest, RolloutResult, canonical, integer


class Runtime(Protocol):
    """A worker owns this object; reset must reconstruct a fresh episode.

    advance always performs one policy forward before the forced/chosen action,
    and records the actual executed action afterwards. snapshot and observe must
    not step physics, run policy inference or consume randomness. The public
    observation must exclude audit-only privileged state.
    """
    def reset(self, task_id: int, seed: int, episode_limit: int) -> None: ...
    def advance(self, forced_action: int | None = None) -> Transition: ...
    def seed_suffix(self, seed: int) -> None: ...
    def snapshot(self) -> dict: ...
    def observe(self) -> dict: ...
    def end_episode(self) -> None: ...
    def close(self) -> None: ...


def build_runtime(spec: RuntimeSpec) -> Runtime:
    module, name = spec.factory.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    return factory(**json.loads(spec.config_json))


class Episode:
    """Live runtime owner; search workers never receive this mutable instance."""
    def __init__(self, spec: RuntimeSpec, *, task_id=0, seed=0, max_steps=200):
        integer(task_id, "task_id")
        integer(seed, "seed")
        integer(max_steps, "max_steps", 1)
        self.spec, self.task_id, self.seed, self.max_steps = spec, task_id, seed, max_steps
        self.episode_id = uuid.uuid4().hex
        self.actions = []
        self.cost = 0.0
        self.done = False
        self.closed = False
        self.runtime = build_runtime(spec)
        try:
            self.runtime.reset(task_id, seed, max_steps)
        except BaseException:
            self.runtime.close()
            raise

    @property
    def observation(self):
        return self.runtime.observe()

    def step(self, action=None):
        if self.closed or self.done:
            raise RuntimeError("Episode is closed or finished")
        if action is not None:
            integer(action, "action")
        try:
            transition = self.runtime.advance(action)
            if action is not None and transition.action != action:
                raise RuntimeError("Runtime did not execute the requested action")
        except BaseException:
            self.done = True  # It may already have moved; do not continue silently.
            raise
        self.actions.append(transition.action)
        self.cost += transition.cost
        self.done = transition.terminated or transition.truncated or len(self.actions) >= self.max_steps
        return transition

    def matches(self, root: Root):
        return (not self.closed and not self.done and self.episode_id == root.episode_id
                and self.spec.identity == root.runtime_id
                and self.task_id == root.task_id and self.seed == root.seed
                and self.max_steps == root.episode_limit and tuple(self.actions) == root.prefix
                and self.cost == root.cost and canonical(self.runtime.snapshot()) == root.snapshot_json)

    def close(self):
        if not self.closed:
            self.closed = True
            self.runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def execute(runtime: Runtime, request: RolloutRequest) -> RolloutResult:
    """Replay an immutable root, validate it, then run the candidate and suffix."""
    started = time.monotonic()
    root = request.root
    replay_steps, actions, rewards, costs = 0, [], [], []
    checks = {}
    base = dict(request_id=request.key, root_id=root.identity, candidate=request.candidate,
                sample_seed=request.sample_seed, root_cost=root.cost)
    result = None
    try:
        runtime.reset(root.task_id, root.seed, root.episode_limit)
        replay_cost = 0.0
        for index, action in enumerate(root.prefix):
            replay_steps += 1
            step = runtime.advance(action)
            replay_cost += step.cost
            if step.action != action or step.terminated or step.truncated:
                raise ValueError(f"Replay diverged/terminated at prefix step {index}")
        actual = canonical(runtime.snapshot())
        expected = json.loads(root.snapshot_json)
        observed = json.loads(actual)
        fields = sorted(set(expected) | set(observed))
        mismatches = [key for key in fields if key not in expected or key not in observed
                      or canonical(expected[key]) != canonical(observed[key])]
        if replay_cost != root.cost:
            mismatches.append("accumulated_cost")
        checks = {"checked_fields": fields, "mismatches": mismatches,
                  "runtime_id": root.runtime_id, "exact_hidden_state_claim": False}
        if mismatches:
            result = RolloutResult(**base, status="root_mismatch", stop_reason="root_mismatch")
        elif request.inspect_only:
            result = RolloutResult(**base, status="valid", stop_reason="candidate_preview",
                                   endpoint_json=canonical(runtime.candidate_preview()))
        elif request.validate_only:
            result = RolloutResult(**base, status="valid", stop_reason="validated",
                                   endpoint_json=canonical(runtime.observe()))
        else:
            last = None
            stop = "horizon"
            if request.budget.stop_on_cost and root.cost > 0:
                stop = "positive_cost"
            else:
                for index in range(request.budget.max_steps):
                    if index >= len(request.candidate) and not request.continuation:
                        stop = "prefix_complete"
                        break
                    forced = request.candidate[index] if index < len(request.candidate) else None
                    step = runtime.advance(forced)
                    if forced is not None and step.action != forced:
                        raise ValueError("Candidate action was not executed")
                    actions.append(step.action)
                    rewards.append(step.reward)
                    costs.append(step.cost)
                    last = step
                    if len(actions) == len(request.candidate):
                        # Fork continuation randomness after candidate action feedback.
                        runtime.seed_suffix(request.sample_seed)
                    if step.terminated:
                        stop = "terminal"
                        break
                    if step.truncated or root.step + len(actions) >= root.episode_limit:
                        stop = "episode_limit"
                        break
                    if request.budget.stop_on_cost and root.cost + sum(costs) > 0:
                        stop = "positive_cost"
                        break
                else:
                    if not request.continuation:
                        stop = "prefix_complete"
            success = (last.success if last and last.terminated else
                       False if stop == "episode_limit" else None)
            total_cost = root.cost + sum(costs)
            safe = False if total_cost > 0 else success
            result = RolloutResult(**base, status="valid", stop_reason=stop,
                terminated=bool(last and last.terminated),
                truncated=stop in {"horizon", "episode_limit", "prefix_complete", "positive_cost"},
                success=success, safe_success=safe, endpoint_json=canonical(runtime.observe()))
    except Exception as exc:
        result = RolloutResult(**base, status="error", stop_reason="runtime_error",
                               error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            runtime.end_episode()
        except Exception as exc:
            result = RolloutResult(**base, status="error", stop_reason="cleanup_error",
                                   error=f"{type(exc).__name__}: {exc}")
    # A failed advance may have moved the simulator before raising. Its exact
    # work is unknown, so do not refund the reservation for an invalid execution.
    native_steps = request.reservation if result.status == 'error' else replay_steps + len(actions)
    return replace(result, replay_steps=replay_steps, native_steps=native_steps,
                   actions=tuple(actions), rewards=tuple(rewards), costs=tuple(costs),
                   checks_json=canonical(checks), elapsed_seconds=time.monotonic() - started)
