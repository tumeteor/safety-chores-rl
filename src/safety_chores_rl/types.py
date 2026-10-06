"""Versioned, immutable request identities and portable result records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from functools import cached_property
from typing import Any

API_VERSION = 1


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class RuntimeSpec:
    """An importable factory and frozen JSON configuration, safe to spawn."""
    factory: str
    config_json: str = "{}"

    def __post_init__(self):
        if ":" not in self.factory or not isinstance(json.loads(self.config_json), dict):
            raise ValueError("Use 'package.module:factory' and a JSON object")
        object.__setattr__(self, "config_json", canonical(json.loads(self.config_json)))

    @classmethod
    def create(cls, factory: str, **config):
        return cls(factory, canonical(config))

    @cached_property
    def identity(self) -> str:
        return digest([API_VERSION, self.factory, self.config_json])


@dataclass(frozen=True)
class Transition:
    action: int
    reward: float
    cost: float
    terminated: bool = False
    truncated: bool = False
    success: bool | None = None

    def __post_init__(self):
        integer(self.action, "action")
        if not math.isfinite(self.reward) or not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("Reward must be finite; cost must be finite and nonnegative")
        if self.success is not None and type(self.success) is not bool:
            raise ValueError("success must be bool or None")
        if self.success is True and not self.terminated:
            raise ValueError("Successful completion must terminate the task")


@dataclass(frozen=True)
class Root:
    owner: str
    episode_id: str
    runtime_id: str
    task_id: int
    seed: int
    episode_limit: int
    prefix: tuple[int, ...]
    cost: float
    snapshot_json: str
    observation_json: str

    @cached_property
    def identity(self):
        return digest(asdict(self))

    @property
    def step(self):
        return len(self.prefix)

    @property
    def observation(self):
        return json.loads(self.observation_json)


@dataclass(frozen=True)
class RolloutBudget:
    max_steps: int
    timeout_seconds: float = 60.0
    stop_on_cost: bool = True

    def __post_init__(self):
        integer(self.max_steps, "max_steps", 1)
        positive(self.timeout_seconds, "timeout_seconds")


@dataclass(frozen=True)
class SearchBudget:
    max_rollouts: int = 128
    max_native_steps: int = 20000
    wall_seconds: float = 60.0
    max_candidates: int = 1024

    def __post_init__(self):
        for name in ("max_rollouts", "max_native_steps", "max_candidates"):
            integer(getattr(self, name), name, 1)
        positive(self.wall_seconds, "wall_seconds")


@dataclass(frozen=True)
class RolloutRequest:
    root: Root
    candidate: tuple[int, ...]
    sample_seed: int
    continuation: bool
    budget: RolloutBudget
    validate_only: bool = False
    inspect_only: bool = False

    @cached_property
    def key(self):
        return digest([API_VERSION, self.root.identity, self.candidate, self.sample_seed,
                       self.continuation, asdict(self.budget), self.validate_only, self.inspect_only])

    @property
    def reservation(self):
        suffix = self.budget.max_steps if self.continuation else len(self.candidate)
        return self.root.step + (0 if self.validate_only or self.inspect_only else suffix)


@dataclass(frozen=True)
class RolloutResult:
    request_id: str
    root_id: str
    candidate: tuple[int, ...]
    sample_seed: int
    status: str
    stop_reason: str
    actions: tuple[int, ...] = ()
    rewards: tuple[float, ...] = ()
    costs: tuple[float, ...] = ()
    replay_steps: int = 0
    native_steps: int = 0
    root_cost: float = 0.0
    terminated: bool = False
    truncated: bool = False
    success: bool | None = None
    safe_success: bool | None = None
    endpoint_json: str = "{}"
    checks_json: str = "{}"
    error: str | None = None
    elapsed_seconds: float = 0.0
    cache_hit: bool = False
    schema_version: int = API_VERSION

    @property
    def valid(self):
        return self.status == "valid"

    @property
    def endpoint(self):
        return json.loads(self.endpoint_json)

    @property
    def total_cost(self):
        return self.root_cost + sum(self.costs)

    def to_dict(self):
        result = asdict(self)
        result["endpoint"] = result.pop("endpoint_json")
        result["endpoint"] = json.loads(result["endpoint"])
        result["checks"] = json.loads(result.pop("checks_json"))
        return result

    @classmethod
    def failed(cls, request, status, reason, *, native_steps=0):
        return cls(request.key, request.root.identity, request.candidate,
                   request.sample_seed, status, reason, root_cost=request.root.cost,
                   native_steps=native_steps)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: tuple[int, ...]
    score: float | None
    standard_error: float | None
    samples: int
    results: tuple[RolloutResult, ...]


@dataclass(frozen=True)
class Plan:
    root_id: str
    candidate: tuple[int, ...] | None
    status: str
    evaluations: tuple[CandidateEvaluation, ...]
    usage: dict[str, Any] = field(default_factory=dict)
    method: str = "best_of_n"

    @property
    def action(self):
        return self.candidate[0] if self.candidate else None

    def to_dict(self):
        return asdict(self)
