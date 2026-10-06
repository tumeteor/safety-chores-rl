"""Independent continuation policy interface and a lightweight random baseline."""
import random
from typing import Protocol


class Policy(Protocol):
    def reset(self, action_names: list[str], seed: int) -> None: ...
    def propose(self, observation: dict) -> int: ...
    def feedback(self, action: int) -> None: ...
    def seed_suffix(self, seed: int) -> None: ...
    def snapshot(self) -> dict: ...
    def close(self) -> None: ...


class UniformPolicy:
    """Uniform random actions; a baseline for exercising the benchmark API."""
    def reset(self, action_names, seed):
        self.actions = list(action_names)
        if not self.actions:
            raise ValueError('Empty action vocabulary')
        self.rng = random.Random(seed)
        self.history = []

    def propose(self, observation):
        return self.rng.randrange(len(self.actions))

    def feedback(self, action):
        self.history.append(action)

    def seed_suffix(self, seed):
        self.rng.seed(seed)

    def snapshot(self):
        return dict(actions=self.actions, rng=self.rng.getstate(), history=list(self.history))

    def close(self):
        pass
