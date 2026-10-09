"""Small deterministic maze runtime for API examples; no robotics claims."""
import random
import time
from .types import Transition


class GridRuntime:
    def __init__(self, *, size=5, obstacles=(), hazards=(), goal=None, step_delay=0.0):
        if size < 2:
            raise ValueError("size must be >= 2")
        self.size, self.goal = size, tuple(goal or (size-1, size-1))
        self.obstacles, self.hazards = set(map(tuple, obstacles)), set(map(tuple, hazards))
        self.step_delay = step_delay

    def reset(self, task_id, seed, episode_limit):
        if task_id != 0:
            raise ValueError("GridRuntime has only task_id=0")
        self.pos, self.steps, self.cost, self.limit = (0, 0), 0, 0.0, episode_limit
        self.rng = random.Random(seed)
        self.history = []

    def advance(self, forced_action=None):
        proposal = self.rng.randrange(4)
        action = proposal if forced_action is None else forced_action
        if action not in range(4):
            raise ValueError("Grid actions are 0:right, 1:up, 2:left, 3:down")
        if self.step_delay:
            time.sleep(self.step_delay)
        dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[action]
        next_pos = (max(0, min(self.size-1, self.pos[0]+dx)), max(0, min(self.size-1, self.pos[1]+dy)))
        if next_pos not in self.obstacles:
            self.pos = next_pos
        self.steps += 1
        cost = float(self.pos in self.hazards)
        self.cost += cost
        self.history.append(action)
        success = self.pos == self.goal
        return Transition(action, 10.0 if success else -0.1, cost,
                          terminated=success, truncated=not success and self.steps >= self.limit,
                          success=True if success else False if self.steps >= self.limit else None)

    def seed_suffix(self, seed):
        self.rng.seed(seed)

    def snapshot(self):
        return dict(position=self.pos, steps=self.steps, cost=self.cost,
                    rng=self.rng.getstate(), policy_history=self.history,
                    task=dict(size=self.size, goal=self.goal, limit=self.limit,
                              obstacles=sorted(self.obstacles), hazards=sorted(self.hazards)))

    def observe(self):
        return dict(position=self.pos, goal=self.goal, steps=self.steps,
                    size=self.size, obstacles=sorted(self.obstacles))

    def execution_signature(self):
        state = self.snapshot()
        state.pop('rng')
        return state

    def end_episode(self):
        pass

    def close(self):
        pass


def goal_distance(result):
    """Explicit heuristic, not a measured safe-completion probability."""
    if not result.valid:
        return None
    observation = result.endpoint
    distance = sum(abs(a-b) for a, b in zip(observation["position"], observation["goal"]))
    return sum(result.rewards) - distance


class ChoiceRuntime:
    """One-step deterministic reward task for orchestration benchmarks only."""
    def __init__(self, delay=0.0):
        import math
        if not math.isfinite(delay) or delay < 0:
            raise ValueError('delay must be finite and nonnegative')
        self.delay = delay

    def reset(self, task_id, seed, episode_limit):
        self.rng = random.Random(seed)
        self.actions = []

    def advance(self, forced_action=None):
        import time
        proposal = self.rng.randrange(8)
        action = proposal if forced_action is None else forced_action
        if action not in range(8):
            raise ValueError('Action must be in 0..7')
        time.sleep(self.delay)
        self.actions.append(action)
        return Transition(action, float(action), 0.0, terminated=True, success=True)

    def snapshot(self):
        return dict(actions=self.actions, rng=self.rng.getstate())

    def observe(self):
        return dict(actions=list(self.actions))

    def seed_suffix(self, seed):
        self.rng.seed(seed)

    def end_episode(self):
        pass

    def close(self):
        pass
