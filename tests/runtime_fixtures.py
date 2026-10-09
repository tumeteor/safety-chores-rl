import multiprocessing
import random
import time
from safety_chores_rl.toy import GridRuntime
from safety_chores_rl import Transition


class BanditRuntime:
    def __init__(self, *, delay=0.0, fail_action=None, drift=False):
        self.delay, self.fail_action, self.drift = delay, fail_action, drift

    def reset(self, task_id, seed, episode_limit):
        self.rng = random.Random(seed)
        self.actions = []
        self.limit = episode_limit

    def snapshot(self):
        result = dict(actions=self.actions, rng=self.rng.getstate())
        if self.drift:
            result['process'] = multiprocessing.current_process().name
        return result

    def observe(self):
        return dict(actions=list(self.actions))

    def advance(self, forced_action=None):
        proposal = self.rng.randrange(8)
        action = proposal if forced_action is None else forced_action
        time.sleep(self.delay)
        if action == self.fail_action:
            raise RuntimeError('injected infrastructure error')
        self.actions.append(action)
        return Transition(action, float(action), 0.0, terminated=True, success=True)

    def seed_suffix(self, seed):
        self.rng.seed(seed)

    def end_episode(self):
        pass

    def close(self):
        pass


class SeedTimingRuntime(GridRuntime):
    def seed_suffix(self, seed):
        if len(self.history) != 2:
            raise RuntimeError('suffix seed before complete prefix feedback')
        super().seed_suffix(seed)


class InspectableGridRuntime(GridRuntime):
    def candidate_preview(self):
        return dict(actions=[0, 1, 2, 3])


class DescendantRuntime(BanditRuntime):
    def __init__(self, pid_path):
        super().__init__()
        self.pid_path = pid_path

    def advance(self, forced_action=None):
        import subprocess
        import sys
        from pathlib import Path
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        Path(self.pid_path).write_text(str(child.pid))
        time.sleep(60)
        return super().advance(forced_action)
