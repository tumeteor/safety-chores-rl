"""Synthetic rollout-count benchmark; not native simulator performance evidence."""
import argparse
import json
import platform
import time

from safety_chores_rl import (RuntimeSpec, Episode, RolloutSession, RolloutBudget,
                             BestOfN, SuccessiveHalving, reward_return)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--delay', type=float, default=0.01,
                        help='Artificial seconds per simulated action')
    args = parser.parse_args()
    spec = RuntimeSpec.create('safety_chores_rl.toy:ChoiceRuntime', delay=args.delay)
    report = dict(runtime='synthetic deterministic choice', python=platform.python_version(),
                  platform=platform.system(), workers=args.workers, delay_seconds=args.delay,
                  includes_worker_startup=True, native_simulator_benchmark=False)
    with Episode(spec) as env:
        for name, planner_type in [('exhaustive', BestOfN), ('halving', SuccessiveHalving)]:
            with RolloutSession(spec, workers=args.workers) as session:
                root = session.capture_root(env)
                start = time.perf_counter()
                plan = planner_type(session, objective=reward_return).plan(root,
                    [(i,) for i in range(8)], sample_seeds=range(8), rollout_budget=RolloutBudget(1))
                report[name] = dict(action=plan.action, status=plan.status,
                                    seconds=time.perf_counter()-start, **plan.usage)
                if plan.status != 'ok' or plan.action != 7:
                    raise RuntimeError('Synthetic known-winner check failed')
                if name == 'halving':
                    full = BestOfN(session, objective=reward_return).plan(root,
                        [(i,) for i in range(8)], sample_seeds=range(8), rollout_budget=RolloutBudget(1))
                    report['exhaustive_after_halving'] = full.usage
    report['rollout_reduction_fraction'] = 1-report['halving']['rollouts']/report['exhaustive']['rollouts']
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
