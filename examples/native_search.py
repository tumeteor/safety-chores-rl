"""Bounded native plan/execute/replan probe, with explicit outcome reporting.

The default random continuation exercises the API; it is not a trained agent or
a task-success claim. All candidate evaluations can continue to the episode cap.
"""
import argparse
import json
from pathlib import Path
import time

from safety_chores_rl import (RuntimeSpec, Episode, RolloutSession, RolloutBudget,
                             SearchBudget, BestOfN, SuccessiveHalving)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=892)
    parser.add_argument('--vulkan-device', type=int, default=0)
    parser.add_argument('--candidates', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--live-steps', type=int, default=2)
    parser.add_argument('--episode-limit', type=int, default=200)
    parser.add_argument('--wall-seconds', type=float, default=600)
    parser.add_argument('--planner', choices=['exhaustive', 'halving'], default='exhaustive')
    parser.add_argument('--policy-factory', default='safety_chores_rl.native.policy:UniformPolicy')
    parser.add_argument('--policy-config', type=Path)
    args = parser.parse_args()
    if (not 1 <= args.samples <= 16 or not 1 <= args.live_steps <= args.episode_limit <= 1000
            or not 1 <= len(set(args.candidates)) <= 32 or args.wall_seconds <= 0):
        parser.error('Use 1..16 samples, 1..32 candidates, bounded live steps/cap, and positive wall time')
    spec = RuntimeSpec.create('safety_chores_rl.native.runtime:SafetyChoresRuntime',
        config=json.loads(args.config.read_text(encoding='utf-8')), vulkan_device=args.vulkan_device,
        policy_factory=args.policy_factory,
        policy_config=json.loads(args.policy_config.read_text(encoding='utf-8')) if args.policy_config else {})
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, task_id=args.task_id, seed=args.seed,
                  continuation=args.policy_factory, episode_limit=args.episode_limit,
                  rounds=[], task_success=None, stopped='not_started')
    started = time.monotonic()
    try:
        with Episode(spec, task_id=args.task_id, seed=args.seed, max_steps=args.episode_limit) as env, RolloutSession(spec) as session:
            planner = (BestOfN if args.planner == 'exhaustive' else SuccessiveHalving)(session)
            for _ in range(args.live_steps):
                wall = args.wall_seconds-(time.monotonic()-started)
                if wall <= 0:
                    report['stopped'] = 'wall_budget'
                    break
                root = session.capture_root(env)
                remaining = args.episode_limit-root.step
                try:
                    plan = planner.plan(root, [(a,) for a in args.candidates], sample_seeds=range(args.samples),
                        rollout_budget=RolloutBudget(remaining, timeout_seconds=min(180, wall), stop_on_cost=False),
                        search_budget=SearchBudget(max_rollouts=len(args.candidates)*args.samples,
                            max_native_steps=args.episode_limit*len(args.candidates)*args.samples,
                            wall_seconds=wall))
                    row = dict(plan=plan.to_dict(), live_unchanged=env.matches(root))
                    report['rounds'].append(row)
                    if plan.status != 'ok' or plan.action is None:
                        report['stopped'] = plan.status
                        break
                    transition = session.execute_first(env, plan)
                    row['executed_action'] = transition.action
                    row['live_steps'] = len(env.actions)
                    row['live_cost'] = env.cost
                    print(json.dumps(dict(round=len(report['rounds']), action=plan.action,
                        status=plan.status, usage=plan.usage)), flush=True)
                    if env.done:
                        report['task_success'] = transition.success if transition.terminated else False
                        report['stopped'] = 'terminal' if transition.terminated else 'episode_limit'
                        break
                    report['stopped'] = 'live_step_budget'
                finally:
                    if root.identity in session._roots:
                        session.release_root(root)
            report.update(live_actions=env.actions, live_cost=env.cost)
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(stopped=report['stopped'], task_success=report['task_success'],
                         report=str(args.out/'report.json'))))


if __name__ == '__main__':
    main()
