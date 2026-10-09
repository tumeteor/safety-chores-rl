"""Find one successful native continuation, then execute it with state checks."""
import argparse
import json
from pathlib import Path
import time

from safety_chores_rl import (Episode, RuntimeSpec, RolloutSession, RolloutBudget,
    SearchBudget, FirstSuccess)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--policy-config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1909)
    parser.add_argument('--vulkan-device', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--candidates', type=int, default=2)
    parser.add_argument('--sample-seeds', type=int, nargs='+', default=[4100, 4101, 4102, 4103])
    parser.add_argument('--episode-limit', type=int, default=200)
    parser.add_argument('--save-frames', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 2 or not 1 <= args.candidates <= 4 or len(args.sample_seeds) > 8:
        parser.error('Use at most two workers, four candidates, and eight sample seeds')
    args.out.mkdir(parents=True, exist_ok=False)
    spec = RuntimeSpec.create('safety_chores_rl.native.runtime:SafetyChoresRuntime',
        config=json.loads(args.config.read_text()), vulkan_device=args.vulkan_device,
        policy_factory='safety_chores_rl.native.safevla_policy:SafeVLAPolicy',
        policy_config=json.loads(args.policy_config.read_text()), diagnostics_dir=str(args.out.resolve()/'diagnostics'))
    report = dict(schema_version=1, method='first_success', task_id=args.task_id, seed=args.seed,
                  episode_limit=args.episode_limit, sample_seeds=args.sample_seeds, workers=args.workers,
                  execution=[], success=None, cost=None, status='running')
    started = time.monotonic()
    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        temporary = args.out/'report.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.out/'report.json')
    def frame(env, index):
        if args.save_frames:
            from PIL import Image
            folder = args.out/'frames'
            folder.mkdir(exist_ok=True)
            # Save original current simulator images without re-rendering.
            for name, sensor in [('nav', 'raw_navigation_camera'), ('manip', 'raw_manipulation_camera')]:
                Image.fromarray(env.runtime._last_obs_raw[sensor]).save(folder/f'{name}-{index:04d}.png')
    try:
        with Episode(spec, task_id=args.task_id, seed=args.seed, max_steps=args.episode_limit) as env:
            report['instruction'] = env.observation['instruction']
            report['action_names'] = env.observation['actions']
            frame(env, 0)
            report['live_reset_seconds'] = time.monotonic()-started
            with RolloutSession(spec, workers=args.workers) as session:
                root = session.capture_root(env)
                before = time.monotonic()
                preview = session.inspect_candidates(root, timeout_seconds=420)
                report['preview'] = preview.to_dict()
                report['preview_seconds'] = time.monotonic()-before
                if not preview.valid:
                    raise RuntimeError('Candidate preview did not reconstruct the root')
                choices = preview.endpoint['actions'][:args.candidates]
                report['candidate_actions'] = choices
                plan = FirstSuccess(session).plan(root, [(a,) for a in choices],
                    sample_seeds=args.sample_seeds,
                    rollout_budget=RolloutBudget(args.episode_limit, timeout_seconds=420),
                    search_budget=SearchBudget(max_rollouts=len(choices)*len(args.sample_seeds),
                        max_native_steps=args.episode_limit*len(choices)*len(args.sample_seeds), wall_seconds=1800))
                report['plan'] = plan.to_dict()
                report['live_unchanged'] = env.matches(root)
                save()
                if plan.witness is None:
                    report['status'] = plan.status
                    raise SystemExit(2)
                before = time.monotonic()
                def observed(episode, transition, index, matched):
                    report['execution'].append(dict(step=index+1, action=transition.action,
                        reward=transition.reward, cost=transition.cost, matched=matched,
                        terminated=transition.terminated, success=transition.success))
                    frame(episode, index+1)
                    if index % 10 == 0:
                        print(json.dumps(dict(step=index+1, matched=matched, cost=episode.cost)), flush=True)
                        save()
                session.execute_trajectory(env, plan, on_step=observed)
                report.update(status='complete', success=True, cost=env.cost,
                              execution_seconds=time.monotonic()-before, final_snapshot=env.runtime.snapshot())
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print(json.dumps(dict(status=report['status'], success=report['success'],
                         seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
