"""Bounded reset/replay validation on a caller-supplied native installation."""
import argparse
import json
from pathlib import Path
from .. import RuntimeSpec, Episode, RolloutSession, RolloutBudget, SearchBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=892)
    parser.add_argument('--vulkan-device', type=int, default=0)
    parser.add_argument('--prefix', type=int, nargs='*', default=[0])
    parser.add_argument('--candidates', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--horizon', type=int, default=2)
    args = parser.parse_args()
    if len(args.prefix) > 16 or len(args.candidates) > 4 or not 1 <= args.horizon <= 16:
        parser.error('Smoke limits: prefix <=16, candidates <=4, horizon 1..16')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    spec = RuntimeSpec.create('safety_chores_rl.native.runtime:SafetyChoresRuntime',
                              config=config, vulkan_device=args.vulkan_device)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(native_validation_passed=False, continuation='uniform_random',
                  task_id=args.task_id, seed=args.seed, prefix=args.prefix, horizon=args.horizon)
    try:
        with Episode(spec, task_id=args.task_id, seed=args.seed, max_steps=200) as env, RolloutSession(spec) as session:
            for action in args.prefix:
                env.step(action)
            root = session.capture_root(env)
            validation = session.validate_root(root, timeout_seconds=180)
            report['validation'] = validation.to_dict()
            if not validation.valid:
                raise RuntimeError('Native replay root did not validate: '+str(validation.error))
            results = session.rollout_batch(root, [(a,) for a in args.candidates], sample_seeds=[0, 1],
                budget=RolloutBudget(args.horizon, timeout_seconds=180, stop_on_cost=False),
                search_budget=SearchBudget(max_rollouts=8, max_native_steps=256, wall_seconds=900))
            report['results'] = [r.to_dict() for r in results]
            report['live_unchanged'] = env.matches(root)
            report['native_validation_passed'] = all(r.valid for r in results) and report['live_unchanged']
            if not report['native_validation_passed']:
                raise RuntimeError('Native smoke results invalid or live episode changed')
    finally:
        (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(native_validation_passed=True, report=str(args.out/'report.json'))))


if __name__ == '__main__':
    main()
