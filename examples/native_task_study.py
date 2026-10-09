"""Paired native baseline/search validation with exact full-action replay.

Runs one fixed task/seed pair. This is an experimental release gate, not a
performance claim. Both controllers use the same caller-supplied policy.
"""
import argparse
import json
from pathlib import Path
import time

from safety_chores_rl import Episode, RuntimeSpec, RolloutSession, RolloutBudget, SearchBudget, BestOfN
from safety_chores_rl.types import canonical


def run_arm(spec, args, search, result, save):
    with Episode(spec, task_id=args.task_id, seed=args.seed, max_steps=args.episode_limit) as env:
        with RolloutSession(spec, workers=args.workers) as session:
            for step in range(args.episode_limit):
                if search and step in args.query_steps:
                    root = session.capture_root(env)
                    try:
                        preview = session.inspect_candidates(root, timeout_seconds=args.branch_seconds)
                        if not preview.valid:
                            result['invalid_preview'] = preview.to_dict()
                            raise RuntimeError('Native candidate-preview replay failed')
                        choices = json.loads(preview.endpoint_json)['actions'][:args.candidates]
                        plan = BestOfN(session).plan(root, [(a,) for a in choices],
                            sample_seeds=args.sample_seeds,
                            rollout_budget=RolloutBudget(args.episode_limit-step,
                                timeout_seconds=args.branch_seconds, stop_on_cost=False),
                            search_budget=SearchBudget(max_rollouts=len(choices)*len(args.sample_seeds),
                                max_native_steps=args.episode_limit*len(choices)*len(args.sample_seeds),
                                wall_seconds=args.query_seconds))
                        result['queries'].append(dict(step=step, candidates=choices,
                            preview=preview.to_dict(), plan=plan.to_dict(), live_unchanged=env.matches(root)))
                        save()
                        if (plan.status != 'ok' or plan.action is None or not env.matches(root)
                                or any(not r.valid for e in plan.evaluations for r in e.results)):
                            raise RuntimeError('Incomplete or invalid native search evidence')
                        transition = session.execute_first(env, plan)
                    finally:
                        if root.identity in session._roots:
                            session.release_root(root)
                else:
                    transition = env.step()
                result['actions'].append(transition.action)
                result['costs'].append(transition.cost)
                result['rewards'].append(transition.reward)
                if step % 10 == 0:
                    print(json.dumps(dict(arm=result['arm'], step=step+1, cost=env.cost)), flush=True)
                    save()
                if env.done:
                    result['success'] = transition.success if transition.terminated else False
                    result['stop_reason'] = 'terminal' if transition.terminated else 'episode_limit'
                    break
            result.update(total_cost=env.cost, safe_success=bool(result['success'] and env.cost == 0),
                          final_snapshot=env.runtime.snapshot())
    # Use the original native/policy seeds and feed back every actual action.
    with Episode(spec, task_id=args.task_id, seed=args.seed, max_steps=args.episode_limit) as replay:
        replay_costs, replay_rewards = [], []
        for action in result['actions']:
            transition = replay.step(action)
            replay_costs.append(transition.cost)
            replay_rewards.append(transition.reward)
        snapshot = replay.runtime.snapshot()
        expected = result['final_snapshot']
        mismatches = [key for key in set(expected) | set(snapshot)
                      if canonical(expected.get(key)) != canonical(snapshot.get(key))]
        result['replay_mismatches'] = sorted(mismatches)
        result['replay_exact'] = (not mismatches and replay_costs == result['costs']
                                  and replay_rewards == result['rewards'] and replay.done
                                  and (transition.success if transition.terminated else False) == result['success'])
        result['valid'] = result['replay_exact']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--policy-config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--vulkan-device', type=int, default=3)
    parser.add_argument('--episode-limit', type=int, default=200)
    parser.add_argument('--query-steps', type=int, nargs='+', default=[0, 40])
    parser.add_argument('--sample-seeds', type=int, nargs='+', default=[4100, 4101, 4102, 4103])
    parser.add_argument('--candidates', type=int, default=3)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--branch-seconds', type=float, default=300)
    parser.add_argument('--query-seconds', type=float, default=2100)
    args = parser.parse_args()
    if (not 1 <= args.episode_limit <= 600 or not 1 <= args.workers <= 2
            or not 1 <= args.candidates <= 4 or len(set(args.sample_seeds)) != len(args.sample_seeds)
            or not 1 <= len(args.sample_seeds) <= 8
            or any(s < 0 or s >= args.episode_limit for s in args.query_steps)):
        parser.error('Use bounded distinct samples, candidates, query steps, and at most two workers')
    args.out.mkdir(parents=True, exist_ok=False)
    spec = RuntimeSpec.create('safety_chores_rl.native.runtime:SafetyChoresRuntime',
        config=json.loads(args.config.read_text()), vulkan_device=args.vulkan_device,
        policy_factory='safety_chores_rl.native.safevla_policy:SafeVLAPolicy',
        policy_config=json.loads(args.policy_config.read_text()), diagnostics_dir=str(args.out.resolve()/'diagnostics'))
    report = dict(schema_version=1, task_id=args.task_id, seed=args.seed, episode_limit=args.episode_limit,
                  query_steps=args.query_steps, candidates=args.candidates, sample_seeds=args.sample_seeds,
                  arms=[], integration_passed=False, status='running')
    started = time.monotonic()
    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        temporary = args.out/'report.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.out/'report.json')
    try:
        for search in (False, True):
            arm = dict(arm='search' if search else 'baseline', actions=[], costs=[], rewards=[],
                       queries=[], success=None, valid=False, replay_exact=False)
            report['arms'].append(arm)
            save()
            run_arm(spec, args, search, arm, save)
            save()
            if not arm['valid']:
                raise RuntimeError('Full-action replay failed')
        report['integration_passed'] = bool(report['arms'][1]['queries'])
        report['status'] = 'complete' if report['integration_passed'] else 'no_search_executed'
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                      error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    if not report['integration_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
