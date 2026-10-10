"""Reconstruct selected native study branches, recording cameras and trajectories.

Use the exact runtime sources/configuration that produced the study. Each branch
starts a fresh episode, replays the live action prefix, forces its candidate, and
regenerates its sampled continuation. Saved actions, rewards/costs, termination,
and endpoint observations must match. Older studies lack per-step state hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image

from safety_chores_rl import Episode, RuntimeSpec
from safety_chores_rl.types import canonical, digest
from native_trajectory_replay import pose, scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--policy-config', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--query-step', type=int, required=True)
    parser.add_argument('--selection', nargs='+', required=True, help='Single candidate action and suffix seed, e.g. 0:4102')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--vulkan-device', type=int, default=0)
    args = parser.parse_args()
    source = json.loads(args.report.read_text())
    arm = next(a for a in source['arms'] if a['arm'] == 'search')
    if source['status'] != 'complete' or not arm['valid'] or not arm['replay_exact']:
        parser.error('A completed study with a validated search arm is required')
    query = next(q for q in arm['queries'] if q['step'] == args.query_step)
    available = [r for e in query['plan']['evaluations'] for r in e['results']]
    selections = [tuple(map(int, s.split(':'))) for s in args.selection]
    if not 2 <= len(selections) <= 4 or len(set(selections)) != len(selections):
        parser.error('Choose two to four distinct candidate:seed pairs')
    selected = [next(r for r in available if r['candidate'] == [action] and r['sample_seed'] == seed)
                for action, seed in selections]
    if any(r['status'] != 'valid' or not r['actions'] for r in selected):
        parser.error('Every selected branch must contain valid recorded actions')
    args.out.mkdir(parents=True, exist_ok=False)
    spec = RuntimeSpec.create('safety_chores_rl.native.runtime:SafetyChoresRuntime',
        config=json.loads(args.config.read_text()), vulkan_device=args.vulkan_device,
        policy_factory='safety_chores_rl.native.safevla_policy:SafeVLAPolicy',
        policy_config=json.loads(args.policy_config.read_text()),
        diagnostics_dir=str(args.out.resolve()/'diagnostics'))
    result = dict(schema_version=1, status='running', kind='native_branch_replay',
        original_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        task_id=source['task_id'], seed=source['seed'], episode_limit=source['episode_limit'],
        method='Selected saved BestOfN branches', query_step=args.query_step,
        original_query_root_id=query['plan']['root_id'],
        prefix_actions=arm['actions'][:args.query_step],
        prefix_costs=arm['costs'][:args.query_step],
        branches=[], available_branches=len(available),
        provenance='Fresh reset and forced live-prefix replay for each branch; candidate forced, continuation regenerated from the original suffix seed. Original actions, rewards/costs, terminal outcome, candidate probabilities, and endpoint observations checked. Reconstructed full roots match one another. Original intermediate state hashes and root snapshots were not saved.')
    started = time.monotonic()

    def save():
        result['elapsed_seconds'] = time.monotonic()-started
        temporary = args.out/'branches.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.out/'branches.json')

    def capture(env, branch, index):
        branch['poses'].append(pose(env.runtime, index))
        branch['state_hashes'].append(digest(env.runtime.snapshot()))
        for name, sensor in [('nav', 'raw_navigation_camera'), ('manip', 'raw_manipulation_camera')]:
            image = Image.fromarray(env.runtime._last_obs_raw[sensor]).convert('RGB')
            if hashlib.sha256(image.tobytes()).hexdigest() != branch['poses'][-1]['image_hashes'][name]:
                raise RuntimeError('Captured frame differs from the runtime camera hash')
            image.save(args.out/branch['frames']/f'{name}-{index:04d}.png')

    try:
        for number, original in enumerate(selected):
            branch = dict(candidate=original['candidate'], sample_seed=original['sample_seed'],
                actions=original['actions'], rewards=original['rewards'], costs=original['costs'],
                original_success=original['success'], original_cost=original['root_cost']+sum(original['costs']),
                stop_reason=original['stop_reason'], poses=[], state_hashes=[],
                frames=f'branch-{number}', verified=False, transitions_matched=0)
            result['branches'].append(branch)
            (args.out/branch['frames']).mkdir()
            with Episode(spec, task_id=source['task_id'], seed=source['seed'], max_steps=source['episode_limit']) as env:
                snapshot = env.runtime.snapshot()
                if snapshot['source_identity'] != arm['final_snapshot']['source_identity']:
                    raise RuntimeError('Use the exact runtime sources from the original study')
                if number == 0:
                    result.update(instruction=env.observation['instruction'], action_names=env.observation['actions'],
                                  scene=scene(env.runtime), source_identity=snapshot['source_identity'])
                capture(env, branch, 0)
                for index, action in enumerate(result['prefix_actions']):
                    transition = env.step(action)
                    if (env.done or transition.cost != arm['costs'][index]
                            or transition.reward != arm['rewards'][index]):
                        raise RuntimeError(f'Prefix transition differs at action {index+1}')
                    capture(env, branch, index+1)
                branch['root_snapshot_sha256'] = digest(env.runtime.snapshot())
                if env.cost != original['root_cost']:
                    raise RuntimeError('Reconstructed root cost differs')
                if number and branch['state_hashes'] != result['branches'][0]['state_hashes'][:args.query_step+1]:
                    raise RuntimeError('Fresh reset/prefix states differ between reconstructed branches')
                branch['root_matches_other_reconstructions'] = True
                save()
                for index, expected_action in enumerate(original['actions']):
                    transition = env.step(expected_action if index < len(original['candidate']) else None)
                    if index == 0:
                        expected_preview = query['preview']['endpoint']
                        if env.runtime.policy.probabilities != expected_preview['probabilities']:
                            raise RuntimeError('Reconstructed root candidate probabilities differ from the original preview')
                        branch['original_candidate_probabilities_matched'] = True
                    if index+1 == len(original['candidate']):
                        env.runtime.seed_suffix(original['sample_seed'])
                    if (transition.action != expected_action or transition.reward != original['rewards'][index]
                            or transition.cost != original['costs'][index]
                            or env.done != (index+1 == len(original['actions'])
                                            and (original['terminated'] or original['truncated']))):
                        raise RuntimeError(f'Branch {number} differs at continuation action {index+1}')
                    branch['transitions_matched'] += 1
                    capture(env, branch, args.query_step+index+1)
                    if index % 25 == 0:
                        print(json.dumps(dict(branch=number, step=index+1, cost=env.cost, matched=True)), flush=True)
                        save()
                endpoint = json.loads(original['endpoint_json']) if 'endpoint_json' in original else original['endpoint']
                if canonical(env.observation) != canonical(endpoint):
                    raise RuntimeError('Final observation differs from the original branch')
                success = transition.success if transition.terminated else False
                if (success != original['success'] or env.cost != branch['original_cost']
                        or transition.terminated != original['terminated']):
                    raise RuntimeError('Final outcome differs from the original branch')
                branch.update(verified=True, endpoint_matched=True,
                              original_endpoint_sha256=digest(endpoint), replay_endpoint_sha256=digest(env.observation))
                save()
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print(json.dumps(dict(status=result['status'], seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
