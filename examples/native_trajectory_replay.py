"""Recover world-coordinate paths from recorded rollouts, checking every saved hash.

Read cached event metadata only; do not issue diagnostic movements or renders.
Use the exact runtime package/configuration that produced the input report.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from safety_chores_rl import Episode, RuntimeSpec
from safety_chores_rl.types import digest


def pose(runtime, index):
    metadata = runtime.profile.raw.last_event.metadata
    agent = metadata['agent']
    arm = metadata.get('arm', {})
    hand = arm.get('handSphereCenter')
    return dict(step=index, position=copy.deepcopy(agent['position']),
                yaw=agent['rotation']['y'], hand=copy.deepcopy(hand),
                image_hashes=dict(runtime._last_check['images']))


def scene(runtime):
    info = runtime.task.task_info
    house = runtime.houses[int(info['house_index'])]
    target_ids = {oid for ids in info['broad_synset_to_object_ids'].values() for oid in ids}
    objects = runtime.profile.raw.last_event.metadata.get('objects', [])
    return dict(rooms=[dict(id=r['id'], room_type=r.get('roomType'),
                            floor_polygon=r['floorPolygon']) for r in house['rooms']],
                targets=[dict(id=o['objectId'], position=o['position']) for o in objects
                         if o['objectId'] in target_ids])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--policy-config', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--vulkan-device', type=int, default=0)
    args = parser.parse_args()
    source = json.loads(args.report.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    spec = RuntimeSpec.create('safety_chores_rl.native.runtime:SafetyChoresRuntime',
        config=json.loads(args.config.read_text()), vulkan_device=args.vulkan_device,
        policy_factory='safety_chores_rl.native.safevla_policy:SafeVLAPolicy',
        policy_config=json.loads(args.policy_config.read_text()),
        diagnostics_dir=str(args.out.resolve()/'diagnostics'))
    result = dict(schema_version=1, status='running', coordinate_system='Unity world x/z, metres; y is height',
                  provenance='Coordinates recovered by forced replay; every root/action checked against the original rollout signature.',
                  original_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                  task_id=source['task_id'], seed=source['seed'], branches=[])
    started = time.monotonic()
    def save():
        result['elapsed_seconds'] = time.monotonic()-started
        temporary = args.out/'trajectories.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.out/'trajectories.json')
    try:
        for evaluation in source['plan']['evaluations']:
            for original in evaluation['results']:
                if original['status'] != 'valid' or len(original['trace']) != len(original['actions'])+1:
                    raise ValueError('Every plotted rollout needs a complete valid state trace')
                branch = dict(candidate=original['candidate'], sample_seed=original['sample_seed'],
                              actions=original['actions'], original_success=original['success'],
                              original_cost=original['root_cost']+sum(original['costs']), poses=[], verified=False)
                result['branches'].append(branch)
                with Episode(spec, task_id=source['task_id'], seed=source['seed'],
                             max_steps=source['episode_limit']) as env:
                    if digest(env.runtime.execution_signature()) != original['trace'][0]:
                        raise RuntimeError('Replay root signature differs from the recorded rollout')
                    if 'scene' not in result:
                        result['scene'] = scene(env.runtime)
                        print(json.dumps(dict(metadata_keys=list(env.runtime.profile.raw.last_event.metadata),
                            arm_keys=list(env.runtime.profile.raw.last_event.metadata.get('arm', {})),
                            targets=len(result['scene']['targets']))), flush=True)
                    branch['poses'].append(pose(env.runtime, 0))
                    save()
                    for index, action in enumerate(original['actions']):
                        transition = env.step(action)
                        if index+1 == len(original['candidate']):
                            env.runtime.seed_suffix(original['sample_seed'])
                        if (digest(env.runtime.execution_signature()) != original['trace'][index+1]
                                or transition.reward != original['rewards'][index]
                                or transition.cost != original['costs'][index]
                                or env.done != (index+1 == len(original['actions'])
                                                and (original['terminated'] or original['truncated']))):
                            raise RuntimeError(f'Replay diverged after action {index+1}')
                        branch['poses'].append(pose(env.runtime, index+1))
                        if index % 20 == 0:
                            print(json.dumps(dict(candidate=branch['candidate'], step=index+1, matched=True)), flush=True)
                            save()
                    if transition.success != original['success'] or env.cost != branch['original_cost']:
                        raise RuntimeError('Final outcome differs from recorded rollout')
                    branch['verified'] = True
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
