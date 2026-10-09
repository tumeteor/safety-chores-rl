"""Opt-in ordered-lighting profile, with current-frame and GPU error guards."""
import hashlib
import json
import os
from pathlib import Path

PROFILE = 'ordered_deferred_float32_capacity32_v7'
PHYSICS_PROFILE = 'high_friction_overlap_count_v1'


class ObservationIntegrityError(RuntimeError):
    pass


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def checked_image(current, sensor, *, camera, step, out):
    """Compare existing arrays; failure capture never refreshes a camera or retries."""
    import numpy as np
    if current.shape == sensor.shape and np.array_equal(current, sensor):
        return hashlib.sha256(current.tobytes()).hexdigest()
    evidence = Path(out)/f'image-mismatch-{step}-{camera}'
    capture_error = None
    try:
        np.savez_compressed(str(evidence)+'.npz', current=current, sensor=sensor)
        detail = dict(camera=camera, step=step, current_shape=list(current.shape),
                      sensor_shape=list(sensor.shape),
                      current_sha256=hashlib.sha256(current.tobytes()).hexdigest(),
                      sensor_sha256=hashlib.sha256(sensor.tobytes()).hexdigest())
        if current.shape == sensor.shape:
            detail['different_elements'] = int(np.count_nonzero(current != sensor))
        Path(str(evidence)+'.json').write_text(json.dumps(detail, indent=2)+'\n', encoding='utf-8')
    except Exception as exc:
        capture_error = f'{type(exc).__name__}: {exc}'
    raise ObservationIntegrityError(
        f'Policy sensor is not the current ordered {camera} image; '
        + (f'failure capture failed: {capture_error}' if capture_error else f'evidence: {evidence}'))


def validate_receipts(checks):
    if (not isinstance(checks, list) or len(checks) != 2
            or any(c.get('profile') != PROFILE or c.get('failure')
                   or c.get('error_counts') != [0, 0, 0, 0]
                   or c.get('duplicate_writes') != 0
                   or not 1 <= c.get('lightCount', 0) <= 32 for c in checks)):
        raise ObservationIntegrityError('Ordered renderer rejected its current frame: ' + str(checks))


def configure_process(out):
    """Local process only; CUDA device visibility does not select Vulkan's device."""
    os.environ['SAFETY_CHORES_STABLE_COLLISION_MESSAGES'] = '1'
    os.environ['SAFETY_CHORES_DETERMINISTIC_DITHERING'] = '1'
    os.environ['SAFETY_CHORES_STABLE_HIGH_FRICTION'] = '1'
    os.environ['SAFETY_CHORES_CANONICAL_COLLIDER_ORDER'] = '0'
    os.environ.pop('SAFETY_CHORES_SETTLE_TRACE_DIR', None)
    from ai2thor.controller import Controller
    original = Controller.unity_command
    device = int(os.environ['ORDERED_VULKAN_DEVICE'])
    if device < 0:
        raise ObservationIntegrityError('Vulkan device index must be nonnegative')

    def unity_command(self, width, height, headless):
        command = original(self, width, height, headless)
        if '-force-device-index' in command:
            index = command.index('-force-device-index')
            del command[index:index+2]
        command.extend(['-force-vulkan', '-force-device-index', str(device),
                        '-logFile', str(out/'unity.log')])
        (out/'unity-command.json').write_text(json.dumps(command, indent=2)+'\n')
        return command

    Controller.unity_command = unity_command


class OrderedProfile:
    def __init__(self, env, controller, task, out, report):
        self.env, self.raw, self.task, self.out, self.report = env, controller.controller, task, out, report
        self.checks = 0
        if task.num_steps_taken() != 0 or task.observation_history:
            raise ObservationIntegrityError('Ordered profile must precede task and policy history')
        appearance = self.command('GetObservationAppearanceDiagnostics').metadata['actionReturn']
        from .cost_state import COST_PROFILE
        if appearance.get('collision_message_profile') != COST_PROFILE:
            raise ObservationIntegrityError('Canonical collision message profile is not active')
        report['cost_profile'] = COST_PROFILE
        if appearance['graphics_api'] != 'Vulkan':
            raise ObservationIntegrityError('Requested Vulkan renderer not active')
        self.command('SetObservationOrderedLighting', enabled=True)
        self.command('RenderObservationDiagnostic')
        # Initialization only: invalidate the upstream step-zero cached observation.
        # Subsequent env.step calls own cache invalidation, as in upstream SafeVLA.
        task._observation_cache = None
        env._last_obs_raw = task.get_observations()
        report.update(observation_profile=PROFILE, physics_profile=PHYSICS_PROFILE, graphics_device=appearance['graphics_device'],
                      graphics_api=appearance['graphics_api'],
                      physical_vulkan_device=int(os.environ['ORDERED_VULKAN_DEVICE']))
        (out/'appearance.json').write_text(json.dumps(appearance, indent=2)+'\n')
        self.check(env.current_obs())

    def command(self, action, **kwargs):
        event = self.raw.step(action=action, **kwargs)
        if not event.metadata['lastActionSuccess']:
            raise ObservationIntegrityError(action + ': ' + str(event.metadata.get('errorMessage')))
        return event

    def check(self, obs):
        import numpy as np
        event = self.command('GetObservationDiagnostics')
        state = event.metadata['actionReturn']
        validate_friction(state)
        if state['deterministic_dithering'] is not True or state['physics_auto_simulation'] is not False:
            raise ObservationIntegrityError('Observation/physics operating mode changed')
        images = {}
        for camera, key, array in [('nav', 'raw_navigation_camera', event.frame),
                                  ('manip', 'raw_manipulation_camera', event.third_party_camera_frames[0])]:
            array = np.asarray(array)
            cutoff = round(array.shape[1]*6/396)
            current = array[:, cutoff:-cutoff, :3] if cutoff else array[:, :, :3]
            sensor = np.asarray(obs.raw[key])
            images[camera] = checked_image(current, sensor, camera=camera,
                                           step=self.task.num_steps_taken(), out=self.out)
        receipts = self.command('GetObservationOrderedLighting').metadata['actionReturn']
        validate_receipts(receipts)
        row = dict(step=self.task.num_steps_taken(), images=images,
                   exposed_state_hash=canonical_hash(state), profile=PROFILE, physics_profile=PHYSICS_PROFILE,
                   checks=[{k:c[k] for k in ('profile','failure','lightCount','error_counts','duplicate_writes')}
                           for c in receipts])
        from .cost_state import snapshot
        row['cost_state'] = snapshot(self.task)
        self.env._ordered_cost_state_hash = row['cost_state']['state_hash']
        with (self.out/'observation_checks.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.env._ordered_observation_state_hash = row['exposed_state_hash']
        self.env._ordered_observation_profile = PROFILE
        self.checks += 1
        self.report['observation_checks'] = self.checks
        return row


def require_root_match(actual, expected):
    fields = ('prefix_actions', 'progress', 'metadata', 'image_hashes',
              'policy_semantic_hashes', 'observation_state_hash', 'observation_profile', 'cost_state_hash')
    differences = [k for k in fields if k not in actual or k not in expected or actual[k] != expected[k]]
    if differences:
        raise ObservationIntegrityError('Replay root mismatch: ' + ', '.join(differences))


def validate_friction(state):
    if state.get('high_friction_profile') != PHYSICS_PROFILE or not isinstance(state.get('high_friction'), list):
        raise ObservationIntegrityError('Missing repaired high-friction state/profile')
    for body in state['high_friction']:
        if body.get('negative_counts') is not False or body.get('inactive_references') != 0 or body.get('high_values_match') is not True:
            raise ObservationIntegrityError('High-friction overlap/material mismatch: ' + str(body))


def audit_observations(folder, report):
    rows = [json.loads(s) for s in (folder/'observation_checks.jsonl').read_text().splitlines()]
    if (report.get('observation_profile') != PROFILE or report.get('physics_profile') != PHYSICS_PROFILE or report.get('graphics_api') != 'Vulkan'
            or report.get('observation_checks') != report['steps']+1
            or len(rows) != report['steps']+1 or [r['step'] for r in rows] != list(range(report['steps']+1))):
        raise ObservationIntegrityError('Missing ordered-profile observation receipts')
    for r in rows:
        validate_receipts(r['checks'])
        if r['profile'] != PROFILE or r.get('physics_profile') != PHYSICS_PROFILE or not r['exposed_state_hash']:
            raise ObservationIntegrityError('Missing observation identity')
    return rows
