import gzip
import json

import pytest

from safety_chores_rl.native.config import file_hash, source_manifest, validate_config
from safety_chores_rl.native.policy import UniformPolicy
from safety_chores_rl.native.ordered_profile import checked_image, ObservationIntegrityError


def test_failed_startup_cleanup_does_not_create_a_controller():
    from contextlib import ExitStack
    from safety_chores_rl.native.runtime import SafetyChoresRuntime
    class FailedSampler:
        @property
        def controller(self):
            pytest.fail('cleanup tried to start a simulator')
        def close(self):
            pytest.fail('upstream close would evaluate the lazy controller')
    runtime = SafetyChoresRuntime.__new__(SafetyChoresRuntime)
    runtime.policy = runtime.task = None
    runtime.sampler = FailedSampler()
    runtime._stack = ExitStack()
    runtime.end_episode()
    assert runtime.sampler is runtime.task is runtime._stack is None


def test_policy_replay_restores_history_and_random_stream():
    policies = [UniformPolicy(), UniformPolicy()]
    for policy in policies:
        policy.reset(['left', 'right', 'end'], 892)
        for executed in [1, 0, 1]:
            policy.propose({})
            policy.feedback(executed)
    assert policies[0].snapshot() == policies[1].snapshot()
    before_history = policies[0].snapshot()['history']
    policies[0].seed_suffix(123)
    assert policies[0].snapshot()['history'] == before_history
    assert policies[0].snapshot()['rng'] != policies[1].snapshot()['rng']
    assert policies[0].snapshot()['history'] == [1, 0, 1]


def native_config(tmp_path):
    repo = tmp_path/'external'
    repo.mkdir()
    (repo/'sampler.py').write_text('VALUE = 1\n')
    assets = tmp_path/'assets'
    assets.mkdir()
    config = dict(schema_version=1, task_type='FetchType', asset_root=str(assets),
                  safevla=dict(repo_path=str(repo), source_files=source_manifest(repo)))
    for name in ('benchmark', 'houses', 'build_manifest', 'player'):
        path = tmp_path/name
        if name == 'benchmark':
            with gzip.open(path, 'wt') as stream:
                stream.write(json.dumps(dict(task_type='FetchType', house_index=1))+'\n')
        else:
            path.write_bytes(b'fixture')
        config[name] = dict(path=str(path), sha256=file_hash(path))
    return config


@pytest.mark.parametrize('artifact', ['player', 'build_manifest', 'benchmark', 'houses'])
def test_native_config_rejects_artifact_changes(tmp_path, artifact):
    config = native_config(tmp_path)
    _, tasks = validate_config(config)
    assert tasks[0]['house_index'] == 1
    (tmp_path/artifact).write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        validate_config(config)


def test_native_config_rejects_added_external_source(tmp_path):
    config = native_config(tmp_path)
    (tmp_path/'external'/'extra.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='source manifest'):
        validate_config(config)


def test_camera_failure_preserves_both_arrays_and_still_raises(tmp_path):
    np = pytest.importorskip('numpy')
    current = np.zeros((4, 6, 3), dtype=np.uint8)
    sensor = current.copy()
    sensor[1, 2, 0] = 255
    with pytest.raises(ObservationIntegrityError, match='current ordered manip'):
        checked_image(current, sensor, camera='manip', step=6, out=tmp_path)
    evidence = np.load(tmp_path/'image-mismatch-6-manip.npz', allow_pickle=False)
    assert np.array_equal(evidence['current'], current)
    assert np.array_equal(evidence['sensor'], sensor)
    details = json.loads((tmp_path/'image-mismatch-6-manip.json').read_text())
    assert details['different_elements'] == 1
    assert details['current_sha256'] != details['sensor_sha256']


def test_capture_failure_cannot_hide_camera_failure(tmp_path, monkeypatch):
    np = pytest.importorskip('numpy')
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(np, 'savez_compressed', fail)
    with pytest.raises(ObservationIntegrityError, match='capture failed: OSError: disk full'):
        checked_image(np.zeros((2, 2)), np.zeros((3, 2)), camera='nav', step=1, out=tmp_path)


def test_matching_camera_check_does_not_write_or_mutate(tmp_path):
    np = pytest.importorskip('numpy')
    import hashlib
    current = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    before = current.copy()
    actual = checked_image(current, current.copy(), camera='nav', step=0, out=tmp_path)
    assert actual == hashlib.sha256(before.tobytes()).hexdigest()
    assert np.array_equal(current, before)
    assert list(tmp_path.iterdir()) == []
