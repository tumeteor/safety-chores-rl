import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('native_setup', Path(__file__).resolve().parents[1] / 'tools/setup_native.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


@pytest.fixture
def installation(tmp_path):
    root = tmp_path / 'installation with spaces'
    repo = root / 'SafeVLA'
    for name in ('tasks/multi_task_eval_sampler.py', 'training/online/dinov2_vits_tsfm_base.py'):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture\n')
    uid = 'a' * 32
    assets = root / 'objaverse_assets/2023_07_28/assets'
    (assets / uid).mkdir(parents=True)
    (assets / uid / f'{uid}.pkl.gz').write_bytes(b'asset')
    (assets / uid / 'thor_metadata.json').write_text('{}')
    for texture in ('albedo.jpg', 'normal.jpg', 'emission.jpg'):
        (assets / uid / texture).write_bytes(b'texture fixture')
    for path, rows in [
        (repo / 'benchmark/fetchtype_val.jsonl.gz', [dict(task_type='FetchType', house_index=0)] * 6),
        (root / 'houses_2023_07_28/val.jsonl.gz', [dict(objects=[dict(assetId='Fridge_24'), dict(assetId=uid)])]),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt') as stream:
            stream.write('\n'.join(json.dumps(row) for row in rows) + '\n')
    with gzip.open(assets.parent / 'annotations.json.gz', 'wt') as stream:
        json.dump({uid: {}}, stream)
    player = root / 'player/thor-Linux64-local'
    player.parent.mkdir()
    player.write_bytes(b'player')
    player.chmod(0o755)
    (player.parent / 'data.bin').write_bytes(b'other player data')
    manifest = dict(schema='safety_chores_checkpoint_build_v1', executableSha256=setup.file_hash(player),
                    artifacts={p.name: setup.file_hash(p) for p in player.parent.iterdir()})
    (player.parent / 'checkpoint-build.json').write_text(json.dumps(manifest))
    ckpt = root / 'custom.pt'
    ckpt.write_bytes(b'weights')
    args = setup.parser().parse_args(['--installation-root', str(root), '--checkpoint', str(ckpt),
                                      '--output', str(tmp_path / 'generated config')])
    return args, root


def test_existing_installation_is_wired_without_filtering_episodes(installation):
    args, _ = installation
    config, rows, task = setup.discover(args, {})
    assert len(rows) == 6  # The preflight's default task 0 does not shrink the task set.
    assert setup.verify_player(config) == 2
    assert setup.verify_assets(config, rows, [0, 5])['objaverse_assets'] == 1
    policy = setup.policy_config(args, config, task, {})
    env = setup.environment(config, args, {})
    original_sources = setup.source_manifest(config['safevla']['repo_path'])
    summary = dict(task_count=len(rows))
    out = setup.write_outputs(args, config, policy, env, summary)
    assert setup.write_outputs(args, config, policy, env, summary) == out
    assert setup.source_manifest(config['safevla']['repo_path']) == original_sources
    assert json.loads((out / 'policy.json').read_text())['checkpoint_sha256'] == hashlib.sha256(b'weights').hexdigest()
    assert 'PYTHONPATH=' in (out / 'env.sh').read_text()
    assert '"$@"' in (out / 'run_demo.sh').read_text()
    with pytest.raises(ValueError, match='Refusing to replace'):
        setup.write_outputs(args, config, {**policy, 'greedy': True}, env, summary)
    assert json.loads((out / 'policy.json').read_text())['greedy'] is False


def test_player_and_asset_failures_are_reported_before_setup(installation):
    args, root = installation
    config, rows, _ = setup.discover(args, {})
    (root / 'player/data.bin').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Player artifact'):
        setup.verify_player(config)
    with pytest.raises(ValueError, match='0..5'):
        setup.verify_assets(config, rows, [6])
    uid = 'a' * 32
    (Path(config['asset_root']) / uid / 'thor_metadata.json').unlink()
    with pytest.raises(ValueError, match='1 Objaverse assets missing'):
        setup.verify_assets(config, rows, [0])
    assert not args.output.exists()


def test_ambiguous_checkpoint_needs_an_explicit_choice(tmp_path):
    copies = [tmp_path / 'first.pt', tmp_path / 'second.pt']
    for index, path in enumerate(copies):
        path.write_bytes(f'weights-{index}'.encode())
    with pytest.raises(ValueError, match='Ambiguous checkpoint'):
        setup.choose('checkpoint', None, copies)
    assert setup.choose('checkpoint', copies[1], copies) == copies[1]
    copies[1].write_bytes(copies[0].read_bytes())
    assert setup.choose('checkpoint', None, copies) == copies[0]


def test_reused_native_config_does_not_repin_changed_sources(installation):
    args, root = installation
    config, _, _ = setup.discover(args, {})
    path = root / 'native.json'
    path.write_text(json.dumps(config))
    args.from_config = path
    (Path(config['safevla']['repo_path']) / 'tasks/multi_task_eval_sampler.py').write_text('CHANGED = True\n')
    with pytest.raises(ValueError, match='source manifest'):
        setup.discover(args, {})


def test_checkpoint_download_requires_expected_bytes_and_preserves_existing_file(tmp_path):
    source = tmp_path / 'remote.pt'
    source.write_bytes(b'weights')
    target = tmp_path / 'cache/model.pt'
    with pytest.raises(ValueError, match='SHA256'):
        setup.download_checkpoint(source.as_uri(), target, '0' * 64)
    assert not target.exists()
    assert not target.with_name('model.pt.download').exists()
    digest = setup.file_hash(source)
    assert setup.download_checkpoint(source.as_uri(), target, digest) == target
    source.write_bytes(b'new upstream bytes')
    assert setup.download_checkpoint(source.as_uri(), target, digest) == target
    assert target.read_bytes() == b'weights'
