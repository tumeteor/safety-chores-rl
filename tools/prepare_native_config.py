"""Pin a local simulator installation without any controller or model config."""
import argparse
import json
from pathlib import Path

from safety_chores_rl.native.config import file_hash, source_manifest, validate_config, TASK_TYPES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--safevla-repo', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--task-type', choices=sorted(TASK_TYPES), default='FetchType')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    for name in ('benchmark', 'houses', 'build-manifest', 'player'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = dict(schema_version=1, task_type=args.task_type, split=args.split,
                  asset_root=str(args.asset_root.resolve()),
                  safevla=dict(repo_path=str(args.safevla_repo.resolve()),
                               source_files=source_manifest(args.safevla_repo.resolve())))
    for name in ('benchmark', 'houses', 'build_manifest', 'player'):
        path = getattr(args, name).resolve(strict=True)
        config[name] = dict(path=str(path), sha256=file_hash(path))
    config, tasks = validate_config(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(task_count=len(tasks), config=str(args.output))))


if __name__ == '__main__':
    main()
