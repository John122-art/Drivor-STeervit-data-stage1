#!/usr/bin/env python3
"""Preserve caption caches across the exact empty-GT validator-only fix.

Only the metadata's reviewed validator hash and per-scene cache_version change.
Metadata is backed up and committed last; interrupted migrations can be retried.
No VLM, images, GT, caption payload, rules or masks are modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parents[2]
OLD_VALIDATOR = 'ab28afd545f739ea4c3ed76c457c38ac86f819c16bae4d7f386cc3c3a8306c3c'
FIXED_VALIDATOR = '306be4641e4c6fd38559a6765e8f79bb25d88614a8f9c7b5ede34bff34fe525e'


def fingerprint(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path, data):
    # Different containers can all report PID 2 while sharing this filesystem.
    temporary = path.with_name(path.name + f'.shape_fix.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def plan_metadata(old, source_hashes):
    """Accept exactly one reviewed transition, not arbitrary source edits."""
    identity = {k: v for k, v in old.items() if k != 'cache_version'}
    if fingerprint(identity) != old['cache_version']:
        raise ValueError('Existing metadata fingerprint is inconsistent')
    previous = old['pipeline_source_hashes']
    if previous == source_hashes:
        return None
    if set(previous) != set(source_hashes):
        raise ValueError('Pipeline source file set changed; not a validator-only fix')
    changed = {k: (previous[k], source_hashes[k]) for k in previous if previous[k] != source_hashes[k]}
    if changed != {'validate_pilot.py': (OLD_VALIDATOR, FIXED_VALIDATOR)}:
        raise ValueError(f'Unapproved pipeline changes: {changed}; no caches modified')
    identity['pipeline_source_hashes'] = dict(source_hashes)
    return dict(identity, cache_version=fingerprint(identity))


def stopped(output):
    scripts = {'stage1_worker.py', 'generate_captions.py', 'build_grounding.py', 'validate_pilot.py'}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if any(Path(arg).name in scripts for arg in args if arg) and any(str(output) in arg for arg in args):
            raise RuntimeError(f'Process {entry.name} still uses this output; stop generation before migrating')


def source_hashes(meta):
    scripts = ROOT / 'scripts/data_generation'
    files = list(scripts.glob('*.py'))
    if not meta.get('stage1'):
        files += list((scripts / 'stage2').glob('*.py'))
    return {str(p.relative_to(scripts)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def check_cache(path, old_version, new_version):
    value = json.loads(path.read_bytes())
    if value.get('token') != path.stem or value.get('cache_version') not in {old_version, new_version}:
        raise ValueError(f'Unexpected caption identity: {path}')


def rewrite_cache(path, old_version, new_version):
    original = path.read_bytes()
    before = json.loads(original)
    if before.get('cache_version') == new_version:
        return path.name, 'already_updated', '', ''
    if before.get('cache_version') != old_version or before.get('token') != path.stem:
        raise ValueError(f'Caption changed during migration: {path}')
    pattern = rb'("cache_version"\s*:\s*")' + old_version.encode() + rb'(")'
    updated, count = re.subn(pattern, lambda m: m[1] + new_version.encode() + m[2], original, count=1)
    if count != 1 or json.loads(updated) != dict(before, cache_version=new_version):
        raise ValueError(f'Could not replace only cache_version: {path}')
    atomic_bytes(path, updated)
    return path.name, 'updated', hashlib.sha256(original).hexdigest(), hashlib.sha256(updated).hexdigest()


def migrate(output, workers, ranks=None):
    paths = sorted(set(output.glob('shards/rank_*/captions/metadata.json')) |
                   set(output.glob('*/shards/rank_*/captions/metadata.json')))
    if ranks is not None:
        allowed = {f'rank_{rank:04d}' for rank in ranks}
        paths = [path for path in paths if path.parent.parent.name in allowed]
    plans = []
    for path in paths:
        old = json.loads(path.read_bytes())
        new = plan_metadata(old, source_hashes(old))
        if new is not None:
            plans.append((path, old, new, sorted((path.parent / 'scenes').glob('*.json'))))
    if not plans:
        print('[cache fix] No validator-only cache upgrade needed.', flush=True)
        return
    stopped(output)
    # Preflight every rank before changing any file.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, old, new, files in plans:
            for start in range(0, len(files), 256):
                list(pool.map(lambda p: check_cache(p, old['cache_version'], new['cache_version']), files[start:start+256]))
            print(f'[cache fix] Preflight OK: {path.parent.parent.name}, {len(files)} scenes', flush=True)
    for path, old, new, files in plans:
        backup = path.parent / 'metadata.before_empty_gt_fix.json'
        if backup.exists():
            if json.loads(backup.read_bytes()) != old:
                raise ValueError(f'Unexpected migration backup: {backup}')
        else:
            atomic_bytes(backup, path.read_bytes())
        journal = path.parent / 'empty_gt_fix_versions.json'
        atomic_json(journal, {'old_metadata': old, 'new_metadata': new, 'scene_count': len(files),
            'completed': False, 'meaning': 'Only validator hash/cache_version changes; original caption payload and all labels retained'})
        with (path.parent / 'empty_gt_fix_file_hashes.csv').open('a', newline='') as handle:
            writer = csv.writer(handle)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(0, len(files), 256):
                    for result in pool.map(lambda p: rewrite_cache(p, old['cache_version'], new['cache_version']), files[start:start+256]):
                        writer.writerow(result)
                    handle.flush()
                    if start % 2048 == 0:
                        print(f'[cache fix] {path.parent.parent.name}: {min(start+256,len(files))}/{len(files)}', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(files), 256):
                list(pool.map(lambda p: check_cache(p, new['cache_version'], new['cache_version']), files[start:start+256]))
        atomic_json(path, new)
        journal_data = json.loads(journal.read_bytes())
        journal_data['completed'] = True
        atomic_json(journal, journal_data)
        print(f'[cache fix] Done: {path.parent.parent.name}, {len(files)} caches preserved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--ranks', type=int, nargs='+', help='Optionally migrate only these independent rank directories')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('workers must be between 1 and 32')
    migrate(args.output.resolve(), args.workers, args.ranks)
