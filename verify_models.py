#!/usr/bin/env python3
"""Verify every archived model and package file against its imported SHA-256."""
import argparse
import json
from batting.video import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--defaults-only', action='store_true', help='Verify only the current PT/ONNX pair')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    wanted = {spec[key] for spec in manifest['defaults'].values() for key in ('checkpoint', 'onnx')}
    failures = []
    checked = 0
    for entry in manifest['files']:
        if args.defaults_only and entry['path'] not in wanted:
            continue
        path = ROOT / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            failures.append(entry['path'])
        checked += 1
    if failures:
        print('Missing or changed files (run git lfs pull):\n' + '\n'.join(failures))
        raise SystemExit(1)
    print(f'Verified {checked} model/package files.')


if __name__ == '__main__':
    main()
