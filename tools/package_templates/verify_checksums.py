#!/usr/bin/env python3
"""Verify all distributed package files using only Python's standard library."""
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    count = 0
    for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit(f'Missing or unsafe package path: {name}')
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        if digest != expected:
            raise SystemExit(f'Checksum mismatch: {name}')
        count += 1
    print(f'Verified {count} package files.')


if __name__ == '__main__':
    main()
