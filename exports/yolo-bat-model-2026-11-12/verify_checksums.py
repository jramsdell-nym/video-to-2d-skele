#!/usr/bin/env python3
"""Verify the package's SHA-256 manifest using only the Python standard library."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
errors = []
entries = (root / 'SHA256SUMS.txt').read_text().splitlines()
for entry in entries:
    expected, name = entry.split('  ', 1)
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        errors.append(name + ': missing or invalid path')
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append(name + ': checksum mismatch')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'PASS: {len(entries)} files match SHA256SUMS.txt')
