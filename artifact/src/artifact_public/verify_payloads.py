"""Stream SHA256 over pinned numeric payloads; never load arrays into RAM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root, expected):
    entries = expected['files']
    if isinstance(entries, dict):
        entries = [dict(path=name, **entry) for name, entry in entries.items()]
    results = []
    for entry in entries:
        rel = PurePosixPath(entry['path'])
        if rel.is_absolute() or '..' in rel.parts or '\\' in str(rel):
            raise ValueError('unsafe expectation path')
        path = Path(root).joinpath(*rel.parts)
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError(f'missing payload or size mismatch: {rel}')
        digest = sha256(path)
        if digest != entry['sha256']:
            raise ValueError(f'SHA256 mismatch: {rel}')
        results.append(dict(path=str(rel), bytes=entry['bytes'], sha256=digest))
    return {'schema_version': 1, 'all_payload_hashes_equal': True, 'payload_count': len(results),
            'files': results, 'scope': expected.get('scope'),
            'does_not_claim_equal_generated_timestamps_paths_or_archive_compression': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    result = verify(args.root, json.loads(args.expected.read_text()))
    result['expectation_sha256'] = sha256(args.expected)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'PASS', 'payloads': result['payload_count'], 'receipt': str(args.receipt)}))


if __name__ == '__main__':
    main()
