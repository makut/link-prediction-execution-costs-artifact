"""Official Citation2 ZIP -> unchanged preparer -> 19 exact payload SHA256 checks.

Plan-only by default. Run on a large CPU host after timed experiments finish.
No YT, GPU, cluster lifecycle, checkpoint loading, or training is performed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .download import SOURCES, download
from .prepare_friendster_full import memory_available
from .verify_payloads import sha256, verify

ARTIFACT_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-parent', type=Path, required=True)
    parser.add_argument('--expected', type=Path, default=ARTIFACT_ROOT / 'metadata/citation2_prepared_expected.json')
    parser.add_argument('--code-root', type=Path, default=ARTIFACT_ROOT / 'src')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--memory-budget-gib', type=float, default=64)
    parser.add_argument('--archive-existing', type=Path,
                        help='fresh official ZIP already streamed to this host; never copied or downloaded again')
    parser.add_argument('--official-download-receipt', type=Path,
                        help='required with --archive-existing: URL, bytes, SHA256, source_kind=fresh_official_public_download')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if bool(args.archive_existing) != bool(args.official_download_receipt):
        parser.error('--archive-existing and --official-download-receipt must be supplied together')
    if not 1 <= args.threads <= 64:
        parser.error('threads must be in 1..64')
    if not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib < 32:
        parser.error('finite memory budget of at least32GiB required')
    code = args.code_root.resolve()
    preparer = code / 'experiment_configs/seal_revision_2026-11/prepare_citation2.py'
    expected = json.loads(args.expected.read_text())
    if len(expected['files']) != 19:
        raise ValueError('expected exactly 19 pinned payloads')
    hashes = expected.get('preparer_source_sha256', {})
    if len(hashes) != 4:
        raise ValueError('four pinned preparation sources are required')
    for name, digest in hashes.items():
        if sha256(code / name) != digest:
            raise ValueError(f'preparation source differs from original run: {name}')
    item = SOURCES['citation2'][0]
    plan = {'schema_version': 1, 'scope': 'public_source_reproduction_only',
            'url': item['url'], 'download_bytes': (0 if args.archive_existing else item['bytes']),
            'official_archive_bytes': item['bytes'],
            'free_disk_required_bytes': 64 << 30, 'available_ram_required_bytes': 32 << 30,
            'memory_budget_gib': args.memory_budget_gib, 'payloads_to_verify': 19,
            'execute': args.execute, 'full_public_reproduction_verified': False,
            'source_mode': ('provisioned_official_archive_no_network' if args.archive_existing else 'official_download'),
            'preparer_source_sha256': hashes, 'expected_manifest_sha256': sha256(args.expected)}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    args.work_parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.work_parent).free < plan['free_disk_required_bytes']:
        raise RuntimeError('64GiB free disk required before download; existing results retained')
    if args.memory_budget_gib < 32 or memory_available() < plan['available_ram_required_bytes']:
        raise RuntimeError('32GiB available RAM and explicit budget required')
    run = args.work_parent / ('citation2_public_repro_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    run.mkdir()
    def status(stage, **extra):
        path = run / 'STATUS.json'
        partial = run / 'STATUS.json.tmp'
        partial.write_text(json.dumps({**plan, 'stage': stage, 'updated_unix': time.time(), **extra}, indent=2) + '\n')
        partial.replace(path)
    status('verifying_provisioned_archive' if args.archive_existing else 'downloading')
    try:
        if args.archive_existing:
            archive = args.archive_existing.resolve()
            receipt_path = args.official_download_receipt.resolve()
            receipt = json.loads(receipt_path.read_text())
            digest = receipt.get('sha256', '')
            if (receipt.get('url') != item['url'] or receipt.get('bytes') != item['bytes']
                    or receipt.get('source_kind') != 'fresh_official_public_download'
                    or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                    or digest == expected.get('recorded_archive_sha256')):
                raise ValueError('provisioned archive lacks a fresh official download identity receipt')
            if archive.stat().st_size != item['bytes'] or sha256(archive) != digest:
                raise ValueError('provisioned official archive failed size/SHA256 integrity check')
        else:
            archive = run / item['name']
            receipt = download(item, archive)
            receipt_path = archive.with_name(archive.name + '.download.json')
        (run / 'OFFICIAL_DOWNLOAD_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
        command = [sys.executable, str(preparer), '--zip', str(archive), '--root', str(run / 'dataset'),
                   '--source-manifest', str(receipt_path)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(args.threads),
                   MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
                   NUMBA_NUM_THREADS=str(args.threads))
        status('preparing', official_download=receipt, argv=command)
        with (run / 'preparation.log').open('x') as log:
            subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status('verifying_19_payloads', official_download=receipt, argv=command)
        result = verify(run / 'dataset/prepared', expected)
        result['expectation_sha256'] = sha256(args.expected)
        result['official_download'] = receipt
        result['preparer_source_sha256'] = hashes
        result['full_public_reproduction_verified'] = True
        (run / 'PUBLIC_19_PAYLOAD_IDENTITY.json').write_text(json.dumps(result, indent=2) + '\n')
        status('complete', official_download=receipt, full_public_reproduction_verified=True,
               payload_identity_receipt='PUBLIC_19_PAYLOAD_IDENTITY.json')
        print(json.dumps({'status': 'PASS', 'run_directory': str(run), 'payload_count': 19}), flush=True)
    except BaseException as error:
        status('failed', error_type=type(error).__name__, error=str(error),
               full_public_reproduction_verified=False, retained_directory=str(run))
        raise


if __name__ == '__main__':
    main()
