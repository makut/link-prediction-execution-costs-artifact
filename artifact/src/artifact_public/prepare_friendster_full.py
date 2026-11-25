"""Resource-guarded adapter to the exact dense Friendster converter."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .download import SOURCES, digests
from .verify_payloads import verify

ARTIFACT_ROOT = Path(__file__).resolve().parents[2]


def memory_available():
    try:
        import psutil
        available = psutil.virtual_memory().available
    except ImportError:
        available = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    base = Path('/sys/fs/cgroup')
    if (base / 'memory.max').is_file() and (base / 'memory.current').is_file():
        maximum = (base / 'memory.max').read_text().strip()
        if maximum != 'max':
            available = min(available, int(maximum) - int((base / 'memory.current').read_text()))
    return max(0, available)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--chunk-mb', type=int, default=64)
    parser.add_argument('--memory-budget-gib', type=float, default=64)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f'refusing existing output: {args.out}')
    if not 1 <= args.workers <= 256 or not 1 <= args.chunk_mb <= 256:
        parser.error('invalid worker/chunk budget')
    if not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib <= 0:
        parser.error('memory budget must be finite and positive')
    # Conservative preflight estimate, not an experimentally measured peak.
    required_ram = 56 << 30
    required_disk = 32 << 30
    print(json.dumps({'required_available_ram_bytes': required_ram, 'required_free_disk_bytes': required_disk,
                      'budget_gib': args.memory_budget_gib, 'execute': args.execute,
                      'estimate_is_not_a_measured_full_scale_peak': True}), flush=True)
    if not args.execute:
        return
    if args.memory_budget_gib * (1 << 30) < required_ram or memory_available() < required_ram:
        raise RuntimeError('insufficient available RAM/budget; use a large CPU host, not a laptop')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out.parent).free < required_disk:
        raise RuntimeError('insufficient disk for full CSR including zero timestamps')
    for item in SOURCES['friendster']:
        path = args.inputs / item['name']
        if path.stat().st_size != item['bytes'] or digests(path)['md5'] != item['md5']:
            raise ValueError(f'public source checksum mismatch: {path}')
    command = [sys.executable, '-m', 'scripts.prepare_friendster_csr',
               '--edges', str((args.inputs / 'com-friendster-dense.ez').resolve()),
               '--vertices', str((args.inputs / 'com-friendster-dense.vz').resolve()),
               '--out', str(args.out.resolve()), '--workers', str(args.workers),
               '--chunk-mb', str(args.chunk_mb), '--file-endian', 'little']
    env = dict(os.environ, NUMBA_NUM_THREADS=str(args.workers))
    subprocess.run(command, cwd=ARTIFACT_ROOT / 'src', env=env, check=True)
    expected = json.loads((ARTIFACT_ROOT / 'metadata/friendster_full_expected.json').read_text())
    result = verify(args.out, expected)
    result['public_source'] = 'https://doi.org/10.5281/zenodo.5146230'
    result['private_infrastructure_required'] = False
    (args.out / 'PUBLIC_PAYLOAD_IDENTITY.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
