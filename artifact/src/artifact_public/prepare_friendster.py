"""Prepare exact Friendster learning inputs from verified PUBLIC graph payloads."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import uuid

import numba

from .verify_payloads import sha256, verify
from .prepare_friendster_full import memory_available

ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
PREPARER = ARTIFACT_ROOT / 'src/experiment_configs/seal_revision_2026-11/prepare_friendster.py'


def extract_evaluation(bundle, destination, manifest):
    """Exact eight-file allowlist; bounded, verified, no archived code execution."""
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(bundle, 'r:gz') as archive:
        members = archive.getmembers()
        expected = manifest['evaluation_files']
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('evaluation bundle differs from exact eight-file allowlist')
        if sum(m.size for m in members) > 100 << 20:
            raise ValueError('evaluation bundle exceeds 100MiB uncompressed budget')
        for member in members:
            rel = PurePosixPath(member.name)
            if (rel.is_absolute() or '..' in rel.parts or not rel.parts or '\\' in member.name
                    or ':' in rel.parts[0]):
                raise ValueError('unsafe evaluation bundle path')
            if not member.isfile() or member.size != expected[member.name]['bytes']:
                raise ValueError('unsafe or size-mismatched evaluation member')
        destination.mkdir(parents=True)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, 4 << 20)
            if sha256(target) != expected[member.name]['sha256']:
                raise ValueError(f'evaluation SHA256 mismatch: {member.name}')
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-graph', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--blocks', type=int, default=64)
    parser.add_argument('--memory-budget-gib', type=float, default=64,
                        help='preflight allowance, not an enforced process-memory limit')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f'refusing existing output: {args.out}')
    if not 1 <= args.threads <= numba.config.NUMBA_NUM_THREADS or args.blocks < 1:
        parser.error('invalid thread/block budget')
    if not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib < 64:
        parser.error('finite memory budget of at least64GiB required; this is a preflight allowance')
    if memory_available() < 64 << 30:
        raise RuntimeError('64GiB available RAM required before split preparation')
    args.out = args.out.resolve(); args.out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out.parent).free < 24 << 30:
        raise RuntimeError('24GiB free disk required before reading/building split inputs')
    numba.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    full = args.full_graph.resolve()
    expected = json.loads((ARTIFACT_ROOT / 'metadata/friendster_full_expected.json').read_text())
    identity = verify(full, expected)  # ~15GB streamed, bounded RAM; never skip.
    manifest = json.loads((ARTIFACT_ROOT / 'data/friendster_evaluation_manifest.json').read_text())
    bundle = ARTIFACT_ROOT / 'data/friendster_evaluation_public.tgz'
    lock = args.out.parent / (args.out.name + '.PUBLIC_PREP_RUNNING.lock')
    with lock.open('x') as stream:
        stream.write('public exact preparation\n')
    try:
        extracted = args.out.parent / (args.out.name + '.verified_eval-' + uuid.uuid4().hex[:8])
        extract_evaluation(bundle, extracted, manifest)
        # The original split constructor expects these provenance carriers.
        # All contents here are public and contain no private IDs or access paths.
        reference = {'split_metadata': manifest['split_metadata'], 'evaluation_files': manifest['evaluation_files'],
                     'public_dataset_doi': manifest['source_doi']}
        (extracted / 'reference_manifest.json').write_text(json.dumps(reference, indent=2) + '\n')
        (extracted / 'BUNDLE_MANIFEST.json').write_text(json.dumps({
            'latest_series_evaluation_hashes_confirmed': True,
            'public_evaluation_manifest_sha256': sha256(ARTIFACT_ROOT / 'data/friendster_evaluation_manifest.json'),
            'files': manifest['evaluation_files']}, indent=2) + '\n')
        (extracted / 'provenance').mkdir()
        (extracted / 'provenance/public_payload_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
        source_audit = {'public_numeric_payload_hashes_equal_verified': True,
                        'historical_source_revisions_equal': False,
                        'source': 'https://doi.org/10.5281/zenodo.5146230',
                        'private_infrastructure_required': False, 'files': {}}
        for item in identity['files']:
            path = full / item['path']
            source_audit['files'][item['path']] = {
                'bytes': item['bytes'], 'sha256': item['sha256'], 'mtime_ns': path.stat().st_mtime_ns}
        spec = importlib.util.spec_from_file_location('artifact_friendster_exact_preparer', PREPARER)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        pending = args.out.with_name(args.out.name + '.public_verify_pending-' + uuid.uuid4().hex[:8])
        result = module.build_prepared(full, extracted, pending, reference, source_audit,
                                       blocks=args.blocks, regenerate=True, strict_official=True,
                                       eval_bundle_sha256=sha256(bundle))
        prepared_expected = json.loads((ARTIFACT_ROOT / 'metadata/friendster_prepared_expected.json').read_text())
        prepared_identity = verify(pending, prepared_expected)
        (pending / 'PUBLIC_23_PAYLOAD_IDENTITY.json').write_text(json.dumps(prepared_identity, indent=2) + '\n')
        if args.out.exists():
            raise FileExistsError(f'output appeared during preparation: {args.out}; pending result retained')
        pending.rename(args.out)
        print(json.dumps({'status': 'PASS', 'prepared': str(args.out),
                          'public_payloads_verified': True, 'prepared_payload_count': len(prepared_identity['files']),
                          'query_checks': result['evaluation']['checks']}))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
