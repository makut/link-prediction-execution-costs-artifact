#!/usr/bin/env python3
"""Prepare one source-pinned immutable native graph per dataset; no GPU or YT."""
import argparse, os, platform, signal, time
PREPARE_ENTRY_NS = time.monotonic_ns()
from pathlib import Path
from bindings import binding
from common import pin, read, require, save, sha, verify_sources
from native_storage import promote_native, verify_native_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('friendster', 'citation2'), required=True)
    parser.add_argument('--graph-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=2400)
    args = parser.parse_args()
    require(0 < args.max_seconds <= 3600, 'bounded preparation deadline required')
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('native preparation deadline')))
        signal.alarm(args.max_seconds)
    manifest = verify_sources()
    b = binding(args.dataset)
    graph = args.graph_root.resolve(); out = args.out.resolve()
    require(not out.exists() and not out.is_relative_to(graph) and not graph.is_relative_to(out),
            'fresh native output disjoint from compact graph required')
    admission_start = time.monotonic_ns()
    source_files = {name: pin(graph / name) for name in b['graph']['files']}
    require(source_files == b['graph']['files'], 'source graph differs before preparation')
    config = {'fixture': False, 'graph_path': str(graph), 'file_endian': 'little', 'native_root': str(out), 'preparation_disk_guard': True,
              'native_preparation_scope': 'once_per_dataset'}
    promotion = promote_native(config)
    validation_start = time.monotonic_ns()
    files = verify_native_files(out, b['native'])
    validation_end = time.monotonic_ns()
    require({p.name for p in out.iterdir()} == {'rowptr.i8', 'col.i8', 'READY.json'}, 'unexpected native cache contents')
    file_stats = {}
    for name in files:
        p = out / name; p.chmod(0o444); s = p.stat()
        file_stats[name] = {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino, 'device': s.st_dev}
    receipt = {'kind': 'CLOSURE_V1_IMMUTABLE_NATIVE_REUSE', 'dataset_id': args.dataset,
               'hostname': platform.node(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip() if Path('/proc/sys/kernel/random/boot_id').is_file() else None,
               'observed_unix': time.time(), 'preparation_complete_monotonic_ns': time.monotonic_ns(),
               'clock_scope': 'Monotonic times are comparable only within this preparation host/boot; later fresh application host clocks may differ.',
               'source_manifest_sha256': sha(Path(__file__).parent / 'SOURCE_MANIFEST.json'),
               'source_graph_root': str(graph), 'source_graph_files': source_files,
               'nodes': b['native']['nodes'], 'stored_entries': b['native']['stored_entries'],
               'preparation_entry_monotonic_ns': PREPARE_ENTRY_NS,
               'expected_native': b['native'], 'files': files, 'file_stats': file_stats,
               'source_admission_begin_ns': admission_start, 'native_promotion': promotion,
               'native_validation_begin_ns': validation_start, 'native_validation_end_ns': validation_end,
               'conversion_seconds': (promotion['end_ns'] - promotion['begin_ns']) / 1e9,
               'validation_seconds': (validation_end - validation_start) / 1e9,
               'one_time_per_dataset': True, 'outside_all_application_clocks': True,
               'reusable_arrays_readonly': True, 'no_private_worker_mapper_prepared': True,
               'scope': 'Measured source-to-int64 promotion/fsync plus separate full readback against previously established native content identities. Source/native scans may warm filesystem caches; no cold-I/O or minimum-memory claim.'}
    save(out / 'READY.json', receipt); (out / 'READY.json').chmod(0o444)
    print(__import__('json').dumps({'status': 'PASS', 'dataset_id': args.dataset,
          'root': str(out), 'receipt_sha256': sha(out / 'READY.json'),
          'conversion_seconds': receipt['conversion_seconds'], 'validation_seconds': receipt['validation_seconds']}), flush=True)


if __name__ == '__main__':
    main()
