"""Identical promotion for fresh/reused native; reusable storage stays immutable."""
from pathlib import Path
import os, shutil, sys, time
import numpy as np
from common import GIB, pin, read, require, save, sha


def promote_native(config, check=lambda: None):
    from dataset import v4
    root = Path(config['native_root'])
    require(root.resolve() == root and not root.exists(), 'fresh owned native directory required')
    parent = root.parent
    while not parent.exists():
        parent = parent.parent
    root.mkdir(parents=True)
    began = time.monotonic_ns()
    graph = v4.load_graph(config)
    n, m = graph.n, graph.m
    require(sys.byteorder == 'little', 'little-endian native input required')
    disk_guard = config.get('preparation_disk_guard', False)
    if disk_guard:
        require(shutil.disk_usage(parent).free >= 8 * (n + 1 + m) + 6 * GIB,
                'native files plus sixGiB initial reserve required')
    ptr = np.memmap(root / 'rowptr.i8', dtype='<i8', mode='w+', shape=(n + 1,))
    col = np.memmap(root / 'col.i8', dtype='<i8', mode='w+', shape=(m,))
    for start in range(0, n, 1000000):
        check()
        if disk_guard:
            require(shutil.disk_usage(root).free >= 4 * GIB, 'fourGiB conversion reserve reached')
        ptr[start:min(n, start + 1000000)] = graph.starts[start:min(n, start + 1000000)]
    ptr[n] = m
    for start in range(0, m, 1000000):
        check()
        if disk_guard:
            require(shutil.disk_usage(root).free >= 4 * GIB, 'fourGiB conversion reserve reached')
        col[start:min(m, start + 1000000)] = graph.indices[start:min(m, start + 1000000)]
    ptr.flush(); col.flush()
    del ptr, col, graph
    for name in ('rowptr.i8', 'col.i8'):
        with (root / name).open('rb') as stream:
            os.fsync(stream.fileno())
    result = {'nodes': n, 'stored_entries': m, 'begin_ns': began, 'end_ns': time.monotonic_ns(),
              'rowptr_bytes': 8 * (n + 1), 'col_bytes': 8 * m, 'dtype': 'little-endian int64',
              'sentinel_added': True, 'full_integrity_readback_inside_application': False,
              'full_native_copy_once_per_app': config.get('native_preparation_scope', 'per_app') == 'per_app',
              'promotion_scope': config.get('native_preparation_scope', 'per_app'), 'files_fsynced': True}
    save(root / 'READY.json', result)
    return result


def verify_native_files(root, expected, check=lambda: None):
    root = Path(root)
    observed = {}
    for name, size_key, sha_key in (
        ('rowptr.i8', 'rowptr_bytes', 'rowptr_sha256'),
        ('col.i8', 'col_bytes', 'col_sha256'),
    ):
        observed[name] = pin(root / name, check)
        require(observed[name]['bytes'] == expected[size_key] and
                observed[name]['sha256'] == expected[sha_key], 'native bytes/hash differ: ' + name)
    return observed


def reuse_descriptor(root, expected, receipt_sha256, full=False, check=lambda: None, expected_device=None):
    """Cheap metadata within apps; full readback only in outer preparation/admission."""
    root = Path(root)
    require(root.resolve() == root and root.is_dir() and not root.is_symlink(),
            'canonical immutable reuse root required')
    require(sha(root / 'READY.json') == receipt_sha256, 'reusable receipt content changed')
    receipt = read(root / 'READY.json')
    actual_root_device = root.stat().st_dev
    expected_device = actual_root_device if expected_device is None else expected_device
    require(type(expected_device) is int and actual_root_device == expected_device,
            'current reuse root differs from current admitted PVC device')
    actual_devices = {}
    require(receipt['kind'] == 'CLOSURE_V1_IMMUTABLE_NATIVE_REUSE' and
            receipt['expected_native'] == expected, 'source-bound reusable receipt differs')
    for name, size_key in (('rowptr.i8', 'rowptr_bytes'), ('col.i8', 'col_bytes')):
        p = root / name
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == expected[size_key],
                'reusable file shape changed')
        require(p.stat().st_mode & 0o222 == 0, 'reusable native file must have no write permission')
        actual = p.stat()
        require({k: getattr(actual, v) for k, v in
                 (('bytes', 'st_size'), ('mtime_ns', 'st_mtime_ns'), ('inode', 'st_ino'))}
                == {k: receipt['file_stats'][name][k] for k in ('bytes', 'mtime_ns', 'inode')},
                'reusable native file metadata changed')
        require(actual.st_dev == actual_root_device == expected_device,
                'reusable file differs from current admitted PVC device')
        actual_devices[name] = actual.st_dev
    observed_files = verify_native_files(root, expected, check) if full else None
    if full:
        require(observed_files == receipt['files'], 'full reusable readback differs')
    return {'root': str(root), 'receipt_sha256': receipt_sha256,
            'expected_native': expected, 'immutable_open_only': True,
            'preparation_file_devices': {name: receipt['file_stats'][name]['device'] for name in actual_devices},
            'actual_file_devices': actual_devices, 'actual_reuse_root_device': actual_root_device,
            'bound_expected_device': expected_device, 'verified_files': observed_files,
            'full_integrity_readback_outside_application': full,
            'device_scope': 'Preparation device values are historical host-local observations; current devices are bound to this allocation PVC proof.',
            'preparation_outside_all_application_clocks': True,
            'full_integrity_readback_inside_application': False}
