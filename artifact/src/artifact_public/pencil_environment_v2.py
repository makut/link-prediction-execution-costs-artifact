"""Portable admission for the repaired PENCIL bridge; no installs or model fits.

The historical probe is copied byte-for-byte and used only in its base mode.
The public adapter checks a supplied environment, without historical venv names,
base paths, controllers, private data paths or storage services.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import sys

from . import systems_common as c

BRIDGE = c.ROOT / 'work/vldb_pencil_bridge_v2'
FROZEN = c.ROOT / 'metadata/PENCIL_V2_FROZEN_SOURCES.json'
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')
VERSIONS = {'torch': '2.5.0+cu121', 'numpy': '1.24.4', 'scipy': '1.10.1',
            'numba': '0.58.0', 'torch-geometric': '2.7.0',
            'transformers': '4.46.2', 'wandb': '0.18.7', 'rich': '14.2.0', 'tqdm': '4.67.1'}


def verify_candidate_sources():
    c.require(c.sha(FROZEN) == '558425e6d4d98a31ff32b10366a97d94aa79ea3f7f5c1261e19f59b842904f1e',
              'repaired frozen source manifest differs')
    definition = c.read(FROZEN)
    c.require(definition['source_packet_sha256'] ==
              '56954a16a9d45d66783a82961fe82cffe4144aad5b3bef382ce2d05ed724d26c',
              'repaired frozen source packet differs')
    files = definition['files']
    c.require(len(files) == 82, 'complete80-file bridge plus original probe/protocol required')
    for name, expected in files.items():
        path = c.ROOT / name
        c.require(path.resolve().is_relative_to(c.ROOT) and path.is_file() and not path.is_symlink()
                  and path.stat().st_size == expected['bytes'] and c.sha(path) == expected['sha256'],
                  'repaired frozen source differs: ' + name)
    return {'status': 'pass', 'files': 82, 'manifest_sha256': c.sha(FROZEN),
            'source_packet_sha256': definition['source_packet_sha256']}


def validate_target_versions(result):
    c.require(sys.platform.startswith('linux') and sys.version_info[:2] == (3, 10),
              'execution target is Linux Python3.10; portable offline plan needs no native packages')
    packages = result['packages']
    c.require(all(packages.get(name) == value for name, value in VERSIONS.items()),
              'supplied environment does not match the frozen numerical/HF target')
    c.require(result['torch_sparse_version'].split('+')[0] == '0.6.18'
              and result['torch_cuda'] == '12.1'
              and result['saint_wrapper']['git_blob'] == '440455853e0309b00376cbe9ad25d7226cfa5e0e'
              and bool(result['torch_sparse_binaries']), 'native wrapper/binary/CUDA target differs')


def probe():
    import os
    verify_candidate_sources()
    # This is admission, not package installation or environment repair.
    c.require(all(os.environ.get(name) == '1' for name in THREADS), 'all four numerical thread counts must be1')
    old = c.load_file(c.ROOT / 'metadata/protocols/pencil_runtime_probe_v3.py', 'public_original_pencil_probe_v3')
    result = old.probe('base')
    validate_target_versions(result)
    sys.path.insert(0, str(BRIDGE))
    from graph_only_transformers_v2 import configure_graph_only_transformers
    result['runtime_compatibility'] = configure_graph_only_transformers()
    result['extra_modules'] = {}
    for name in ('transformers', 'wandb', 'rich', 'tqdm'):
        origin = Path(importlib.import_module(name).__file__).resolve()
        result['extra_modules'][name] = {'origin': str(origin), 'sha256': c.sha(origin), 'version': md.version(name)}
    policy = result['runtime_compatibility']
    origin = Path(policy['import_utils_origin'])
    c.require(origin == Path(result['extra_modules']['transformers']['origin']).parent / 'utils/import_utils.py'
              and c.sha(origin) == policy['import_utils_sha256']
              and policy['policy_source_sha256'] == c.sha(BRIDGE / 'graph_only_transformers_v2.py'),
              'loaded HF policy/source origins differ')
    c.require(not any(name == 'torchvision' or name.startswith('torchvision.')
                      or name == 'transformers.models.bert.modeling_bert' for name in sys.modules),
              'environment admission must precede optional vision and BERT imports')
    result['public_admission'] = {'version': 2, 'package_installations': 0,
        'source_mutations': 0, 'models_loaded': 0,
        'scope': 'Supplied numerical/native/HF environment, recorded again after48 scientific jobs.'}
    verify_candidate_sources()
    return result


def validate_driver_environment(raw, admitted):
    policy = raw['runtime_compatibility']
    c.require(policy == admitted['runtime_compatibility']
              and raw['audit_environment']['runtime_compatibility'] == policy,
              'driver graph-only policy differs from the actual public environment probe')
    native = raw['native_setup']
    c.require(native['torch'] == admitted['torch'] and native['torch_sparse'] == admitted['torch_sparse_version']
              and native['pyg'] == admitted['pyg']
              and native['wrapper_sha256'] == admitted['saint_wrapper']['sha256']
              and native['wrapper_git_blob'] == admitted['saint_wrapper']['git_blob'],
              'driver numerical/native wrapper differs from public admission')
    c.require({r['name']: r['sha256'] for r in native['extension_binaries']} ==
              {name: r['sha256'] for name, r in admitted['torch_sparse_binaries'].items()},
              'driver native library hashes differ from public admission')
    c.require(all(native['software_versions'][name] == admitted['packages'][name]
                  for name in ('numpy', 'numba', 'scipy', 'transformers', 'wandb')),
              'driver numerical/HF versions differ from public admission')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    c.require(not args.output.is_symlink() and not output.exists() and not output.is_relative_to(c.ROOT),
              'fresh environment receipt outside source required')
    value = probe()
    c.save(output, value)
    print(json.dumps({'status': 'admitted', 'output': str(output), 'source_mutations': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
