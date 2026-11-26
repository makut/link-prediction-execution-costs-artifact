#!/usr/bin/env python3
"""Reaudit published E25 v6 records using the exact included stdlib tools."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PINS = dict(source_manifest_sha256='3fd8783173e4a850018d860459881aafd1dd49e8d4cda101eb273641e47af086',
    protocol_sha256='e89b371089bd01f71579fdc00ec06004373bb549542658cc223914759f392e18',
    wrapper_sha256='072575b15555a83f223e15643b869874f9a5e4f83172c06e0f4f5d032bdae4c9',
    wrapper_ready_sha256='88b06c57d7de42c062fbd2ea76cd268999d3bbabe8bbd31130a9ec79dfa5a785')
SCRIPTS = {
    'independent/audit_block_v6.py': 'b50f69169d8149d090913dcedf5a15aea8ed6d397650a276f432e1781b41deb5',
    'independent/independent_env.py': '96959b28102ea112538783843b1fe50634a733fac7d19ee77eeff54732ed7c27',
    'aggregate/aggregate_blocks_v6.py': '8159c9cef46d38d0a6e5a48904960255fcba6377f85316fbbeabf21fc71e220c'}
FILES = {'archive': 'block_raw.tgz', 'root_collection': 'ROOT_COLLECTION.json', 'audit': 'ACTUAL_AUDIT.json'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    need(path.resolve() == path and path.is_file() and not path.is_symlink(), 'canonical regular input/output required')
    size = path.stat().st_size
    need(0 < size <= 128 * 2**20, 'bounded nonempty payload required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(2**20), b''):
            h.update(data)
    return dict(bytes=size, sha256=h.hexdigest())


def read(path):
    need(pin(path)['bytes'] <= 32 * 2**20, 'JSON32MiB cap')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'duplicate JSON key'); result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    artifact, out = args.artifact.absolute(), args.out.absolute()
    need(artifact.resolve() == artifact and artifact.is_dir(), 'canonical artifact directory required')
    need(not os.path.lexists(out) and out.resolve() == out and out.parent.is_dir()
         and not out.is_relative_to(artifact), 'fresh canonical output directory outside artifact required')
    extension = artifact / 'extensions/e25'; recorded = extension / 'recorded'
    catalog_path = recorded / 'RECORDED_IDENTITIES.json'; catalog = read(catalog_path)
    need(set(catalog) == {'kind', 'status', 'global24_complete', 'jobs', 'blocks', 'source_hashes',
         'block1', 'block2', 'block3', 'aggregate'}, 'exact recorded catalog schema required')
    need(catalog['kind'] == 'E25_V6_RECORDED_IDENTITIES' and catalog['status'] == 'PASS_COMPLETE_THREE_BLOCKS_24_APPS'
         and catalog['global24_complete'] is True and type(catalog['jobs']) is int and catalog['jobs'] == 24
         and type(catalog['blocks']) is int and catalog['blocks'] == 3 and catalog['source_hashes'] == PINS,
         'all three actual blocks and frozen scientific pins required')
    for relative, sha in SCRIPTS.items():
        need(pin(extension / relative)['sha256'] == sha, 'included exact audit/aggregate source differs')
    aggregate_path = recorded / 'ACTUAL_AGGREGATE.json'
    need(pin(aggregate_path) == catalog['aggregate'], 'recorded aggregate whole identity differs')
    aggregate = read(aggregate_path)
    need(aggregate['kind'] == 'E25_V6_COMPLETE_THREE_ALLOCATION_AGGREGATE'
         and aggregate['status'] == catalog['status'] and aggregate['global24_complete'] is True
         and aggregate['jobs'] == 24 and aggregate['blocks'] == 3 and aggregate['source_hashes'] == PINS
         and aggregate['auditor_source_sha256'] == SCRIPTS['independent/audit_block_v6.py'], 'actual PASS24 aggregate required')
    for block in (1, 2, 3):
        row = catalog['block' + str(block)]; need(set(row) == set(FILES), 'exact three block pins required')
        for key, name in FILES.items():
            need(pin(recorded / ('block' + str(block)) / name) == row[key], 'recorded block whole identity differs')
            field = {'archive': 'raw_archive_pins', 'root_collection': 'root_collection_pins', 'audit': 'block_audit_pins'}[key]
            need({k: aggregate[field][str(block)][k] for k in ('bytes', 'sha256')} == row[key], 'catalog/aggregate pin differs')
        audit = read(recorded / ('block' + str(block)) / FILES['audit'])
        need(audit['kind'] == 'INDEPENDENT_ACTUAL_E25_V6_EIGHT_APP_BLOCK_AUDIT'
             and audit['status'] == 'PASS_ACTUAL_E25_V6_EIGHT_APP_BLOCK_AUDIT' and audit['block'] == block
             and audit['block_jobs'] == 8 and audit['global_protocol_jobs'] == 24 and audit['global24_complete'] is False
             and all(audit[k] == v for k, v in PINS.items())
             and audit['script']['sha256'] == SCRIPTS['independent/audit_block_v6.py'], 'three actual source-bound PASS8 audits required')
    # All catalog, source, actual PASS and whole-input gates precede output creation.
    out.mkdir(); env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONSAFEPATH'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    def run(relative, values):
        subprocess.run([sys.executable, '-B', '-S', str(extension / relative), *map(str, values)],
            cwd=out, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, check=True, timeout=600)
    pairs = []
    for block in (1, 2, 3):
        name = 'block' + str(block); folder = out / name; folder.mkdir(); row = catalog[name]
        fresh = folder / 'ACTUAL_AUDIT.json'  # Aggregator serializes this basename.
        run('independent/audit_block_v6.py', ['--archive', recorded / name / FILES['archive'],
            '--archive-sha256', row['archive']['sha256'], '--root-collection', recorded / name / FILES['root_collection'],
            '--root-collection-sha256', row['root_collection']['sha256'], '--output', fresh])
        need(pin(fresh) == row['audit'], 'fresh audit differs byte-for-byte from published audit')
        pairs.extend(['--block-audit', fresh, row['audit']['sha256']])
    fresh = out / 'ACTUAL_AGGREGATE.json'
    run('aggregate/aggregate_blocks_v6.py', [*pairs, '--auditor-source-sha256', SCRIPTS['independent/audit_block_v6.py'], '--output', fresh])
    need(pin(fresh) == catalog['aggregate'], 'fresh aggregate differs byte-for-byte from published aggregate')
    receipt = dict(status='PASS_RECORDED_E25_V6_OFFLINE_ROUNDTRIP', jobs=24, blocks=3,
        catalog=pin(catalog_path), aggregate=pin(fresh), all_audits_byte_identical=True,
        actual_new_experiments_GPU_graph_model_network_cluster_YT_actions=0)
    with (out / 'VERIFICATION_RECEIPT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
