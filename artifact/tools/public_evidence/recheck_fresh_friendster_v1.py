#!/usr/bin/env python3
"""Bounded local comparison of fresh public E7/E12 with frozen raw evidence.

No graph, checkpoint, full-score file, remote connection or scientific rerun.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import tarfile
import time

import numpy as np

sys.dont_write_bytecode = True
RAW_SHA = '48232549089f3295c893bb33f6cea0d1aac4a0ee45b91a2669392c90ae1bfe40'
MANIFEST_SHA = '2800804cfc2278d399b95a7c1112f05b79e98fe004679466876c7f9185980b2c'
PREFIX = 'lpp_public_systems_evidence_v1/'
NPY_NAMES = ('midpoint_ranks.npy', 'reciprocal_ranks.npy', 'query_ids.npy',
             'source_node.npy', 'target_node.npy', 'warm_query.npy')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(4 << 20):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular nonsymlink evidence required')
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--systems-asset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.candidate_root.resolve()
    FRESH = root / 'metadata/evidence/fresh_friendster_v1'
    ADAPTER = root / 'work/vldb_public_fresh_adapters_v1'
    RAW = args.systems_asset.resolve()
    root_launch = FRESH / 'ROOT_LAUNCH.json'
    output = args.out.resolve()
    require(not output.is_relative_to(root), 'recheck output must be outside immutable candidate')
    require(not output.exists() and not args.out.is_symlink(), 'fresh recheck output required')
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), 'fresh peer receipt required')
    collection = read(FRESH / 'collection_receipt.json')
    results = FRESH / 'results'
    for name, pin in collection['files'].items():
        require(identity(results / name) == pin, 'collected result bytes changed: ' + name)
    require(len(collection['files']) == 26, 'all26 collected scientific/text files required')
    require(read(results / 'actual_launch.json') == read(root_launch),
            'actual launch differs from root launch')
    require(read(results / 'actual_launch.json')['source_packet_sha256'] ==
            'ec47493ce203fb9dbe306062ec2b3c32981f461797f642286aec800894e00a47', 'adapter packet changed')
    done, terminal = read(results / 'COMPLETE.json'), read(results / 'SCIENTIFIC_TERMINAL.json')
    require(all(done[k] is True for k in ('controller_finished', 'success', 'owned_children_reaped', 'own_lock_released')),
            'genuine controller completion required')
    require(terminal['success'] is True and terminal['experiments'] == ['e7', 'e12'] and
            terminal['new_performance_comparison'] is False and collection['controller_quiescence']['absent'] is True,
            'actual scientific completion/quiescence required')
    expected_numeric = read(ADAPTER / 'PREPARED_23_PAYLOAD_IDENTITY.json')['files']
    admissions = []
    for experiment in ('e7', 'e12'):
        exit_record = read(results / (experiment + '.exit.json'))
        require(exit_record['exit_code'] == 0 and exit_record['timed_out'] is False and
                exit_record['owned_process_reaped'] is True, 'clean scientific subprocess exit required')
        admission_path = results / experiment / 'PUBLIC_ADMISSION.json'
        admission = read(admission_path)
        complete = read(results / experiment / 'COMPLETE.json')
        require(complete['status'] == 'PUBLIC_FRESH_SCIENCE_COMPLETE' and
                complete['admission_sha256'] == digest(admission_path), 'actual wrapper admission binding required')
        require(admission['full_numeric_payloads_verified'] == 23 and admission['fixed_validation_queries'] == 512 and
                admission['fixed_test_queries'] == 4096 and admission['no_original_metadata_or_mtime_changes'] is True,
                'full exact numeric admission required')
        actual = {str(Path(k).relative_to(admission['prepared_root'])):
                  {'bytes': v['bytes'], 'sha256': v['sha256']}
                  for k, v in admission['actual_fresh_numeric_file_stats'].items()}
        require(actual == {r['path']: {'bytes': r['bytes'], 'sha256': r['sha256']} for r in expected_numeric},
                'actual23 full numeric hashes differ from frozen public contract')
        for key in ('historical_source_or_metadata_modified', 'model_loaded', 'new_method_claim',
                    'query_selection_changed', 'scientific_parameters_changed'):
            require(complete[key] is False, 'scientific scope/immutability contradiction')
        admissions.append(admission)
    require(admissions[0] == admissions[1], 'two fresh jobs admitted different inputs')
    require(identity(RAW) == {'bytes': 83629661, 'sha256': RAW_SHA}, 'immutable raw public asset changed')
    wanted = {'MANIFEST.json', 'E7/report.json', 'E7/source/semantic_sampler_audit_v1.py',
              'E12/summary.json', 'E12/source/heuristic_baselines.py'} | {'E12/' + n for n in NPY_NAMES}
    payloads = {}
    with tarfile.open(RAW, 'r|gz') as archive:
        for member in archive:
            name = member.name.removeprefix(PREFIX)
            if name in wanted:
                require(member.isfile() and member.size <= 4 << 20, 'bounded original raw payload required')
                payloads[name] = archive.extractfile(member).read()
    require(set(payloads) == wanted, 'missing historical original raw member')
    require(hashlib.sha256(payloads['MANIFEST.json']).hexdigest() == MANIFEST_SHA, 'raw manifest pin changed')
    manifest = json.loads(payloads['MANIFEST.json'])
    for name, data in payloads.items():
        if name == 'MANIFEST.json':
            continue
        pin = manifest['files'][name]
        require(len(data) == pin['bytes'] and hashlib.sha256(data).hexdigest() == pin['sha256'] and
                pin['lineage']['kind'] == 'unmodified_original_member' and pin['lineage']['unchanged_original_bytes'] is True,
                'historical raw lineage or member identity changed')
    old7, new7 = json.loads(payloads['E7/report.json']), read(results / 'e7/results/report.json')
    omitted7 = {'configuration', 'input_sha256', 'graph_input_stats', 'wall_seconds_budget_accounting_only'}
    require(set(old7) == set(new7) and all(old7[k] == new7[k] for k in set(old7) - omitted7),
            'E7 semantic record/summary/scientific source changed')
    paths7 = {'repo', 'prepared_root', 'quality_root', 'out'}
    require({k: v for k, v in old7['configuration'].items() if k not in paths7} ==
            {k: v for k, v in new7['configuration'].items() if k not in paths7}, 'E7 scientific parameters changed')
    require(len(new7['records']) == 512 and all(set(r['cases']) == {'h2_cap', 'h1_cap', 'h2_no_cap'}
                                               for r in new7['records']), 'full512-query three-control semantics required')
    require(new7['implementation_sha256'] == hashlib.sha256(payloads['E7/source/semantic_sampler_audit_v1.py']).hexdigest(),
            'E7 actual original implementation differs')
    valid_ids = np.load(ADAPTER / 'quality_reference/subsets/valid_query_ids.npy', allow_pickle=False)
    valid_edges = np.load(ADAPTER / 'quality_reference/subsets/valid_edge.npy', allow_pickle=False)
    require(np.array_equal(valid_ids, np.array([r['query_id'] for r in new7['records']])) and
            np.array_equal(valid_edges, np.array([[r['source_node'], r['target_node']] for r in new7['records']])),
            'E7 validation IDs/positive pairs differ')
    old12, new12 = json.loads(payloads['E12/summary.json']), read(results / 'e12/results/summary.json')
    omitted12 = {'gate', 'selection', 'provenance', 'timings', 'memory', 'total_seconds', 'output_directory'}
    require(set(old12) == set(new12) and all(old12[k] == new12[k] for k in set(old12) - omitted12),
            'E12 scientific result/parameters/definitions/source changed')
    arrays, comparisons = {}, {}
    for name in NPY_NAMES:
        path = results / 'e12/results' / name
        old = np.load(io.BytesIO(payloads['E12/' + name]), allow_pickle=False)
        fresh = np.load(path, allow_pickle=False)
        require(path.read_bytes() == payloads['E12/' + name] and np.array_equal(old, fresh),
                'E12 exact original array changed: ' + name)
        arrays[name] = fresh
        comparisons[name] = {**identity(path), 'shape': list(fresh.shape), 'dtype': str(fresh.dtype),
                             'bytes_and_all_values_identical_to_historical': True}
    ranks, reciprocal = arrays['midpoint_ranks.npy'], arrays['reciprocal_ranks.npy']
    require(ranks.shape == reciprocal.shape == (4096, 5) and np.array_equal(reciprocal, 1. / ranks),
            'all5x4096 exact reciprocal ranks required')
    positive = np.load(ADAPTER / 'quality_reference/subsets/test_edge.npy', allow_pickle=False)
    require(np.array_equal(positive[:, 0], arrays['source_node.npy']) and
            np.array_equal(positive[:, 1], arrays['target_node.npy']), 'frozen positive pairs differ')
    warm = arrays['warm_query.npy']
    metrics = {}
    for column, method in enumerate(new12['methods']):
        metrics[method] = {'mrr': float(reciprocal[:, column].mean()),
                           'hits_at_10': float(np.mean(ranks[:, column] <= 10)),
                           'warm_queries': int(warm.sum()), 'warm_mrr': float(reciprocal[warm, column].mean()),
                           'cold_queries': int((~warm).sum()), 'cold_mrr': float(reciprocal[~warm, column].mean())}
    require(metrics == new12['metrics'] == old12['metrics'], 'independent rank aggregation differs')
    receipt = {'kind': 'PORTABLE_RECHECK_FRESH_E7_E12_NUMERIC_CLOSURE', 'status': 'PASS',
               'observed_unix': time.time(), 'auditor_sha256': digest(__file__),
               'collection_receipt_sha256': digest(FRESH / 'collection_receipt.json'),
               'all26_collected_file_hashes_verified': True, 'genuine_completion_and_child_exits_verified': True,
               'outer_controller_OS_exit_status': collection['controller_OS_exit_status'],
               'historical_raw_asset_sha256': RAW_SHA, 'historical_raw_manifest_sha256': MANIFEST_SHA,
               'actual23_numeric_payloads_total_bytes': sum(r['bytes'] for r in expected_numeric),
               'both_jobs_full23_numeric_SHA_and_exact_frozen_cohorts_admitted': True,
               'E7': {'all512_records_all_three_controls_exact': True, 'all_summaries_exact': True,
                      'validation_IDs_and_positive_pairs_exact': True, 'scientific_parameters_and_source_exact': True,
                      'summary': new7['summary'], 'hop1_cap_binding': new7['hop1_cap_binding'],
                      'allowed_operational_differences': sorted(omitted7)},
               'E12': {'arrays': comparisons, 'all20480_midpoint_and_reciprocal_ranks_exact': True,
                       'metrics_recomputed_from_all_ranks_exact': metrics,
                       'positive_pairs_exact_to_frozen_quality': True,
                       'all_other_scientific_fields_exact': True, 'allowed_operational_differences': sorted(omitted12)},
               'scope': 'Successful fresh public numerical reproduction, without a new performance comparison. All local reads bounded to4MiB buffers/small arrays; no graph/model/full-score load, remote actions, source changes or historical metadata edits.'}
    with output.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'output': str(output), 'sha256': digest(output),
                      'E7_records': 512, 'E12_rank_values': 20480, 'collected_files': 26}))


if __name__ == '__main__':
    main()
