#!/usr/bin/env python3
"""Independent cross-block closure arithmetic and PSS-coverage peer.

Only stdlib. Does not import/execute producer, auditor or aggregator. Reuses
the six independently raw-recomputed block receipts; checks their current raw
archive identities and directly reads platform/report identities from those
same archives. No extraction, fresh experiment, graph/tensor regeneration,
or pooled cross-dataset estimate.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import os
import shutil
import statistics
import tarfile
import time

HERE = Path(__file__).resolve().parent
ARMS = ('reference_union_compiled', 'reference_union_native_fresh', 'reference_union_native_reuse')
HORIZONS = ('short320', 'long6400')
OUT = HERE / 'FINAL_36_INDEPENDENT_RAW_PEER.json'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    need(path.is_file() and not path.is_symlink(), 'plain local file required: ' + str(path))
    return json.loads(path.read_text())


def pin(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    after = path.stat()
    need((before.st_size, before.st_ino, before.st_mtime_ns) ==
         (after.st_size, after.st_ino, after.st_mtime_ns), 'input changed')
    return {'bytes': before.st_size, 'sha256': digest.hexdigest()}


def stats(values):
    need(len(values) == 3 and all(math.isfinite(v) for v in values), 'three finite block observations required')
    return {'by_block_1_2_3': values, 'median': statistics.median(values), 'min': min(values), 'max': max(values),
            'sample_sd_descriptive_only': statistics.stdev(values)}


def close(a, b, what):
    need(type(a) in (int, float) and type(b) in (int, float)
         and math.isfinite(a) and math.isfinite(b)
         and math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-12), what)


def main():
    # Required actual terminal and both real separate dataset aggregates.
    terminal_path = HERE / 'blocks/FINISH_SEQUENCE_TERMINAL.json'
    aggregate_paths = {d: HERE / 'blocks' / ('ACTUAL_AGGREGATE_' + d + '.json') for d in ('friendster', 'citation2')}
    terminal = read(terminal_path)
    formal = {d: read(p) for d, p in aggregate_paths.items()}
    need(not OUT.exists(), 'fresh peer output required')
    need(shutil.disk_usage(HERE).free - (1 << 20) >= 10 * 2**30, 'local disk reserve')
    datasets = {}
    all_pods = set()
    all_source_pins = []
    all_counts = Counter()
    all_pss = Counter()
    all_phase_coverage = Counter()
    compared = []
    direct_platforms = {}
    direct_preparation = {}
    for dataset, prefix in (('friendster', 'F'), ('citation2', 'C')):
        blocks = []
        rows = []
        direct_platforms[dataset] = {}
        direct_preparation[dataset] = []
        for block in (1, 2, 3):
            label = prefix + str(block)
            peer_path = HERE / (label + '_INDEPENDENT_RAW_PEER' + ('_v3' if label == 'F1' else '') + '.json')
            peer = read(peer_path)
            need(peer['dataset_id'] == dataset and peer['repetition_block'] == block
                 and peer['applications'] == 6
                 and peer['status'] == 'PASS_ONE_COMPLETE_' + label + '_BLOCK_RAW_RECOMPUTATION', 'complete actual independent block required')
            collected = HERE / 'blocks' / dataset / ('block' + str(block)) / 'collected'
            archive = collected / 'block_raw.tgz'
            need(pin(archive) == peer['archive'], 'raw archive changed since independent block peer')
            need(pin(collected / 'ACTUAL_AUDIT.json') == peer['formal_audit_compared'], 'formal block audit changed')
            need(pin(collected / 'ROOT_COLLECTION.json') == peer['root_collection'], 'collection changed')
            all_source_pins.append(peer['source_pins'])
            appnames = {r['name'] for r in peer['actual_6_clocks_and_PSS']}
            need(len(appnames) == 6 and all(n.startswith(dataset + '_') for n in appnames), 'dataset-bound applications required')
            wanted = {'closure/results/PLATFORM_ADMISSION.json', 'closure/results/native_reuse_admission.json'} | {
                'closure/results/jobs/' + name + '/report.json' for name in appnames}
            direct = {}
            with tarfile.open(archive, 'r:gz') as tar:
                for member in tar:
                    if member.name in wanted:
                        need(member.isfile() and member.size <= 2 << 20 and member.name not in direct, 'bounded unique direct member required')
                        with tar.extractfile(member) as stream:
                            direct[member.name] = json.load(stream)
            need(set(direct) == wanted, 'missing direct platform/report identity')
            platform = direct['closure/results/PLATFORM_ADMISSION.json']
            direct_platforms[dataset][str(block)] = platform
            reuse_admission = direct['closure/results/native_reuse_admission.json']
            prep = reuse_admission['preparation']
            need(reuse_admission['outside_application_lifetimes'] is True and prep['one_time_per_dataset'] is True
                 and prep['outside_all_application_clocks'] is True and prep['no_private_worker_mapper_prepared'] is True,
                 'once-per-dataset preparation boundary differs')
            promotion = prep['native_promotion']
            conversion_seconds = (promotion['end_ns'] - promotion['begin_ns']) / 1_000_000_000
            validation_seconds = (prep['native_validation_end_ns'] - prep['native_validation_begin_ns']) / 1_000_000_000
            close(prep['conversion_seconds'], conversion_seconds, 'preparation conversion nanosecond arithmetic differs')
            close(prep['validation_seconds'], validation_seconds, 'preparation validation nanosecond arithmetic differs')
            direct_preparation[dataset].append({'conversion_seconds': conversion_seconds, 'validation_seconds': validation_seconds,
                                               'native_total_bytes': promotion['rowptr_bytes'] + promotion['col_bytes'],
                                               'receipt_sha256': reuse_admission['descriptor']['receipt_sha256']})
            need(platform['dataset_id'] == dataset and platform['block'] == block and platform['pod_uid'] == peer['pod_uid'], 'platform dataset/block/allocation binding differs')
            need(platform['pod_uid'] not in all_pods, 'allocation UID reused across blocks')
            all_pods.add(platform['pod_uid'])
            topology = platform['topology']
            cpus = platform['cpu_affinity']
            need(len(cpus) == len(set(cpus)) == 27 and {t['cpu'] for t in topology} == set(cpus), 'recorded CPU topology differs from pool')
            physical_cores = {(t['physical_package_id'], t['core_id']) for t in topology}
            numa = {n for t in topology for n in t['numa_nodes']}
            need(platform['cpu_max'] == '2700000 100000' and platform['memory_max'] == '274877906944'
                 and 'H100' in platform['gpu']['name'], 'actual resource admission differs')
            need(peer['native_worker_private_maps_verified'] == 32, 'native worker maps per block differ')
            matrix = {(r['horizon'], r['arm']) for r in peer['actual_6_clocks_and_PSS']}
            need(matrix == {(h, a) for h in HORIZONS for a in ARMS}, 'incomplete actual within-block matrix')
            block_models = set()
            for row in peer['actual_6_clocks_and_PSS']:
                report = direct['closure/results/jobs/' + row['name'] + '/report.json']
                block_models.add(report['model']['initial_state_sha256'])
                need(report['semantic_input_prefixes'] == peer['source_bound_ordered_typed_shaped_consumed_prefixes'][row['horizon']], 'direct report input digest differs')
                measured = 320 if row['horizon'] == 'short320' else 6400
                need(row['consumed_graphs'] == measured + 256 and row['measured_graphs'] == measured
                     and row['actual_measured_batch_losses'] == measured // 32, 'raw count cell differs')
                memory = row['pss']
                need(sum(memory['counts'].values()) == memory['total_samples'], 'PSS coverage count partition differs')
                need(sum(p['eligible_samples'] for p in memory['phases'].values()) == memory['counts']['eligible'], 'phase eligible totals differ')
                all_pss.update(memory['counts'])
                all_counts.update(applications=1, consumed_graphs=row['consumed_graphs'], measured_graphs=measured,
                                  completed_extractions_including_unconsumed_prefetch=row['completed_extractions_including_unconsumed_prefetch'])
                for phase in ('warmup', 'checkpointed_measured_window', 'shutdown'):
                    if phase in memory['phases']:
                        all_phase_coverage[phase + '_apps_with_eligible_samples'] += 1
                    else:
                        all_phase_coverage[phase + '_apps_without_eligible_samples'] += 1
                metric_values = dict(row['raw_ns_metrics_seconds'])
                metric_values['input_hash_seconds_included'] = row['hash_observer_seconds_reported_included_not_independently_retimed']
                rows.append({'block': block, 'name': row['name'], 'horizon': row['horizon'], 'arm': row['arm'],
                             'metrics': metric_values, 'consumed_graphs': row['consumed_graphs'],
                             'measured_graphs': measured,
                             'completed_extractions_including_unconsumed_prefetch': row['completed_extractions_including_unconsumed_prefetch'],
                             'PSS_counts': memory['counts'], 'PSS_total_samples': memory['total_samples'],
                             'PSS_observed_maximum': memory['maximum'],
                             'PSS_eligible_phases': memory['phases'],
                             'PSS_seen_phases_without_eligible_sample': memory['seen_phases_without_any_eligible_sample'],
                             'PSS_key_phases_without_eligible_sample': sorted({'warmup', 'checkpointed_measured_window', 'shutdown'} - memory['phases'].keys())})
            need(len(block_models) == 1, 'common model within block differs')
            blocks.append({'block': block, 'independent_raw_peer': {'path': str(peer_path.relative_to(HERE)), **pin(peer_path)},
                           'raw_archive': peer['archive'], 'pod_uid': peer['pod_uid'], 'node': platform['node'],
                           'boot_id': platform['boot_id'], 'cpu_affinity': cpus,
                           'logical_CPUs': len(cpus), 'physical_cores_spanned': len(physical_cores),
                           'NUMA_nodes_spanned': sorted(numa), 'CPU_quota': platform['cpu_max'], 'memory_max': platform['memory_max'],
                           'GPU_name': platform['gpu']['name'], 'CPU_marketing_SKU': 'not recorded in this platform evidence',
                           'source_bound_input_prefixes': peer['source_bound_ordered_typed_shaped_consumed_prefixes'],
                           'source_bound_initial_model_hash': next(iter(block_models))})
        need(len(rows) == 18 and all(b['source_bound_input_prefixes'] == blocks[0]['source_bound_input_prefixes'] for b in blocks), 'cross-block consumed inputs differ')
        need(len({b['source_bound_initial_model_hash'] for b in blocks}) == 1, 'initial model differs across same-dataset blocks')
        index = {(r['block'], r['horizon'], r['arm']): r for r in rows}
        ratios = []
        cells = []
        for horizon in HORIZONS:
            for arm in ARMS:
                metric_names = set(index[(1, horizon, arm)]['metrics'])
                need(all(set(index[(b, horizon, arm)]['metrics']) == metric_names for b in (1, 2, 3)), 'metric domains differ across blocks')
                cells.append({'horizon': horizon, 'arm': arm,
                              'metrics': {m: stats([index[(b, horizon, arm)]['metrics'][m] for b in (1, 2, 3)]) for m in sorted(metric_names)}})
            for numerator, denominator in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[2], ARMS[1])):
                metric_names = set(index[(1, horizon, numerator)]['metrics'])
                ratios.append({'horizon': horizon, 'numerator': numerator, 'denominator': denominator,
                               'metrics': {m: stats([index[(b, horizon, numerator)]['metrics'][m] / index[(b, horizon, denominator)]['metrics'][m]
                                                     for b in (1, 2, 3)]) for m in sorted(metric_names)}})
        datasets[dataset] = {'actual_applications': 18, 'blocks': blocks, 'raw_derived_application_rows': rows,
                             'raw_clock_cell_statistics': cells, 'within_block_paired_ratio_statistics': ratios,
                             'one_time_native_preparation_raw_clock_summary': direct_preparation[dataset][0],
                             'formal_aggregate_input': {'path': str(aggregate_paths[dataset].relative_to(HERE)), **pin(aggregate_paths[dataset])}}
        need(all(p == direct_preparation[dataset][0] for p in direct_preparation[dataset]), 'same dataset preparation differs across fresh allocations')
    need(all(s == all_source_pins[0] for s in all_source_pins), 'source protocol/envelope pins differ across the six allocations')
    need(all_counts['applications'] == 36 and len(all_pods) == 6, 'final complete 36-app six-allocation matrix missing')
    # Formal aggregate schema comparisons are intentionally explicit and are
    # filled from the actual completed aggregate, never guessed from its source.
    compare_actual(formal, datasets, terminal, compared, direct_platforms)
    result = {'kind': 'INDEPENDENT_TWO_SEPARATE_DATASET_AGGREGATES_36_RAW_APPLICATION_PEER',
              'status': 'PASS_ALL_36_RAW_DERIVED_ARITHMETIC_FINGERPRINTS_AND_PSS_COVERAGE',
              'created_unix': time.time(), 'peer_source': pin(Path(__file__)),
              'actual_terminal': {'path': str(terminal_path.relative_to(HERE)), **pin(terminal_path)},
              'source_pins_all_six_equal': all_source_pins[0], 'actual_counts': dict(all_counts),
              'PSS_counts_all_36': dict(all_pss), 'PSS_key_phase_coverage_all_36': dict(all_phase_coverage),
              'datasets': datasets, 'explicit_formal_comparisons': compared,
              'independence': __doc__,
              'memory_limits': 'All maxima are sampled, non-atomic and residency-dependent. Missing phase coverage remains unknown, never zero. No exact peak, minimum RAM, full-lifetime coverage or native-reuse memory ranking is established.',
              'inference_limits': 'Separate dataset aggregates only; no cross-dataset pooled result, confidence interval, population/hardware causal claim, time-to-quality or universal winner.',
              'source_bound_graph_native_or_tensor_payloads_independently_regenerated_or_rehashed': False,
              'new_experiments': 0, 'remote_calls': 0, 'ML_imports': 0, 'bulk_extractions': 0}
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    need(len(raw) <= 256 << 10, 'bounded independent result required')
    with OUT.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': result['status'], 'output': str(OUT), 'identity': pin(OUT), 'counts': result['actual_counts'],
                      'PSS_counts': result['PSS_counts_all_36'], 'phase_coverage': result['PSS_key_phase_coverage_all_36']}, sort_keys=True))


def compare_actual(formal, datasets, terminal, compared, direct_platforms):
    need(terminal['success'] is True and terminal['local_sequence_stopped'] is True
         and terminal['applications'] == 36 and terminal['blocks'] == 6
         and terminal['global_two_dataset_complete'] is True
         and terminal['all_owned_GPU_allocations_released'] is True
         and terminal['old_E25_or_failed_apps_pooled'] is False, 'real complete terminal proof missing')
    releases = terminal['blocks_collected_audited_released']
    need(len(releases) == 6 and {(r['dataset_id'], r['repetition_block']) for r in releases} ==
         {(d, b) for d in ('friendster', 'citation2') for b in (1, 2, 3)}, 'terminal six release matrix differs')
    maximum_binary64_difference = 0.0
    numeric_comparisons = 0

    def number(actual, expected, what):
        nonlocal maximum_binary64_difference, numeric_comparisons
        close(actual, expected, what)
        maximum_binary64_difference = max(maximum_binary64_difference, abs(actual - expected))
        numeric_comparisons += 1

    def stat(target, expected, what):
        need(target['n'] == 3 and set(target['by_block']) == {'1', '2', '3'}, what + ' block support differs')
        for b, value in enumerate(expected['by_block_1_2_3'], 1):
            number(target['by_block'][str(b)], value, what + ' block value differs')
        for a, e in [('median', 'median'), ('minimum', 'min'), ('maximum', 'max'), ('sample_sd', 'sample_sd_descriptive_only')]:
            number(target[a], expected[e], what + ' statistic differs')

    for dataset, rebuilt in datasets.items():
        target = formal[dataset]
        need(target['dataset_id'] == dataset and target['blocks'] == 3 and target['jobs'] == 18
             and target['status'] == 'PASS_THREE_BLOCKS_18_APPS' and target['global_dataset_complete'] is True
             and target['global_two_dataset_complete'] is False and target['failed_or_historical_applications_pooled'] is False,
             'separate actual 18-app dataset aggregate required')
        aggregate_pin = rebuilt['formal_aggregate_input']
        need({k: terminal['dataset_aggregates'][dataset][k] for k in ('bytes', 'sha256')} ==
             {k: aggregate_pin[k] for k in ('bytes', 'sha256')}, 'terminal actual aggregate identity differs')
        prefix_map = rebuilt['blocks'][0]['source_bound_input_prefixes']
        need(target['source_bound_semantic_prefixes_by_horizon'] == prefix_map, 'final aggregate semantic input prefixes differ')
        need(target['source_bound_consumed_input_streams'] ==
             {h: prefix_map[h]['576' if h == 'short320' else '6656'] for h in HORIZONS}, 'final aggregate full consumed streams differ')
        need(target['platforms'] == direct_platforms[dataset], 'formal platforms differ from directly read raw platform objects')
        for block in (1, 2, 3):
            folder = HERE / 'blocks' / dataset / ('block' + str(block))
            need(target['block_audit_pins'][block - 1] == pin(folder / 'collected/ACTUAL_AUDIT.json')
                 and target['raw_archive_pins'][str(block)] == pin(folder / 'collected/block_raw.tgz')
                 and target['root_collection_pins'][str(block)] == pin(folder / 'collected/ROOT_COLLECTION.json'),
                 'formal dataset input identity differs')
            entry = next(e for e in releases if (e['dataset_id'], e['repetition_block']) == (dataset, block))
            need(entry['applications'] == 6 and entry['audit'] == target['block_audit_pins'][block - 1]
                 and entry['resource_gone'] == pin(folder / 'RESOURCE_GONE.json'), 'terminal release/audit receipt identity differs')
            release = read(folder / 'RESOURCE_GONE.json')
            need(release['old_pod_absent'] is True and release['matching_pods'] == [] and release['replicas'] == 0
                 and release['owned_GPU_reservation_absent'] is True and release['PVC_retained'] is True
                 and release['original_pod_uid'] == rebuilt['blocks'][block - 1]['pod_uid'], 'allocation release proof differs')
        actual_cells = {(r['horizon'], r['arm']): r for r in target['rows']}
        expected_cells = {(r['horizon'], r['arm']): r for r in rebuilt['raw_clock_cell_statistics']}
        need(set(actual_cells) == set(expected_cells) and len(target['rows']) == 6, 'formal cell matrix differs')
        for key, cell in expected_cells.items():
            original = actual_cells[key]
            need(original['workers'] == 8 and set(original['metrics']) == set(cell['metrics']), 'formal cell metric domain differs')
            for name, metric in cell['metrics'].items():
                stat(original['metrics'][name], metric, dataset + '/' + '/'.join(key) + '/' + name)
            observed = original['sampled_application_PSS_bytes']
            matches = sorted((r for r in rebuilt['raw_derived_application_rows'] if (r['horizon'], r['arm']) == key), key=lambda r: r['block'])
            expected_memory = {str(r['block']): None if r['PSS_observed_maximum'] is None else r['PSS_observed_maximum']['tree_pss_bytes'] for r in matches}
            known = [v for v in expected_memory.values() if v is not None]
            need(observed['by_block'] == expected_memory and observed['observed_blocks'] == len(known)
                 and observed['unobserved_blocks'] == [int(k) for k, v in expected_memory.items() if v is None]
                 and observed['minimum_of_observed'] == (min(known) if known else None)
                 and observed['maximum_of_observed'] == (max(known) if known else None)
                 and observed['complete_three_block_range'] is (len(known) == 3), 'sampled PSS maxima/unknown block propagation differs')
        actual_ratios = {(r['horizon'], r['numerator'], r['denominator']): r for r in target['paired_arm_ratios']}
        expected_ratios = {(r['horizon'], r['numerator'], r['denominator']): r for r in rebuilt['within_block_paired_ratio_statistics']}
        need(set(actual_ratios) == set(expected_ratios) and len(target['paired_arm_ratios']) == 6, 'paired ratio matrix differs')
        for key, pair in expected_ratios.items():
            original = actual_ratios[key]
            need(original['workers'] == 8 and set(original['metrics']) == set(pair['metrics']), 'paired metric domain differs')
            for name, metric in pair['metrics'].items():
                stat(original['metrics'][name], metric, dataset + '/' + '/'.join(key) + '/' + name)
        prep = rebuilt['one_time_native_preparation_raw_clock_summary']
        for key in ('conversion_seconds', 'validation_seconds'):
            number(target['native_reuse_preparation'][key], prep[key], 'once-per-dataset preparation differs')
        for key in ('native_total_bytes', 'receipt_sha256'):
            need(target['native_reuse_preparation'][key] == prep[key], 'preparation identity differs')
        compared.append({'dataset': dataset, 'actual_cells': len(actual_cells), 'actual_pair_groups': len(actual_ratios),
                         'all_raw_block_values_medians_ranges_and_descriptive_SD_verified': True,
                         'all_platforms_equal_raw_objects': True, 'all_six_sampled_memory_cells_preserve_unknowns': True,
                         'all_input_and_release_receipts_bound': True})
    compared.append({'numeric_comparisons': numeric_comparisons, 'maximum_binary64_difference': maximum_binary64_difference,
                     'tolerance_relative': 1e-13, 'tolerance_absolute': 1e-12,
                     'input_hash_seconds_included': 'Compared as reported observer duration; not independently retimed'})


if __name__ == '__main__':
    main()
