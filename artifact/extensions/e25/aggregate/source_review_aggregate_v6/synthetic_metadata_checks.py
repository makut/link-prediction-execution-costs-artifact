#!/usr/bin/env python3
"""SOURCE ONLY: synthetic metadata checks, never scientific observations.

Actual-shaped report dictionaries exist only in memory to exercise aggregation.
No raw archive, recorded measurement, cluster, producer or ML module is read.
Only the source-only test outcome is serialized.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('synthetic_reviewed_aggregate', ROOT / 'aggregate_blocks_v6.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AUDITOR = 'a' * 64  # Deliberately synthetic; not an actual auditor identity.


def fixtures():
    reports = {}
    for block in (1, 2, 3):
        compiled, native = ((2., 1.), (100., 10.), (100., 1000.))[block - 1]
        r = dict(fixture_notice='SYNTHETIC METADATA ONLY; NOT SCIENTIFIC EVIDENCE',
            kind='INDEPENDENT_ACTUAL_E25_V6_EIGHT_APP_BLOCK_AUDIT',
            status='PASS_ACTUAL_E25_V6_EIGHT_APP_BLOCK_AUDIT', block=block,
            block_jobs=8, global_protocol_jobs=24, global24_complete=False,
            all_three_complete_blocks_required=True, script={'sha256': AUDITOR},
            independent_environment_module={'sha256': '96959b28102ea112538783843b1fe50634a733fac7d19ee77eeff54732ed7c27'},
            all_app_waits_zero_normal_worker_close=True, all_initial_models_exact=True,
            original_summary_independently_recomputed=True,
            source_bound_consumed_digest_equality_checked=True,
            consumed_tensor_bytes_independently_rehashed=False,
            platform={'block': block, 'pod_uid': f'SYNTHETIC_POD_{block}'},
            archive={'sha256': str(block) * 64}, root_collection={'sha256': str(block + 3) * 64},
            source_bound_consumed_input_streams={'short320': 'SYNTHETIC_SHORT', 'long6400': 'SYNTHETIC_LONG'},
            source_bound_semantic_prefixes_by_horizon={
                'short320': {'576': 'SYNTHETIC_SHORT'},
                'long6400': {'576': 'SYNTHETIC_SHORT', '6656': 'SYNTHETIC_LONG'}},
            cross_horizon_input_prefix576={'short_final_stream_sha256': 'SYNTHETIC_SHORT'},
            raw8_metrics=[], rows=[{}, {}, {}, {}], paired_compiled_over_native=[],
            headline_application_PSS_all_phases={'runs': []}, **module.PINS)
        for horizon in module.HORIZONS:
            for workers in (1, 8):
                for arm in module.ARMS:
                    job = dict(horizon=horizon, workers=workers, arm=arm, repetition=block,
                        name=f'{horizon}_w{workers}_r{block}_{arm}')
                    metric = compiled if arm == module.ARMS[0] else native
                    r['raw8_metrics'].append({'job': job, 'metrics': {'launch_to_actual_wait_seconds': metric}})
                    absent = horizon == 'short320' and workers == 1 and (arm == module.ARMS[1] or block != 2)
                    r['headline_application_PSS_all_phases']['runs'].append(dict(
                        job=job, status='unobserved' if absent else 'observed',
                        eligible_samples=0 if absent else 1,
                        maximum=None if absent else {'tree_pss_bytes': 10 * block}))
                r['paired_compiled_over_native'].append(dict(horizon=horizon, workers=workers,
                    repetition=block, numerator=module.ARMS[0], denominator=module.ARMS[1],
                    metrics={'launch_to_actual_wait_seconds': compiled / native}))
        reports[block] = r
    return reports


def check_rejected(reports, mutate):
    value = copy.deepcopy(reports)
    mutate(value)
    try:
        module.aggregate(value, {}, AUDITOR)
    except (ValueError, KeyError):
        return True
    raise AssertionError('Synthetic negative fixture was accepted')


def run():
    reports = fixtures()
    result = module.aggregate(reports, {}, AUDITOR)
    ratios = next(r for r in result['paired_compiled_over_native'] if r['horizon'] == 'short320' and r['workers'] == 1)
    ratio = ratios['metrics']['launch_to_actual_wait_seconds']
    assert ratio['median'] == 2. and ratio['minimum'] == .1 and ratio['maximum'] == 10.
    assert ratio['sample_sd'] == statistics.stdev([2., 10., .1])
    assert ratio['median'] != statistics.median([2., 100., 100.]) / statistics.median([1., 10., 1000.])
    def pss(horizon, workers, arm):
        return next(r['sampled_application_PSS_bytes'] for r in result['rows']
            if (r['horizon'], r['workers'], r['arm']) == (horizon, workers, arm))
    partial = pss('short320', 1, module.ARMS[0])
    assert partial['by_block'] == {'1': None, '2': 20, '3': None}
    assert partial['observed_blocks'] == 1 and partial['unobserved_blocks'] == ['1', '3']
    assert partial['complete_three_block_range'] is False
    assert partial['minimum_of_observed'] == partial['maximum_of_observed'] == 20
    missing = pss('short320', 1, module.ARMS[1])
    assert missing['observed_blocks'] == 0 and missing['minimum_of_observed'] is missing['maximum_of_observed'] is None
    full = pss('long6400', 8, module.ARMS[1])
    assert full['observed_blocks'] == 3 and full['complete_three_block_range'] is True
    assert full['minimum_of_observed'] == 10 and full['maximum_of_observed'] == 30
    negatives = {
        'missing_block_rejected': lambda x: x.pop(3),
        'duplicate_actual_block_ID_rejected': lambda x: x[3].update(block=2),
        'duplicate_PodUID_rejected': lambda x: x[3]['platform'].update(pod_uid=x[1]['platform']['pod_uid']),
        'duplicate_raw_archive_rejected': lambda x: x[3]['archive'].update(sha256=x[1]['archive']['sha256']),
        'duplicate_job_rejected': lambda x: x[2]['raw8_metrics'].__setitem__(1, copy.deepcopy(x[2]['raw8_metrics'][0])),
        'cross_block_stream_change_rejected': lambda x: x[2]['source_bound_consumed_input_streams'].update(short320='SYNTHETIC_DIFFERENT'),
        'incorrect_paired_ratio_rejected': lambda x: x[2]['paired_compiled_over_native'][0]['metrics'].update(launch_to_actual_wait_seconds=999.),
    }
    cases = {name: check_rejected(reports, mutate) for name, mutate in negatives.items()}
    cases.update(median_of_pairs_not_ratio_of_medians=True, descriptive_SD_of_pairs_correct=True,
        partial_PSS_None_and_coverage_preserved=True, all_missing_PSS_not_zero=True, complete_PSS_range_retained=True)
    return dict(kind='SOURCE_ONLY_SYNTHETIC_AGGREGATOR_REVIEW', status='PASS_SYNTHETIC_METADATA_CHECKS',
        scope='Synthetic metadata only; no completed actual block or complete24 cohort was verified.',
        actual_raw_archives_or_measurements_read=0, actual_experiments_cluster_GPU_YT_calls=0,
        serialized_scientific_aggregate_reports=0, cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, indent=2, sort_keys=True))
