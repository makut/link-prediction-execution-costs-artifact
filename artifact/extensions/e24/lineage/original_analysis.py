#!/usr/bin/env python3
"""Descriptive posthoc scan/pair analysis of two existing matched128 cohorts.

Stdlib only. No experiment, full graph, model, policy fitting or new tests.
All timing values come solely from the actual E20/E22 Numba32 reports.
"""
from __future__ import annotations
import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
MIB = 1 << 20
FR_PIN = dict(bytes=2121570, sha256='546cad6f6b14c9f56844dc3414804e97f0dd303028532910dd74bee9d10f9374')
C2_PINS = {
    'results/citation2_neighbors32.json': dict(bytes=497684, sha256='2ee782fa962d1bab7799136da8c455b134d2809e215ddba862add8b580de5176'),
    'results/cohort/citation2_exact_int32.json': dict(bytes=456993, sha256='9da827fa6076f9716ea4edbcfbdc70afc56364d15730a6075690941d88dac98e'),
    'results/cohort/FROZEN_128_SELECTION.json': dict(bytes=73714, sha256='4c6093e6ac941a5b6d03c3e40cb82005e63df39188cfe07574c1957fd446d7eb'),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, maximum=16 * MIB):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum, 'bounded regular input required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def decode(raw):
    need(len(raw) <= 16 * MIB, 'bounded JSON metadata required')
    return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def load_friendster(path):
    need(pin(path) == FR_PIN, 'actual published E20 archive identity differs')
    prefix = 'public_fr_cpu_matched_v2_20261125/'
    names = ('results/friendster_neighbors32.json', 'sources/records/friendster_exact_int32.json',
             'sources/FROZEN_128_SELECTION.json', 'MANIFEST.json')
    payloads = {}
    with tarfile.open(path, 'r|gz') as tar:
        for member in tar:
            if member.name in {prefix + name for name in names}:
                name = member.name[len(prefix):]
                need(member.isfile() and name not in payloads and member.size <= 16 * MIB, 'bounded unique selected raw member required')
                payloads[name] = tar.extractfile(member).read()
    need(set(payloads) == set(names), 'all E20 report/geometry/selection/manifest members required')
    manifest = decode(payloads['MANIFEST.json'])
    identities = {}
    for name in names[:-1]:
        identities[name] = dict(bytes=len(payloads[name]), sha256=hashlib.sha256(payloads[name]).hexdigest())
        need(identities[name] == manifest['files'][name], 'E20 selected member differs from actual manifest')
    return (decode(payloads[names[0]]), decode(payloads[names[1]]), decode(payloads[names[2]]),
            dict(archive=FR_PIN, selected_members=identities))


def load_citation2(folder):
    folder = Path(folder)
    manifest = decode((folder / 'MANIFEST.json').read_bytes())
    values = []
    for name, expected in C2_PINS.items():
        need(pin(folder / name) == expected == manifest['files'][name], 'actual E22 report/geometry/selection identity differs')
        values.append(decode((folder / name).read_bytes()))
    return (*values, dict(selected_members=C2_PINS, raw_manifest=pin(folder / 'MANIFEST.json')))


def ranks(values):
    """One-based average ranks for exact ties; rational ratios avoid false ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = ((start + 1) + stop) / 2
        for index in order[start:stop]:
            result[index] = average
        start = stop
    return result


def spearman(x, y):
    need(len(x) == len(y) == 128, '128 separately ranked cases required')
    rx, ry = ranks(x), ranks(y)
    mx, my = math.fsum(rx) / len(rx), math.fsum(ry) / len(ry)
    xx = math.fsum((v - mx) ** 2 for v in rx)
    yy = math.fsum((v - my) ** 2 for v in ry)
    need(xx > 0 and yy > 0, 'nonconstant ranks required')
    return math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(xx * yy)


def median3(values):
    need(len(values) == 3 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values),
         'three positive finite raw edge-construction times required')
    return Decimal(str(sorted(values)[1]))


def analyze(label, report, geometry, selection, sources):
    need(report['status'] == 'complete' and report['neighbor_width'] == 32 and report['record_count'] == 128
         and report['all_executed_outputs_exact'] is True and report['numba_threads'] == 1
         and report['torch_imported'] is False and report['physical_arrays']['neighbor_dtype'] == 'int32',
         'actual complete single-thread Numba32 matched report required')
    need(report['configuration']['warmup'] == 1 and report['configuration']['repeats'] == 3, 'original repetition boundary differs')
    need(report['replay_plan'] == selection or
         {**report['replay_plan'], 'archive': {k: v for k, v in report['replay_plan']['archive'].items() if k != 'original_local_path'}} ==
         {**selection, 'archive': {k: v for k, v in selection['archive'].items() if k != 'original_local_path'}}, 'actual frozen replay selection differs')
    records, selected = report['records'], selection['selected']
    need(len(records) == len(selected) == 128, 'exact128 matched selected cases required')
    geom_identity = report['replay_plan']['archive']
    need(len(geometry['records']) == geom_identity['record_count'], 'geometry archive record count differs')
    rows, sums = [], {key: Decimal(0) for key in ('always_scan', 'always_pairwise', 'frozen_proxy', 'hindsight_best')}
    pair_set, unordered_pairs, set_hashes = set(), set(), set()
    for timed, expected in zip(records, selected):
        index = expected['archive_record_index']
        raw = geometry['records'][index]
        need(timed['archive_record_index'] == index and timed['exact_archived_oracles'] is True, 'timing cohort order/semantic proof differs')
        for key in ('node_set_sha256', 'canonical_edges_sha256', 'drnl_sha256', 'sum_out_degrees'):
            need(timed[key] == raw[key] == expected[key], 'selected geometry/timing identity differs: ' + key)
        nodes = raw['nodes']
        k, ds = len(nodes), timed['sum_out_degrees']
        need(k >= 2 and type(ds) is int and ds >= 0 and len(set(nodes)) == k, 'valid ordered S and degree sum required')
        q = k * (k - 1) // 2
        medians = {}
        repeats = {}
        for method in ('scan', 'pairwise'):
            data = timed['methods'][method]
            need(data['status'] == 'ok' and data['exact_edges'] is True and data['exact_drnl'] is True,
                 'actual original method exactness required')
            repeats[method] = data['edge_seconds']
            medians[method] = median3(repeats[method])
        scan, pair = medians['scan'], medians['pairwise']
        winner = 'scan' if scan < pair else 'pairwise' if pair < scan else 'tie'
        proxy = 'pairwise' if q <= max(1, ds) else 'scan'
        best = min(scan, pair)
        for name, value in (('always_scan', scan), ('always_pairwise', pair), ('frozen_proxy', medians[proxy]), ('hindsight_best', best)):
            sums[name] += value
        physical_pair = tuple(raw.get('pair', nodes[:2]))
        pair_set.add(physical_pair); unordered_pairs.add(tuple(sorted(physical_pair))); set_hashes.add(raw['node_set_sha256'])
        rows.append(dict(archive_record_index=index, node_set_sha256=raw['node_set_sha256'], pair=list(physical_pair),
            sampler=raw.get('sampler'), cap=raw.get('cap'), workload=raw.get('workload'),
            k=k, Ds=ds, Q=q, Ds_over_Q=float(Fraction(ds, q)),
            edge_seconds_raw=repeats, edge_median_seconds={key: float(value) for key, value in medians.items()},
            scan_over_pair=float(scan / pair), winner=winner, frozen_proxy_choice=proxy,
            proxy_agrees_with_strict_winner=None if winner == 'tie' else proxy == winner))
    x = [Fraction(r['Ds'], r['Q']) for r in rows]
    y = [Fraction(str(r['edge_median_seconds']['scan'])) / Fraction(str(r['edge_median_seconds']['pairwise'])) for r in rows]
    ties = sum(r['winner'] == 'tie' for r in rows)
    agrees = sum(r['proxy_agrees_with_strict_winner'] is True for r in rows)
    output = dict(dataset=label, matched_cases=128, distinct_ordered_pairs=len(pair_set),
        distinct_unordered_physical_pairs=len(unordered_pairs), distinct_ordered_sets=len(set_hashes),
        geometry_archive_records_available=len(geometry['records']), geometry_unselected_timings_not_used=True,
        sources=sources, host=report['host'], original_timing_contract=report['timing_contract'],
        wins={method: sum(r['winner'] == method for r in rows) for method in ('scan', 'pairwise', 'tie')},
        spearman_Ds_over_Q_vs_scan_over_pair=spearman(x, y),
        fixed_proxy=dict(rule='choose pairwise iff Q <= max(1, Ds), otherwise scan; alpha=1 fixed without fitting',
            choices={method: sum(r['frozen_proxy_choice'] == method for r in rows) for method in ('scan', 'pairwise')},
            agreement=dict(strict_matches=agrees, strict_mismatches=128 - ties - agrees, exact_ties=ties,
                strict_total=128 - ties, strict_fraction=agrees / (128 - ties) if ties < 128 else None,
                tie_compatible_fraction=(agrees + ties) / 128)),
        sums_of_case_median_edge_seconds={key: float(value) for key, value in sums.items()},
        exact_decimal_sums_seconds={key: str(value) for key, value in sums.items()},
        factors_baseline_time_over_policy_time={policy: {baseline: float(sums[baseline] / sums[policy])
            for baseline in ('always_pairwise', 'always_scan')} for policy in ('hindsight_best', 'frozen_proxy')},
        proxy_over_hindsight_time=float(sums['frozen_proxy'] / sums['hindsight_best']),
        cases=rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--friendster-archive', type=Path, default=ROOT/'work/vldb_artifact_v0_5_portable_v1/assets/lpp_public_friendster_cpu_matched_v2_raw.tgz')
    parser.add_argument('--citation2-extracted', type=Path, default=ROOT/'work/vldb_science_revision_20261126/e22_raw/extracted/citation2_cpu_matched_v1_20261126')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists() and not args.output.is_symlink(), 'fresh derived output required')
    fr = load_friendster(args.friendster_archive)
    c2 = load_citation2(args.citation2_extracted)
    output = dict(kind='DESCRIPTIVE_POSTHOC_MATCHED128_NUMBA32_COST_REGIMES',
        status='DERIVED_FROM_EXISTING_ACTUAL_RAW_TIMINGS',
        script_identity=pin(Path(__file__)),
        scope='Separate E20 Friendster128 and E22 strict-clean Citation2 128 matched cohorts; Numba32 edge construction only. No pooled E17/768 timing observations, new experiment, independently sampled queries, confidence interval or fitted dispatch threshold.',
        definitions=dict(Ds='sum of out-degrees over the frozen ordered set S', k='number of vertices in S', Q='k*(k-1)/2 unordered candidate pairs',
            observed_cost='median of the three original raw edge_seconds values per case/method; then sum case medians',
            correlation='Spearman correlation of average ranks for exact ties; ratios ranked as exact rationals of serialized medians',
            factors='baseline sum / selected-policy sum; >1 means lower hypothetical selected-policy edge cost',
            hindsight='per-case minimum of the two observed medians, a hindsight lower bound',
            proxy='Q<=max(1,Ds) chooses pairwise; otherwise scan. Fixed alpha1, no fitting or search.'),
        limitations=['All observations are repeated-cache fixed-cohort posthoc summaries; cases may repeat physical pairs or ordered sets.',
            'Policy sums reuse measured per-case medians; dispatch/feature-computation overhead is not measured.',
            'Only edge-construction boundary is compared, excluding DRNL, conversion, JIT, warmup, representation setup, models and training.',
            'The two datasets have separate cohorts and timings and are not pooled or treated as independent query samples.'],
        datasets=[analyze('Friendster_E20', *fr), analyze('Citation2_strict_clean_E22', *c2)],
        new_experiments=0, graph_model_native_library_loads=0, network_cluster_YT_actions=0, thresholds_optimized=0)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(args.output), datasets=[{key: item[key] for key in ('dataset','wins','spearman_Ds_over_Q_vs_scan_over_pair','fixed_proxy','sums_of_case_median_edge_seconds','factors_baseline_time_over_policy_time','proxy_over_hindsight_time')} for item in output['datasets']]), indent=2))


if __name__ == '__main__':
    main()
