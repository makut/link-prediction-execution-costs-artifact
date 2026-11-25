"""Independently recompute published E16 sums from the saved public raw timings."""
import collections
import json
import math
import statistics

from . import systems_common as c


def verify():
    folder = c.ROOT / 'metadata/evidence/E16'
    report_path = folder / 'dispatch_report.json'
    c.require(c.sha(report_path) == '9657f9b745863333dfc11c30716854c3f03f46e776ea580c910a657a4674263b', 'E16raw report SHA differs')
    report = c.read(report_path)
    reference = c.read(c.ROOT / 'data/friendster_exact_int32_reference.json')
    rows = report['records']
    saved_lines = [json.loads(line) for line in (folder / 'dispatch_records.jsonl').read_text().splitlines()]
    c.require(len(rows) == 768 and rows == saved_lines, 'all768 raw records and JSONL copies required')
    c.require(report['status'] == 'complete' and report['configuration']['repeats'] == 5
              and report['configuration']['warmup'] == 1 and report['configuration']['auto_ratio'] == 1
              and not report['configuration']['include_precomputed_lower_bound'], 'frozen E16protocol differs')
    methods = ('fixed_pairwise', 'fixed_scan', 'exact_python_auto', 'production_vectorized_auto')
    routes = ('exact_python_auto', 'production_vectorized_auto')
    edge_values, route_values = {m: [] for m in methods}, {m: [] for m in routes}
    for row in rows:
        original = reference['records'][row['input_index']]
        c.require(row['pair'] == original['pair'] and row['node_set_sha256'] == original['node_set_sha256']
                  and row['canonical_edges_sha256'] == original['canonical_edges_sha256']
                  and row['drnl_sha256'] == original['drnl_sha256']
                  and row['all_results_match_archived_edges_and_drnl'] is True, 'public frozen query/structural identity differs')
        for name in methods:
            samples = row['methods'][name]['edge_seconds']
            c.require(len(samples) == 5 and all(math.isfinite(x) and x > 0 for x in samples), 'five positive finite timing samples required')
            value = statistics.median(samples)
            c.require(value == row['methods'][name]['edge_summary']['p50_s'], 'per-record median differs')
            edge_values[name].append(value)
        for name in routes:
            samples = row['route_only'][name]['seconds']
            c.require(len(samples) == 5 and all(math.isfinite(x) and x > 0 for x in samples), 'five routing samples required')
            value = statistics.median(samples)
            c.require(value == row['route_only'][name]['summary']['p50_s'], 'routing median differs')
            route_values[name].append(value)
    edge = {k: math.fsum(v) for k, v in edge_values.items()}
    route = {k: math.fsum(v) for k, v in route_values.items()}
    for actual, expected in ((edge, report['summary']['edge_sum_of_per_record_medians_seconds']),
                             (route, report['summary']['route_only_sum_of_per_record_medians_seconds'])):
        c.require(set(actual) == set(expected) and all(math.isclose(actual[k], expected[k], rel_tol=1e-12) for k in actual), 'saved sums differ')
    c.require(dict(collections.Counter(r['selected_route'] for r in rows)) == report['summary']['auto_route_counts'], 'routing counts differ')
    return {'status': 'verified', 'records': 768, 'edge_sum_of_per_record_medians_seconds': edge,
            'route_only_sum_of_per_record_medians_seconds': route,
            'python_auto_over_vectorized_auto_sum_ratio': edge['exact_python_auto'] / edge['production_vectorized_auto'],
            'scope': 'Independent raw timing aggregation; no new performance experiment or production/pipeline speedup claim'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
