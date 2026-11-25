"""Public Friendster projection of the existing E17 posthoc analysis; no new timings."""
import argparse
import json
import math
from pathlib import Path
import shutil

from . import systems_common as c


def analyze(out):
    original = c.load_file(c.ROOT / 'work/vldb_literature/analyze_fixed_s_cost_regimes_v1.py', 'public_original_cost_analysis')
    checker = c.load_file(c.ROOT / 'work/vldb_literature/check_fixed_s_cost_regimes_v1.py', 'public_original_cost_checker')
    # Only filesystem roots and graph selection change; the numerical functions are unchanged.
    shutil.copyfile(c.ROOT / 'data/friendster_exact_int32_reference.json', out / 'friendster_exact_int32.json')
    original.ROOT = original.INPUT = out
    summary, rows = original.analyze('friendster')
    source = c.read(out / 'friendster_exact_int32.json')
    independent = []
    for record in source['records']:
        a = sorted(record['methods']['ours_scan']['edge_seconds'])[1]
        b = sorted(record['methods']['ours_pairwise']['edge_seconds'])[1]
        q = record['num_nodes'] * (record['num_nodes'] - 1) // 2
        d = record['sum_out_degrees']
        independent.append((a, b, min(a, b), b if q <= max(1, d) else a, d / q, a / b))
    sums = [math.fsum(row[i] for row in independent) for i in range(4)]
    for key, value in zip(('always_scan', 'always_pair', 'observed_best_branch', 'declared_proxy_branch'), sums):
        c.require(math.isclose(value, summary['sum_per_record_medians_s'][key], rel_tol=1e-12), 'independent sum differs')
    rho = checker.spearman([r[4] for r in independent], [r[5] for r in independent])
    c.require(abs(rho - summary['spearman_D_over_Q_vs_scan_over_pair']) < 1e-12, 'independent tie-aware Spearman differs')
    result = {'status': 'verified_public_friendster_projection', 'scope': 'E17posthoc stored-timing diagnostic only; no experiment/HPO/selector change',
              'projection': 'Friendster only; no private VK records required or included', 'graphs': [summary], 'rows': rows,
              'independent_check': {'sums': sums, 'tie_aware_spearman': rho},
              'original_analysis_source_sha256': c.sha(c.ROOT / 'work/vldb_literature/analyze_fixed_s_cost_regimes_v1.py')}
    c.save(out / 'analysis.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    c.require(not out.exists() and not out.is_relative_to(c.ROOT), 'fresh output outside the artifact required')
    c.frozen_sources()
    if not args.execute:
        print(json.dumps({'execute': False, 'out': str(out), 'graph': 'friendster', 'rows': 768, 'new_timings': 0}))
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    c.require(shutil.disk_usage(out.parent).free >= 32 << 20, '32MiB free disk required for public analysis')
    out.mkdir()
    result = analyze(out)
    print(json.dumps({'status': result['status'], 'records': len(result['rows']), 'new_timings': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
