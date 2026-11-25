"""Public Citation2/Friendster projection of the unchanged frozen V4 protocol.

Plan by default. Execution runs24sequential jobs,20timings and12prefix checks.
There is no private graph, historical quality gate, remote access or deployment.
"""
from __future__ import annotations
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

from . import systems_common as c

SUITE = c.CODE / 'experiment_configs/seal_revision_2026-11/run_pipeline_suite_v4.py'
REFERENCES = ('reference_union_stock', 'reference_union_corrected', 'reference_union_compiled')


def make_plan(args):
    suite = c.load_file(SUITE, 'public_frozen_pipeline_suite_v4')
    original = suite.build_plan(args.out, c.CODE, sys.executable)
    replacements = {
        str(args.out / 'data/citation2/prepared/train_csr'): str(args.citation2_root / 'train_csr'),
        str(args.out / 'data/citation2/prepared/train_pairs_csr'): str(args.citation2_root / 'train_pairs_csr'),
        str(args.out / 'data/friendster/full_csr'): str(args.friendster_root),
    }
    jobs = []
    for original_job in original['jobs']:
        if original_job['dataset'] not in (None, 'citation2', 'friendster'):
            continue
        job = dict(original_job)
        job['argv'] = [replacements.get(value, value) for value in job['argv']]
        job['timeout_seconds'] = 3600
        jobs.append(job)
    c.require(len(jobs) == 24 and sum(j['kind'] == 'pipeline' for j in jobs) == 20, 'exact public24job projection required')
    return {'schema_version': 1, 'scope': 'public2graph V4 short featureless training control; no quality/time-to-quality claim',
            'protocol': original['protocol'], 'original_protocol_sha256': original['protocol_sha256'],
            'projection': {'datasets': ['citation2', 'friendster'], 'jobs': 24, 'timings': 20,
                           'manifests': 2, 'fixtures': 2, 'real_prefix_comparisons': 12},
            'jobs': jobs, 'execute': args.execute, 'input_identity_checked': False,
            'coordinator_limits': c.coordinator_limits(args),
            'frozen_sources': c.frozen_sources()}


def validate_job(job, exit_data):
    output = Path(job['out'])
    report = c.read(output / 'report.json')
    c.require(report.get('benchmark_version') == 4, 'V4report required')
    runs = report.get('runs', [])
    if job['kind'] == 'expected_gpu_refusal':
        messages = '\n'.join(p.read_text() for p in output.rglob('process.log') if p.stat().st_size <= 8 << 20)
        c.require(exit_data['returncode'] != 0 and not exit_data.get('timed_out')
                  and exit_data.get('guard_abort_reason') is None and exit_data.get('owned_tree_quiescent') is True
                  and not exit_data.get('orphan_cleanup') and len(runs) == 1
                  and runs[0].get('status') == 'failed' and 'GPU preflight refused' in messages,
                  'only the declared GPU capacity refusal is expected')
        return report
    c.require(c.clean_exit(exit_data), 'owned phase failed or was guarded')
    pins = c.read(c.ROOT / 'metadata/SYSTEMS_FROZEN_INPUTS.json')['files']
    for name, actual in report.get('source_sha256', {}).items():
        c.require(pins.get('src/' + name) == actual, 'reported frozen runtime source differs')
    c.require(len(report.get('source_sha256', {})) == 7, 'complete runtime source bindings required')
    if job['kind'] == 'manifest':
        manifest = report['manifest']
        c.require(not runs and manifest['rows'] == 8000 and manifest['positive_occurrences'] == 4000
                  and Path(manifest['path']) == Path(job['pairs']) and c.sha(job['pairs']) == manifest['sha256'],
                  'exact8000row public workload manifest required')
        return report
    expected_runs = 5 if job['kind'] == 'gpu_smoke' else 1
    c.require(len(runs) == expected_runs, 'exact arm count required')
    for run in runs:
        c.require(run.get('status') == 'complete' and run.get('returncode') == 0 and run.get('abort_reason') is None,
                  'status alone cannot override failed or guarded timing')
        c.require(run['device']['type'] == 'cuda' and run['device']['visible_gpu_count'] == 1,
                  'oneGPU and no CPU fallback required')
        guard = run['gpu_memory']['guard']
        c.require(guard['status'] == 'configured_before_model_allocation' and guard['allocator_budget_bytes'] == 24 * c.GIB
                  and 0 < guard['allocator_fraction'] <= 1 and run['gpu_memory']['reserved_peak_gib'] <= 24,
                  'exact24GiB before-model allocator gate required')
        c.require(run['model']['sortpool_k'] == 281 and run['model']['use_feature'] is False,
                  'featureless k281 model required')
        if job['kind'] == 'pipeline':
            cfg, timing = run['config'], run['timing']
            c.require(cfg['arm'] == job['arm'] and cfg['num_workers'] == job['workers']
                      and cfg['stock_index_dtype'] == 'auto' and cfg['batch_size'] == 32
                      and cfg['seed'] == 20261124 and cfg['audit_pairs'] == 32
                      and timing['measured_steps'] == 100 and timing['measured_graphs'] == 3200
                      and timing['warmup_graphs_excluded'] == 256 and timing['steady_seconds'] > 0
                      and abs(timing['graphs_per_second'] - 3200 / timing['steady_seconds']) < 1e-9
                      and cfg['manifest_sha256'] == c.sha(job['pairs']), 'frozen V4timing denominator/protocol differs')
            c.require([a['index'] for a in run['audits']] == list(range(32)), 'all32warmup-prefix audits required')
            if job['arm'] == 'reference_union_stock':
                c.require(run['stock_physical_route']['measured_temporary_row_dtype_counts'] == {'int32': 3200},
                          'native-auto measured stock temporary must be int32')
    if job['kind'] == 'gpu_smoke':
        pairs = report['equality_audits']
        c.require({(a['reference_arm'], a['candidate_arm']) for a in pairs} == set(combinations(REFERENCES, 2))
                  and len(pairs) == 3 and all(a['same_nodes_edges_drnl_and_initial_model'] for a in pairs),
                  'fivearm fixture must pass all three reference-pair checks')
    return report


def comparisons(reports):
    checks = []
    for dataset in ('citation2', 'friendster'):
        for workers in (1, 8):
            for left, right in combinations(REFERENCES, 2):
                a = reports[f'{dataset}_{left}_workers{workers}']['runs'][0]
                b = reports[f'{dataset}_{right}_workers{workers}']['runs'][0]
                c.require(a['audits'] == b['audits'] and a['model']['initial_state_sha256'] == b['model']['initial_state_sha256']
                          and a['config']['manifest_sha256'] == b['config']['manifest_sha256'], 'fixed-prefix reference equivalence failed')
                checks.append({'dataset': dataset, 'workers': workers, 'reference_arm': left, 'candidate_arm': right,
                               'queries': 32, 'status': 'matched', 'scope': 'excluded warmup prefix only; not all3200measured graphs'})
    c.require(len(checks) == 12, 'twelve public prefix comparisons required')
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    c.common_arguments(parser)
    parser.add_argument('--citation2-root', type=Path, required=True, help='strict-clean prepared Citation2 directory')
    parser.add_argument('--friendster-root', type=Path, required=True, help='public full CSR, not quality train split')
    args = parser.parse_args(argv)
    args.citation2_root, args.friendster_root = args.citation2_root.resolve(), args.friendster_root.resolve()
    c.validate_arguments(args, (args.citation2_root, args.friendster_root))
    plan = make_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    with c.Execution(args, plan, inputs=(args.citation2_root, args.friendster_root), gpu=True, h100=True) as execution:
        c.save(args.out / 'input_identity.json', {
            'citation2': c.verify_inputs(args.citation2_root, 'metadata/citation2_prepared_expected.json', names={
                'num_nodes.txt', 'train_csr/edge_starts.bin', 'train_csr/edge_ends.bin',
                'train_pairs_csr/edge_starts.bin', 'train_pairs_csr/edge_ends.bin'}),
            'friendster': c.verify_inputs(args.friendster_root, 'metadata/friendster_full_expected.json')})
        reports = {}
        for job in plan['jobs']:
            report = validate_job(job, execution.run(job))
            if job['kind'] == 'expected_gpu_refusal':
                execution.records[-1]['expected_guard_refusal'] = True
            reports[job['name']] = report
        c.save(args.out / 'scientific_summary.json', {'status': 'complete', 'jobs': 24, 'timing_jobs': 20,
             'reference_correction_audits': comparisons(reports), 'scope': plan['scope']})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
