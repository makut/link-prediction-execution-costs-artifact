"""Protocol/validator tests; fabricated JSON here is test data, never science."""
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import run_pipeline_repeats as r


class ProtocolTests(unittest.TestCase):
    def args(self, out=None):
        return r.arguments(['--citation2-root', '/public/citation2/prepared',
            '--friendster-root', '/public/friendster/full_csr', '--out', out or '/scratch/new_repeat_run'])

    def test_exact_public_matrix_and_latin_position_balance(self):
        plan = r.make_plan(self.args())
        self.assertEqual(40, len(plan['jobs']))
        timings = [j for j in plan['jobs'] if j['kind'] == 'pipeline']
        self.assertEqual(36, len(timings))
        self.assertEqual(40, len({j['name'] for j in plan['jobs']}))
        self.assertEqual({'citation2', 'friendster'}, {j['dataset'] for j in timings})
        self.assertEqual(set(r.REFERENCES), {j['arm'] for j in timings})
        self.assertEqual(12, len(plan['blocks']))
        for ds in r.DATASETS:
            for w in r.WORKERS:
                subset = [j for j in timings if j['dataset'] == ds and j['workers'] == w]
                self.assertEqual(Counter({(arm, pos): 1 for arm in r.REFERENCES for pos in (1, 2, 3)}),
                                 Counter((j['arm'], j['position']) for j in subset))
                self.assertEqual({1, 2, 3}, {j['repetition'] for j in subset})
        self.assertEqual(plan, r.make_plan(self.args()))

    def test_exact_finite_window_arguments_and_original_workload(self):
        plan = r.make_plan(self.args())
        for j in plan['jobs']:
            if j['kind'] != 'pipeline':
                continue
            a = j['argv']
            for option, value in {'--seed': '20261124', '--batch-sizes': '32', '--steps': '100',
                '--warmup-steps': '8', '--prefetch-factor': '1', '--cap': '200', '--hops': '2',
                '--stock-index-dtype': 'auto', '--device': 'cuda', '--memory-budget-gib': '180',
                '--gpu-memory-budget-gib': '24'}.items():
                self.assertEqual(value, a[a.index(option) + 1])
            self.assertEqual(j['arm'], a[a.index('--arms') + 1])
            self.assertEqual(str(j['workers']), a[a.index('--workers') + 1])
            self.assertNotIn('--fixture', a)
        pins = r.verify_sources()
        self.assertEqual('648332c22cb6a25825e8487e2e82e39e2cb5dc53c8890cb3acb71b6324d7b750',
                         pins['workloads']['citation2']['sha256'])
        self.assertEqual('340e61cdd1eb8d329d52be164f919480e6a41feb4c620852df0393c2e9a192f0',
                         pins['workloads']['friendster']['sha256'])

    def test_output_is_disjoint_and_budgets_finite_bounded(self):
        with self.assertRaises(RuntimeError):
            self.args(str(r.HERE / 'results'))
        with self.assertRaises(RuntimeError):
            self.args('/public/citation2/prepared/results')
        for flag, value in [('--max-hours', '7'), ('--memory-budget-gib', '181'),
                            ('--max-output-gib', 'nan')]:
            with self.assertRaises(RuntimeError):
                r.arguments(['--citation2-root', '/c2', '--friendster-root', '/fr', '--out', '/scratch/r', flag, value])

    def test_no_plan_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not_created'
            r.make_plan(self.args(str(out)))
            self.assertFalse(out.exists())


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.pins = r.verify_sources()
        self.job = dict(kind='pipeline', out=str(self.out), dataset='citation2', arm=r.REFERENCES[0], workers=8)
        self.exit = dict(returncode=0, timed_out=False, owned_tree_quiescent=True,
                         orphan_cleanup=False, guard_abort_reason=None)
        sources = ['scripts/seal_revision/pipeline_benchmark_v4.py', 'scripts/seal_revision/exact_benchmark.py',
                   'scripts/seal_revision/memory_benchmark.py', 'seal/models.py', 'seal/utils.py',
                   'seal/graph_csr_kernels.py', 'seal/graph_csr_extraction.py']
        self.report = dict(benchmark_version=4, runtime_versions=self.pins['original_runtime_versions'],
            source_sha256={n:self.pins['frozen_sources']['frozen_src/' + n]['sha256'] for n in sources},
            runs=[dict(status='complete', returncode=0, abort_reason=None,
                device=dict(type='cuda', visible_gpu_count=1),
                gpu_memory=dict(reserved_peak_gib=1, guard=dict(status='configured_before_model_allocation',
                    allocator_budget_bytes=24 * r.GIB, allocator_fraction=.3)),
                model=dict(sortpool_k=281, use_feature=False,
                    initial_state_sha256=self.pins['original_initial_model_state_sha256']),
                config=dict(arm=r.REFERENCES[0], num_workers=8, batch_size=32, seed=20261124,
                    audit_pairs=32, stock_index_dtype='auto',
                    manifest_sha256=self.pins['workloads']['citation2']['sha256']),
                timing=dict(measured_steps=100, measured_graphs=3200, warmup_graphs_excluded=256,
                    steady_seconds=2, graphs_per_second=1600),
                audits=[dict(index=i, nodes_sha256='fixture-node', edges_sha256='fixture-edge', drnl_sha256='fixture-label')
                        for i in range(32)],
                stock_physical_route=dict(measured_temporary_row_dtype_counts={'int32': 3200}))])

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, report=None, exit_data=None):
        (self.out / 'report.json').write_text(json.dumps(report if report is not None else self.report))
        return r.validate_job(self.job, exit_data if exit_data is not None else self.exit, self.pins)

    def test_positive_protocol_fixture_is_accepted(self):
        self.validate()

    def test_complete_status_does_not_override_abort_or_exit(self):
        for scope, key, value in [('inner', 'returncode', 1), ('inner', 'abort_reason', 'guard'),
                                 ('outer', 'returncode', 1), ('outer', 'returncode', False),
                                 ('outer', 'guard_abort_reason', 'memory'), ('outer', 'owned_tree_quiescent', False),
                                 ('outer', 'orphan_cleanup', True), ('outer', 'timed_out', True)]:
            report, ex = deepcopy(self.report), deepcopy(self.exit)
            (report['runs'][0] if scope == 'inner' else ex)[key] = value
            with self.assertRaises(RuntimeError):
                self.validate(report, ex)

    def test_protocol_source_denominator_model_and_width_changes_rejected(self):
        cases = [(['runs', 0, 'timing', 'measured_graphs'], 3199),
                 (['runs', 0, 'config', 'seed'], 20261125),
                 (['runs', 0, 'config', 'arm'], 'production_pv'),
                 (['runs', 0, 'model', 'initial_state_sha256'], 'different'),
                 (['runs', 0, 'device', 'type'], 'cpu'),
                 (['runs', 0, 'stock_physical_route', 'measured_temporary_row_dtype_counts'], {'int64': 3200}),
                 (['runtime_versions', 'torch'], '2.9.0'),
                 (['source_sha256', 'seal/models.py'], 'different')]
        for path, value in cases:
            report = deepcopy(self.report)
            obj = report
            for component in path[:-1]:
                obj = obj[component]
            obj[path[-1]] = value
            with self.assertRaises(RuntimeError):
                self.validate(report)

    def test_nonfinite_or_zero_wall_refused(self):
        for value in (0, -1):
            report = deepcopy(self.report)
            report['runs'][0]['timing']['steady_seconds'] = value
            with self.assertRaises(RuntimeError):
                self.validate(report)
        (self.out / 'report.json').write_text(json.dumps(self.report).replace('"steady_seconds": 2', '"steady_seconds": NaN'))
        with self.assertRaises(ValueError):
            r.validate_job(self.job, self.exit, self.pins)

    def test_cross_arm_prefix_mismatch_refused(self):
        jobs, reports = [], {}
        for arm in r.REFERENCES:
            name = 'block_' + arm
            jobs.append(dict(name=name, arm=arm, block='block'))
            reports[name] = deepcopy(self.report)
        self.assertEqual(3, len(r.compare_block(jobs, reports)))
        reports[jobs[-1]['name']]['runs'][0]['audits'][0]['nodes_sha256'] = 'changed'
        with self.assertRaises(RuntimeError):
            r.compare_block(jobs, reports)

    def test_summary_retains_three_observations_and_pairs_ratios_before_aggregation(self):
        args = r.arguments(['--citation2-root', '/c2', '--friendster-root', '/fr', '--out', '/scratch/newrun'])
        plan = r.make_plan(args)
        values = {r.REFERENCES[0]: [10., 20., 30.], r.REFERENCES[1]: [9., 5., 10.],
                  r.REFERENCES[2]: [2., 3., 4.]}
        reports = {}
        for job in plan['jobs']:
            if job['kind'] != 'pipeline':
                continue
            report = deepcopy(self.report)
            report['runs'][0]['timing']['steady_seconds'] = values[job['arm']][job['repetition'] - 1]
            reports[job['name']] = report
        summary = r.summarize(plan, reports)
        self.assertEqual(36, len(summary['reference_prefix_comparisons']))
        self.assertEqual(12, len(summary['rows']))
        row = next(x for x in summary['rows'] if x['dataset'] == 'citation2' and x['workers'] == 1
                   and x['arm'] == r.REFERENCES[0])
        self.assertEqual([10., 20., 30.], row['steady_seconds'])
        self.assertEqual(20., row['mean'])
        self.assertEqual(10., row['sample_sd'])
        ratio = next(x for x in summary['paired_time_ratios'] if x['dataset'] == 'citation2' and x['workers'] == 1
                     and x['numerator'] == r.REFERENCES[0] and x['denominator'] == r.REFERENCES[1])
        self.assertEqual([10 / 9, 4., 3.], ratio['paired_time_ratios'])
        self.assertEqual(3., ratio['median'])
        del reports[next(j['name'] for j in plan['jobs'] if j['kind'] == 'pipeline')]
        with self.assertRaises(RuntimeError):
            r.summarize(plan, reports)


if __name__ == '__main__':
    unittest.main()
