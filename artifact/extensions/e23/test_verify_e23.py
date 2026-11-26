"""Small synthetic validator checks. No archived/scientific results generated."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import verify_e23 as v


class VerifierChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pins, cls.protocol, cls.workloads, cls.rows = v.local_sources()

    def exit_fixture(self):
        return dict(pid=123, returncode=0, timed_out=False, orphan_cleanup=False,
                    owned_tree_quiescent=True, owned_pgids_signaled=[],
                    timeout_seconds=3600, guard_abort_reason=None)

    def report_fixture(self):
        """One in-memory synthetic run; values are validator test data only."""
        workload = self.workloads['citation2']
        cfg = dict(fixture=False, file_endian='little', seed=20261124, hops=2,
                   cap=200, warmup_pairs=256, steps=100, prefetch_factor=1,
                   audit_pairs=32, device='cuda', torch_threads=1, gpu_memory_budget_gib=24.,
                   stock_index_dtype='auto', batch_size=32, num_workers=8, arm=v.ARMS[0],
                   max_neighbors=20000000,
                   manifest_sha256=self.pins['workloads']['citation2']['sha256'])
        total, free = 80 * v.GIB, 70 * v.GIB
        return dict(status='complete', returncode=0, abort_reason=None, config=cfg, arm=v.ARMS[0],
            device=dict(type='cuda', name='SYNTHETIC H100 VALIDATOR FIXTURE', visible_gpu_count=1,
                        used_devices=['cuda:0'], torch_version=self.pins['original_runtime_versions']['torch']),
            model=dict(v.MODEL, initial_state_sha256=self.pins['original_initial_model_state_sha256'],
                       final_state_sha256='a' * 64),
            gpu_memory=dict(allocated_peak_gib=1., reserved_peak_gib=2., guard=dict(
                status='configured_before_model_allocation', cuda_api_called=True,
                allocator_budget_bytes=24 * v.GIB, requested_budget_gib=24.,
                total_bytes=total, free_bytes_before_model=free, allocator_fraction=24 * v.GIB / total)),
            timing=dict(measured_graphs=3200, measured_steps=100, warmup_graphs_excluded=256,
                        warmup_steps=8, steady_seconds=2., graphs_per_second=1600.,
                        measured_positive_occurrences=sum(r[2] for r in workload[256:3456]),
                        prefetch_batches_capacity=8, prefetch_boundary_fraction=.08),
            losses=[.5] * 100,
            audits=[dict(index=i, pair=list(row[:2]), label=row[2], nodes_sha256='b' * 64,
                         edges_sha256='c' * 64, drnl_sha256='d' * 64)
                    for i, row in enumerate(workload[:32])],
            stock_physical_route=dict(measured_temporary_row_dtype_counts={'int32': 3200}))

    def test_source_grid_and_latin_balance(self):
        blocks, jobs = v.schedule()
        self.assertEqual((12, 40), (len(blocks), len(jobs)))
        self.assertEqual(40, len({j['name'] for j in jobs}))
        actual_source_only_plan = v.read_json(v.HERE / 'PLAN_ONLY_EXAMPLE.json')
        self.assertFalse(actual_source_only_plan['execute'])
        self.assertEqual(blocks, actual_source_only_plan['blocks'])
        for actual, expected in zip(actual_source_only_plan['jobs'], jobs):
            self.assertTrue(all(actual.get(k) == value for k, value in expected.items()))
        for dataset in v.DATASETS:
            for workers in v.WORKERS:
                subset = [j for j in jobs if j.get('dataset') == dataset and j.get('workers') == workers]
                self.assertEqual(9, len(subset))
                self.assertEqual({(a, p) for a in v.ARMS for p in (1, 2, 3)},
                                 {(j['arm'], j['position']) for j in subset})

    def test_strict_exit_rejects_bool_abort_or_cleanup(self):
        v.check_clean_exit(self.exit_fixture())
        for key, value in [('returncode', False), ('returncode', 1), ('timed_out', True),
                           ('orphan_cleanup', True), ('owned_tree_quiescent', False),
                           ('owned_pgids_signaled', [9]), ('guard_abort_reason', 'memory')]:
            fixture = self.exit_fixture()
            fixture[key] = value
            with self.assertRaises(v.VerificationError):
                v.check_clean_exit(fixture)
        fixture = self.exit_fixture()
        fixture['returncode'] = 1
        v.check_clean_exit(fixture, expected_code=1)

    def test_one_report_protocol_and_meaningful_mutation_refusals(self):
        job = dict(dataset='citation2', arm=v.ARMS[0], workers=8)
        run = self.report_fixture()
        v.validate_run(run, job, self.pins, self.workloads['citation2'])
        changes = [('returncode', 1), ('abort_reason', 'OOM'),
            ('model.initial_state_sha256', 'f' * 64), ('config.manifest_sha256', 'f' * 64),
            ('config.arm', 'production_pv'), ('config.num_workers', 1),
            ('timing.measured_graphs', 3199), ('timing.steady_seconds', 0),
            ('gpu_memory.guard.allocator_budget_bytes', 25 * v.GIB),
            ('gpu_memory.reserved_peak_gib', 25), ('device.type', 'cpu'),
            ('stock_physical_route.measured_temporary_row_dtype_counts', {'int64': 3200})]
        for path, value in changes:
            altered = deepcopy(run)
            target = altered
            keys = path.split('.')
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            with self.assertRaises(v.VerificationError, msg=path):
                v.validate_run(altered, job, self.pins, self.workloads['citation2'])
        altered = deepcopy(run)
        altered['audits'][0]['pair'].reverse()
        with self.assertRaises(v.VerificationError):
            v.validate_run(altered, job, self.pins, self.workloads['citation2'])

    def test_independent_arithmetic_and_pair_before_aggregation(self):
        self.assertEqual(dict(mean=20., sample_sd=10., median=20., minimum=10., maximum=30.),
                         v.descriptive([10., 20., 30.]))
        ratios = [a / b for a, b in zip([10., 20., 30.], [9., 5., 10.])]
        self.assertEqual(3., v.descriptive(ratios)['median'])
        self.assertNotEqual(v.descriptive([10., 20., 30.])['median'] /
                            v.descriptive([9., 5., 10.])['median'], 3.)
        with self.assertRaises(v.VerificationError):
            v.descriptive([1., 2.])
        with self.assertRaises(v.VerificationError):
            v.matching([dict(mean=20.)], [dict(mean=21.)], 'synthetic')

    def test_json_nonfinite_duplicates_and_paths_refused(self):
        for data in (b'{"x":NaN}', b'{"x":1,"x":2}'):
            with self.assertRaises(v.VerificationError):
                v.decode(data)
        for name in ('/absolute', '../escape', 'e23/../escape', 'e23//bad', 'e23\\bad'):
            with self.assertRaises(v.VerificationError):
                v.safe_name(name)

    def test_small_archive_hash_binding_and_tamper_detection(self):
        """Byte-level test archive has no E23 result/report structure."""
        content = b'{"synthetic_validator_test_only":true}\n'
        name = 'e23/' + 'synthetic_long_path_' * 6 + '/validator_test.json'
        manifest = dict(files={name: dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())},
                        graphs_included=False, numba_cache_included=False)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic_validator_only.tgz'
            with tarfile.open(path, 'w:gz') as tar:
                for member_name, body in ((name, content), ('SHA256_MANIFEST.json', json.dumps(manifest).encode())):
                    member = tarfile.TarInfo(member_name)
                    member.size = len(body)
                    tar.addfile(member, io.BytesIO(body))
            actual = v.digest_file(path, v.MAX_COMPRESSED)['sha256']
            loaded = v.Archive(path, actual)
            self.assertEqual(True, loaded.get(name)['synthetic_validator_test_only'])
            with self.assertRaises(v.VerificationError):
                v.Archive(path, '0' * 64)


if __name__ == '__main__':
    unittest.main()
