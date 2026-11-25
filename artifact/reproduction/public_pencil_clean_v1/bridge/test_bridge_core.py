import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import bridge_core as b
import pencil_bridge as driver


class Operators(unittest.TestCase):
    def test_actual_numba_both_widths_against_independent_raw_slot_oracle(self):
        rng = np.random.default_rng(712)
        for width in (np.int32, np.int64):
            for graph_seed in range(5):
                n = 23
                mask = rng.random((n, n)) < .12
                np.fill_diagonal(mask, False)
                rows, cols = np.nonzero(mask)
                ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))].astype(np.int64)
                col = cols.astype(width)
                for nodes in (np.array([17, 4, 0, 11, 8], dtype=np.int64),
                              np.array([9], dtype=np.int64), np.arange(n, dtype=np.int64)):
                    expected = b.oracle(ptr, col, nodes)
                    for mode in ("scan", "pairwise"):
                        actual = b.CompiledBackend(ptr, col, mode).induce(nodes)
                        for a, e in zip(actual, expected):
                            self.assertEqual(a.dtype, np.int64)
                            np.testing.assert_array_equal(a, e)

    def test_unsorted_S_preserves_native_csr_scan_order_not_local_sort(self):
        g, _, _, _ = b.tiny_graph()
        nodes = np.array([4, 0, 3, 2, 1], dtype=np.int64)
        raw = b.CompiledBackend(g.rowptr, g.col, "scan").induce(nodes)
        np.testing.assert_array_equal(raw[1][:2], [3, 2])  # row4 neighbors2,3 map to3,2
        np.testing.assert_array_equal(raw[2][:2], [8, 9])

    def test_single_cold_node_and_equal_roots(self):
        g, _, _, _ = b.tiny_graph()
        nodes = np.array([8], dtype=np.int64)
        for mode in ("scan", "pairwise"):
            raw = b.CompiledBackend(g.rowptr, g.col, mode).induce(nodes)
            self.assertTrue(all(len(a) == 0 for a in raw))
            self.assertEqual(b.canonical_postmask(raw, [0, 0]).shape, (2, 0))

    def test_both_target_directions_removed_at_arbitrary_positions(self):
        g, _, _, _ = b.tiny_graph()
        nodes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        raw = b.oracle(g.rowptr, g.col, nodes)
        edges = b.canonical_postmask(raw, [3, 1])
        self.assertFalse(any(tuple(x) in ((3, 1), (1, 3)) for x in edges.T))
        self.assertEqual(edges.shape[1], len(raw[0]) - 2)

    def test_invalid_subset_dtype_duplicates_bounds_rejected(self):
        for nodes in (np.array([], dtype=np.int64), np.array([0, 0], dtype=np.int64),
                      np.array([0, 9], dtype=np.int64), np.array([0], dtype=np.int32),
                      np.array([0, 1, 2, 3], dtype=np.int64)[::2]):
            with self.assertRaises(ValueError): b.validate_nodes(nodes, 9)

    def test_touched_source_rows_contract_and_query_budget(self):
        g, _, _, _ = b.tiny_graph()
        with self.assertRaises(b.GuardError): g.validate_rows(np.array([0, 1]), 1)
        for cols in ([2, 1], [1, 1], [0, 2], [1, 5]):
            graph = b.Graph(np.array([0, 2, 2, 2], dtype=np.int64), np.array(cols, dtype=np.int32), {})
            with self.assertRaises(ValueError): graph.validate_rows(np.array([0]))


class CohortAndTrace(unittest.TestCase):
    def arrays(self, root):
        pos = np.array([[0, 2], [1, 4], [3, 0]], dtype=np.int64)
        neg = np.array([[[0, 4], [0, 0]], [[1, 0], [1, 3]], [[3, 3], [3, 1]]], dtype=np.int64)
        np.save(root / "pos.npy", pos); np.save(root / "neg.npy", neg.reshape(-1, 2))
        return pos, neg

    def test_fixed_candidate_zero_and_all_positive_order(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); pos, neg = self.arrays(root)
            pairs, labels, meta = b.cohort(root / "pos.npy", root / "neg.npy", 5, expected_sources=3, expected_negatives=2)
            np.testing.assert_array_equal(pairs, np.concatenate([pos, neg[:, 0]]))
            np.testing.assert_array_equal(labels, [1, 1, 1, 0, 0, 0])
            self.assertFalse(meta["selection_depends_on_timing_or_scores"])
            self.assertEqual(pairs[-1].tolist(), [3, 3])  # Self-query retained.

    def test_later_negative_source_corruption_is_checked(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); _, neg = self.arrays(root); neg[2, 1, 0] = 0
            np.save(root / "neg.npy", neg.reshape(-1, 2))
            with self.assertRaisesRegex(ValueError, "grouping"):
                b.cohort(root / "pos.npy", root / "neg.npy", 5, expected_sources=3, expected_negatives=2)

    def trace(self, root):
        pairs = np.array([[3, 0], [8, 8]], dtype=np.int64)
        labels = np.array([0, 0], dtype=np.int64)
        nodes = [np.array([0, 1, 3], dtype=np.int64), np.array([8], dtype=np.int64)]
        roots = [np.array([2, 0]), np.array([0, 0])]
        return b.save_trace(root, pairs, labels, nodes, roots, [{"query_index": 0}, {"query_index": 1}], {"fixture": True}, 100000)

    def test_ragged_roundtrip_arbitrary_equal_root_positions(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "trace"; self.trace(root); trace = b.Trace(root)
            self.assertEqual(len(trace), 2)
            np.testing.assert_array_equal(trace.nodes(0), [0, 1, 3])
            np.testing.assert_array_equal(trace.nodes(1)[trace.roots[1]], [8, 8])

    def test_same_size_trace_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "trace"; self.trace(root)
            with (root / "nodes.npy").open("r+b") as f:
                f.seek(-4, 2); f.write(np.array([7], dtype=np.int32).tobytes())
            with self.assertRaisesRegex(ValueError, "hash"):
                b.Trace(root)

    def test_bad_roots_and_unsorted_S_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as t:
            for nodes, roots in (([2, 0], [0, 1]), ([0, 2], [-1, 0])):
                with self.assertRaises(ValueError):
                    b.save_trace(Path(t) / "trace", np.array([[2, 0]], dtype=np.int64),
                        np.array([0]), [np.array(nodes)], [np.array(roots)], [{}], {}, 1000)
            self.assertFalse((Path(t) / "trace").exists())

    def test_trace_budget_and_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "trace"; self.trace(root)
            with self.assertRaises(FileExistsError): self.trace(root)
            with self.assertRaises(b.GuardError):
                b.save_trace(Path(t) / "new", np.array([[0, 1]]), np.array([1]),
                    [np.array([0, 1])], [np.array([0, 1])], [{}], {}, 1)


class Safety(unittest.TestCase):
    def test_output_cannot_overwrite_any_consumed_file_or_subtree_on_failure(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); graph = root / "graph"; graph.mkdir()
            trace = root / "trace"; trace.mkdir(); config = root / "config"; config.mkdir()
            names = [graph / "edge_starts.bin", graph / "edge_ends.bin", trace / "manifest.json",
                     trace / "nodes.npy", config / "config.json", config / "source.json", root / "audit.json"]
            for p in names: p.write_bytes(b"preserved input bytes")
            common = ["audit", "--fixture", "--graph-path", str(graph), "--trace-dir", str(trace),
                      "--bert-config", str(config), "--audit-receipt", str(root / "audit.json")]
            for path in names:
                self.assertEqual(driver.main([*common, "--output", str(path)]), 2)
                self.assertEqual(path.read_bytes(), b"preserved input bytes")
            new_input_child = graph / "new_status.json"
            self.assertEqual(driver.main([*common, "--output", str(new_input_child)]), 2)
            self.assertFalse(new_input_child.exists())

    def test_any_existing_output_and_capture_trace_are_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); output = root / "failed.json"; output.write_text('{"status":"failed"}')
            self.assertEqual(driver.main(["preflight", "--fixture", "--output", str(output)]), 2)
            self.assertEqual(output.read_text(), '{"status":"failed"}')
            trace = root / "trace"; trace.mkdir(); (trace / "partial").write_text("keep")
            fresh = root / "new.json"
            self.assertEqual(driver.main(["capture", "--fixture", "--trace-dir", str(trace), "--output", str(fresh)]), 2)
            self.assertFalse(fresh.exists()); self.assertEqual((trace / "partial").read_text(), "keep")

    def test_payloads_bound_to_strict_manifest_and_original_subset_contract(self):
        contract = {"payloads": {}}
        actual, items = {}, []
        for i, name in enumerate(("graph_starts", "graph_columns", "features", "valid_positive", "valid_negative")):
            digest = str(i) * 64
            contract["payloads"][name] = {"sha256": digest}
            actual[name] = {"sha256": digest, "bytes": 100 + i}
            if i < 3:
                contract["payloads"][name].update(path=name + ".bin", bytes=100 + i)
                items.append({"path": name + ".bin", "bytes": 100 + i, "sha256": digest})
        strict = {"variant": "strict_clean", "quality_dataset": True, "files": items,
                  "clean_heldout_audits": {"train_csr": {k: {"forward": 0, "reverse": 0} for k in ("valid", "test")}}}
        self.assertEqual(driver.verify_payload_binding(actual, strict, contract)["status"], "pass")
        for key in actual:
            bad = copy.deepcopy(actual); bad[key]["sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "actual payload"):
                driver.verify_payload_binding(bad, strict, contract)
        bad_manifest = copy.deepcopy(strict); bad_manifest["files"][0]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "strict manifest"):
            driver.verify_payload_binding(actual, bad_manifest, contract)
        bad_audit = copy.deepcopy(strict); bad_audit["clean_heldout_audits"]["train_csr"]["valid"]["reverse"] = 1
        with self.assertRaisesRegex(ValueError, "held-out"):
            driver.verify_payload_binding(actual, bad_audit, contract)

    def test_nonfinite_budgets_and_headroom(self):
        for value in (float("nan"), float("inf"), -1, 0):
            with self.assertRaises(b.GuardError): b.finite_budget(value, "test")
        with mock.patch.object(b, "capacity", return_value={"effective_available_bytes": 100}):
            with self.assertRaisesRegex(b.GuardError, "85%"):
                b.enforce_memory(90, 1, check_available=True)

    def test_preflight_does_not_load_native_or_gpu(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(driver, "setup", side_effect=AssertionError("native allocation")) as setup:
            rc = driver.main(["capture", "--fixture", "--output", str(Path(t) / "out.json")])
            self.assertEqual(rc, 0); setup.assert_not_called()
            self.assertEqual(json.loads((Path(t) / "out.json").read_text())["status"], "preflight")

    def test_tiny_model_cannot_be_real_cohort_result(self):
        with tempfile.TemporaryDirectory() as t:
            rc = driver.main(["audit", "--tiny-model", "--output", str(Path(t) / "out.json")])
            self.assertEqual(rc, 2)
            self.assertIn("fixture", json.loads((Path(t) / "out.json").read_text())["error"]["message"])

    def test_excessive_microbatch_guard(self):
        with tempfile.TemporaryDirectory() as t:
            rc = driver.main(["benchmark", "--fixture", "--batch-size", "8192", "--output", str(Path(t) / "out.json")])
            self.assertEqual(rc, 2)
            self.assertEqual(json.loads((Path(t) / "out.json").read_text())["status"], "guarded_skip")

    def test_memory_estimate_charges_native_rows_features_and_perworker_maps(self):
        x = driver.public_estimate({"n": 2927963, "m": 60000000}, 2927963 * 128 * 4,
                                  workers=1, batch_size=128, model=True)
        self.assertEqual(x["native_coo_row_bytes"], 480000000)
        self.assertEqual(x["private_assoc_upper_bytes"], 2927963 * 8)
        self.assertGreater(x["conservative_host_peak_bytes"], x["feature_bytes"] + x["native_coo_row_bytes"])


if __name__ == "__main__": unittest.main()
