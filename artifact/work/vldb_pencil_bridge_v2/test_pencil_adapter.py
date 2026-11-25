"""Actual pinned PENCIL consumer/model tests; simulated native sparse API only.

Local Mac lacks torch-sparse. The sparse simulator is an independent oracle,
not a native-extension reproduction. Actual source __getitem__, target mask,
wrapper, collator, BERTLP and compiled NumPy/Numba candidate operators do execute.
Actual native admission remains mandatory in the driver before full allocation.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import bridge_core as core
import pencil_runtime as rt
import pencil_bridge as driver

HERE = Path(__file__).resolve().parent


def dependencies_available():
    return all(importlib.util.find_spec(n) is not None for n in
               ("torch", "torch_geometric", "transformers", "wandb", "rich"))


class FakeSparseTensor:
    """Tiny independent CSR oracle as stand-in for absent compiled package."""
    def __init__(self, row=None, col=None, value=None, sparse_sizes=None, **_):
        self.row, self.col, self.value, self.shape = row, col, value, sparse_sizes
    def coo(self): return self.row, self.col, self.value
    def t(self):
        import torch
        # Match sparse transpose's row-major local order.
        r, c = self.col, self.row
        order = torch.argsort(r * self.shape[0] + c, stable=True)
        return FakeSparseTensor(row=r[order], col=c[order],
            value=None if self.value is None else self.value[order], sparse_sizes=self.shape[::-1])


class FakeGlobalAdj:
    def __init__(self, ptr, col, value): self.ptr, self.col, self.value = ptr, col, value
    def saint_subgraph(self, idx):
        import torch
        r, c, ids = core.oracle(self.ptr.numpy(), self.col.numpy(), idx.numpy())
        r, c, ids = (torch.from_numpy(a) for a in (r, c, ids))
        return FakeSparseTensor(row=r, col=c, value=self.value[ids], sparse_sizes=(len(idx), len(idx))), ids


def fake_native():
    import torch
    graph, features, pairs, labels = core.tiny_graph()
    ptr, col = torch.from_numpy(graph.rowptr), torch.from_numpy(graph.col.astype(np.int64))
    # Values intentionally do NOT equal CSR slots: catches accidental ID conflation.
    value = torch.arange(graph.m - 1, -1, -1, dtype=torch.int64)
    native = SimpleNamespace(torch=torch, torch_sparse=SimpleNamespace(SparseTensor=FakeSparseTensor),
        graph=graph, ptr=ptr, col=col, value=value, ptr_np=ptr.numpy(), col_np=col.numpy(),
        adj=FakeGlobalAdj(ptr, col, value))
    return native, features, pairs, labels


def fake_ego(ptr, col, roots, depth, neighbors, replace):
    import torch
    # Fixture degrees<=40: actual1-hop native sampler would retain all neighbors.
    assert depth == 1 and neighbors == 40 and not replace
    nodes = []
    for root in roots:
        root = int(root); nodes.append(root)
        nodes.extend(col[int(ptr[root]):int(ptr[root + 1])].tolist())
    return (None, None, torch.tensor(nodes, dtype=torch.int64))


@unittest.skipUnless(dependencies_available(), "actual pinned consumer/model dependencies missing")
class ActualConsumer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.u = rt.load_upstream(HERE / "upstream_pencil", HERE / "upstream_manifest.json")

    def test_unmodified_source_path_all_compiled_arms_eval_and_training_wrapper(self):
        import torch
        native, features, pairs, labels = fake_native()
        arms = ["torch_sparse_public", "compiled_scan_int32", "compiled_scan_int64",
                "compiled_pairwise_int32", "compiled_pairwise_int64"]
        with mock.patch.object(torch.ops.torch_sparse, "ego_k_hop_sample_adj", fake_ego, create=True):
            for training in (False, True):
                states = {arm: rt.QueryRuntime(self.u, native, features, pairs, labels, arm,
                                               training_wrapper=training) for arm in arms}
                wrapped = {arm: [] for arm in arms}
                for i in range(len(pairs)):
                    baseline, data, stages = states[arms[0]].wrapped(i)
                    S = stages["nodes"]
                    expected = core.oracle(native.ptr_np, native.col_np, S)
                    for a, b in zip(stages["raw"], expected): np.testing.assert_array_equal(a, b)
                    wrapped[arms[0]].append(baseline)
                    for arm in arms[1:]:
                        item, actual_data, actual_stages = states[arm].wrapped(i, S)
                        rt.assert_equal(actual_data, data, arm + ".rawData")
                        rt.assert_equal(item, baseline, arm + ".wrapper")
                        for a, b in zip(actual_stages["raw"], expected): np.testing.assert_array_equal(a, b)
                        wrapped[arm].append(item)
                    if i == 2:
                        self.assertEqual(data.edge_index.shape[1], 0)  # Two cold roots retained.
                    if i == 4:
                        self.assertEqual(data.root_n_id[0], data.root_n_id[1])
                batches = {arm: states[arm].collator(wrapped[arm]) for arm in arms}
                for arm in arms: rt.assert_equal(batches[arm], batches[arms[0]], arm + ".collator")
                self.assertEqual(batches[arms[0]]["feature_embeds"].shape[-1], 128)
                self.assertEqual(batches[arms[0]]["input_embeds"].shape[-1], 166)
                for state in states.values(): state.close()

    def test_actual_BERT_forward_identical_inputs_and_weights(self):
        import torch
        native, features, pairs, labels = fake_native()
        with mock.patch.object(torch.ops.torch_sparse, "ego_k_hop_sample_adj", fake_ego, create=True):
            reference = rt.QueryRuntime(self.u, native, features, pairs, labels, "torch_sparse_public")
            candidate = rt.QueryRuntime(self.u, native, features, pairs, labels, "compiled_pairwise_int32")
            a, b = [], []
            for i in range(len(pairs)):
                ref, _, stages = reference.wrapped(i)
                actual, _, _ = candidate.wrapped(i, stages["nodes"])
                a.append(ref); b.append(actual)
            batch_a, batch_b = reference.collator(a), candidate.collator(b)
            rt.assert_equal(batch_a, batch_b)
            with contextlib.redirect_stdout(io.StringIO()):
                model, meta = rt.build_model(self.u, HERE / "bert_config", tiny=True)
            x, y = rt.forward(model, batch_a, "cpu", "fp32"), rt.forward(model, batch_b, "cpu", "fp32")
            self.assertTrue(torch.isfinite(x).all())
            self.assertTrue(torch.equal(x, y))
            self.assertEqual(meta["features"], 128)
            self.assertIsNone(meta["checkpoint"])

    def test_training_randperm_is_real_even_with_node_remapping_false(self):
        import torch
        native, features, pairs, labels = fake_native()
        with mock.patch.object(torch.ops.torch_sparse, "ego_k_hop_sample_adj", fake_ego, create=True):
            state = rt.QueryRuntime(self.u, native, features, pairs, labels, "compiled_scan_int32", training_wrapper=True)
            raw, _ = state.raw(0)
            state.one.item = raw
            rt.set_query_seed(91); first = state.wrapper[0]
            rt.set_query_seed(97); second = state.wrapper[0]
            self.assertNotEqual(rt.fingerprint(first), rt.fingerprint(second))
            rt.set_query_seed(91); repeated = state.wrapper[0]
            rt.assert_equal(first, repeated)

    def test_native_postraw_rng_restoration_preserves_online_training_wrapper(self):
        import torch
        native, features, pairs, labels = fake_native()
        def consuming_ego(*args):
            torch.rand(17)  # Demonstrate the actual sampler-RNG versus replay distinction.
            return fake_ego(*args)
        observed_difference = False
        with mock.patch.object(torch.ops.torch_sparse, "ego_k_hop_sample_adj", consuming_ego, create=True):
            for seed in range(13, 20):
                ref = rt.QueryRuntime(self.u, native, features, pairs, labels, "torch_sparse_public", eval_seed=seed, training_wrapper=True)
                alt = rt.QueryRuntime(self.u, native, features, pairs, labels, "compiled_scan_int32", eval_seed=seed, training_wrapper=True)
                expected, _, stages = ref.wrapped(0, capture_rng=True)
                uncontrolled, _, _ = alt.wrapped(0, stages["nodes"])
                restored, _, _ = alt.wrapped(0, stages["nodes"], wrapper_rng_override=stages["post_raw_rng"])
                observed_difference |= rt.fingerprint(uncontrolled) != rt.fingerprint(expected)
                rt.assert_equal(restored, expected)
            self.assertTrue(observed_difference)

    def test_pickle_free_rng_roundtrip_restores_all_CPU_generators(self):
        import random
        import torch
        rt.set_query_seed(745)
        state = rt.snapshot_rng()
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "trace"
            core.save_trace(root, np.array([[8, 8]], dtype=np.int64), np.array([0], dtype=np.int64),
                [np.array([8], dtype=np.int64)], [np.array([0, 0], dtype=np.int64)], [{}], {}, 100000,
                rng_states=[state])
            decoded = core.Trace(root).rng(0)
            self.assertEqual(rt.fingerprint(state), rt.fingerprint(decoded))
            expected = (random.random(), np.random.random(), torch.rand(4))
            rt.set_query_seed(1); rt.restore_rng(decoded)
            actual = (random.random(), np.random.random(), torch.rand(4))
            self.assertEqual(actual[0], expected[0]); self.assertEqual(actual[1], expected[1])
            self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_data_and_collated_order_corruption_is_detected(self):
        import torch
        a = {"x": torch.arange(8).reshape(4, 2), "edge_index": torch.tensor([[0, 1], [1, 0]])}
        for key in a:
            b = {k: v.clone() for k, v in a.items()}; b[key] = b[key].flip(-1)
            with self.assertRaises(AssertionError): rt.assert_equal(a, b)

    def test_actual_CPP_matched_controls_feed_native_source_and_wrapper(self):
        import torch
        module = HERE / "corrected_native/native_adapter_v1.py"
        default_name = "native_assoc_v1.dylib" if os.sys.platform == "darwin" else "native_assoc_v1.so"
        library = Path(os.environ.get("PENCIL_BRIDGE_CPP_LIBRARY", str(HERE / "_local_cpp_build" / default_name)))
        if not module.exists() or not library.exists(): self.skipTest("local corrected C++ build absent")
        native, features, pairs, labels = fake_native()
        with mock.patch.object(torch.ops.torch_sparse, "ego_k_hop_sample_adj", fake_ego, create=True):
            ref = rt.QueryRuntime(self.u, native, features, pairs, labels, "torch_sparse_public")
            states = [rt.QueryRuntime(self.u, native, features, pairs, labels, arm,
                          cpp_module=module, cpp_library=library) for arm in ("cpp_full_clear", "cpp_touched_reset")]
            for i in range(len(pairs)):
                expected, raw, st = ref.wrapped(i)
                for state in states:
                    actual, data, ast = state.wrapped(i, st["nodes"])
                    rt.assert_equal(raw, data)
                    rt.assert_equal(actual, expected)
                    for a, b in zip(ast["raw"], st["raw"]): np.testing.assert_array_equal(a, b)
            for state in states:
                self.assertIsNotNone(state.facade.initializations[0]["metadata"])
                state.close()

    def test_source_and_manifest_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            manifest = root / "manifest.json"; manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError, "manifest content"):
                rt.verify_upstream(root, manifest)
        actual = rt.verify_upstream(HERE / "upstream_pencil", HERE / "upstream_manifest.json")
        self.assertEqual(actual["all_source_files_verified"], 53)

    def test_features_dimension_guard(self):
        import torch
        native, features, pairs, labels = fake_native()
        with mock.patch.object(torch.ops.torch_sparse, "ego_k_hop_sample_adj", fake_ego, create=True):
            with self.assertRaisesRegex(ValueError, "128"):
                rt.QueryRuntime(self.u, native, features[:, :64], pairs, labels, "compiled_scan_int32")

    def test_full_driver_capture_audit_forward_and_throughput_orchestration(self):
        import torch
        native, _, _, _ = fake_native()
        native.provenance = {"test_double": "independent sparse API simulator; NOT native C++ execution"}
        admission = {"test_double": True, "actual_native_execution": False}
        with tempfile.TemporaryDirectory() as t, mock.patch.object(torch.ops.torch_sparse, "ego_k_hop_sample_adj", fake_ego, create=True), mock.patch.object(driver, "setup", return_value=(self.u, native, admission)):
            root = Path(t)
            common = ["--fixture", "--execute", "--trace-dir", str(root / "trace"), "--batch-size", "2"]
            self.assertEqual(driver.main(["capture", *common, "--output", str(root / "capture.json")]), 0)
            arms = "torch_sparse_public,compiled_scan_int32,compiled_pairwise_int64"
            self.assertEqual(driver.main(["audit", *common, "--arms", arms, "--forward", "--tiny-model",
                "--training-wrapper", "--output", str(root / "audit.json")]), 0)
            audit = json.loads((root / "audit.json").read_text())
            self.assertTrue(audit["result"]["model_forward_checked"])
            self.assertEqual(len(audit["result"]["semantic_gates"]), 6)
            self.assertEqual(driver.main(["benchmark", *common, "--arm", "compiled_scan_int32", "--forward", "--tiny-model",
                "--audit-receipt", str(root / "audit.json"), "--warmup", "1", "--repeats", "1",
                "--output", str(root / "bench.json")]), 0)
            result = json.loads((root / "bench.json").read_text())
            self.assertEqual(result["result"]["sweeps"][-1]["queries"], 7)
            self.assertEqual(result["quality_claim"], "none")
            # Completed receipts must never be replaced by later failure/status.
            old = (root / "audit.json").read_bytes()
            self.assertEqual(driver.main(["capture", *common, "--output", str(root / "audit.json")]), 2)
            self.assertEqual((root / "audit.json").read_bytes(), old)

    def test_forward_receipt_rejects_changed_model_setting(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); (root / "manifest.json").write_text('{}')
            receipt = root / "audit.json"
            receipt.write_text(json.dumps({"status": "complete", "command": "audit",
                "trace_manifest_sha256": core.sha_file(root / "manifest.json"),
                "configuration": {"precision": "fp32", "device": "cpu", "model_seed": 42, "tiny_model": True,
                                  "batch_size": 2, "num_workers": 0, "eval_seed": 1, "max_logit_abs": 0},
                "audit_environment": driver.audit_environment(SimpleNamespace(provenance={"test": True})),
                "source_files": driver.source_files(),
                "result": {"model_forward_checked": True, "semantic_gates": [{"arm": "compiled_scan_int32",
                    "training_wrapper": False, "wrapped_and_collated_inputs_exact": True}]}}))
            args = SimpleNamespace(audit_receipt=receipt, trace_dir=root, arm="compiled_scan_int32", forward=True,
                precision="bf16", device="cpu", model_seed=42, tiny_model=True,
                batch_size=2, num_workers=0, eval_seed=1, max_logit_abs=0)
            with self.assertRaisesRegex(ValueError, "setting mismatch"):
                driver.verify_audit_receipt(args, None, SimpleNamespace(provenance={"test": True}))

    def test_receipt_binds_batch_worker_tolerance_native_binary_and_bridge_source(self):
        native = SimpleNamespace(provenance={"torch": "test", "extension_binaries": [{"name": "native.so", "sha256": "a"}],
                                             "native_setup_seconds": 17, "native_setup_memory": {"rss": 123}})
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); (root / "manifest.json").write_text('{}'); receipt = root / "audit.json"
            args = driver.parser().parse_args(["benchmark", "--fixture", "--trace-dir", str(root),
                "--output", str(root / "unused.json"), "--audit-receipt", str(receipt), "--arm", "compiled_scan_int32"])
            base = {"status": "complete", "command": "audit", "trace_manifest_sha256": core.sha_file(root / "manifest.json"),
                "configuration": vars(args).copy(), "source_files": driver.source_files(),
                "audit_environment": driver.audit_environment(native),
                "result": {"semantic_gates": [{"arm": args.arm, "training_wrapper": False,
                                                "wrapped_and_collated_inputs_exact": True}]}}
            # Path configuration is irrelevant to equality here; serialize explicitly.
            base["configuration"] = {k: str(v) if isinstance(v, Path) else v for k, v in base["configuration"].items()}
            receipt.write_text(json.dumps(base)); driver.verify_audit_receipt(args, None, native)
            changed_times = copy.deepcopy(native); changed_times.provenance["native_setup_seconds"] = 999
            driver.verify_audit_receipt(args, None, changed_times)  # Setup cost is not an identity.
            for field in ("batch_size", "num_workers", "eval_seed", "max_logit_abs"):
                bad = copy.deepcopy(base); bad["configuration"][field] += 1; receipt.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "execution setting"):
                    driver.verify_audit_receipt(args, None, native)
            receipt.write_text(json.dumps(base)); bad_native = copy.deepcopy(native)
            bad_native.provenance["extension_binaries"][0]["sha256"] = "b"
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                driver.verify_audit_receipt(args, None, bad_native)
            bad = copy.deepcopy(base); bad["source_files"]["pencil_runtime.py"] = "different"
            receipt.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "bridge source changed"):
                driver.verify_audit_receipt(args, None, native)


if __name__ == "__main__": unittest.main()
