#!/usr/bin/env python3
"""Safe PENCIL native-S capture, exact model-input audit and one-GPU throughput.

Default is preflight only, even for capture/audit/benchmark commands. No fitting,
HPO, test-cohort access, network requests or multi-GPU launch. Root separately
authorizes actual controlled CPU/GPU execution. The real primary cohort is fixed:
512 validation positives followed by candidate0 of each original1000-negative row.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from bridge_core import (DEFAULT_ARMS, GuardError, Graph, Trace, array_hash, atomic_json,
    canonical_postmask, cohort, enforce_memory, finite_budget, identity, inspect_graph,
    memory, open_graph, oracle, save_trace, selected_symmetric, sha_file, summary, tiny_graph)
from pencil_runtime import (NativeGraph, QueryRuntime, assert_equal, build_model,
                            fingerprint, forward, load_upstream, verify_upstream)

HERE = Path(__file__).resolve().parent
INPUT_CONTRACT_SHA256 = "9eb90a67ca5b2e3d5dcbcb80421c32e53278e5714f7fd15a9d076d50fca6b0e6"
CPP_MODULE_SHA256 = "4aa4f51fcc6621ec42c6890be42439ef98132cc0554efbde3295e1d1f5e29549"
CPP_SOURCE_SHA256 = "67a9e651b7bad7889a73751bff3b33b4309fb2ca8ddfe2660acf851141045d20"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def verify_cpp_artifacts(args):
    """Bind the audited adapter and locally compiled library to the frozen source."""
    if sha_file(args.cpp_module) != CPP_MODULE_SHA256:
        raise ValueError("C++ adapter differs from frozen reviewed source")
    receipt_path = args.cpp_build_receipt or args.cpp_library.parent / "build_receipt_v1.json"
    receipt = json.loads(receipt_path.read_text())
    library_sha = sha_file(args.cpp_library)
    if (receipt.get("kind") != "corrected_native_build_v1" or
            receipt.get("source_sha256") != CPP_SOURCE_SHA256 or
            receipt.get("library_sha256") != library_sha):
        raise ValueError("C++ build receipt does not bind frozen source and actual library")
    return {"module_sha256": CPP_MODULE_SHA256, "source_sha256": CPP_SOURCE_SHA256,
            "library_sha256": library_sha, "build_receipt_sha256": sha_file(receipt_path)}


def audit_environment(native):
    """Stable execution provenance; setup time/memory deliberately excluded."""
    import torch
    stable_native = {k: v for k, v in native.provenance.items()
                     if k not in ("native_setup_seconds", "native_setup_memory")}
    return {"native": stable_native, "python": sys.version,
            "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {k: os.getenv(k) for k in THREAD_VARIABLES}}


def verify_payload_binding(actual: dict, strict: dict, contract: dict, quality_plan: dict | None = None):
    if "dataset_provenance" in strict:
        strict = strict["dataset_provenance"]["content"]
    if strict.get("variant") != "strict_clean" or strict.get("quality_dataset") is not True:
        raise ValueError("manifest does not declare the strict-clean quality dataset")
    files = strict.get("files", [])
    lookup = {x["path"]: x for x in files}
    if len(lookup) != len(files): raise ValueError("duplicate strict manifest paths")
    for name in ("graph_starts", "graph_columns", "features"):
        expected = contract["payloads"][name]
        item = lookup.get(expected["path"])
        if item is None or item.get("sha256") != expected["sha256"] or item.get("bytes") != expected["bytes"]:
            raise ValueError("strict manifest differs from original frozen payload: " + name)
    for name, observed in actual.items():
        expected = contract["payloads"][name]
        if observed["sha256"] != expected["sha256"] or ("bytes" in expected and observed["bytes"] != expected["bytes"]):
            raise ValueError("actual payload differs from original frozen contract: " + name)
    audits = strict.get("clean_heldout_audits", {}).get("train_csr", {})
    for split in ("valid", "test"):
        if audits.get(split, {}).get("forward") != 0 or audits.get(split, {}).get("reverse") != 0:
            raise ValueError("strict-clean manifest has no passed bidirectional held-out audit")
    if quality_plan is not None:
        original = quality_plan["dataset_provenance"]["content"]
        original_lookup = {x["path"]: x for x in original["files"]}
        for name in ("graph_starts", "graph_columns", "features"):
            expected = contract["payloads"][name]
            if original_lookup[expected["path"]]["sha256"] != actual[name]["sha256"]:
                raise ValueError("quality plan graph/feature payload mismatch")
        val = quality_plan["subsets"]["val"]
        if val["source_count"] != 512 or val["negatives_per_source"] != 1000 or val["subset_seed"] != 20261124:
            raise ValueError("quality plan validation cohort definition changed")
        for name, field in (("valid_positive", "positive"), ("valid_negative", "negative")):
            if val["outputs"][field]["sha256"] != actual[name]["sha256"]:
                raise ValueError("quality plan fixed validation payload mismatch")
    return {"status": "pass", "scope": "actual graph/features/fixed validation SHA matched original contract and strict manifest",
            "payloads": actual, "quality_plan_checked": quality_plan is not None}


def public_estimate(graph_meta: dict, feature_bytes: int, *, workers: int,
                    batch_size: int, model: bool) -> dict:
    n, m = graph_meta["n"], graph_meta["m"]
    # Source compact graph + native transpose/values/COO/lazy caches + node IDs.
    global_bytes = 8 * (4 * (n + 1) + 8 * m) + 4 * m + feature_bytes + 8 * n
    per_batch = batch_size * 84 * (166 + 128) * 4
    queued = max(1, workers) * per_batch * 2
    private_assoc = max(1, workers) * n * 8
    estimate = global_bytes + queued + private_assoc + 2 * 2**30
    if model: estimate += 2**30  # Host model/states; GPU live memory measured separately.
    return {"conservative_host_peak_bytes": int(estimate), "native_coo_row_bytes": 8 * m,
            "native_transpose_values_cache_upper_bytes": 8 * (4 * (n + 1) + 8 * m),
            "compact_source_bytes": 8 * (n + 1) + 4 * m,
            "feature_bytes": feature_bytes, "private_assoc_upper_bytes": private_assoc,
            "queued_batch_upper_bytes": queued,
            "note": "Conservative byte accounting, not observed RSS/GPU peak; public full-E COO explicitly included"}


def preflight(args) -> dict:
    source = verify_upstream(args.upstream, args.upstream_manifest)
    if args.fixture:
        graph, features, pairs, labels = tiny_graph()
        meta = graph.provenance
        binding = {"fixture": True, "rowptr": array_hash(graph.rowptr), "col": array_hash(graph.col), "features": array_hash(features)}
        input_meta = {"fixture": True, "queries": len(pairs), "quality_claim": "none"}
    else:
        for key in ("graph_path", "features", "valid_positive", "valid_negative", "strict_manifest"):
            if getattr(args, key) is None: raise ValueError("missing required strict-clean input --" + key.replace("_", "-"))
        meta = inspect_graph(args.graph_path, args.file_endian)
        if meta["n"] != 2_927_963: raise ValueError("this adapter is scoped to full citation2 with2927963 vertices")
        features = np.load(args.features, mmap_mode="c", allow_pickle=False)
        if features.shape != (meta["n"], 128) or features.dtype != np.float32:
            raise ValueError("features must be full N by128 float32")
        pairs, labels, input_meta = cohort(args.valid_positive, args.valid_negative, meta["n"])
        if len(pairs) != 1024: raise ValueError("real cohort must have exactly1024 prespecified queries")
        if args.strict_manifest.stat().st_size > 16 * 2**20: raise GuardError("strict input manifest unexpectedly large")
        strict = json.loads(args.strict_manifest.read_text())
        if sha_file(args.input_contract) != INPUT_CONTRACT_SHA256:
            raise ValueError("frozen input contract content changed")
        contract = json.loads(args.input_contract.read_text())
        if meta["n"] != contract["n"] or meta["m"] != contract["m"]:
            raise ValueError("actual graph dimensions differ from frozen strict-clean graph")
        if args.eval_seed != contract["eval_extraction_seed"]:
            raise ValueError("extraction seed differs from frozen input contract")
        input_meta["strict_manifest"] = {**identity(args.strict_manifest), "sha256": sha_file(args.strict_manifest),
                                         "content": strict}
        # All full payload hashing is bounded streaming and setup-only.
        paths = {"graph_starts": args.graph_path / "edge_starts.bin", "graph_columns": args.graph_path / "edge_ends.bin",
                 "features": args.features, "valid_positive": args.valid_positive, "valid_negative": args.valid_negative}
        actual = {name: {"bytes": path.stat().st_size, "sha256": sha_file(path)} for name, path in paths.items()}
        quality = None
        if args.quality_plan is not None:
            if sha_file(args.quality_plan) != contract["source_quality_plan_sha256"]:
                raise ValueError("quality plan differs from original frozen plan")
            quality = json.loads(args.quality_plan.read_text())
        input_meta["payload_binding_verification"] = verify_payload_binding(actual, strict, contract, quality)
        binding = {"n": meta["n"], "m": meta["m"], "payloads": actual,
                   "input_contract_sha256": INPUT_CONTRACT_SHA256,
                   "strict_manifest_sha256": sha_file(args.strict_manifest)}
        graph = None
    estimate = public_estimate(meta, features.nbytes, workers=args.num_workers,
                               batch_size=args.batch_size, model=args.forward)
    finite_budget(args.max_public_coo_gib, "max-public-coo-gib", allow_zero=True)
    if estimate["native_coo_row_bytes"] > args.max_public_coo_gib * 2**30:
        raise GuardError("native public API full-E COO exceeds its explicit byte budget")
    current = enforce_memory(estimate["conservative_host_peak_bytes"], args.max_memory_gib,
                             check_available=args.execute and not args.fixture)
    return {"graph": graph, "features": features, "pairs": pairs, "labels": labels,
            "graph_metadata": meta, "input_metadata": input_meta, "binding": binding,
            "estimate": estimate, "capacity": current, "upstream": source}


def raw_fingerprint(raw):
    return {name: fingerprint(a) for name, a in zip(("rows", "cols", "csr_edge_slots"), raw)}


def setup(args, inputs):
    import torch
    torch.set_num_threads(1)
    if args.device == "cuda":
        if not torch.cuda.is_available(): raise ValueError("CUDA unavailable")
        if torch.cuda.device_count() != 1: raise GuardError("bridge requires exactly one visible GPU")
    elif args.precision == "bf16":
        raise ValueError("bf16 requires the declared single CUDA device")
    upstream = load_upstream(args.upstream, args.upstream_manifest)
    # Actual tiny native admission happens before constructing any full graph.
    toy, toy_features, toy_pairs, toy_labels = tiny_graph()
    toy_native = NativeGraph(toy)
    rt = QueryRuntime(upstream, toy_native, toy_features, toy_pairs, toy_labels, "torch_sparse_public")
    for i in range(len(toy_pairs)):
        data, stages = rt.raw(i)
        expected = oracle(toy_native.ptr_np, toy_native.col_np, stages["nodes"])
        for a, b in zip(stages["raw"], expected):
            if not np.array_equal(a, b): raise AssertionError("actual native admission raw output mismatch")
        expected_edges = canonical_postmask(expected, data.root_n_id.numpy())
        if not np.array_equal(data.edge_index.numpy(), expected_edges):
            raise AssertionError("actual native admission target/order mismatch")
    rt.close()
    graph = inputs["graph"] or open_graph(args.graph_path, args.file_endian)
    # Fixture setup can reuse the admitted toy representation; real graph is new.
    native = toy_native if args.fixture else NativeGraph(graph)
    return upstream, native, {"status": "pass", "actual_native_cases": len(toy_pairs),
                              "before_full_graph_allocation": True}


def runtime(args, upstream, native, inputs, arm, *, training=False):
    return QueryRuntime(upstream, native, inputs["features"], inputs["pairs"], inputs["labels"], arm,
        eval_seed=args.eval_seed, training_wrapper=training, cpp_library=args.cpp_library,
        cpp_module=args.cpp_module, max_assoc_bytes=int(args.max_assoc_gib * 2**30))


def capture(args, inputs, upstream, native):
    rt = runtime(args, upstream, native, inputs, "torch_sparse_public")
    nodes, roots, records, rng_states = [], [], [], []
    for i in range(len(inputs["pairs"])):
        u, v = map(int, inputs["pairs"][i])
        if not args.fixture and inputs["labels"][i] == 1 and (native.graph.has_edge(u, v) or native.graph.has_edge(v, u)):
            raise AssertionError("held-out validation target is present in strict-clean graph")
        data, stages = rt.raw(i, capture_rng=True)
        S, raw = stages["nodes"], stages["raw"]
        touched = native.graph.validate_rows(S, args.max_query_neighbors)
        expected = oracle(native.ptr_np, native.col_np, S)
        for a, b in zip(raw, expected):
            if not np.array_equal(a, b): raise AssertionError("actual native capture differs from independent CSR oracle")
        if not selected_symmetric(raw): raise AssertionError("selected C2 graph is not symmetric")
        if not np.array_equal(data.edge_index.numpy(), canonical_postmask(raw, data.root_n_id.numpy())):
            raise AssertionError("native post-target-mask directed edge order mismatch")
        nodes.append(S.copy()); roots.append(data.root_n_id.numpy().copy())
        rng_states.append(stages["post_raw_rng"])
        records.append({"query_index": i, "nodes_hash": array_hash(S),
            "sum_selected_degrees": touched, "raw": raw_fingerprint(raw), "data": fingerprint(data),
            "stored_values": fingerprint(stages["stored_values"]),
            "post_raw_rng": fingerprint(stages["post_raw_rng"]),
            "native_sampler_seconds": stages["sampler_seconds"],
            "native_public_induction_seconds": stages["induction_seconds"],
            "native_raw_data_seconds": stages["raw_data_seconds"]})
    rt.close()
    meta = {"binding": inputs["binding"], "cohort": inputs["input_metadata"],
            "upstream": upstream.provenance, "native": native.provenance,
            "eval_seed": args.eval_seed, "query_seed_policy": "SeedSequence([base,0,query_index])",
            "sampler": {"actual_native_operator": "ego_k_hop_sample_adj", "depth": 1,
                        "neighbors_per_endpoint": 40, "replacement": False, "torch_threads": 1},
            "scope": "Actual sorted native S and exact directed raw/post-mask outputs; no model quality result",
            "fixture": args.fixture}
    manifest = save_trace(args.trace_dir, inputs["pairs"], inputs["labels"], nodes, roots, records,
                          meta, int(args.max_trace_mib * 2**20), rng_states=rng_states)
    return {"trace": str(args.trace_dir), "trace_manifest_sha256": sha_file(args.trace_dir / "manifest.json"),
            "queries": len(nodes), "native_sampler_summary": summary([r["native_sampler_seconds"] for r in records]),
            "native_induction_summary": summary([r["native_public_induction_seconds"] for r in records]),
            "trace_files": manifest["files"]}


def load_bound_trace(args, inputs):
    trace = Trace(args.trace_dir)
    meta = trace.manifest["metadata"]
    if meta["binding"] != inputs["binding"]: raise ValueError("trace observation graph/features/input manifest binding mismatch")
    if bool(meta.get("fixture")) != args.fixture: raise ValueError("fixture/real trace cannot be interchanged")
    if len(trace) != len(inputs["pairs"]) or not np.array_equal(trace.pairs, inputs["pairs"]) or not np.array_equal(trace.labels, inputs["labels"]):
        raise ValueError("trace query/candidate order mismatch")
    if meta["eval_seed"] != args.eval_seed: raise ValueError("trace extraction seed mismatch")
    return trace


def compare_trace(raw, data, stages, trace, i):
    if not np.array_equal(stages["nodes"], trace.nodes(i)):
        raise AssertionError("actual sorted S differs from frozen native trace")
    if raw_fingerprint(raw) != trace.records[i]["raw"]:
        raise AssertionError("raw rows/columns/CSR edge slot identities/order differ")
    if fingerprint(stages["stored_values"]) != trace.records[i]["stored_values"]:
        raise AssertionError("native stored edge-ID payload/order differs")
    if fingerprint(data) != trace.records[i]["data"]:
        raise AssertionError("post-mask native Data fields differ from frozen trace")


def audit(args, inputs, upstream, native, trace, progress):
    import torch
    arms = args.arms.split(",")
    if len(arms) != len(set(arms)) or "torch_sparse_public" not in arms:
        raise ValueError("audit requires unique arms including primary torch_sparse_public")
    modes = [False, True] if args.training_wrapper else [False]
    results, full_batch_checks = [], []
    model = None
    if args.forward:
        with contextlib.redirect_stdout(io.StringIO()):
            model, info = build_model(upstream, args.bert_config, seed=args.model_seed,
                                      checkpoint=args.checkpoint, tiny=args.tiny_model)
        model.to(args.device)
    else: info = None
    for training in modes:
        states = {arm: runtime(args, upstream, native, inputs, arm, training=training) for arm in arms}
        pending = {arm: [] for arm in arms}
        for i in range(len(trace)):
            # Strong reference: actual online native sampler consumes its real
            # RNG stream, then the unchanged wrapper. Replay candidates restore
            # the captured native post-raw RNG before their unchanged wrapper.
            reference_wrapped, reference_data, reference_stages = states["torch_sparse_public"].wrapped(i, capture_rng=True)
            compare_trace(reference_stages["raw"], reference_data, reference_stages, trace, i)
            if fingerprint(reference_stages["post_raw_rng"]) != trace.records[i]["post_raw_rng"]:
                raise AssertionError("actual online native post-raw CPU RNG differs from capture")
            pending["torch_sparse_public"].append(reference_wrapped)
            for arm in arms:
                if arm == "torch_sparse_public": continue
                wrapped, data, stages = states[arm].wrapped(i, trace.nodes(i), wrapper_rng_override=trace.rng(i))
                compare_trace(stages["raw"], data, stages, trace, i)
                assert_equal(data, reference_data, arm + ".Data")
                assert_equal(wrapped, reference_wrapped, arm + ".wrapped")
                pending[arm].append(wrapped)
            if len(pending[arms[0]]) == args.batch_size or i + 1 == len(trace):
                batches = {arm: states[arm].collator(pending[arm]) for arm in arms}
                baseline = batches["torch_sparse_public"]
                for arm in arms: assert_equal(batches[arm], baseline, arm + ".collated")
                logits = None
                if model is not None: logits = forward(model, baseline, args.device, args.precision)
                check = {"through_query_index": i, "training_wrapper": training,
                         "batch_size": len(pending[arms[0]]), "batch_fingerprint": fingerprint(baseline),
                         "all_input_tensors_exact": True, "logits": {}}
                for arm in arms:
                    if model is not None:
                        actual = forward(model, batches[arm], args.device, args.precision)
                        if not torch.isfinite(actual).all(): raise AssertionError("nonfinite model logits")
                        maximum = float((actual - logits).abs().max()) if actual.numel() else 0.
                        if maximum > args.max_logit_abs:
                            raise AssertionError(f"{arm} logits exceed predeclared tolerance: {maximum}")
                        check["logits"][arm] = {"max_abs": maximum,
                            "bitwise_equal": bool(torch.equal(actual, logits)), "hash": fingerprint(actual)}
                full_batch_checks.append(check)
                for arm in arms: pending[arm].clear()
            if i % 128 == 0: progress({"phase": "semantic_audit", "training_wrapper": training,
                                      "through_query_index": i, "cohort_queries": len(trace)})
        for arm, state in states.items():
            results.append({"arm": arm, "training_wrapper": training, "queries": len(trace),
                "raw_csr_slots_order_exact": True, "postmask_data_exact": True,
                "wrapped_and_collated_inputs_exact": True,
                "reference": "actual online native public sampler and wrapper",
                "native_postraw_cpu_rng_restored_before_candidate_wrapper": True,
                "cpp_initializations": state.facade.initializations})
            state.close()
    return {"semantic_gates": results, "batch_checks": full_batch_checks, "model": info,
            "model_forward_checked": args.forward, "max_logit_abs_tolerance": args.max_logit_abs,
            "quality_claim": "none; 1024-query systems cohort is not a1000-negative ranking evaluation"}


class PipelineDataset:
    """One process-local native source/wrapper path; explicit fork workers only."""
    def __init__(self, args, upstream, native, inputs, trace):
        self.args, self.upstream, self.native, self.inputs, self.trace = args, upstream, native, inputs, trace
        self.state, self.pid = None, None
    def __len__(self): return len(self.trace)
    def __getitem__(self, i):
        if self.pid != os.getpid():
            self.state = runtime(self.args, self.upstream, self.native, self.inputs, self.args.arm)
            self.pid = os.getpid()
        frozen = None if self.args.pipeline_mode == "online" else self.trace.nodes(i)
        wrapped, data, stages = self.state.wrapped(i, frozen)
        check_started = time.perf_counter_ns()
        if not np.array_equal(stages["nodes"], self.trace.nodes(i)):
            raise AssertionError("online/replay S changed from trace")
        # Semantic audit is a mandatory separate completed artifact. Do not hash
        # all features in timed execution; keep this small S check explicit.
        verification = (time.perf_counter_ns() - check_started) / 1e9
        return wrapped, {"query_index": i, "pid": self.pid,
            "sampler_seconds": stages["sampler_seconds"], "induction_seconds": stages["induction_seconds"],
            "raw_data_seconds": stages["raw_data_seconds"], "wrapper_seconds": stages["wrapper_seconds"],
            "S_verification_seconds": verification, "memory": memory(),
            "cpp_initializations": self.state.facade.initializations}


class PipelineCollator:
    def __init__(self, collator): self.collator = collator
    def __call__(self, items):
        began = time.perf_counter_ns()
        batch = self.collator([item[0] for item in items])
        seconds = (time.perf_counter_ns() - began) / 1e9
        return batch, [item[1] for item in items], seconds


def verify_audit_receipt(args, trace, native):
    if args.audit_receipt is None: raise ValueError("benchmark requires explicit completed --audit-receipt")
    raw = json.loads(args.audit_receipt.read_text())
    if raw.get("status") != "complete" or raw.get("command") != "audit":
        raise ValueError("audit receipt is not a completed semantic audit")
    if raw.get("trace_manifest_sha256") != sha_file(args.trace_dir / "manifest.json"):
        raise ValueError("audit receipt belongs to a different frozen trace")
    gates = raw.get("result", {}).get("semantic_gates", [])
    matching = [g for g in gates if g["arm"] == args.arm and not g["training_wrapper"]]
    if not matching or not all(g["wrapped_and_collated_inputs_exact"] for g in matching):
        raise ValueError("requested physical arm has no passed input-equivalence audit")
    config = raw["configuration"]
    for key in ("batch_size", "num_workers", "eval_seed", "max_logit_abs"):
        if config.get(key) != getattr(args, key):
            raise ValueError("audit execution setting mismatch: " + key)
    if raw.get("audit_environment") != audit_environment(native):
        raise ValueError("native/software/thread execution provenance changed since audit")
    if args.forward:
        if not raw.get("result", {}).get("model_forward_checked"):
            raise ValueError("model throughput requires a passed forward-equivalence audit")
        for key in ("precision", "device", "model_seed", "tiny_model"):
            if config.get(key) != getattr(args, key):
                raise ValueError("forward audit setting mismatch: " + key)
        checkpoint = raw["result"]["model"]["checkpoint"]
        if (checkpoint is None) != (args.checkpoint is None):
            raise ValueError("forward checkpoint presence changed since audit")
        if checkpoint is not None and checkpoint["sha256"] != sha_file(args.checkpoint):
            raise ValueError("checkpoint content changed since forward audit")
        if raw["result"]["model"]["bert_config_sha256"] != sha_file(args.bert_config / "config.json"):
            raise ValueError("offline BERT config changed since forward audit")
    if args.arm.startswith("cpp_"):
        current_cpp = verify_cpp_artifacts(args)
        if raw.get("cpp_artifacts") != current_cpp:
            raise ValueError("C++ adapter/library changed since audit")
    # Bind both code and exact upstream source; fresh code requires a new audit.
    if raw.get("source_files") != source_files(): raise ValueError("bridge source changed since audit")
    return {"path": str(args.audit_receipt), "sha256": sha_file(args.audit_receipt)}


def benchmark(args, inputs, upstream, native, trace, progress):
    import torch
    from torch.utils.data import DataLoader
    receipt = verify_audit_receipt(args, trace, native)
    if args.num_workers and sys.platform != "linux":
        raise GuardError("forked native throughput workers are admitted only on Linux")
    if args.pipeline_mode == "online" and args.eval_seed != trace.manifest["metadata"]["eval_seed"]:
        raise ValueError("online extraction seed must match capture")
    state = runtime(args, upstream, native, inputs, args.arm)
    # JIT/state warmup outside steady-state timers; worker-private state init may
    # recur after fork and is covered by the explicit warmup sweep.
    state.wrapped(0, trace.nodes(0)); state.close()
    model, info = None, None
    if args.forward:
        with contextlib.redirect_stdout(io.StringIO()):
            model, info = build_model(upstream, args.bert_config, seed=args.model_seed,
                checkpoint=args.checkpoint, tiny=args.tiny_model)
        model.to(args.device)
    dataset = PipelineDataset(args, upstream, native, inputs, trace)
    tokenizer = upstream.tokenizer(num_nodes=82)
    collate = PipelineCollator(upstream.collator(tokenizer))
    kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers,
              "shuffle": False, "collate_fn": collate, "pin_memory": args.device == "cuda"}
    if args.num_workers:
        kwargs.update(multiprocessing_context="fork", persistent_workers=True, prefetch_factor=1)
    loader = DataLoader(dataset, **kwargs)
    sweeps = []
    if args.device == "cuda": torch.cuda.reset_peak_memory_stats()
    try:
        for repetition in range(args.warmup + args.repeats):
            if args.device == "cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            indices, stages, batch_times, gpu_times, score_parts = [], [], [], [], []
            for batch, meta, collate_seconds in loader:
                indices.extend(x["query_index"] for x in meta); stages.extend(meta)
                batch_times.append(collate_seconds)
                if model is not None:
                    if args.device == "cuda": torch.cuda.synchronize()
                    model_start = time.perf_counter()
                    score = forward(model, batch, args.device, args.precision)
                    if args.device == "cuda": torch.cuda.synchronize()
                    gpu_times.append(time.perf_counter() - model_start)
                    score_parts.append(score)
            if args.device == "cuda": torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if indices != list(range(len(trace))): raise AssertionError("loader omitted, duplicated or reordered query indices")
            scores = torch.cat(score_parts) if score_parts else None
            row = {"repetition": repetition, "warmup": repetition < args.warmup,
                "queries": len(indices), "elapsed_seconds": elapsed, "queries_per_second": len(indices) / elapsed,
                "stages": {key: summary([x[key] for x in stages]) for key in
                    ("sampler_seconds", "induction_seconds", "raw_data_seconds", "wrapper_seconds", "S_verification_seconds")},
                "collator_seconds_total": sum(batch_times),
                "h2d_forward_score_copy_seconds_total": sum(gpu_times),
                "parent_memory": memory(), "worker_memory": [dict(pid=pid,
                    max_pss_bytes=max((x["memory"].get("pss_bytes", 0) for x in stages if x["pid"] == pid), default=0),
                    maxrss_bytes=max(x["memory"]["maxrss_bytes"] for x in stages if x["pid"] == pid)) for pid in sorted(set(x["pid"] for x in stages))],
                "scores": fingerprint(scores) if scores is not None else None,
                "cpp_initializations": [x["cpp_initializations"] for x in stages if x["cpp_initializations"]][:1]}
            if args.device == "cuda": row["gpu"] = {"max_allocated_bytes": torch.cuda.max_memory_allocated(),
                                                        "max_reserved_bytes": torch.cuda.max_memory_reserved()}
            sweeps.append(row); progress({"phase": "throughput", "sweeps": sweeps})
    finally:
        # Owned workers must stop even after failed loader/forward/verification.
        if getattr(loader, "_iterator", None) is not None: loader._iterator._shutdown_workers()
    return {"arm": args.arm, "pipeline_mode": args.pipeline_mode, "audit_receipt": receipt,
        "batch_size": args.batch_size, "num_workers": args.num_workers, "prefetch_factor": 1 if args.num_workers else None,
        "model": info, "sweeps": sweeps,
        "elapsed_summary": summary([r["elapsed_seconds"] for r in sweeps if not r["warmup"]]),
        "scope": "fixed-S loader/wrapper/collator/optional model forward; sampler replay explicitly bypassed" if args.pipeline_mode == "replay" else "online native sampler + physical replacement + wrapper/collator/optional forward, actual S checked per query",
        "measurement_note": "Instrumented throughput: per-query memory/smaps measurement, S verification, IPC, and forward score copy are included in wall time; use separately recorded induction stage for operator latency. All native graph representations remain resident for every arm; no memory-saving comparison.",
        "quality_claim": "none; no fitting or full1000-candidate ranking evaluation"}


def source_files():
    names = ("bridge_core.py", "pencil_runtime.py", "pencil_bridge.py", "upstream_manifest.json", "input_contract.json",
             "bert_config/config.json", "bert_config/source.json")
    return {name: sha_file(HERE / name) for name in names}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "capture", "audit", "benchmark"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--graph-path", type=Path); p.add_argument("--features", type=Path)
    p.add_argument("--valid-positive", type=Path); p.add_argument("--valid-negative", type=Path)
    p.add_argument("--strict-manifest", type=Path)
    p.add_argument("--quality-plan", type=Path, help="Optional original frozen SEAL quality plan, content-hash pinned")
    p.add_argument("--input-contract", type=Path, default=HERE / "input_contract.json")
    p.add_argument("--trace-dir", type=Path, default=HERE.parent / "pencil_bridge_trace")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream", type=Path, default=HERE / "upstream_pencil")
    p.add_argument("--upstream-manifest", type=Path, default=HERE / "upstream_manifest.json")
    p.add_argument("--bert-config", type=Path, default=HERE / "bert_config")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--arm", default="torch_sparse_public")
    p.add_argument("--cpp-module", type=Path); p.add_argument("--cpp-library", type=Path)
    p.add_argument("--cpp-build-receipt", type=Path, help="Default: build_receipt_v1.json beside --cpp-library")
    p.add_argument("--max-assoc-gib", type=float, default=1)
    p.add_argument("--max-memory-gib", type=float, default=32)
    p.add_argument("--max-public-coo-gib", type=float, default=1)
    p.add_argument("--max-trace-mib", type=float, default=512)
    p.add_argument("--max-query-neighbors", type=int, default=20_000_000)
    p.add_argument("--eval-seed", type=int, default=20261126)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, choices=(0, 1), default=0)
    p.add_argument("--training-wrapper", action="store_true", help="Audit both eval and training randperm input paths")
    p.add_argument("--forward", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--tiny-model", action="store_true", help="Fixture-only small BERT; never a real-cohort result")
    p.add_argument("--max-logit-abs", type=float, default=0)
    p.add_argument("--audit-receipt", type=Path)
    p.add_argument("--pipeline-mode", choices=("replay", "online"), default="replay")
    p.add_argument("--warmup", type=int, default=1); p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--file-endian", choices=("little", "big"), default="little")
    return p


def validate_output_paths(args):
    """Run before any error/status write; immutable inputs always take precedence."""
    protected = {p.resolve() for p in (args.features, args.valid_positive, args.valid_negative,
                 args.strict_manifest, args.cpp_module, args.cpp_library, args.cpp_build_receipt,
                 args.upstream_manifest, args.quality_plan, args.input_contract,
                 args.audit_receipt, args.checkpoint, args.bert_config / "config.json", args.bert_config / "source.json") if p is not None}
    if args.cpp_library is not None:
        protected.add((args.cpp_library.parent / "build_receipt_v1.json").resolve())
    roots = [p.resolve() for p in (HERE, args.upstream, args.bert_config, args.graph_path) if p is not None]
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if any(p in protected or any(p.is_relative_to(root) for root in roots + [args.trace_dir.resolve()])
           for p in (output, partial)):
        raise ValueError("output must be outside every source/input/trace subtree and consumed input file")
    if args.output.exists() or partial.exists():
        raise FileExistsError("preserve existing output; choose a fresh output path")
    if args.command == "capture":
        trace = args.trace_dir.resolve()
        if trace.exists() or any(trace.is_relative_to(root) or root.is_relative_to(trace) for root in roots):
            raise ValueError("capture requires a fresh trace directory outside source/input subtrees")
        if any(p.is_relative_to(trace) for p in protected):
            raise ValueError("capture trace must not contain a consumed input")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        validate_output_paths(args)
    except (ValueError, OSError) as e:
        print(json.dumps({"status": "refused", "error": str(e)}), file=sys.stderr)
        return 2
    result = {"schema_version": 1, "command": args.command, "status": "preflight",
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_files": source_files(), "provenance": {"python": sys.version, "platform": platform.platform(),
            "initial_memory": memory(), "thread_environment": {k: os.getenv(k) for k in THREAD_VARIABLES}},
        "quality_claim": "none"}
    try:
        for name in ("max_memory_gib", "max_public_coo_gib", "max_trace_mib", "max_assoc_gib", "max_logit_abs"):
            finite_budget(getattr(args, name), name, allow_zero=name in ("max_public_coo_gib", "max_logit_abs"))
        if args.batch_size <= 0 or args.repeats <= 0 or args.warmup < 0 or args.max_query_neighbors <= 0:
            raise ValueError("invalid batch/repeat/query budget")
        if args.tiny_model and not args.fixture: raise ValueError("tiny model is fixture-only")
        known_arms = set(DEFAULT_ARMS) | {"cpp_full_clear", "cpp_touched_reset"}
        requested = args.arms.split(",") if args.command == "audit" else [args.arm]
        if not requested or any(a not in known_arms for a in requested): raise ValueError("unknown physical arm")
        if len(requested) != len(set(requested)): raise ValueError("duplicate physical arm")
        if args.execute and any(a.startswith("cpp_") for a in requested):
            if args.cpp_module is None or args.cpp_library is None or not args.cpp_module.is_file() or not args.cpp_library.is_file():
                raise ValueError("C++ arms require staged module and compiled library before full-graph allocation")
            result["cpp_artifacts"] = verify_cpp_artifacts(args)
        if args.batch_size > 512: raise GuardError("controlled bridge admits batch<=512; native8192 is not inferred safe")
        inputs = preflight(args)
        result.update(input_binding=inputs["binding"], input_metadata=inputs["input_metadata"],
                      graph_metadata=inputs["graph_metadata"], memory_estimate=inputs["estimate"],
                      capacity=inputs["capacity"], upstream=inputs["upstream"])
        if not args.execute or args.command == "preflight":
            result["preflight_passed"] = True
            atomic_json(args.output, result)
            print(json.dumps({"status": "preflight", "command": args.command, "queries": len(inputs["pairs"])}))
            return 0
        upstream, native, admission = setup(args, inputs)
        result.update(status="running", native_admission=admission, native_setup=native.provenance)
        result["audit_environment"] = audit_environment(native)
        atomic_json(args.output, result)
        def progress(row):
            result["progress"] = row; atomic_json(args.output, result)
        if args.command == "capture":
            result["result"] = capture(args, inputs, upstream, native)
        else:
            trace = load_bound_trace(args, inputs)
            result["trace_manifest_sha256"] = sha_file(args.trace_dir / "manifest.json")
            if args.command == "audit": result["result"] = audit(args, inputs, upstream, native, trace, progress)
            else: result["result"] = benchmark(args, inputs, upstream, native, trace, progress)
        result.update(status="complete", final_memory=memory())
        result.pop("progress", None)
        atomic_json(args.output, result)
        print(json.dumps({"status": "complete", "command": args.command, "output": str(args.output)}))
        return 0
    except Exception as e:
        result.update(status="guarded_skip" if isinstance(e, GuardError) else "failed",
                      error={"type": type(e).__name__, "message": str(e)})
        atomic_json(args.output, result)
        print(json.dumps({"status": result["status"], "error": str(e)}), file=sys.stderr)
        return 2


if __name__ == "__main__": raise SystemExit(main())
