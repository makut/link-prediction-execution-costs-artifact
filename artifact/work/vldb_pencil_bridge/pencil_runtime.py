"""Unmodified pinned PENCIL classes behind a strict-clean CSR consumer adapter.

No repository files are patched. The sampler hook changes only the private
module's Torch namespace for an explicit frozen-S replay; online mode calls the
real registered ego sampler. The actual PENCIL __getitem__, target mask,
DatasetWrapper, Collator and BERTLP execute without source edits.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bridge_core import (UPSTREAM_COMMIT, CompiledBackend, GuardError, array_hash,
                         memory, query_seed, sha_file, validate_nodes)

PINNED_MANIFEST_SHA256 = "6020d2bc7c2290cf251e751b97a841202c1904a3dd4e89f5521abda876e0598e"


def verify_upstream(root: Path, manifest_path: Path) -> dict:
    if sha_file(manifest_path) != PINNED_MANIFEST_SHA256:
        raise ValueError("upstream manifest content differs from frozen pin")
    m = json.loads(manifest_path.read_text())
    if m["commit"] != UPSTREAM_COMMIT or m.get("modified_upstream_files"):
        raise ValueError("upstream source pin/modified-file manifest mismatch")
    for item in m["files"]:
        p = root / item["path"]
        if p.stat().st_size != item["bytes"] or sha_file(p) != item["sha256"]:
            raise ValueError("pinned PENCIL source mismatch: " + item["path"])
    return {"commit": m["commit"], "repository": m["repository"], "license": m["license"],
            "manifest_sha256": sha_file(manifest_path), "all_source_files_verified": len(m["files"])}


def _file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream(root: Path, manifest_path: Path):
    provenance = verify_upstream(root, manifest_path)
    if importlib.metadata.version("transformers") != "4.46.2":
        raise ValueError("PENCIL bridge pins transformers4.46.2 private SDPA API")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = root.resolve()
    # Refuse a silent collision with the user's repository or another package.
    for key in ("utils", "models"):
        old = sys.modules.get(key)
        if old is not None and getattr(old, "__file__", None):
            if root not in Path(old.__file__).resolve().parents:
                raise RuntimeError("use a fresh process; conflicting upstream import: " + key)
    sys.path.insert(0, str(root))
    utils = importlib.import_module("utils")
    tokenizer = importlib.import_module("models.transformers.stokenizer")
    model = importlib.import_module("models.transformers.bert_lp")
    # Avoid datasets/__init__.py importing unrelated raw data loaders/dependencies.
    wrapper = _file_module("_pencil_bridge_wrapper", root / "datasets/dataset_wrapper.py")
    collator = _file_module("_pencil_bridge_collator", root / "datasets/collator.py")
    dataset = _file_module("_pencil_bridge_dataset_map", root / "datasets/dataset_map.py")
    return SimpleNamespace(utils=utils, tokenizer=tokenizer.STokenizer,
                           model=model.BERTLP, wrapper=wrapper.DatasetWrapper,
                           collator=collator.Collator, dataset=dataset,
                           provenance=provenance)


def _tensor_bytes(t):
    import torch
    a = t.detach().cpu().contiguous()
    return a.view(torch.uint16).numpy().tobytes() if a.dtype == torch.bfloat16 else a.numpy().tobytes()


def fingerprint(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "dtype": str(value.dtype), "shape": list(value.shape),
                "stride": list(value.stride()), "sha256": hashlib.sha256(_tensor_bytes(value)).hexdigest()}
    if isinstance(value, np.ndarray):
        return {"kind": "numpy", "dtype": value.dtype.str, "shape": list(value.shape), "hash": array_hash(value)}
    if hasattr(value, "to_dict"): return fingerprint(value.to_dict())
    if isinstance(value, dict): return {str(k): fingerprint(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)): return [fingerprint(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    if value is None or isinstance(value, (str, int, bool, float)): return value
    raise TypeError("unsupported fingerprint value: " + str(type(value)))


def assert_equal(actual, expected, where="root"):
    import torch
    if hasattr(actual, "to_dict"): actual = actual.to_dict()
    if hasattr(expected, "to_dict"): expected = expected.to_dict()
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            raise AssertionError(where + ": tensor type mismatch")
        if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.stride() != expected.stride() or not torch.equal(actual, expected):
            raise AssertionError(where + ": tensor dtype/shape/stride/value mismatch")
    elif isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected): raise AssertionError(where + ": dictionary keys mismatch")
        for key in actual: assert_equal(actual[key], expected[key], where + "." + str(key))
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected): raise AssertionError(where + ": sequence length mismatch")
        for i, (a, b) in enumerate(zip(actual, expected)): assert_equal(a, b, where + f"[{i}]")
    elif actual != expected:
        raise AssertionError(where + ": scalar mismatch")


def set_query_seed(seed: int):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def snapshot_rng() -> dict:
    """Pickle-free CPU RNG state after native sampling and before CPU wrapper."""
    import torch
    py, ns = random.getstate(), np.random.get_state()
    return {"py_version": int(py[0]), "py_words": np.asarray(py[1], dtype=np.uint32),
            "py_gauss": py[2], "np_algorithm": ns[0], "np_words": np.asarray(ns[1], dtype=np.uint32).copy(),
            "np_position": int(ns[2]), "np_has_gauss": int(ns[3]), "np_cached_gauss": float(ns[4]),
            "torch_cpu": torch.get_rng_state().numpy().copy()}


def restore_rng(state: dict):
    import torch
    if state["py_version"] != 3 or state["np_algorithm"] != "MT19937":
        raise ValueError("unsupported captured CPU RNG schema")
    if np.asarray(state["py_words"]).shape != (625,) or np.asarray(state["np_words"]).shape != (624,) or np.asarray(state["torch_cpu"]).dtype != np.uint8:
        raise ValueError("malformed captured CPU RNG payload")
    random.setstate((3, tuple(int(x) for x in state["py_words"]), state["py_gauss"]))
    np.random.set_state(("MT19937", np.asarray(state["np_words"], dtype=np.uint32),
                         state["np_position"], state["np_has_gauss"], state["np_cached_gauss"]))
    torch.set_rng_state(torch.from_numpy(np.array(state["torch_cpu"], dtype=np.uint8)))


class _Proxy:
    def __init__(self, base, **overrides): self._base, self._overrides = base, overrides
    def __getattr__(self, name): return self._overrides[name] if name in self._overrides else getattr(self._base, name)


class SamplerHook:
    def __init__(self, real): self.real, self.frozen, self.last_raw, self.last_seconds = real, None, None, None
    def __call__(self, *args):
        import torch
        start = time.perf_counter_ns()
        if self.frozen is None:
            result = self.real(*args)
            self.last_raw = result[2]
            self.last_seconds = (time.perf_counter_ns() - start) / 1e9
            return result
        self.last_raw = torch.from_numpy(np.array(self.frozen, dtype=np.int64))
        self.last_seconds = (time.perf_counter_ns() - start) / 1e9
        # Native PENCIL uses only out[2]; do not claim this tuple is a generic sampler API.
        return (None, None, self.last_raw)


class NativeGraph:
    def __init__(self, graph, *, threads: int = 1):
        import torch
        import torch_sparse
        version = importlib.metadata.version("torch-sparse")
        if version.split("+")[0] != "0.6.18": raise ValueError("pinned torch-sparse0.6.18 required")
        torch.set_num_threads(threads)
        self.torch, self.torch_sparse, self.graph = torch, torch_sparse, graph
        began = time.perf_counter()
        col64 = np.empty(graph.m, dtype=np.int64)
        for a in range(0, graph.m, 1_000_000): col64[a:a + 1_000_000] = graph.col[a:a + 1_000_000]
        self.source_ptr = torch.from_numpy(graph.rowptr)
        self.source_col = torch.from_numpy(col64)
        # Match PENCIL's stored edge-ID values and transpose construction, without
        # constructing a separate full edge_index. All global caches are charged.
        base = torch_sparse.SparseTensor(rowptr=self.source_ptr, col=self.source_col,
                  value=torch.arange(graph.m, dtype=torch.int64),
                  sparse_sizes=(graph.n, graph.n), is_sorted=True, trust_data=True)
        self.adj = base.t()
        self.ptr, self.col, self.value = self.adj.csr()
        self.adj.coo()  # Public API's global COO row is charged to setup, warmed once.
        if not torch.equal(self.ptr, self.source_ptr): raise ValueError("strict-clean graph is not symmetric (rowptr)")
        for a in range(0, graph.m, 1_000_000):
            if not torch.equal(self.col[a:a + 1_000_000], self.source_col[a:a + 1_000_000]):
                raise ValueError("strict-clean graph is not symmetric (columns)")
        self.ptr_np, self.col_np = self.ptr.numpy(), self.col.numpy()
        package = Path(torch_sparse.__file__).parent
        self.provenance = {"torch": torch.__version__, "torch_sparse": version,
            "pyg": importlib.metadata.version("torch-geometric"),
            "native_setup_seconds": time.perf_counter() - began,
            "native_setup_memory": memory(), "int64_csr_bytes": self.ptr.numel() * 8 + self.col.numel() * 8,
            "public_coo_row_bytes": graph.m * 8, "stored_value_bytes": graph.m * 8,
            "symmetric_full_CSR_verified": True,
            "extension_binaries": [{"name": p.name, "sha256": sha_file(p)}
                for p in sorted(package.iterdir()) if p.is_file() and p.suffix in (".so", ".pyd", ".dylib")],
            "wrapper_sha256": sha_file(package / "saint.py"),
            "software_versions": {name: importlib.metadata.version(name) for name in
                ("numpy", "numba", "scipy", "transformers", "wandb")},
            "note": "Source/version pin is recorded; wheel build commit is not inferred"}
        self.provenance["wrapper_git_blob"] = hashlib.sha1(b"blob " + str((package / "saint.py").stat().st_size).encode() + b"\0" + (package / "saint.py").read_bytes()).hexdigest()
        if self.provenance["wrapper_git_blob"] != "440455853e0309b00376cbe9ad25d7226cfa5e0e":
            raise ValueError("installed saint wrapper differs from pinned0.6.18 source")


class Facade:
    """Native CSR sampling plus selectable exact raw induction backend."""
    def __init__(self, native: NativeGraph, arm: str, cpp_library: Path | None = None,
                 cpp_module: Path | None = None, max_assoc_bytes: int = 2**30):
        self.native, self.arm = native, arm
        self.last_nodes, self.last_raw, self.last_values, self.last_seconds = None, None, None, None
        self.cpp_library, self.cpp_module = cpp_library, cpp_module
        self.max_assoc_bytes, self.cpp_state, self.cpp_pid = max_assoc_bytes, None, None
        self.initializations = []
        if arm.startswith("compiled_"):
            col = native.graph.col if arm.endswith("int32") else native.col_np
            mode = "pairwise" if "pairwise" in arm else "scan"
            self.compiled = CompiledBackend(native.ptr_np, col, mode)
        elif arm not in ("torch_sparse_public", "torch_sparse_direct", "cpp_full_clear", "cpp_touched_reset"):
            raise ValueError("unknown physical arm: " + arm)

    def csr(self): return self.native.ptr, self.native.col, self.native.value

    def _cpp(self):
        if self.cpp_pid != os.getpid():
            if self.cpp_module is None or self.cpp_library is None:
                raise ValueError("C++ arm requires explicit adapter module and compiled library")
            module = _file_module("_pencil_corrected_native_adapter", self.cpp_module)
            mode = "reference_full_clear" if self.arm == "cpp_full_clear" else "persistent_touched_reset"
            self.cpp_state = module.NativeCSR(self.native.ptr_np, self.native.col_np,
                self.cpp_library, mode=mode, max_assoc_bytes=self.max_assoc_bytes)
            self.cpp_pid = os.getpid()
            self.initializations.append({"pid": self.cpp_pid, "mode": mode,
                                          "assoc_budget_bytes": self.max_assoc_bytes,
                                          "metadata": getattr(self.cpp_state, "initialization", None)})
        return self.cpp_state

    def saint_subgraph(self, node_idx):
        torch, sparse = self.native.torch, self.native.torch_sparse
        nodes = np.array(node_idx.numpy(), dtype=np.int64)
        validate_nodes(nodes, self.native.graph.n)
        if np.any(nodes[1:] <= nodes[:-1]): raise ValueError("PENCIL S must remain sorted unique")
        self.last_nodes = nodes
        cpp = self._cpp() if self.arm.startswith("cpp_") else None
        start = time.perf_counter_ns()
        if self.arm == "torch_sparse_public":
            out, ids = self.native.adj.saint_subgraph(node_idx)
        else:
            if self.arm == "torch_sparse_direct":
                row, col, ids = torch.ops.torch_sparse.saint_subgraph(node_idx, self.native.ptr,
                                             torch.empty(0, dtype=torch.int64), self.native.col)
            else:
                if cpp is not None:
                    result = cpp.induce(nodes)
                    try: arrays = result.copy_arrays()
                    finally: result.close()
                else:
                    arrays = self.compiled.induce(nodes)
                row, col, ids = (torch.from_numpy(a) for a in arrays)
            value = self.native.value[ids] if self.native.value is not None else None
            out = sparse.SparseTensor(row=row, col=col, value=value,
                    sparse_sizes=(len(nodes), len(nodes)), is_sorted=True, trust_data=True)
        self.last_seconds = (time.perf_counter_ns() - start) / 1e9
        if self.arm == "torch_sparse_public": row, col, _ = out.coo()
        # Numpy views retain their originating tensors; no output validation in timer.
        self.last_raw = (row.numpy(), col.numpy(), ids.numpy())
        out_value = out.coo()[2]
        self.last_values = None if out_value is None else out_value.numpy()
        return out, ids

    def close(self):
        if self.cpp_state is not None and self.cpp_pid == os.getpid(): self.cpp_state.close()


class GlobalDataView:
    """C2 live node attributes only; native __getitem__ skips global edge_index."""
    def __init__(self, x, ids, n, m): self.x, self.id, self.num_nodes, self.num_edges = x, ids, n, m
    def __iter__(self): return iter((("x", self.x), ("id", self.id)))


class _OneSample:
    def __init__(self): self.item = None
    def __len__(self): return 1
    def __getitem__(self, idx): return idx, self.item


class QueryRuntime:
    def __init__(self, upstream, native: NativeGraph, features: np.ndarray,
                 pairs: np.ndarray, labels: np.ndarray, arm: str, *,
                 eval_seed: int = 20261126, training_wrapper: bool = False,
                 cpp_library: Path | None = None, cpp_module: Path | None = None,
                 max_assoc_bytes: int = 2**30):
        torch = native.torch
        if features.shape != (native.graph.n, 128) or features.dtype != np.float32:
            raise ValueError("strict-clean C2 features must be N by128 float32")
        self.upstream, self.native, self.features, self.pairs = upstream, native, features, pairs
        self.eval_seed, self.training_wrapper = eval_seed, training_wrapper
        self.facade = Facade(native, arm, cpp_library, cpp_module, max_assoc_bytes)
        self.hook = SamplerHook(torch.ops.torch_sparse.ego_k_hop_sample_adj)
        self._real_torch = upstream.dataset.torch
        # Private file-module namespace only; global Torch operators remain untouched.
        self.proxy = _Proxy(torch, ops=_Proxy(torch.ops,
            torch_sparse=_Proxy(torch.ops.torch_sparse, ego_k_hop_sample_adj=self.hook)))
        source = upstream.dataset.ShaDowKHopSeqFromEdgesMapDataset.__new__(upstream.dataset.ShaDowKHopSeqFromEdgesMapDataset)
        source.all_edges_with_y = torch.from_numpy(np.column_stack([pairs, labels]).astype(np.int64))
        source.labels, source.all_edge_attr, source.wgt = None, None, None
        source.depth_neighbors, source.replace = [[1, 40]], False
        source.adj_t, source.pretrain_mode, source.heuristic_labels = self.facade, False, None
        source.allow_zero_edges = True
        # Copy-on-write feature mappings permit Torch's bridge; actual queries only read.
        source.data = GlobalDataView(torch.from_numpy(features), torch.arange(native.graph.n),
                                     native.graph.n, native.graph.m)
        self.source = source
        self.tokenizer = upstream.tokenizer(num_nodes=82)
        self.one = _OneSample()
        self.wrapper = upstream.wrapper(self.one, self.tokenizer, max_sequence_length=84,
            node_remapping=False, is_eval=not training_wrapper,
            use_features=True, encoding_scheme="adjacency_row", use_edge_weight=False)
        self.collator = upstream.collator(self.tokenizer)

    def raw(self, index: int, frozen_nodes: np.ndarray | None = None, *, capture_rng: bool = False):
        self.hook.frozen = frozen_nodes
        set_query_seed(query_seed(self.eval_seed, index))
        old = self.upstream.dataset.torch
        start = time.perf_counter_ns()
        self.upstream.dataset.torch = self.proxy
        try:
            idx, data = self.source[index]  # Actual unmodified native method.
        finally:
            self.upstream.dataset.torch = old
        elapsed = (time.perf_counter_ns() - start) / 1e9
        if idx != index: raise AssertionError("query index changed")
        if not np.array_equal(self.facade.last_nodes[data.root_n_id.numpy()], self.pairs[index]):
            raise AssertionError("native root/S order changed")
        if data.x.shape[1] != 128 or not self.native.torch.isfinite(data.x).all():
            raise ValueError("selected features invalid")
        stages = {"raw_data_seconds": elapsed, "induction_seconds": self.facade.last_seconds,
                      "sampler_seconds": self.hook.last_seconds,
                      "nodes": self.facade.last_nodes, "raw": self.facade.last_raw,
                      "stored_values": self.facade.last_values}
        if capture_rng: stages["post_raw_rng"] = snapshot_rng()
        return data, stages

    def wrapped(self, index: int, frozen_nodes: np.ndarray | None = None, *,
                wrapper_rng_override: dict | None = None, capture_rng: bool = False):
        data, stages = self.raw(index, frozen_nodes, capture_rng=capture_rng)
        self.one.item = data
        if wrapper_rng_override is not None: restore_rng(wrapper_rng_override)
        began = time.perf_counter_ns()
        wrapped = self.wrapper[index]  # Actual native wrapper; train randperm preserved.
        stages["wrapper_seconds"] = (time.perf_counter_ns() - began) / 1e9
        return wrapped, data, stages

    def close(self): self.facade.close()


def build_model(upstream, config_dir: Path, *, seed: int = 42, checkpoint: Path | None = None,
                tiny: bool = False):
    import torch
    config_path = config_dir / "config.json"
    expected = json.loads((config_dir / "source.json").read_text())["sha256"]
    if sha_file(config_path) != expected: raise ValueError("offline BERT config hash mismatch")
    set_query_seed(seed)
    tokenizer = upstream.tokenizer(num_nodes=82)
    settings = {"name": str(config_dir.resolve()), "hidden_size": 32 if tiny else 512,
        "num_hidden_layers": 1 if tiny else 3, "num_attention_heads": 4 if tiny else 8,
        "max_position_embeddings": 2048, "intermediate_size": 64 if tiny else 2048,
        "from_pretrained": False}
    model = upstream.model.from_default_configs_and_yaml(tokenizer, settings,
        use_features=True, feature_dim=128, feature_fusion="early", encoding_scheme="adjacency_row")
    checkpoint_meta = None
    if checkpoint is not None:
        # Weights-only avoids running arbitrary checkpoint pickle code.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state: state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
        if not isinstance(state, dict): raise ValueError("checkpoint must contain a state dict")
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        checkpoint_meta = {"path": str(checkpoint), "sha256": sha_file(checkpoint)}
    model.eval()
    return model, {"settings": settings, "tiny_fixture_model": tiny,
        "initialization_seed": seed, "checkpoint": checkpoint_meta,
        "quality_claim": "none; randomly initialized frozen model" if checkpoint is None else "declared checkpoint only; no score computed yet",
        "bert_config_sha256": expected, "parameters": sum(p.numel() for p in model.parameters()),
        "features": 128, "feature_fusion": "early", "encoding": "adjacency_row"}


def model_inputs(batch, device):
    keys = ("input_embeds", "attention_mask", "feature_embeds", "num_nodes", "labels")
    return {key: batch[key].to(device) for key in keys if key in batch}


def forward(model, batch, device: str, precision: str):
    import torch
    if precision == "bf16" and device != "cuda": raise ValueError("bf16 bridge intended for CUDA only")
    inputs = model_inputs(batch, device)
    with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                         dtype=torch.bfloat16, enabled=precision == "bf16"):
        return model(**inputs).logits.detach().float().cpu()
