#!/usr/bin/env python3
"""Matched short SEAL training-throughput control, not a full upstream pipeline.

Only mmap GraphCSR and query-local matrices are constructed. Reference/corrected
use verified reference_union and identical per-pair seeds. See pipeline/README.md
for scope, stage-timer overlap, prefetch boundaries, and resource guards.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.seal_revision import exact_benchmark as exact
from scripts.seal_revision import memory_benchmark as memory

ROUTING_VERSION = "production_matched_vectorized_selected_degree_once_v4_reference_bridge"
ROUTING_CONTRACT = {
    "version": ROUTING_VERSION,
    "bridge_arm": "reference_union_compiled",
    "bridge_control": "reference_union sampling and shared S; compiled production induction with the same selected-row guard",
    "selected_degree": "guarded NumPy selected-row starts/ends and int64 sum, once per sampled S",
    "reuse": "shared selected_nnz for guards, stock dtype resolver and production route selection",
    "production_dispatch": "choose scan/pairwise from known degree; invoke exact helper with fixed mode",
    "timing": "selected-degree collection and dispatch included in induction service time",
    "reference": "seal/graph_csr_extraction.py automatic selected-degree calculation",
}

ARMS = ("reference_union_stock", "reference_union_corrected", "production_union", "production_pv", "reference_union_compiled")
MODEL_CONFIG = {"name": "DGCNN", "sortpool_k": 281, "hidden_channels": 32,
                "num_layers": 3, "max_z": 1000, "learning_rate": 1e-4,
                "optimizer": "Adam", "use_feature": False, "node_embedding": False}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dependency_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def configure_gpu_budget(torch_module, device, budget_gib: float) -> dict:
    """Snapshot free VRAM and cap Torch's caching allocator before model.to().

    This is not a device-wide limit: CUDA contexts/libraries, allocations outside
    Torch's caching allocator and other processes remain outside this cap.
    """
    if not math.isfinite(budget_gib) or budget_gib <= 0:
        raise ValueError("GPU memory budget must be finite and positive")
    if device.type != "cuda":
        return {"status": "not_applicable_cpu", "requested_budget_gib": budget_gib,
                "allocator_fraction": None, "free_bytes_before_model": None,
                "total_bytes": None, "cuda_api_called": False}
    requested = int(budget_gib * memory.GIB)
    free, total = (int(v) for v in torch_module.cuda.mem_get_info(device))
    if total <= 0 or free < 0 or free > total:
        raise RuntimeError("CUDA returned invalid free/total memory information")
    if requested > total or requested > int(.8 * free):
        raise RuntimeError(f"GPU preflight refused {budget_gib:.3f} GiB budget: "
                           f"free={free / memory.GIB:.3f} GiB, total={total / memory.GIB:.3f} GiB; "
                           "budget must fit capacity and 80% of current free VRAM")
    fraction = requested / total
    torch_module.cuda.set_per_process_memory_fraction(fraction, device=device)
    return {"status": "configured_before_model_allocation", "requested_budget_gib": budget_gib,
            "allocator_budget_bytes": requested, "allocator_fraction": fraction,
            "free_bytes_before_model": free, "free_gib_before_model": free / memory.GIB,
            "total_bytes": total, "total_gib": total / memory.GIB,
            "free_headroom_fraction": .2, "cuda_api_called": True,
            "scope": "PyTorch caching allocator cap; excludes context/library/direct-CUDA allocations and all external processes",
            "snapshot_note": "free-memory preflight is a snapshot; later external allocations can still cause OOM"}


def pair_seed(seed: int, src: int, dst: int) -> int:
    # Ordered-pair seed; independent of worker scheduling and Python hash salt.
    payload = f"{seed}:{src}:{dst}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def load_graph(config: dict):
    return exact.fixture_graph() if config["fixture"] else exact.load_graph(Path(config["graph_path"]), config["file_endian"])


def graph_metadata(config: dict) -> dict:
    if config["fixture"]:
        graph = exact.fixture_graph()
        return {"path": "fixture", "nodes": graph.n, "adjacency_entries": graph.m,
                "file_endian": sys.byteorder, "timestamp_entries": 0}
    return memory.inspect_graph(Path(config["graph_path"]), config["file_endian"])


def prepare_manifest(path: Path, config: dict, needed: int, n_positives: int | None,
                     train_pairs_csr: Path | None, negative_kind: str) -> dict:
    graph = load_graph(config)
    if not path.exists():
        count = n_positives if n_positives is not None else (needed + 1) // 2
        if count < 1 or count * 2 < needed:
            raise ValueError("--n-positives must supply enough pairs for warmup + measured + prefetch-tail batches")
        if count > 1_000_000:
            raise ValueError("short control limits manifest to one million positives")
        positives = exact.load_graph(train_pairs_csr, config["file_endian"]) if train_pairs_csr else graph
        if positives.n != graph.n or positives.m == 0:
            raise ValueError("positive CSR must have the same N and contain edges")
        rng = np.random.default_rng(config["seed"])
        pairs = np.empty((count * 2, 3), dtype=np.int64)
        for i in range(count):
            offset = int(rng.integers(positives.m))
            src = int(np.searchsorted(positives.starts, offset, side="right") - 1)
            dst = int(positives.indices[offset])
            if src == dst or not graph.has_edge(src, dst):
                raise ValueError("train_pairs_csr positive absent from train graph or is a self loop")
            pairs[2 * i] = (src, dst, 1)
            for _ in range(10000):
                u, v = (int(x) for x in rng.integers(graph.n, size=2))
                if negative_kind == "twohop":
                    first = graph.neighbors(u)
                    if not first.size:
                        continue
                    middle = int(first[int(rng.integers(first.size))])
                    second = graph.neighbors(middle)
                    if not second.size:
                        continue
                    v = int(second[int(rng.integers(second.size))])
                if u != v and not graph.has_edge(u, v):
                    pairs[2 * i + 1] = (u, v, 0)
                    break
            else:
                raise ValueError("could not draw a non-edge for frozen manifest")
        rng.shuffle(pairs)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.save(stream, pairs, allow_pickle=False)
        Path(str(path) + ".json").write_text(json.dumps({
            "created_by": "pipeline_benchmark", "seed": config["seed"],
            "graph": graph_metadata(config), "positive_csr": str(train_pairs_csr) if train_pairs_csr else "train graph",
            "n_positives": count, "negative_kind": negative_kind,
            "negative_filter": "self and existing TRAIN edges excluded; no future-edge/heldout filtering",
            "positive_sampling": "uniform adjacency occurrences with replacement; duplicates allowed",
        }, indent=2) + "\n")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 3 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("manifest must be integer .npy [src,dst,label]")
    if array.shape[0] < needed:
        raise ValueError(f"manifest has {array.shape[0]} rows, requires {needed}")
    for start in range(0, array.shape[0], 65536):
        block = array[start:start + 65536]
        if np.any(block[:, :2] < 0) or np.any(block[:, :2] >= graph.n) or np.any(block[:, 0] == block[:, 1]):
            raise ValueError("manifest contains invalid endpoint or self pair")
        if np.any((block[:, 2] != 0) & (block[:, 2] != 1)):
            raise ValueError("manifest labels must be 0 or 1")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": int(array.shape[0]),
            "positive_occurrences": int(np.count_nonzero(array[:, 2])),
            "columns": ["src", "dst", "label"], "split_scope": "training workload only, no validation"}


def sample_production(graph, pair: tuple[int, int], config: dict, pv: bool) -> np.ndarray:
    """Staged production kernel path, with retained-only visited semantics.

    Uses the actual production Numba collect/dedup/setdiff/merge kernels. Per-query
    RNG initialization is measured; sampling is deterministic across workers.
    """
    kernels = exact.kernels
    seed = pair_seed(config["seed"], *pair)
    rng = np.random.RandomState(seed)
    kernels.seed_numba_rng(seed)
    chunks = [np.asarray(pair, dtype=np.int64)]
    frontier = chunks[0]
    visited = kernels._unique_sorted_int64(frontier)
    for _ in range(config["hops"]):
        if pv:
            per_vertex = max(1, math.ceil(config["cap"] * config["oversample"] / frontier.size))
            estimate = frontier.size * per_vertex
            if estimate > config["max_neighbors"]:
                raise exact.WorkspaceLimit("PV temporary exceeds --max-neighbors")
            values, _ = kernels._sample_neighbors_per_vertex_numba(graph.starts, graph.indices, graph.m, frontier, per_vertex)
        else:
            degree = graph.degree_sum(frontier)
            if degree > config["max_neighbors"]:
                raise exact.WorkspaceLimit("production union expansion exceeds --max-neighbors")
            values = kernels._fill_neighbors_numba(graph.starts, graph.indices, graph.m, frontier)
        fringe = kernels._sorted_setdiff_int64(kernels._unique_sorted_int64(values), visited)
        if fringe.size > config["cap"]:
            fringe = np.sort(fringe[rng.permutation(fringe.size)[:config["cap"]]])
        visited = kernels._merge_sorted_unique_int64(visited, fringe)
        if not fringe.size:
            break
        chunks.append(fringe)
        frontier = fringe
    return np.concatenate(chunks)


def resolve_stock_index_dtype(policy: str, graph_nodes: int, selected_nnz: int) -> np.dtype:
    """Match csr_matrix row restriction's constructor width, not full-A width.

    In SciPy 1.15.3 an int64 full CSR's A[S,:] is reconstructed with int32
    indices/indptr whenever its shape and selected row nnz fit int32. The
    original full graph nnz no longer determines this temporary's width.
    """
    fits32 = max(graph_nodes, selected_nnz) <= np.iinfo(np.int32).max
    if policy == "auto":
        return np.dtype(np.int32 if fits32 else np.int64)
    if policy not in ("int32", "int64"):
        raise ValueError(f"Unknown stock index dtype policy: {policy}")
    if policy == "int32" and not fits32:
        raise exact.WorkspaceLimit("forced int32 cannot represent selected-row shape/nnz")
    return np.dtype(policy)


def selected_degree_sum(graph, nodes: np.ndarray) -> int:
    """Production-matched vectorized degree sum; only O(|S|) local arrays.

    GraphCSR stores N starts without a terminal sentinel. Guard the final row's
    end lookup before np.where, which eagerly evaluates both branches.
    """
    nodes = np.asarray(nodes)
    if nodes.ndim != 1 or nodes.dtype.kind not in "iu":
        raise ValueError("selected nodes must be a one-dimensional integer array")
    if not nodes.size:
        return 0
    starts = graph.starts
    n_starts = int(starts.size)
    if n_starts == 0 or np.any(nodes < 0) or np.any(nodes >= n_starts):
        raise ValueError("selected node outside CSR offset range")
    row_start = starts[nodes]
    row_end = np.where(nodes + 1 < n_starts,
                       starts[np.minimum(nodes + 1, n_starts - 1)], int(graph.m))
    if np.any(row_start < 0) or np.any(row_end < row_start) or np.any(row_end > graph.m):
        raise ValueError("invalid selected CSR row offsets")
    return int((row_end - row_start).sum(dtype=np.int64))


def stock_induced(graph, nodes: np.ndarray, policy: str, *, selected_nnz: int):
    dtype = resolve_stock_index_dtype(policy, graph.n, selected_nnz)
    selected_rows = exact.gather_selected_rows(graph, nodes, graph.n, dtype)
    actual = str(selected_rows.indices.dtype)
    if selected_rows.indptr.dtype != dtype or selected_rows.indices.dtype != dtype:
        raise RuntimeError("Actual selected-row index dtype differs from requested physical route")
    return exact.remove_target(selected_rows[:, nodes]), actual


def induce_selected(graph, nodes: np.ndarray, config: dict, selected_nnz: int):
    """One shared known degree, unchanged guards and exact fixed-mode kernels."""
    arm = config["arm"]
    if arm in ("reference_union_stock", "reference_union_corrected", "reference_union_compiled") and selected_nnz > config["max_neighbors"]:
        raise exact.WorkspaceLimit("selected-row induction exceeds --max-neighbors")
    if arm == "reference_union_stock":
        return stock_induced(graph, nodes, config.get("stock_index_dtype", "auto"), selected_nnz=selected_nnz)
    if arm == "reference_union_corrected":
        return exact.corrected_local(graph, nodes), None
    pairwise = nodes.size * (nodes.size - 1) // 2 <= config["auto_ratio"] * max(1, selected_nnz)
    if not pairwise and selected_nnz > config["max_neighbors"]:
        raise exact.WorkspaceLimit("production scan exceeds --max-neighbors")
    return exact.production_induced(graph, nodes, "pairwise" if pairwise else "scan"), None


def warm_production_kernels() -> None:
    # Both extraction strategies and both array mutabilities compile before the
    # measured window; an auto branch first encountered later must not JIT there.
    config = {"seed": 0, "hops": 2, "cap": 2, "oversample": 1.5, "max_neighbors": 10000}
    for readonly in (False, True):
        graph = exact.fixture_graph()
        if readonly:
            graph.starts.flags.writeable = False
            graph.indices.flags.writeable = False
        sample_production(graph, (0, 1), config, False)
        sample_production(graph, (0, 1), config, True)
        for mode in ("scan", "pairwise"):
            exact.production_induced(graph, np.array([0, 1, 2], dtype=np.int64), mode)


class PipelineDataset:
    """Lazy mmap inside each spawn worker: no graph-array pickle or CUDA fork."""

    def __init__(self, config: dict):
        self.config = config
        self.graph = None
        self.pairs = None
        tail = max(1, config["num_workers"]) * config["prefetch_factor"]
        self.length = config["warmup_pairs"] + (config["steps"] + tail) * config["batch_size"]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        import torch
        from seal.utils import construct_pyg_graph, drnl_node_labeling
        if self.graph is None:
            self.graph = load_graph(self.config)
            self.pairs = np.load(self.config["pairs_path"], mmap_mode="r", allow_pickle=False)
        src, dst, label = (int(v) for v in self.pairs[index])
        pair = (src, dst)
        arm = self.config["arm"]
        started = time.perf_counter()
        if arm.startswith("reference_union_"):
            nodes = exact.sample_node_set(self.graph, pair, self.config["hops"], self.config["cap"],
                                          np.random.default_rng(pair_seed(self.config["seed"], *pair)),
                                          sampler="reference_union", max_neighbors=self.config["max_neighbors"],
                                          max_nodes=2 + self.config["hops"] * self.config["cap"])
        else:
            nodes = sample_production(self.graph, pair, self.config, arm == "production_pv")
        sampled = time.perf_counter()
        selected_nnz = selected_degree_sum(self.graph, nodes)
        adjacency, stock_dtype = induce_selected(self.graph, nodes, self.config, selected_nnz)
        induced = time.perf_counter()
        labels = drnl_node_labeling(adjacency, 0, 1)
        drnl_done = time.perf_counter()
        if int(labels.max()) >= MODEL_CONFIG["max_z"] or int(labels.min()) < 0:
            raise ValueError("DRNL labels outside fixed DGCNN embedding range")
        # Avoid a second DRNL call: construct an equivalent Data with hop labels,
        # then attach exactly the labels computed above. Features are disabled.
        data = construct_pyg_graph(nodes.tolist(), adjacency, np.zeros(nodes.size, dtype=np.int64),
                                   None, label, node_label="hop")
        data.z = labels
        constructed = time.perf_counter()
        audit = None
        if index < self.config["audit_pairs"]:
            if (adjacency != adjacency.T).nnz:
                raise ValueError("undirected control requires symmetric selected adjacency")
            audit = {"index": index, "pair": [src, dst], "label": label,
                     "nodes_sha256": exact.digest(nodes),
                     "edges_sha256": exact.digest(exact.canonical_edges(adjacency)),
                     "drnl_sha256": exact.digest(labels.numpy())}
        return data, np.array([sampled - started, induced - sampled, drnl_done - induced,
                               constructed - drnl_done], dtype=np.float64), audit, stock_dtype


class Collate:
    def __call__(self, records):
        from torch_geometric.data import Batch
        started = time.perf_counter()
        batch = Batch.from_data_list([record[0] for record in records])
        dtype_counts = {}
        for record in records:
            if record[3] is not None:
                dtype_counts[record[3]] = dtype_counts.get(record[3], 0) + 1
        return {"batch": batch, "worker_stage_seconds": np.stack([record[1] for record in records]).sum(axis=0),
                "collate_seconds": time.perf_counter() - started,
                "audits": [record[2] for record in records if record[2] is not None],
                "stock_row_index_dtype_counts": dtype_counts}


def worker_init(_worker_id):
    import torch
    torch.set_num_threads(1)
    warm_production_kernels()


def train_child(config: dict, output: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from seal.models import DGCNN
    from seal.reproducibility import model_state_sha256
    torch.set_num_threads(config["torch_threads"])
    if not config["num_workers"]:
        warm_production_kernels()
    torch.manual_seed(config["seed"])
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    device = torch.device("cuda:0" if config["device"] == "cuda" else "cpu")
    gpu_guard = configure_gpu_budget(torch, device, config["gpu_memory_budget_gib"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    model = DGCNN(32, 3, 1000, 281, train_dataset=None, dynamic_train=True,
                  use_feature=False, node_embedding=None).to(device)
    initial_hash = model_state_sha256(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataset = PipelineDataset(config)
    loader_options = {"batch_size": config["batch_size"], "shuffle": False,
                      "num_workers": config["num_workers"], "collate_fn": Collate(),
                      "worker_init_fn": worker_init, "drop_last": True,
                      "pin_memory": device.type == "cuda",
                      "generator": torch.Generator().manual_seed(config["seed"])}
    if config["num_workers"]:
        loader_options.update(multiprocessing_context="spawn", prefetch_factor=config["prefetch_factor"],
                              persistent_workers=True)
    loader = DataLoader(dataset, **loader_options)
    iterator = None
    started = time.perf_counter()
    audits, losses, gpu_events = [], [], []
    model.train()

    def step(record, measured):
        batch = record["batch"]
        if device.type == "cuda":
            transfer_start, model_start, model_end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            transfer_start.record()
        transfer_cpu = time.perf_counter()
        batch = batch.to(device, non_blocking=device.type == "cuda")
        transfer_cpu = time.perf_counter() - transfer_cpu
        if device.type == "cuda":
            model_start.record()
        model_cpu = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.z, batch.edge_index, batch.batch, None, batch.edge_weight, None).view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.y.float())
        loss.backward()
        optimizer.step()
        model_cpu = time.perf_counter() - model_cpu
        if device.type == "cuda":
            model_end.record()
            if measured:
                gpu_events.append((transfer_start, model_start, model_end))
        if measured:
            losses.append(loss.detach())
        return batch.num_graphs, transfer_cpu, model_cpu

    worker_stages = np.zeros(4)
    loader_wait = collate = transfer_cpu_sum = model_cpu_sum = 0.0
    graphs = measured_positives = 0
    stock_dtype_counts = {}
    try:
        iterator_start = time.perf_counter()
        iterator = iter(loader)
        first_wait_start = time.perf_counter()
        first_record = next(iterator)
        first_batch_wait_seconds = time.perf_counter() - first_wait_start
        first_batch_seconds = time.perf_counter() - iterator_start
        warmup_steps = config["warmup_pairs"] // config["batch_size"]
        for index in range(warmup_steps):
            record = first_record if index == 0 else next(iterator)
            audits.extend(record["audits"])
            step(record, False)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        measured_start = time.perf_counter()
        for _ in range(config["steps"]):
            wait_start = time.perf_counter()
            record = next(iterator)
            loader_wait += time.perf_counter() - wait_start
            audits.extend(record["audits"])
            measured_positives += int(record["batch"].y.sum())
            for dtype, count in record["stock_row_index_dtype_counts"].items():
                stock_dtype_counts[dtype] = stock_dtype_counts.get(dtype, 0) + count
            count, transfer_cpu, model_cpu = step(record, True)
            graphs += count
            worker_stages += record["worker_stage_seconds"]
            collate += record["collate_seconds"]
            transfer_cpu_sum += transfer_cpu
            model_cpu_sum += model_cpu
        if device.type == "cuda":
            torch.cuda.synchronize()
        measured_end = time.perf_counter()
        elapsed = measured_end - measured_start
        loss_values = [float(loss.cpu()) for loss in losses]
        if not all(math.isfinite(loss) for loss in loss_values):
            raise RuntimeError("non-finite training loss in throughput control")
        transfer_gpu = sum(a.elapsed_time(b) for a, b, _ in gpu_events) / 1000 if gpu_events else None
        model_gpu = sum(b.elapsed_time(c) for _, b, c in gpu_events) / 1000 if gpu_events else None
        final_hash = model_state_sha256(model)
        report = {"schema_version": 1, "benchmark_version": 4, "routing_contract": ROUTING_CONTRACT, "status": "complete", "arm": config["arm"], "config": config,
                  "scope": "local reference_union semantic training harness; not full upstream SEAL-OGB",
                  "model": {**MODEL_CONFIG, "parameters": sum(p.numel() for p in model.parameters()),
                            "initial_state_sha256": initial_hash, "final_state_sha256": final_hash},
                  "device": {"type": device.type, "name": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
                             "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
                             "visible_gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
                             "used_devices": [str(device)], "training_parallelism": "one GPU, no DDP"},
                  "timing": {"first_batch_seconds": first_batch_seconds,
                             "first_batch_next_wait_seconds": first_batch_wait_seconds,
                             "warmup_steps": warmup_steps, "warmup_graphs_excluded": config["warmup_pairs"],
                             "measured_steps": config["steps"], "measured_graphs": graphs,
                             "measured_positive_occurrences": measured_positives,
                             "steady_seconds": elapsed, "graphs_per_second": graphs / elapsed,
                             "worker_service_seconds_sum": dict(zip(("sampling", "induction", "drnl", "construct"), worker_stages.tolist())),
                             "collate_worker_service_seconds_sum": collate, "loader_wait_main_seconds": loader_wait,
                             "h2d_cpu_enqueue_seconds": transfer_cpu_sum, "model_cpu_enqueue_seconds": model_cpu_sum,
                             "h2d_cuda_event_seconds": transfer_gpu, "model_cuda_event_seconds": model_gpu,
                             "train_loop_through_measured_seconds": measured_end - started,
                             "prefetch_batches_capacity": config["num_workers"] * config["prefetch_factor"],
                             "prefetch_boundary_fraction": config["num_workers"] * config["prefetch_factor"] / config["steps"]},
                  "losses": loss_values, "audits": sorted(audits, key=lambda a: a["index"]),
                  "stock_physical_route": {"index_dtype_policy": config["stock_index_dtype"],
                                           "measured_temporary_row_dtype_counts": stock_dtype_counts,
                                           "auto_rule": "int32 if max(N, selected-row nnz)<=2**31-1; otherwise int64. Global full-A nnz does not set row width.",
                                           "forced64_scope": "explicit physical variant; not the native-auto headline baseline"},
                  "gpu_memory": {"allocated_peak_gib": torch.cuda.max_memory_allocated() / memory.GIB if device.type == "cuda" else None,
                                 "reserved_peak_gib": torch.cuda.max_memory_reserved() / memory.GIB if device.type == "cuda" else None,
                                 "guard": gpu_guard,
                                 "scope": "whole training process GPU memory; not graph representation size"}}
        output.write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        if iterator is not None and config["num_workers"]:
            iterator._shutdown_workers()


def preflight(meta: dict, config: dict) -> dict:
    workers = max(1, config["num_workers"])
    nodes = 2 + config["hops"] * config["cap"]
    graph_bytes = meta["nodes"] * 8 + meta["adjacency_entries"] * 4
    # Conservative 8*N column-index scratch per active stock worker, even when
    # native-auto uses 4*N. Local buffers and queued graphs remain separate.
    global_scratch = meta["nodes"] * 8 if config["arm"] == "reference_union_stock" else 0
    stock_policy = config.get("stock_index_dtype", "auto")
    expected_width = None
    if config["arm"] == "reference_union_stock":
        expected_width = resolve_stock_index_dtype(stock_policy, meta["nodes"], config["max_neighbors"]).itemsize
    local_scratch = config["max_neighbors"] * 256 + nodes * 256
    prefetch = workers * config["prefetch_factor"] * config["batch_size"] * nodes * nodes * 24
    reserve = (workers + 1) * (1536 << 20)
    estimate = graph_bytes + workers * (global_scratch + local_scratch) + prefetch + reserve
    return {"estimated_peak_bytes": estimate, "estimated_peak_gib": estimate / memory.GIB,
            "graph_logical_bytes_once": graph_bytes, "stock_global_scratch_bytes_per_worker": global_scratch,
            "stock_global_scratch_scope": "conservative 8*N upper bound even for native-auto/int32; not actual row width",
            "stock_expected_column_scratch_bytes_per_worker": meta["nodes"] * expected_width if expected_width else 0,
            "stock_expected_row_index_dtype": f"int{expected_width * 8}" if expected_width else None,
            "local_scratch_bound_bytes_per_worker": local_scratch,
            "queued_local_graphs_bound_bytes": prefetch, "interpreter_library_reserve_bytes": reserve,
            "conservative_estimate_not_measured": True}


def run_isolated(config: dict, out: Path, budget_gib: float, timeout: float, meta: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    estimate = preflight(meta, config)
    limit = int(budget_gib * memory.GIB)
    available = memory.available_memory_bytes()
    base = {"config": config, "preflight": estimate, "directory": str(out)}
    if estimate["estimated_peak_bytes"] > limit or (available is not None and estimate["estimated_peak_bytes"] > .8 * available):
        return {**base, "status": "skipped_preflight", "reason": "estimated CPU footprint exceeds memory budget/headroom"}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", SEAL_DEBUG_SUBGRAPH="0")
    start, samples = time.monotonic(), []
    aborted = None
    output = out / "result.json"
    with (out / "process.log").open("w") as log, (out / "memory.samples.jsonl").open("w") as sample_log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--_child-config", str(config_path),
                                    "--out", str(output)], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
        try:
            while process.poll() is None:
                sample = memory.sample_tree(process.pid)
                sample["elapsed_seconds"] = time.monotonic() - start
                sample_log.write(json.dumps(sample) + "\n")
                sample_log.flush()
                samples.append(sample)
                if sample["tree_pss_bytes"] is not None and sample["tree_pss_bytes"] > limit:
                    aborted = "sampled CPU process-tree PSS exceeded budget"
                if time.monotonic() - start > timeout:
                    aborted = "scenario timeout"
                if aborted:
                    memory.kill_process_group(process.pid)
                    break
                time.sleep(.25)
            process.wait()
        finally:
            memory.kill_process_group(process.pid)
            process.wait()
    result = json.loads(output.read_text()) if output.exists() and process.returncode == 0 else {"status": "failed"}
    return {**base, **result, "returncode": process.returncode, "abort_reason": aborted,
            "cpu_memory_sampled_peak": memory.maxima(samples),
            "memory_note": "PSS tree is proportional; RSS sum is non-unique. Includes model/worker runtime, not graph-only memory."}


def equality_audit(runs: list[dict]) -> list[dict]:
    from itertools import combinations
    reference_arms = ("reference_union_stock", "reference_union_corrected", "reference_union_compiled")
    comparisons = []
    for left_arm, right_arm in combinations(reference_arms, 2):
        for left in runs:
            if left.get("status") != "complete" or left["config"]["arm"] != left_arm:
                continue
            candidates = [r for r in runs if r.get("status") == "complete" and r["config"]["arm"] == right_arm
                          and all(r["config"][k] == left["config"][k] for k in ("num_workers", "batch_size"))]
            for right in candidates:
                matched = (left["audits"] == right["audits"]
                           and left["model"]["initial_state_sha256"] == right["model"]["initial_state_sha256"]
                           and left["config"].get("manifest_sha256") == right["config"].get("manifest_sha256"))
                comparisons.append({"reference_arm": left_arm, "candidate_arm": right_arm,
                                    "num_workers": left["config"]["num_workers"], "batch_size": left["config"]["batch_size"],
                                    "query_audits": len(left["audits"]), "same_nodes_edges_drnl_and_initial_model": matched,
                                    "scope": "audited warmup prefix; does not independently audit every measured graph"})
    return comparisons


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-path", type=Path)
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--file-endian", choices=("little", "big"), default="little")
    p.add_argument("--pairs-path", type=Path)
    p.add_argument("--train-pairs-csr", type=Path)
    p.add_argument("--n-positives", type=int)
    p.add_argument("--negative-kind", choices=("uniform", "twohop"), default="uniform")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--workers", type=int, nargs="+", default=[1])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--prefetch-factor", type=int, default=1)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--cap", type=int, default=200)
    p.add_argument("--oversample", type=float, default=1.5)
    p.add_argument("--auto-ratio", type=float, default=1)
    p.add_argument("--max-neighbors", type=int, default=2_000_000)
    p.add_argument("--memory-budget-gib", type=float, default=180)
    p.add_argument("--gpu-memory-budget-gib", type=float, default=24,
                   help="CUDA caching allocator cap; must fit 80%% of free VRAM, context/external allocations excluded")
    p.add_argument("--stock-index-dtype", choices=("auto", "int32", "int64"), default="auto",
                   help="Temporary selected-row CSR width; auto matches native csr_matrix row restriction")
    p.add_argument("--timeout-seconds", type=float, default=1800)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--audit-pairs", type=int, default=32)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--_child-config", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args._child_config:
        train_child(json.loads(args._child_config.read_text()), args.out)
        return 0
    if bool(args.graph_path) == args.fixture:
        p.error("select exactly one of --graph-path or --fixture")
    if (min(args.workers) < 0 or min(args.batch_sizes) < 1 or args.steps < 1 or args.warmup_steps < 1
            or args.cap < 1 or args.hops < 1 or args.prefetch_factor < 1 or args.max_neighbors < 1
            or args.memory_budget_gib <= 0 or args.timeout_seconds <= 0 or args.audit_pairs < 1
            or not math.isfinite(args.gpu_memory_budget_gib) or args.gpu_memory_budget_gib <= 0
            or args.torch_threads < 1 or args.oversample <= 0 or args.auto_ratio <= 0):
        p.error("workers>=0; batch/steps/cap/hops/budgets/audit/thread values must be positive")
    if args.batch_sizes != [32] and not args.fixture and any(b not in (32, 128) for b in args.batch_sizes):
        p.error("real-data matched controls support fixed batch sizes 32 and 128")
    args.out.mkdir(parents=True, exist_ok=True)
    # Same warmup and measured query prefix across worker settings. For the
    # declared 32/128 batch sizes, this is divisible by every batch size.
    quantum = math.lcm(*args.batch_sizes)
    reserve = max(args.warmup_steps, max(args.workers), 1) * max(args.batch_sizes)
    warmup_pairs = math.ceil(reserve / quantum) * quantum
    needed = warmup_pairs + (args.steps + max(1, max(args.workers)) * args.prefetch_factor) * max(args.batch_sizes)
    config = {"fixture": args.fixture, "graph_path": str(args.graph_path.resolve()) if args.graph_path else None,
              "file_endian": args.file_endian, "seed": args.seed, "hops": args.hops, "cap": args.cap,
              "oversample": args.oversample, "auto_ratio": args.auto_ratio,
              "max_neighbors": args.max_neighbors, "warmup_pairs": warmup_pairs,
              "steps": args.steps, "prefetch_factor": args.prefetch_factor, "audit_pairs": min(args.audit_pairs, warmup_pairs),
              "device": args.device, "torch_threads": args.torch_threads,
              "gpu_memory_budget_gib": args.gpu_memory_budget_gib, "stock_index_dtype": args.stock_index_dtype}
    path = args.pairs_path.resolve() if args.pairs_path else (args.out / "pairs.npy").resolve()
    manifest = prepare_manifest(path, config, needed, args.n_positives, args.train_pairs_csr, args.negative_kind)
    config["pairs_path"] = manifest["path"]
    config["manifest_sha256"] = manifest["sha256"]
    report = {"schema_version": 1, "benchmark_version": 4, "routing_contract": ROUTING_CONTRACT, "scope": "undirected simple binary sorted CSR; features disabled; single-GPU short training control, no validation",
              "stock_index_dtype": args.stock_index_dtype,
              "model": MODEL_CONFIG, "graph": graph_metadata(config), "manifest": manifest,
              "runtime_versions": {name: dependency_version(name) for name in
                                   ("numpy", "scipy", "numba", "torch", "torch-geometric")},
              "host": {"platform": platform.platform(), "hostname": platform.node(),
                       "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
              "contract": {"reference": "verified reference_union semantic implementation + native-auto row-restricted SciPy route by default; not full upstream timing",
                           "correction": "reference/corrected change induction only, with same per-pair S and RNG",
                           "bridge": "reference_union_compiled keeps reference sampling/S/DRNL/model and replaces only induction with production fixed-mode compiled kernels",
                           "production": "staged actual Numba sampler/edge kernels; no production DataLoader/pipeline identity claim",
                           "timers": "worker-stage wall-clock service sums overlap parallel workers, prefetch and GPU; cannot add to wall time",
                           "steady": "only measured steps in numerator and timer, after excluded training warmup; synchronize CUDA at both boundaries",
                           "prefetch": "one iterator continues across boundary and beyond measured end; queued unmeasured tail prevents deliberate drain; finite-window bias reported",
                           "quality": "no validation or quality-preservation claim from short BCE steps",
                           "workers": "DataLoader worker count; parent is additional; spawn with lazy read-only mmap, no CUDA fork/graph pickle"},
              "source_sha256": {str(file.relative_to(ROOT)): sha256_file(file) for file in
                                (Path(__file__).resolve(), ROOT / "scripts/seal_revision/exact_benchmark.py",
                                 ROOT / "scripts/seal_revision/memory_benchmark.py", ROOT / "seal/models.py",
                                 ROOT / "seal/utils.py", ROOT / "seal/graph_csr_kernels.py",
                                 ROOT / "seal/graph_csr_extraction.py")},
              "runs": [], "equality_audits": []}
    if args.manifest_only:
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(manifest))
        return 0
    for batch_size in args.batch_sizes:
        for workers in args.workers:
            for arm in args.arms:
                job = {**config, "batch_size": batch_size, "num_workers": workers, "arm": arm}
                directory = args.out / f"{arm}.batch{batch_size}.workers{workers}"
                run = run_isolated(job, directory, args.memory_budget_gib, args.timeout_seconds, report["graph"])
                report["runs"].append(run)
                report["equality_audits"] = equality_audit(report["runs"])
                temp = args.out / "report.json.tmp"
                temp.write_text(json.dumps(report, indent=2) + "\n")
                temp.replace(args.out / "report.json")
                print(json.dumps({"arm": arm, "batch_size": batch_size, "workers": workers, "status": run["status"]}), flush=True)
    return int(any(r["status"] != "complete" for r in report["runs"]) or
               any(not a["same_nodes_edges_drnl_and_initial_model"] for a in report["equality_audits"]))


if __name__ == "__main__":
    raise SystemExit(main())
