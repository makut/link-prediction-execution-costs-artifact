#!/usr/bin/env python3
"""Frozen-node-set SEAL extraction benchmark; no Torch/PyG/GPU imports.

Examples (run from the repository root)::

    python -m scripts.seal_revision.exact_benchmark --fixture --output /tmp/exact.json
    python -m scripts.seal_revision.exact_benchmark --fixture --directed \
        --pad-isolates 0,10000,1000000 --output /tmp/padding.json
    python -m scripts.seal_revision.exact_benchmark --graph-path DATA/train_csr \
        --pairs-path pairs.npy --caps 50,200 --hops 2 --limit 100 --output exact.json

GraphCSR files are mapped read-only, without timestamps or whole-graph conversion.
Stock is the row-gather + SciPy column-fancy-index route equivalent to
A[S, :][:, S]. The row gather is implemented here over mmap rows; it does NOT
measure conversion to a whole-graph SciPy matrix or the original SEAL pipeline.
Its index width is explicit. A workspace guard prevents accidental O(N) scratch
allocations. Virtual isolates change only the column count of this row-restricted
matrix; no padded global offset array is allocated.

Contract: simple binary graphs, sorted unique CSR rows, no self loops; symmetric
CSR in undirected mode; exact incoming transpose in directed mode. Contracts are
checked on touched rows / selected induced edges, not by scanning the full graph.
Sampling is performed once per pair/cap/sampler, outside all execution timers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

# These production kernels depend on NumPy/Numba only, not Torch or GraphCSR.
from seal import graph_csr_kernels as kernels

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("stock_scipy", "corrected_local_csr", "ours_scan", "ours_pairwise", "ours_auto")


class WorkspaceLimit(ValueError):
    """An arm was skipped before allocating its declared scratch budget."""


@dataclass
class CSRGraph:
    starts: np.ndarray  # N offsets, without the final sentinel
    indices: np.ndarray
    origin: str = "in_memory"

    @property
    def n(self) -> int:
        return int(self.starts.size)

    @property
    def m(self) -> int:
        return int(self.indices.size)

    def bounds(self, u: int) -> tuple[int, int]:
        if not 0 <= u < self.n:
            raise ValueError(f"node {u} outside [0,{self.n})")
        a = int(self.starts[u])
        b = int(self.starts[u + 1]) if u + 1 < self.n else self.m
        if a < 0 or b < a or b > self.m:
            raise ValueError(f"invalid CSR offsets for node {u}: {a}, {b}")
        return a, b

    def neighbors(self, u: int) -> np.ndarray:
        a, b = self.bounds(int(u))
        return self.indices[a:b]

    def has_edge(self, u: int, v: int) -> bool:
        row = self.neighbors(u)
        j = int(np.searchsorted(row, v))
        return j < row.size and int(row[j]) == v

    def validate_row(self, u: int) -> None:
        row = self.neighbors(u)
        if row.size and (int(row[0]) < 0 or int(row[-1]) >= self.n):
            raise ValueError(f"out-of-range neighbor in row {u}")
        # Chunk validation avoids a degree-sized temporary boolean array.
        for start in range(1, row.size, 65536):
            stop = min(row.size, start + 65536)
            if np.any(row[start:stop] <= row[start - 1:stop - 1]):
                raise ValueError(f"row {u} must be sorted and unique")
        if self.has_edge(u, u):
            raise ValueError(f"self loop at {u}: simple loop-free graph required")

    def degree_sum(self, nodes: np.ndarray) -> int:
        return sum(self.bounds(int(u))[1] - self.bounds(int(u))[0] for u in nodes)


def load_graph(path: Path, endian: str = "little") -> CSRGraph:
    """Load GraphCSR's N raw int64 offsets + int64-header/int32-neighbor file."""
    path = path.resolve()
    if path.suffix == ".npz":
        a = sparse.load_npz(path).tocsr()
        if a.shape[0] != a.shape[1]:
            raise ValueError("input NPZ must be a square adjacency matrix")
        if np.any(a.data != 1):
            raise ValueError("only binary adjacency values equal to one are supported")
        return CSRGraph(a.indptr[:-1], a.indices, str(path))
    prefix = "<" if endian == "little" else ">"
    offset_path, index_path = path / "edge_starts.bin", path / "edge_ends.bin"
    size_offsets = offset_path.stat().st_size
    if size_offsets == 0 or size_offsets % 8:
        raise ValueError("edge_starts.bin must contain N>0 raw int64 offsets")
    with index_path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise ValueError("missing int64 neighbor-count header")
    count = int.from_bytes(header, endian, signed=True)
    if count < 0 or index_path.stat().st_size != 8 + count * 4:
        raise ValueError("edge_ends.bin size does not match its int64 header")
    starts = np.memmap(offset_path, dtype=prefix + "i8", mode="r")
    indices = (np.memmap(index_path, dtype=prefix + "i4", offset=8, shape=(count,), mode="r")
               if count else np.empty(0, dtype=prefix + "i4"))
    if not starts.dtype.isnative or not indices.dtype.isnative:
        raise ValueError("production Numba arms require native endian; convert the dataset outside this benchmark")
    graph = CSRGraph(starts, indices, str(path))
    if int(starts[0]) != 0 or int(starts[-1]) > count:
        raise ValueError("invalid initial/final CSR offsets")
    return graph


def fixture_graph(directed: bool = False) -> CSRGraph:
    # Includes asymmetric arcs, a bridge, disconnected components and isolates.
    edges = [(0, 1), (0, 2), (2, 1), (2, 3), (3, 0), (1, 4),
             (4, 2), (5, 3), (6, 7)]
    if not directed:
        edges += [(v, u) for u, v in edges]
    rows, cols = zip(*edges)
    a = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(9, 9))
    a.sum_duplicates()
    a.data[:] = 1
    a.sort_indices()
    return CSRGraph(a.indptr[:-1].astype(np.int64), a.indices.astype(np.int32), "fixture")


def transpose_small(graph: CSRGraph, edge_limit: int) -> CSRGraph:
    if graph.m > edge_limit:
        raise ValueError("directed input requires --in-graph-path; refusing a whole-graph transpose above --max-transpose-edges")
    ptr = np.append(graph.starts, graph.m)
    a = sparse.csr_matrix((np.ones(graph.m, dtype=np.int8), graph.indices, ptr), shape=(graph.n, graph.n))
    t = a.T.tocsr()
    return CSRGraph(t.indptr[:-1].astype(np.int64), t.indices, "small_transpose:" + graph.origin)


def deterministic_pairs(graph: CSRGraph, count: int, seed: int, kind: str) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    pairs = []
    if graph.n < 2:
        raise ValueError("at least two vertices required")
    for i in range(count):
        positive = kind == "edges" or (kind in ("mixed", "mixed_twohop") and i % 2 == 0)
        if positive:
            if graph.m == 0:
                raise ValueError("cannot sample existing edges from an empty graph")
            j = int(rng.integers(graph.m))
            u = int(np.searchsorted(graph.starts, j, side="right") - 1)
            v = int(graph.indices[j])
        else:
            for _ in range(10000):
                u, v = (int(x) for x in rng.integers(graph.n, size=2))
                if kind in ("twohop", "mixed_twohop"):
                    first = graph.neighbors(u)
                    if not first.size:
                        continue
                    mid = int(first[int(rng.integers(first.size))])
                    second = graph.neighbors(mid)
                    if not second.size:
                        continue
                    v = int(second[int(rng.integers(second.size))])
                if u != v and (kind == "uniform" or not graph.has_edge(u, v)):
                    break
            else:
                raise ValueError("failed to draw a non-edge from this graph")
        if u == v:
            raise ValueError("sampled a self loop; loop-free input required")
        pairs.append((u, v))
    return pairs


def read_pairs(path: Path, limit: int) -> list[tuple[int, int]]:
    if path.suffix == ".npy":
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        if a.ndim != 2 or a.shape[1] < 2 or not np.issubdtype(a.dtype, np.integer):
            raise ValueError("pairs.npy must be a two-dimensional integer array with >=2 columns")
        return [(int(u), int(v)) for u, v in a[:limit, :2]]
    pairs = []
    with path.open() as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.replace(",", " ").split()
            pairs.append((int(fields[0]), int(fields[1])))
            if len(pairs) >= limit:
                break
    return pairs


def sample_node_set(graph: CSRGraph, pair: tuple[int, int], hops: int, cap: int | None,
                    rng: np.random.Generator, sampler: str = "reference_union",
                    incoming: CSRGraph | None = None, oversample: float = 1.5,
                    max_neighbors: int = 2_000_000, max_nodes: int = 10000) -> np.ndarray:
    """Choose S once. Reference union marks all discovered vertices visited before capping.

    production_union marks only retained vertices visited (as current LPP does).
    per_vertex uses bounded replacement draws before deduplication, also as LPP.
    No Numba RNG is consumed, making the frozen S independently reproducible.
    """
    src, dst = pair
    if src == dst:
        raise ValueError("source and destination must differ")
    graph.bounds(src); graph.bounds(dst)
    nodes, frontier, visited = [src, dst], [src, dst], {src, dst}
    for _ in range(hops):
        chunks = []
        budget = max(1, math.ceil(cap * oversample / len(frontier))) if cap and frontier else None
        for u in frontier:
            graph.validate_row(u)
            rows = [graph.neighbors(u)]
            if incoming is not None:
                incoming.validate_row(u)
                rows.append(incoming.neighbors(u))
            degree = sum(int(x.size) for x in rows)
            if sampler == "per_vertex" and budget is not None and degree > budget:
                # Sampling the concatenated out/in multiset without materializing it.
                positions = rng.integers(degree, size=budget)
                out = rows[0]
                if len(rows) == 1:
                    chunk = np.asarray(out[positions], dtype=np.int64)
                else:
                    chunk = np.empty(budget, dtype=np.int64)
                    mask = positions < out.size
                    chunk[mask] = out[positions[mask]]
                    chunk[~mask] = rows[1][positions[~mask] - out.size]
                chunks.append(chunk)
            else:
                if degree > max_neighbors:
                    raise WorkspaceLimit(f"expansion row {u} has {degree} neighbors > --max-neighbors")
                chunks.extend(rows)
        size = sum(int(x.size) for x in chunks)
        if size > max_neighbors:
            raise WorkspaceLimit(f"expansion needs {size} neighbors > --max-neighbors")
        candidates = sorted(set(int(x) for row in chunks for x in row).difference(visited))
        if sampler == "reference_union":
            visited.update(candidates)
        if cap is not None and len(candidates) > cap:
            take = rng.choice(len(candidates), cap, replace=False)
            candidates = sorted(candidates[int(i)] for i in take)
        if sampler != "reference_union":
            visited.update(candidates)
        if len(nodes) + len(candidates) > max_nodes:
            raise WorkspaceLimit("selected S exceeds --max-subgraph-nodes")
        if not candidates:
            break
        nodes.extend(candidates)
        frontier = candidates
    return np.asarray(nodes, dtype=np.int64)


def remove_target(a: sparse.csr_matrix) -> sparse.csr_matrix:
    # Avoid SparseEfficiencyWarning / inserting absent zeros into CSR.
    a = a.tocoo()
    keep = ~(((a.row == 0) & (a.col == 1)) | ((a.row == 1) & (a.col == 0)))
    return sparse.csr_matrix((a.data[keep], (a.row[keep], a.col[keep])), shape=a.shape)


def gather_selected_rows(graph: CSRGraph, nodes: np.ndarray, columns: int,
                         index_dtype: np.dtype) -> sparse.csr_matrix:
    """Materialize only A[S,:], with explicit SciPy index dtype (including int64)."""
    degrees = [graph.bounds(int(u))[1] - graph.bounds(int(u))[0] for u in nodes]
    ptr = np.zeros(nodes.size + 1, dtype=index_dtype)
    np.cumsum(degrees, out=ptr[1:])
    cols = np.empty(int(ptr[-1]), dtype=index_dtype)
    for i, u in enumerate(nodes):
        cols[int(ptr[i]):int(ptr[i + 1])] = graph.neighbors(int(u))
    a = sparse.csr_matrix((nodes.size, columns), dtype=np.int8)
    # Direct assignment preserves the selected reference index width; the public
    # tuple constructor may silently downcast int64 arrays on small shapes.
    a.indptr, a.indices, a.data = ptr, cols, np.ones(cols.size, dtype=np.int8)
    a.has_sorted_indices = True
    return a


def stock_scipy(graph: CSRGraph, nodes: np.ndarray, virtual_n: int, index_dtype: np.dtype) -> sparse.csr_matrix:
    selected_rows = gather_selected_rows(graph, nodes, virtual_n, index_dtype)
    return remove_target(selected_rows[:, nodes])


def corrected_local(graph: CSRGraph, nodes: np.ndarray) -> sparse.csr_matrix:
    """Patch column selection with compact local column IDs, never an N-sized map."""
    order = np.argsort(nodes)
    ordered = nodes[order]
    rows, cols = [], []
    for i, u in enumerate(nodes):
        neighbors = graph.neighbors(int(u))
        positions = np.searchsorted(ordered, neighbors)
        valid = positions < ordered.size
        matched = np.zeros(neighbors.size, dtype=bool)
        matched[valid] = ordered[positions[valid]] == neighbors[valid]
        local = order[positions[matched]]
        keep = ~(((i == 0) & (local == 1)) | ((i == 1) & (local == 0)))
        local = local[keep]
        rows.extend([i] * local.size)
        cols.extend(local.tolist())
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(nodes.size, nodes.size))


def production_induced(graph: CSRGraph, nodes: np.ndarray, mode: str,
                       incoming: CSRGraph | None = None, auto_ratio: float = 1.0) -> sparse.csr_matrix:
    pairwise = mode == "pairwise"
    if mode == "auto":
        pairs = nodes.size * (nodes.size - 1) // 2
        pairwise = (2 if incoming is not None else 1) * pairs <= auto_ratio * max(1, graph.degree_sum(nodes))
    if pairwise:
        if incoming is None:
            rows, cols, data = kernels._build_induced_edges_pairwise(graph.starts, graph.indices, graph.m, nodes, 0, 1)
        else:
            rows, cols, data = kernels._build_induced_edges_pairwise_directed(
                graph.starts, graph.indices, graph.m, incoming.starts, incoming.indices, incoming.m, nodes, 0, 1)
    else:
        neighbors, owners = kernels._fill_neighbors_with_owner_numba(graph.starts, graph.indices, graph.m, nodes)
        order = np.argsort(nodes)
        rows, cols, data = kernels._build_induced_edges_neighbor_scan(owners, neighbors, nodes[order], order, 0, 1)
    return sparse.csr_matrix((data, (rows, cols)), shape=(nodes.size, nodes.size))


def oracle_induced(graph: CSRGraph, nodes: np.ndarray) -> sparse.csr_matrix:
    """Simple independent Python edge-set oracle, outside benchmark timers."""
    mapping = {int(u): i for i, u in enumerate(nodes)}
    edges = []
    for i, u in enumerate(nodes):
        for v in graph.neighbors(int(u)):
            j = mapping.get(int(v))
            if j is not None and (i, j) not in ((0, 1), (1, 0)):
                edges.append((i, j))
    if not edges:
        return sparse.csr_matrix((nodes.size, nodes.size), dtype=np.int8)
    rows, cols = zip(*edges)
    return sparse.csr_matrix((np.ones(len(edges), dtype=np.int8), (rows, cols)), shape=(nodes.size, nodes.size))


def canonical_edges(a: sparse.csr_matrix) -> np.ndarray:
    row, col = a.nonzero()
    order = np.lexsort((col, row))
    return np.stack((row[order], col[order]), axis=1).astype(np.int64)


def drnl_labels(a: sparse.csr_matrix) -> np.ndarray:
    """NumPy equivalent of SEAL-OGB DRNL; endpoint-deleted, undirected distances."""
    n = a.shape[0]
    without_dst = np.r_[0, np.arange(2, n)]
    without_src = np.arange(1, n)
    d_src = shortest_path(a[without_dst, :][:, without_dst], directed=False, unweighted=True, indices=0)
    d_dst = shortest_path(a[without_src, :][:, without_src], directed=False, unweighted=True, indices=0)
    d_src = np.insert(d_src, 1, 0)
    d_dst = np.insert(d_dst, 0, 0)
    labels = np.zeros(n, dtype=np.int64)
    finite = np.isfinite(d_src) & np.isfinite(d_dst)
    distance = d_src[finite] + d_dst[finite]
    half, mod = np.floor_divide(distance, 2), np.mod(distance, 2)
    labels[finite] = (1 + np.minimum(d_src[finite], d_dst[finite]) + half * (half + mod - 1)).astype(np.int64)
    labels[:2] = 1
    return labels


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean_s": float(a.mean()), "p50_s": float(np.percentile(a, 50)),
            "p95_s": float(np.percentile(a, 95)), "p99_s": float(np.percentile(a, 99)),
            "min_s": float(a.min()), "max_s": float(a.max())}


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<i8").tobytes()).hexdigest()


def measure(method: Callable[[], sparse.csr_matrix], expected: np.ndarray,
            expected_drnl: np.ndarray | None, warmup: int, repeats: int) -> dict:
    times, labels_times, combined = [], [], []
    for i in range(warmup + repeats):
        t0 = time.perf_counter_ns()
        actual = method()
        t1 = time.perf_counter_ns()
        labels = drnl_labels(actual) if expected_drnl is not None else None
        t2 = time.perf_counter_ns()
        # Validation never enters execution/DRNL timers and is repeated for every result.
        if not np.array_equal(canonical_edges(actual), expected):
            raise AssertionError("canonical induced edges differ from the independent frozen-S oracle")
        if actual[0, 1] or actual[1, 0]:
            raise AssertionError("target edge survived removal")
        if labels is not None and not np.array_equal(labels, expected_drnl):
            raise AssertionError("DRNL labels differ from frozen-S oracle")
        if i >= warmup:
            times.append((t1 - t0) / 1e9)
            labels_times.append((t2 - t1) / 1e9)
            combined.append((t2 - t0) / 1e9)
    return {"status": "ok", "edge_seconds": times,
            "drnl_seconds": labels_times if expected_drnl is not None else None,
            "edge_plus_drnl_seconds": combined if expected_drnl is not None else None,
            "edge_summary": stats(times),
            "drnl_summary": stats(labels_times) if expected_drnl is not None else None,
            "edge_plus_drnl_summary": stats(combined) if expected_drnl is not None else None,
            "exact_edges": True, "exact_drnl": expected_drnl is not None}


def benchmark_pair(graph: CSRGraph, nodes: np.ndarray, virtual_n: int, *,
                   incoming: CSRGraph | None = None, warmup: int = 1, repeats: int = 3,
                   stock_index_dtype: str = "int64", workspace_bytes: int = 128 << 20,
                   max_neighbors: int = 2_000_000, do_drnl: bool = True,
                   auto_ratio: float = 1.0, order_seed: int = 0,
                   max_working_bytes: int = 512 << 20, max_pairwise_checks: int = 20_000_000) -> dict:
    if nodes.ndim != 1 or nodes.size < 2 or len(set(nodes.tolist())) != nodes.size:
        raise ValueError("S must be a unique one-dimensional array, beginning with src,dst")
    if virtual_n < graph.n:
        raise ValueError("virtual N cannot remove original graph nodes")
    if incoming is not None and incoming.n != graph.n:
        raise ValueError("incoming and outgoing CSR shapes differ")
    degree_sum = graph.degree_sum(nodes)
    incoming_degree_sum = incoming.degree_sum(nodes) if incoming is not None else 0
    if degree_sum + incoming_degree_sum > max_neighbors:
        raise WorkspaceLimit("selected out/in rows exceed --max-neighbors; no execution arm allocated")
    # Conservative local-scratch allowance also covers Python oracle/list objects.
    local_estimate = (degree_sum + incoming_degree_sum) * 256 + nodes.size * 256
    if local_estimate > max_working_bytes:
        raise WorkspaceLimit("selected rows exceed --max-working-set-mib before oracle/arm allocation")
    for u in nodes:
        graph.validate_row(int(u))
        if incoming is not None:
            incoming.validate_row(int(u))
    oracle = oracle_induced(graph, nodes)
    if incoming is None and (oracle != oracle.T).nnz:
        raise ValueError("undirected mode requires symmetric induced adjacency")
    if incoming is not None:
        # Check the relevant transpose without a quadratic Python membership loop.
        # Target arcs are removed in both oracles; all execution arms discard them.
        reverse = oracle_induced(incoming, nodes)
        if (oracle != reverse.T).nnz:
            raise ValueError("incoming CSR is not the transpose on the selected node set")
    expected = canonical_edges(oracle)
    expected_drnl = drnl_labels(oracle) if do_drnl else None
    dtype = np.dtype(stock_index_dtype)
    if dtype.itemsize == 4 and max(virtual_n, degree_sum) > np.iinfo(np.int32).max:
        raise ValueError("int32 stock indices cannot represent requested shape/rows")
    # Conservative declared bound: col_offsets plus row data/indices, local maps,
    # output CSR and temporary index copies. Not a measured memory statistic.
    stock_estimate = virtual_n * dtype.itemsize + degree_sum * (3 * dtype.itemsize + 3) + nodes.size * 64
    methods = {
        "stock_scipy": lambda: stock_scipy(graph, nodes, virtual_n, dtype),
        "corrected_local_csr": lambda: corrected_local(graph, nodes),
        "ours_scan": lambda: production_induced(graph, nodes, "scan", incoming, auto_ratio),
        "ours_pairwise": lambda: production_induced(graph, nodes, "pairwise", incoming, auto_ratio),
        "ours_auto": lambda: production_induced(graph, nodes, "auto", incoming, auto_ratio),
    }
    order = list(METHODS)
    np.random.default_rng(order_seed).shuffle(order)
    result = {"nodes": nodes.tolist(), "node_set_sha256": digest(nodes), "num_nodes": int(nodes.size),
              "virtual_graph_nodes": virtual_n, "sum_out_degrees": degree_sum,
              "sum_in_degrees": incoming_degree_sum,
              "induced_directed_entries": int(expected.shape[0]), "canonical_edges_sha256": digest(expected),
              "drnl_sha256": digest(expected_drnl) if expected_drnl is not None else None,
              "stock_workspace_upper_estimate_bytes": stock_estimate,
              "local_scratch_upper_estimate_bytes": local_estimate, "method_order": order,
              "auto_selected": "pairwise" if (2 if incoming is not None else 1) * nodes.size * (nodes.size - 1) // 2
              <= auto_ratio * max(1, degree_sum) else "scan", "methods": {}}
    for name in order:
        if name == "stock_scipy" and (stock_estimate > workspace_bytes or stock_estimate + local_estimate > max_working_bytes):
            result["methods"][name] = {"status": "skipped_workspace_guard", "estimated_bytes": stock_estimate,
                                        "limit_bytes": workspace_bytes, "total_limit_bytes": max_working_bytes}
            continue
        pair_checks = (2 if incoming is not None else 1) * nodes.size * (nodes.size - 1) // 2
        if pair_checks > max_pairwise_checks and (name == "ours_pairwise" or
                                                 (name == "ours_auto" and result["auto_selected"] == "pairwise")):
            result["methods"][name] = {"status": "skipped_pairwise_work_guard", "pair_checks": pair_checks,
                                        "limit_checks": max_pairwise_checks}
            continue
        result["methods"][name] = measure(methods[name], expected, expected_drnl, warmup, repeats)
    return result


def provenance(graph: CSRGraph, incoming: CSRGraph | None) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    paths = [Path(__file__), ROOT / "seal/graph_csr_kernels.py", ROOT / "seal/graph_csr_extraction.py"]
    source_metadata = []
    for item in (graph, incoming):
        if item is None:
            continue
        path = Path(item.origin)
        candidates = [path / name for name in ("edge_starts.bin", "edge_ends.bin", "meta.json")] if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file():
                st = candidate.stat()
                source_metadata.append({"path": str(candidate), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns})
    return {"python": sys.version, "platform": platform.platform(), "hostname": platform.node(),
            "logical_cpu_count": os.cpu_count(), "numpy": np.__version__,
            "scipy": scipy.__version__, "numba_available": kernels.NUMBA_AVAILABLE,
            "numba": importlib.metadata.version("numba") if kernels.NUMBA_AVAILABLE else None,
            "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "graph": {"origin": graph.origin, "nodes": graph.n, "stored_entries": graph.m,
                      "offset_dtype": str(graph.starts.dtype), "index_dtype": str(graph.indices.dtype),
                      "mmap_offsets": isinstance(graph.starts, np.memmap), "mmap_indices": isinstance(graph.indices, np.memmap)},
            "incoming_origin": incoming.origin if incoming is not None else None,
            "graph_source_metadata": source_metadata,
            "dataset_content_hash": "not computed; immutable source/split manifest must be supplied separately",
            "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def comma_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--graph-path", type=Path)
    source.add_argument("--fixture", action="store_true")
    parser.add_argument("--in-graph-path", type=Path)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--file-endian", choices=("little", "big"), default="little")
    parser.add_argument("--pairs-path", type=Path)
    parser.add_argument("--pair-sampling", choices=("edges", "mixed", "uniform", "twohop", "mixed_twohop"), default="mixed")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--caps", type=comma_ints, default=[50, 200], help="comma-separated; 0 means uncapped")
    parser.add_argument("--samplers", default="reference_union", help="reference_union,production_union,per_vertex")
    parser.add_argument("--oversample", type=float, default=1.5)
    parser.add_argument("--pad-isolates", type=comma_ints, default=[0], help="virtual added isolate counts; mmap sizes stay unchanged")
    parser.add_argument("--virtual-num-nodes", type=comma_ints, help="absolute virtual N grid, alternative to --pad-isolates")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2, help="at least one call per arm excludes Numba compilation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stock-index-dtype", choices=("int32", "int64"), default="int64")
    parser.add_argument("--max-stock-workspace-mib", type=float, default=128)
    parser.add_argument("--max-neighbors", type=int, default=2_000_000)
    parser.add_argument("--max-working-set-mib", type=float, default=512,
                        help="conservative per-arm local scratch plus stock scratch guard; excludes mmap page cache/runtime")
    parser.add_argument("--max-pairwise-checks", type=int, default=20_000_000)
    parser.add_argument("--max-subgraph-nodes", type=int, default=10000)
    parser.add_argument("--max-transpose-edges", type=int, default=1_000_000)
    parser.add_argument("--auto-ratio", type=float, default=1.0)
    parser.add_argument("--no-drnl", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    samplers = [x.strip() for x in args.samplers.split(",")]
    if (args.limit < 1 or args.repeats < 1 or args.warmup < 1 or args.hops < 0 or not args.caps
            or min(args.caps) < 0 or not args.pad_isolates or min(args.pad_isolates) < 0
            or args.max_stock_workspace_mib < 0 or args.max_working_set_mib <= 0 or args.max_pairwise_checks < 1
            or args.max_neighbors < 1 or args.max_subgraph_nodes < 2 or args.max_transpose_edges < 0
            or args.auto_ratio <= 0 or args.oversample <= 0
            or any(x not in ("reference_union", "production_union", "per_vertex") for x in samplers)):
        parser.error("invalid limits, repetitions, sampler, cap or padding configuration")
    graph = fixture_graph(args.directed) if args.fixture else load_graph(args.graph_path, args.file_endian)
    if args.virtual_num_nodes is not None:
        if args.pad_isolates != [0] or not args.virtual_num_nodes or min(args.virtual_num_nodes) < graph.n:
            parser.error("--virtual-num-nodes must be >= original N and cannot be combined with --pad-isolates")
        args.pad_isolates = [n - graph.n for n in args.virtual_num_nodes]
    incoming = None
    if args.directed:
        incoming = load_graph(args.in_graph_path, args.file_endian) if args.in_graph_path else transpose_small(graph, args.max_transpose_edges)
    elif args.in_graph_path:
        parser.error("--in-graph-path requires --directed")
    pairs = read_pairs(args.pairs_path, args.limit) if args.pairs_path else deterministic_pairs(graph, args.limit, args.seed, args.pair_sampling)
    if not pairs:
        raise ValueError("no input pairs")
    result = {"schema_version": 1, "provenance": provenance(graph, incoming),
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "contract": {"simple_binary_graph": True, "no_self_loops": True, "sorted_unique_rows": True,
                           "checks": "touched rows and selected induced adjacency, not a full-graph scan",
                           "target_removal": "both src->dst and dst->src, in directed and undirected modes",
                           "drnl": "endpoint-deleted shortest paths on undirected view",
                           "scope": "frozen S -> induced CSR; sampling/setup/oracle/validation excluded",
                           "stock": "selected-row gather + SciPy column fancy indexing; full-graph SciPy construction excluded",
                           "padding": "virtual isolated vertices; S, original arrays and local structure fixed",
                           "uncertainty": "timing quantiles, not confidence intervals; repeated samples share graph/cache"},
              "records": [], "summary": []}
    aggregates: dict[tuple, list[float]] = {}
    for sampler_index, sampler in enumerate(samplers):
        for cap in args.caps:
            for pair_index, pair in enumerate(pairs):
                sampling_seed = int(np.random.SeedSequence([args.seed, sampler_index, cap, pair_index]).generate_state(1)[0])
                t0 = time.perf_counter()
                nodes = sample_node_set(graph, pair, args.hops, cap or None, np.random.default_rng(sampling_seed), sampler,
                                        incoming, args.oversample, args.max_neighbors, args.max_subgraph_nodes)
                sample_seconds = time.perf_counter() - t0
                for padding in args.pad_isolates:
                    record = benchmark_pair(graph, nodes, graph.n + padding, incoming=incoming, warmup=args.warmup,
                                            repeats=args.repeats, stock_index_dtype=args.stock_index_dtype,
                                            workspace_bytes=int(args.max_stock_workspace_mib * 2**20), max_neighbors=args.max_neighbors,
                                            do_drnl=not args.no_drnl, auto_ratio=args.auto_ratio,
                                            order_seed=sampling_seed + padding,
                                            max_working_bytes=int(args.max_working_set_mib * 2**20),
                                            max_pairwise_checks=args.max_pairwise_checks)
                    workload = ("positive" if graph.has_edge(*pair) else
                                "twohop_nonedge" if args.pairs_path is None and args.pair_sampling in ("twohop", "mixed_twohop")
                                else "provided_nonedge" if args.pairs_path is not None else "uniform_nonedge")
                    record.update(pair=list(pair), pair_index=pair_index, workload=workload, sampler=sampler, cap=cap,
                                  sampling_seed=sampling_seed, sampling_seconds=sample_seconds, added_isolates=padding)
                    result["records"].append(record)
                    for name, arm in record["methods"].items():
                        if arm["status"] == "ok":
                            aggregates.setdefault((sampler, cap, padding, name, workload), []).extend(arm["edge_seconds"])
    for (sampler, cap, padding, method, workload), values in aggregates.items():
        result["summary"].append({"sampler": sampler, "cap": cap, "added_isolates": padding, "method": method, "workload": workload,
                                  "edge_summary": stats(values)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "records": len(result["records"]),
                      "all_executed_arms_exact": True, "torch_imported": "torch" in sys.modules}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
