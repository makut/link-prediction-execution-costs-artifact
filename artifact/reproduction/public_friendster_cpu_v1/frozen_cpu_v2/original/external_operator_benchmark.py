#!/usr/bin/env python3
"""Replay frozen S against unmodified torch-sparse 0.6.18 CPU induction.

No downloads, GPU operations, model fitting, full-graph COO construction by
default, or repository-code imports. Default is preflight only. Execution and
streaming int32 -> int64 disk conversion require separate explicit flags.

Input is an exact_benchmark.py JSON archive (records contain S and oracle
edge/DRNL hashes), plus its original GraphCSR directory. Only one archive width
is needed: the int32/int64 archives often repeat the same S, not independent
workloads. Conversion/setup and per-query induction have separate timers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

PIN = {
    "distribution": "torch-sparse", "version": "0.6.18", "license": "MIT",
    "repository": "https://github.com/rusty1s/pytorch_sparse",
    "commit": "7d22892fb47626d2c5b0171769d44bdc8edd606c",
    "cpu_source": "csrc/cpu/saint_cpu.cpp",
    "cpu_source_git_blob": "21a13be8067e66ef4d6f9ae57f98407332f31929",
    "wrapper_source_git_blob": "440455853e0309b00376cbe9ad25d7226cfa5e0e",
    "device": "CPU only; this operator has no CUDA implementation",
}
GIB = 2**30


class GuardError(ValueError):
    pass


def sha_file(path: Path, chunk: int = 8 * 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<i8").tobytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def identity(path: Path) -> dict:
    s = path.stat()
    return {"path": str(path.resolve()), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def inspect_graph(path: Path, endian: str = "little") -> dict:
    starts, indices = path / "edge_starts.bin", path / "edge_ends.bin"
    if starts.stat().st_size == 0 or starts.stat().st_size % 8:
        raise ValueError("GraphCSR starts must be N>0 raw int64 offsets")
    with indices.open("rb") as f:
        h = f.read(8)
    if len(h) != 8:
        raise ValueError("GraphCSR neighbor-count header missing")
    m = int.from_bytes(h, endian, signed=True)
    if m < 0 or indices.stat().st_size != 8 + 4 * m:
        raise ValueError("GraphCSR neighbor-count/file-size mismatch")
    n = starts.stat().st_size // 8
    return {"n": n, "m": m, "endian": endian,
            "source_files": [identity(starts), identity(indices)],
            "source_storage_bytes": 8 * n + 4 * m + 8,
            "native_int64_storage_bytes": 8 * (n + 1 + m)}


def load_records(path: Path, graph: dict, *, limit: int = 0,
                 max_json_bytes: int = 64 * 2**20) -> tuple[list[dict], dict]:
    if path.stat().st_size > max_json_bytes:
        raise GuardError("record archive exceeds JSON size budget")
    raw = json.loads(path.read_text())
    p = raw.get("provenance", {}).get("graph", {})
    if (p.get("nodes"), p.get("stored_entries")) != (graph["n"], graph["m"]):
        raise ValueError("archive graph dimensions disagree with original GraphCSR")
    records = raw.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("archive has no frozen-S records")
    if limit:
        records = records[:limit]
    for r in records:
        validate_record(r, graph["n"])
    return records, {"path": str(path.resolve()), "sha256": sha_file(path),
                     "selected_records": len(records), "limit": limit,
                     "original_graph": p, "original_configuration": raw.get("configuration")}


def validate_record(record: dict, n: int) -> np.ndarray:
    raw = record["nodes"]
    if not isinstance(raw, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in raw):
        raise ValueError("S must contain integer vertex IDs")
    nodes = np.asarray(raw, dtype=np.int64)
    if nodes.size < 2 or nodes.size > 10000 or len(set(raw)) != nodes.size:
        raise ValueError("S must have 2..10000 distinct vertices")
    if np.any(nodes < 0) or np.any(nodes >= n):
        raise ValueError("S vertex outside original graph")
    if record.get("virtual_graph_nodes", n) != n or record.get("added_isolates", 0):
        raise ValueError("this native full-graph control requires unpadded N")
    if record.get("pair", raw[:2]) != raw[:2]:
        raise ValueError("S[:2] must be the ordered query endpoints")
    if record.get("node_set_sha256") != digest(nodes):
        raise ValueError("frozen-S hash mismatch")
    for key in ("canonical_edges_sha256", "drnl_sha256"):
        val = record.get(key)
        if not isinstance(val, str) or len(val) != 64:
            raise ValueError(f"required archive oracle hash missing: {key}")
    return nodes


def estimate_memory(graph: dict, public_api: bool, chunk_entries: int) -> dict:
    n, m = graph["n"], graph["m"]
    # Conservative accounting includes both source and native file working sets.
    # Page-cache residency is reported separately; it is not guaranteed resident.
    return {"source_storage_bytes": graph["source_storage_bytes"],
            "native_int64_storage_bytes": graph["native_int64_storage_bytes"],
            "per_query_assoc_bytes": 8 * n,
            "public_api_global_row_bytes": 8 * m if public_api else 0,
            "conversion_chunk_upper_bytes": 24 * chunk_entries,
            "conservative_peak_bytes": graph["source_storage_bytes"] +
                graph["native_int64_storage_bytes"] + 8 * n +
                (8 * m if public_api else 0) + 24 * chunk_entries + GIB}


def enforce_budgets(graph: dict, estimate: dict, *, max_native_gib: float,
                    max_peak_gib: float, public_api: bool,
                    max_public_coo_gib: float) -> None:
    if (not all(math.isfinite(x) for x in (max_native_gib, max_peak_gib, max_public_coo_gib))
            or min(max_native_gib, max_peak_gib) <= 0 or max_public_coo_gib < 0):
        raise GuardError("memory budgets must be finite and nonnegative (native/peak strictly positive)")
    if graph["native_int64_storage_bytes"] > max_native_gib * GIB:
        raise GuardError("native int64 full-graph storage exceeds declared --max-native-gib")
    if estimate["conservative_peak_bytes"] > max_peak_gib * GIB:
        raise GuardError("conservative source/native/workspace estimate exceeds --max-peak-gib")
    if public_api and estimate["public_api_global_row_bytes"] > max_public_coo_gib * GIB:
        raise GuardError("public API full-E COO row exceeds explicit --max-public-coo-gib")


def capacity_snapshot() -> dict:
    out = {}
    info = Path("/proc/meminfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                out["host_available_bytes"] = int(line.split()[1]) * 1024
    for root, maximum, current in ((Path("/sys/fs/cgroup"), "memory.max", "memory.current"),
                                    (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes")):
        if (root / maximum).exists() and (root / current).exists():
            value = (root / maximum).read_text().strip()
            if value != "max" and int(value) < 2**60:
                out["cgroup_limit_bytes"] = int(value)
                out["cgroup_current_bytes"] = int((root / current).read_text())
                out["cgroup_available_bytes"] = max(0, out["cgroup_limit_bytes"] - out["cgroup_current_bytes"])
                break
    candidates = [v for k, v in out.items() if k in ("host_available_bytes", "cgroup_available_bytes")]
    out["effective_available_bytes"] = min(candidates) if candidates else None
    return out


def enforce_capacity(estimate: dict, capacity: dict) -> None:
    available = capacity["effective_available_bytes"]
    if available is not None and estimate["conservative_peak_bytes"] > 0.85 * available:
        raise GuardError("conservative working-set estimate exceeds 85% of current host/cgroup available memory")


def prepare_native(source: Path, target: Path, graph: dict, chunk_entries: int) -> dict:
    if target.exists():
        raise FileExistsError("native directory already exists; never overwrite a conversion")
    target.parent.mkdir(parents=True, exist_ok=True)
    required = graph["native_int64_storage_bytes"] + 2 * GIB
    if shutil.disk_usage(target.parent).free < required:
        raise GuardError("insufficient disk for native int64 files plus 2 GiB reserve")
    partial = target.with_name(target.name + ".partial")
    partial.mkdir()  # Interrupted output requires explicit inspection; never reuse it.
    prefix = "<" if graph["endian"] == "little" else ">"
    n, m = graph["n"], graph["m"]
    src_ptr = np.memmap(source / "edge_starts.bin", dtype=prefix + "i8", mode="r")
    src_col = (np.memmap(source / "edge_ends.bin", dtype=prefix + "i4", mode="r", offset=8)
               if m else np.empty(0, dtype=prefix + "i4"))
    began = time.perf_counter()
    hptr, hcol = hashlib.sha256(), hashlib.sha256()
    out_ptr = np.memmap(partial / "rowptr.i8", dtype="<i8", mode="w+", shape=(n + 1,))
    previous = 0
    for a in range(0, n, chunk_entries):
        b = min(n, a + chunk_entries)
        block = np.asarray(src_ptr[a:b], dtype="<i8")
        if (a == 0 and int(block[0]) != 0) or int(block[0]) < previous or np.any(block[1:] < block[:-1]) or int(block[-1]) > m:
            raise ValueError("invalid original GraphCSR offsets")
        previous = int(block[-1])
        out_ptr[a:b] = block
        hptr.update(block.tobytes())
    out_ptr[n] = m
    hptr.update(np.asarray([m], dtype="<i8").tobytes())
    out_ptr.flush()
    if m:
        out_col = np.memmap(partial / "col.i8", dtype="<i8", mode="w+", shape=(m,))
        for a in range(0, m, chunk_entries):
            b = min(m, a + chunk_entries)
            block = np.asarray(src_col[a:b], dtype="<i8")
            if np.any(block < 0) or np.any(block >= n):
                raise ValueError("out-of-range original GraphCSR neighbor")
            out_col[a:b] = block
            hcol.update(block.tobytes())
        out_col.flush()
        del out_col
    else:
        (partial / "col.i8").touch()
    del out_ptr, src_ptr, src_col
    if inspect_graph(source, graph["endian"]) != graph:
        raise ValueError("original GraphCSR changed during conversion")
    manifest = {"schema_version": 1, "graph": graph,
                "conversion_seconds": time.perf_counter() - began,
                "rowptr_sha256": hptr.hexdigest(), "col_sha256": hcol.hexdigest(),
                "native_dtype": "little-endian int64", "sentinel_added": True}
    for name in ("rowptr.i8", "col.i8"):
        with (partial / name).open("rb") as f:
            os.fsync(f.fileno())
    write_json(partial / "manifest.json", manifest)
    os.replace(partial, target)
    return manifest


def open_native(path: Path, graph: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = json.loads((path / "manifest.json").read_text())
    if meta["graph"] != graph:
        raise ValueError("native conversion does not match current original GraphCSR identity")
    n, m = graph["n"], graph["m"]
    if (path / "rowptr.i8").stat().st_size != 8 * (n + 1) or (path / "col.i8").stat().st_size != 8 * m:
        raise ValueError("native conversion file size mismatch")
    began = time.perf_counter()
    verified_files = []
    for name, field in (("rowptr.i8", "rowptr_sha256"), ("col.i8", "col_sha256")):
        before = identity(path / name)
        actual_hash = sha_file(path / name)
        if actual_hash != meta[field]:
            raise ValueError(f"native conversion content hash mismatch: {name}")
        if identity(path / name) != before:
            raise ValueError(f"native conversion changed during verification: {name}")
        verified_files.append({**before, "sha256": actual_hash})
    meta = {**meta, "runtime_content_verification": {
        "status": "pass", "method": "streaming SHA256 of both native payloads outside timers",
        "seconds": time.perf_counter() - began, "files": verified_files}}
    # Copy-on-write mappings permit Torch's NumPy bridge without modifying files.
    ptr = np.memmap(path / "rowptr.i8", dtype="<i8", mode="c", shape=(n + 1,))
    col = np.memmap(path / "col.i8", dtype="<i8", mode="c", shape=(m,)) if m else np.empty(0, dtype=np.int64)
    if not ptr.dtype.isnative or not col.dtype.isnative:
        raise ValueError("Torch operator requires native-endian int64")
    if int(ptr[0]) != 0 or int(ptr[-1]) != m:
        raise ValueError("invalid native rowptr sentinel")
    return ptr, col, meta


def canonical_edges(a: sparse.csr_matrix) -> np.ndarray:
    rows, cols = a.nonzero()
    order = np.lexsort((cols, rows))
    return np.stack([rows[order], cols[order]], axis=1).astype(np.int64)


def finish_local(rows: np.ndarray, cols: np.ndarray, size: int) -> sparse.csr_matrix:
    # Includes target removal and local SciPy packaging inside the edge timer.
    keep = ~(((rows == 0) & (cols == 1)) | ((rows == 1) & (cols == 0)))
    return sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.int8),
                              (rows[keep], cols[keep])), shape=(size, size))


def drnl_labels(a: sparse.csr_matrix) -> np.ndarray:
    # Same mathematical contract as archived SEAL-OGB audit, independently local.
    n = a.shape[0]
    keep_dst, keep_src = np.r_[0, np.arange(2, n)], np.arange(1, n)
    ds = shortest_path(a[keep_dst, :][:, keep_dst], directed=False, unweighted=True, indices=0)
    dt = shortest_path(a[keep_src, :][:, keep_src], directed=False, unweighted=True, indices=0)
    ds, dt = np.insert(ds, 1, 0), np.insert(dt, 0, 0)
    z = np.zeros(n, dtype=np.int64)
    valid = np.isfinite(ds) & np.isfinite(dt)
    distance = ds[valid] + dt[valid]
    half, mod = np.floor_divide(distance, 2), np.mod(distance, 2)
    z[valid] = (1 + np.minimum(ds[valid], dt[valid]) + half * (half + mod - 1)).astype(np.int64)
    z[:2] = 1
    return z


def memory_snapshot() -> dict:
    r = resource.getrusage(resource.RUSAGE_SELF)
    out = {"ru_maxrss_bytes": int(r.ru_maxrss * (1 if sys.platform == "darwin" else 1024))}
    p = Path("/proc/self/smaps_rollup")
    if p.exists():
        for line in p.read_text().splitlines():
            key = line.split(":")[0]
            if key in ("Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty"):
                out[key.lower() + "_bytes"] = int(line.split()[1]) * 1024
    return out


def stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "p50_s": float(np.percentile(a, 50)),
            "p95_s": float(np.percentile(a, 95)), "min_s": float(a.min()),
            "max_s": float(a.max()), "mean_s": float(a.mean())} if values else {"n": 0}


def validate_selected_rows(ptr: np.ndarray, col: np.ndarray, nodes: np.ndarray) -> None:
    n, m = len(ptr) - 1, len(col)
    for u in nodes:
        u = int(u)
        a, b = int(ptr[u]), int(ptr[u + 1])
        if not 0 <= a <= b <= m:
            raise ValueError("invalid touched native CSR row bounds")
        row = col[a:b]
        if len(row) and (int(row[0]) < 0 or int(row[-1]) >= n):
            raise ValueError("out-of-range touched native CSR neighbor")
        for start in range(1, len(row), 65536):
            stop = min(len(row), start + 65536)
            if np.any(row[start:stop] <= row[start - 1:stop - 1]):
                raise ValueError("touched native rows must be sorted and unique")
        at = np.searchsorted(row, u)
        if at < len(row) and int(row[at]) == u:
            raise ValueError("touched native rows must not contain self loops")


def load_operator(threads: int) -> tuple[object, object, dict]:
    version = importlib.metadata.version("torch-sparse")
    if version.split("+")[0] != PIN["version"]:
        raise ValueError(f"required torch-sparse {PIN['version']}, found {version}")
    import torch
    import torch_sparse
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    package = Path(torch_sparse.__file__).parent
    binaries = [{"path": str(p), "sha256": sha_file(p)} for p in sorted(package.glob("_saint*")) if p.is_file()]
    if not binaries:
        raise ValueError("loaded torch-sparse has no recorded saint extension binary")
    wrapper = package / "saint.py"
    body = wrapper.read_bytes()
    wrapper_blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    if wrapper_blob != PIN["wrapper_source_git_blob"]:
        raise ValueError("installed public saint wrapper differs from pinned source")
    return torch, torch_sparse, {"torch": torch.__version__, "torch_sparse": version,
                                  "extension_binaries": binaries,
                                  "installed_wrapper_sha256": sha_file(wrapper),
                                  "installed_wrapper_git_blob": wrapper_blob,
                                  "torch_num_threads": torch.get_num_threads(),
                                  "source_pin": PIN,
                                  "binary_source_build_commit": "not inferred from version; binary SHA recorded"}


def run_record(record: dict, torch, ptr, col, *, mode: str, adj=None,
               warmup: int = 1, repeats: int = 3) -> dict:
    nodes = validate_record(record, ptr.numel() - 1)
    idx = torch.from_numpy(nodes)
    empty = torch.empty(0, dtype=torch.int64)
    degrees = ptr[idx + 1] - ptr[idx]
    if int(degrees.sum()) != record["sum_out_degrees"]:
        raise ValueError("archive selected-row degree sum does not match native graph")
    times, raw_times, labels_times = [], [], []
    for i in range(warmup + repeats):
        start = time.perf_counter_ns()
        if mode == "direct":
            rows, cols, edge_ids = torch.ops.torch_sparse.saint_subgraph(idx, ptr, empty, col)
        else:
            out, edge_ids = adj.saint_subgraph(idx)
            rows, cols, _ = out.coo()
        kernel_end = time.perf_counter_ns()
        actual = finish_local(rows.numpy(), cols.numpy(), len(nodes))
        edge_end = time.perf_counter_ns()
        labels = drnl_labels(actual)
        end = time.perf_counter_ns()
        if digest(canonical_edges(actual)) != record["canonical_edges_sha256"]:
            raise AssertionError("external induced edges differ from archived exact-S oracle")
        if digest(labels) != record["drnl_sha256"]:
            raise AssertionError("external DRNL differs from archived exact-S oracle")
        if int(actual.nnz) != record["induced_directed_entries"] or actual[0, 1] or actual[1, 0]:
            raise AssertionError("external adjacency violates target-removal/entry-count contract")
        if i >= warmup:
            raw_times.append((kernel_end - start) / 1e9)
            times.append((edge_end - start) / 1e9)
            labels_times.append((end - edge_end) / 1e9)
        del actual, rows, cols, edge_ids
    return {"status": "ok", "method": "torch_sparse_0.6.18_" + mode + "_cpu",
            "raw_operator_seconds": raw_times, "edge_seconds": times,
            "drnl_seconds": labels_times,
            "edge_plus_drnl_seconds": [a + b for a, b in zip(times, labels_times)],
            "edge_summary": stats(times), "raw_operator_summary": stats(raw_times),
            "exact_edges": True, "exact_drnl": True, "memory_after": memory_snapshot()}


def fixture_data() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    # Non-sorted S, asymmetric arcs, both target arcs, bridge, isolates and cold endpoints.
    edges = [(0, 1), (1, 0), (0, 2), (2, 3), (3, 1), (4, 2), (3, 4), (6, 7)]
    a = sparse.csr_matrix((np.ones(len(edges), dtype=np.int8), tuple(zip(*edges))), shape=(9, 9))
    a.sort_indices()
    records = []
    for values in ([0, 1, 4, 2, 3, 8], [8, 7, 6, 0], [0, 1], [6, 7, 8]):
        nodes = np.asarray(values, dtype=np.int64)
        local = a[nodes, :][:, nodes].tocoo()
        local = finish_local(local.row, local.col, len(nodes))
        records.append({"nodes": values, "pair": values[:2], "virtual_graph_nodes": 9,
                        "node_set_sha256": digest(nodes),
                        "canonical_edges_sha256": digest(canonical_edges(local)),
                        "drnl_sha256": digest(drnl_labels(local)),
                        "induced_directed_entries": int(local.nnz),
                        "sum_out_degrees": int(np.diff(a.indptr)[nodes].sum())})
    return a.indptr.astype(np.int64), a.indices.astype(np.int64), records


def native_admission_smoke(torch, torch_sparse, public_api: bool) -> dict:
    ptr_np, col_np, records = fixture_data()
    ptr, col = torch.from_numpy(ptr_np), torch.from_numpy(col_np)
    adj = None
    if public_api:
        adj = torch_sparse.SparseTensor(rowptr=ptr, col=col, sparse_sizes=(9, 9), is_sorted=True, trust_data=True)
        adj.coo()
    modes = ["direct", "public_api"] if public_api else ["direct"]
    for record in records:
        for mode in modes:
            run_record(record, torch, ptr, col, mode=mode, adj=adj, warmup=0, repeats=1)
    return {"status": "pass", "cases_per_arm": len(records), "arms": modes,
            "scope": "Actual installed extension, before full-graph conversion/allocation"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-path", type=Path)
    p.add_argument("--records", type=Path)
    p.add_argument("--native-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true", help="Without this flag, preflight only")
    p.add_argument("--prepare-native", action="store_true", help="Explicit streaming disk conversion")
    p.add_argument("--fixture", action="store_true", help="Tiny actual-extension smoke; no full graph")
    p.add_argument("--public-api", action="store_true", help="Optional guarded public SparseTensor arm")
    p.add_argument("--max-native-gib", type=float, default=8)
    p.add_argument("--max-peak-gib", type=float, default=16)
    p.add_argument("--max-public-coo-gib", type=float, default=0)
    p.add_argument("--max-query-neighbors", type=int, default=20_000_000)
    p.add_argument("--limit", type=int, default=0, help="Predeclare prefix, 0=all records")
    p.add_argument("--chunk-entries", type=int, default=1_000_000)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--file-endian", choices=["little", "big"], default="little")
    args = p.parse_args(argv)
    if (not all(math.isfinite(x) for x in (args.max_native_gib, args.max_peak_gib, args.max_public_coo_gib))
            or min(args.max_native_gib, args.max_peak_gib, args.threads, args.repeats, args.chunk_entries, args.max_query_neighbors) <= 0
            or min(args.warmup, args.limit, args.max_public_coo_gib) < 0):
        p.error("invalid negative or zero budget/repeat argument")
    result = {"schema_version": 1, "status": "preflight", "source_pin": PIN,
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "provenance": {"python": sys.version, "platform": platform.platform(),
                             "script_sha256": sha_file(Path(__file__)), "numpy": np.__version__,
                             "scipy": scipy.__version__, "memory_initial": memory_snapshot()},
              "contract": {"scope": "Frozen original S -> exact local CSR; DRNL separate. No model/quality claims.",
                           "target_removal": "Both directions between S[0] and S[1]",
                           "drnl": "Endpoint-deleted shortest paths on undirected local view",
                           "setup": "Int64 graph conversion/storage and optional full-E COO row are separately reported",
                           "timing": "edge_seconds includes operator, output transfer-free NumPy view, target removal and local CSR construction; raw_operator excludes local packaging",
                           "uncertainty": "Repeated-cache timing quantiles, not independent-seed confidence intervals"},
              "records": []}
    try:
        if args.fixture:
            ptr_np, col_np, records = fixture_data()
            graph = {"n": len(ptr_np) - 1, "m": len(col_np)}
            result["fixture"] = True
        else:
            if args.graph_path is None or args.records is None or args.native_dir is None:
                p.error("--graph-path, --records and --native-dir required outside fixture")
            graph = inspect_graph(args.graph_path, args.file_endian)
            records, archive = load_records(args.records, graph, limit=args.limit)
            result.update(graph=graph, archive=archive)
            estimate = estimate_memory(graph, args.public_api, args.chunk_entries)
            result["memory_estimate"] = estimate
            enforce_budgets(graph, estimate, max_native_gib=args.max_native_gib,
                            max_peak_gib=args.max_peak_gib, public_api=args.public_api,
                            max_public_coo_gib=args.max_public_coo_gib)
            if any(r["sum_out_degrees"] > args.max_query_neighbors for r in records):
                raise GuardError("predeclared record set exceeds selected-row query budget")
        if not args.execute:
            result["preflight_passed"] = True
            write_json(args.output, result)
            print(json.dumps({"status": "preflight", "records": len(records), "output": str(args.output)}))
            return 0
        torch, torch_sparse, software = load_operator(args.threads)
        result["provenance"].update(software)
        result["native_admission_fixture"] = native_admission_smoke(torch, torch_sparse, args.public_api)
        if not args.fixture:
            result["capacity_before_execution"] = capacity_snapshot()
            enforce_capacity(estimate, result["capacity_before_execution"])
            if args.prepare_native:
                result["conversion"] = prepare_native(args.graph_path, args.native_dir, graph, args.chunk_entries)
            ptr_np, col_np, conversion = open_native(args.native_dir, graph)
            result["conversion"] = conversion
        ptr, col = torch.from_numpy(ptr_np), torch.from_numpy(col_np)
        adj = None
        if args.public_api:
            # Explicit opt-in + full-E byte guard above. Never entered by default.
            start = time.perf_counter()
            adj = torch_sparse.SparseTensor(rowptr=ptr, col=col, sparse_sizes=(graph["n"], graph["n"]), is_sorted=True, trust_data=True)
            adj.coo()  # Charge global row materialization to setup, not first query.
            result["public_api_setup_seconds"] = time.perf_counter() - start
            result["public_api_setup_memory"] = memory_snapshot()
        result["status"] = "running"
        write_json(args.output, result)
        for number, record in enumerate(records):
            identity_fields = {k: record[k] for k in ("nodes", "pair", "node_set_sha256", "canonical_edges_sha256", "drnl_sha256", "sampler", "cap", "pair_index", "workload", "sum_out_degrees") if k in record}
            arms = {}
            validate_selected_rows(ptr_np, col_np, np.asarray(record["nodes"], dtype=np.int64))
            mode_order = (["direct", "public_api"] if number % 2 == 0 else ["public_api", "direct"]) if args.public_api else ["direct"]
            for mode in mode_order:
                arms[mode] = run_record(record, torch, ptr, col, mode=mode, adj=adj,
                                        warmup=args.warmup, repeats=args.repeats)
            result["records"].append({"archive_record_index": number, **identity_fields, "method_order": mode_order, "methods": arms})
            # Durable after each bounded record; no claims based on incomplete output.
            write_json(args.output, result)
        result["status"] = "complete"
        result["all_executed_arms_exact"] = True
        result["provenance"]["memory_final"] = memory_snapshot()
        result["summary"] = {mode: stats([x for r in result["records"] for x in r["methods"][mode]["edge_seconds"]]) for mode in result["records"][0]["methods"]}
        write_json(args.output, result)
        print(json.dumps({"status": result["status"], "records": len(records), "output": str(args.output)}))
        return 0
    except Exception as e:
        result.update(status="guarded_skip" if isinstance(e, GuardError) else "failed",
                      error={"type": type(e).__name__, "message": str(e)})
        write_json(args.output, result)
        print(json.dumps({"status": result["status"], "error": str(e)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
