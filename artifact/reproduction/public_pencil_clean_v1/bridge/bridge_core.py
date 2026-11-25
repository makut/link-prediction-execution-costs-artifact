"""Bounded CSR primitives and immutable PENCIL trace I/O; no GPU imports."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

UPSTREAM_COMMIT = "2d32e29dbed533288d9d758138e07547a0a7d8a9"
DEFAULT_ARMS = ("torch_sparse_public", "torch_sparse_direct", "compiled_scan_int64",
                "compiled_pairwise_int64", "compiled_scan_int32", "compiled_pairwise_int32")


class GuardError(ValueError):
    pass


def sha_file(path: Path, chunk: int = 8 * 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def identity(path: Path) -> dict:
    s = path.stat()
    return {"path": str(path.resolve()), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def capacity() -> dict:
    out = {}
    mem = Path("/proc/meminfo")
    if mem.exists():
        for line in mem.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                out["host_available_bytes"] = int(line.split()[1]) * 1024
    for root, maximum, current in ((Path("/sys/fs/cgroup"), "memory.max", "memory.current"),
                                  (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes")):
        if (root / maximum).exists() and (root / current).exists():
            value = (root / maximum).read_text().strip()
            if value != "max" and int(value) < 2**60:
                out["cgroup_limit_bytes"] = int(value)
                out["cgroup_current_bytes"] = int((root / current).read_text())
                out["cgroup_available_bytes"] = max(0, int(value) - out["cgroup_current_bytes"])
                break
    vals = [v for k, v in out.items() if k.endswith("available_bytes")]
    out["effective_available_bytes"] = min(vals) if vals else None
    return out


def memory() -> dict:
    out = {"maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                              (1 if sys.platform == "darwin" else 1024))}
    f = Path("/proc/self/smaps_rollup")
    if f.exists():
        for line in f.read_text().splitlines():
            k = line.split(":")[0]
            if k in ("Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty"):
                out[k.lower() + "_bytes"] = int(line.split()[1]) * 1024
    return out


def finite_budget(value: float, name: str, *, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise GuardError(name + " must be a finite positive budget")


def enforce_memory(estimated: int, max_gib: float, *, check_available: bool) -> dict:
    finite_budget(max_gib, "max-memory-gib")
    if estimated > max_gib * 2**30:
        raise GuardError(f"conservative {estimated} byte estimate exceeds declared memory budget")
    snapshot = capacity()
    available = snapshot["effective_available_bytes"]
    if check_available and available is not None and estimated > .85 * available:
        raise GuardError("conservative estimate exceeds 85% current host/cgroup available memory")
    return snapshot


@dataclass
class Graph:
    rowptr: np.ndarray  # includes final sentinel; int64
    col: np.ndarray     # original int32 compact columns
    provenance: dict

    @property
    def n(self): return len(self.rowptr) - 1
    @property
    def m(self): return len(self.col)

    def row(self, u: int) -> np.ndarray:
        if not 0 <= u < self.n: raise ValueError("vertex outside graph")
        a, z = int(self.rowptr[u]), int(self.rowptr[u + 1])
        if not 0 <= a <= z <= self.m: raise ValueError("invalid selected CSR row bounds")
        return self.col[a:z]

    def has_edge(self, u: int, v: int) -> bool:
        row = self.row(u); at = int(np.searchsorted(row, v))
        return at < len(row) and int(row[at]) == v

    def validate_rows(self, nodes: np.ndarray, max_neighbors: int = 20_000_000) -> int:
        count = 0
        for u in nodes:
            row = self.row(int(u)); count += len(row)
            if count > max_neighbors: raise GuardError("selected rows exceed neighbor budget")
            if len(row) and (int(row[0]) < 0 or int(row[-1]) >= self.n):
                raise ValueError("out-of-range selected neighbor")
            for a in range(1, len(row), 65536):
                z = min(len(row), a + 65536)
                if np.any(row[a:z] <= row[a - 1:z - 1]):
                    raise ValueError("selected rows must be sorted and unique")
            if self.has_edge(int(u), int(u)): raise ValueError("self loop in source graph")
        return count


def inspect_graph(path: Path, endian: str = "little") -> dict:
    starts, cols = path / "edge_starts.bin", path / "edge_ends.bin"
    if starts.stat().st_size == 0 or starts.stat().st_size % 8:
        raise ValueError("GraphCSR starts must contain N>0 raw int64 values")
    with cols.open("rb") as f: header = f.read(8)
    if len(header) != 8: raise ValueError("neighbor count header missing")
    m = int.from_bytes(header, endian, signed=True)
    if m < 0 or cols.stat().st_size != 8 + 4 * m:
        raise ValueError("GraphCSR neighbor count/file size mismatch")
    n = starts.stat().st_size // 8
    if n > np.iinfo(np.int32).max: raise GuardError("compact trace requires int32 global IDs")
    return {"n": n, "m": m, "endian": endian,
            "source_files": [identity(starts), identity(cols)],
            "compact_bytes": 8 * (n + 1) + 4 * m,
            "native_csr_bytes": 8 * (n + 1 + m),
            # Public native construction includes values, transpose/caches and COO.
            "conservative_native_setup_bytes": 8 * (4 * (n + 1) + 8 * m),
            "scope": "strict-clean simple symmetric C2 train graph required; no timestamps mapped"}


def open_graph(path: Path, endian: str = "little") -> Graph:
    meta = inspect_graph(path, endian)
    prefix = "<" if endian == "little" else ">"
    starts = np.memmap(path / "edge_starts.bin", dtype=prefix + "i8", mode="r")
    ptr = np.empty(meta["n"] + 1, dtype=np.int64)
    for a in range(0, len(starts), 1_000_000):
        ptr[a:min(len(starts), a + 1_000_000)] = starts[a:a + 1_000_000]
    ptr[-1] = meta["m"]
    if int(ptr[0]) != 0 or np.any(ptr[1:] < ptr[:-1]) or int(ptr[-2]) > meta["m"]:
        raise ValueError("invalid GraphCSR offsets")
    col = (np.memmap(path / "edge_ends.bin", dtype=prefix + "i4", offset=8,
                     shape=(meta["m"],), mode="c") if meta["m"] else np.empty(0, dtype=np.int32))
    if not col.dtype.isnative:
        raise ValueError("convert source to native endianness before benchmark")
    meta["rowptr_sha256"] = array_hash(ptr)
    return Graph(ptr, col, meta)


def validate_nodes(nodes: np.ndarray, n: int, *, max_nodes: int = 82) -> np.ndarray:
    if nodes.ndim != 1 or nodes.dtype != np.dtype("int64") or not nodes.flags.c_contiguous:
        raise ValueError("S must be a contiguous int64 vector")
    if not 1 <= len(nodes) <= max_nodes or int(nodes.min()) < 0 or int(nodes.max()) >= n:
        raise ValueError("S size or vertex range invalid")
    if len(np.unique(nodes)) != len(nodes): raise ValueError("S must contain unique vertices")
    return nodes


def oracle(ptr: np.ndarray, col: np.ndarray, nodes: np.ndarray):
    mapping = {int(v): i for i, v in enumerate(nodes)}
    rows, columns, ids = [], [], []
    for i, u in enumerate(nodes):
        for at in range(int(ptr[u]), int(ptr[u + 1])):
            local = mapping.get(int(col[at]))
            if local is not None:
                rows.append(i); columns.append(local); ids.append(at)
    return tuple(np.asarray(a, dtype=np.int64) for a in (rows, columns, ids))


@njit(cache=True)
def _find(a, lo, hi, value):
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < value: lo = mid + 1
        else: hi = mid
    return lo


@njit(cache=True)
def compiled_scan(ptr, col, nodes, sorted_nodes, sorted_local):
    count = 0
    for i in range(nodes.size):
        u = nodes[i]
        for at in range(ptr[u], ptr[u + 1]):
            j = _find(sorted_nodes, 0, sorted_nodes.size, col[at])
            if j < sorted_nodes.size and sorted_nodes[j] == col[at]: count += 1
    rows, columns, ids = np.empty(count, np.int64), np.empty(count, np.int64), np.empty(count, np.int64)
    k = 0
    for i in range(nodes.size):
        u = nodes[i]
        for at in range(ptr[u], ptr[u + 1]):
            j = _find(sorted_nodes, 0, sorted_nodes.size, col[at])
            if j < sorted_nodes.size and sorted_nodes[j] == col[at]:
                rows[k], columns[k], ids[k] = i, sorted_local[j], at; k += 1
    return rows, columns, ids


@njit(cache=True)
def compiled_pairwise(ptr, col, nodes, sorted_nodes, sorted_local):
    count = 0
    for i in range(nodes.size):
        u = nodes[i]
        for j in range(sorted_nodes.size):
            at = _find(col, ptr[u], ptr[u + 1], sorted_nodes[j])
            if at < ptr[u + 1] and col[at] == sorted_nodes[j]: count += 1
    rows, columns, ids = np.empty(count, np.int64), np.empty(count, np.int64), np.empty(count, np.int64)
    k = 0
    for i in range(nodes.size):
        u = nodes[i]
        for j in range(sorted_nodes.size):
            at = _find(col, ptr[u], ptr[u + 1], sorted_nodes[j])
            if at < ptr[u + 1] and col[at] == sorted_nodes[j]:
                rows[k], columns[k], ids[k] = i, sorted_local[j], at; k += 1
    return rows, columns, ids


class CompiledBackend:
    """Owned raw outputs matching native idx-order CSR scan, including CSR slot IDs."""
    def __init__(self, ptr: np.ndarray, col: np.ndarray, mode: str):
        self.ptr, self.col, self.mode = ptr, col, mode
        if mode not in ("scan", "pairwise"): raise ValueError("unknown compiled mode")

    def induce(self, nodes: np.ndarray):
        validate_nodes(nodes, len(self.ptr) - 1)
        order = np.argsort(nodes).astype(np.int64)
        # Sorting/membership preparation is part of every operator invocation.
        sorted_nodes = np.ascontiguousarray(nodes[order])
        fn = compiled_scan if self.mode == "scan" else compiled_pairwise
        return fn(self.ptr, self.col, nodes, sorted_nodes, order)


def canonical_postmask(raw, roots):
    rows, cols, _ = raw
    s, t = (int(x) for x in roots)
    keep = ~(((rows == s) & (cols == t)) | ((rows == t) & (cols == s)))
    rows, cols = rows[keep], cols[keep]
    order = np.lexsort((cols, rows))
    return np.stack([rows[order], cols[order]], axis=0)


def selected_symmetric(raw) -> bool:
    r, c, _ = raw
    return set(zip(r.tolist(), c.tolist())) == set(zip(c.tolist(), r.tolist()))


def cohort(pos_path: Path, neg_path: Path, n: int, *, expected_sources: int = 512,
           expected_negatives: int = 1000) -> tuple[np.ndarray, np.ndarray, dict]:
    pos = np.load(pos_path, mmap_mode="r", allow_pickle=False)
    neg = np.load(neg_path, mmap_mode="r", allow_pickle=False)
    for a in (pos, neg):
        if not np.issubdtype(a.dtype, np.integer): raise ValueError("candidate arrays must be integer")
    if pos.shape != (expected_sources, 2): raise ValueError("expected512 fixed positive validation queries")
    if neg.shape == (expected_sources, expected_negatives, 2): neg = neg.reshape(-1, 2)
    if neg.shape != (expected_sources * expected_negatives, 2):
        raise ValueError("expected all1000 candidates for every fixed validation source")
    if len(np.unique(pos[:, 0])) != expected_sources:
        raise ValueError("expected one positive per distinct validation source")
    for i in range(expected_sources):
        block = neg[i * expected_negatives:(i + 1) * expected_negatives]
        if np.any(block[:, 0] != pos[i, 0]): raise ValueError("negative grouping/source order mismatch")
        if np.any(block < 0) or np.any(block >= n): raise ValueError("negative vertex outside graph")
    if np.any(pos < 0) or np.any(pos >= n): raise ValueError("positive vertex outside graph")
    pairs = np.concatenate([np.asarray(pos, dtype=np.int64),
                            np.asarray(neg[::expected_negatives], dtype=np.int64)])
    labels = np.r_[np.ones(expected_sources, dtype=np.int64), np.zeros(expected_sources, dtype=np.int64)]
    meta = {"policy": "all512 frozen validation positives, then negative candidate index0 per source",
            "sources": expected_sources, "original_negatives_per_source": expected_negatives,
            "queries": len(pairs), "selection_depends_on_timing_or_scores": False,
            "positive_file": {**identity(pos_path), "sha256": sha_file(pos_path)},
            "negative_file": {**identity(neg_path), "sha256": sha_file(neg_path)},
            "pairs_hash": array_hash(pairs), "labels_hash": array_hash(labels)}
    return pairs, labels, meta


def query_seed(base: int, index: int, domain: int = 0) -> int:
    return int(np.random.SeedSequence([base, domain, index]).generate_state(1)[0])


def pack_rng(states: list[dict]) -> tuple[dict, dict]:
    if not states or any(s["py_version"] != 3 or s["np_algorithm"] != "MT19937" for s in states):
        raise ValueError("unsupported captured RNG states")
    lengths = {len(s["torch_cpu"]) for s in states}
    if len(lengths) != 1: raise ValueError("Torch CPU RNG payload lengths differ")
    scalars = [[int(s["py_gauss"] is not None), 0. if s["py_gauss"] is None else s["py_gauss"],
                s["np_position"], s["np_has_gauss"], s["np_cached_gauss"]] for s in states]
    arrays = {"rng_python_words.npy": np.asarray([s["py_words"] for s in states], dtype=np.uint32),
              "rng_numpy_words.npy": np.asarray([s["np_words"] for s in states], dtype=np.uint32),
              "rng_torch_cpu.npy": np.asarray([s["torch_cpu"] for s in states], dtype=np.uint8),
              "rng_scalars.npy": np.asarray(scalars, dtype=np.float64)}
    if arrays["rng_python_words.npy"].shape != (len(states), 625) or arrays["rng_numpy_words.npy"].shape != (len(states), 624):
        raise ValueError("captured RNG word shape mismatch")
    if not np.isfinite(arrays["rng_scalars.npy"]).all(): raise ValueError("nonfinite captured RNG scalar")
    return arrays, {"py_version": 3, "np_algorithm": "MT19937", "torch_cpu_bytes": lengths.pop(),
                    "scope": "post-native-raw CPU RNG before wrapper; no CUDA training/dropout state claim"}


def save_trace(path: Path, pairs: np.ndarray, labels: np.ndarray, nodes: list[np.ndarray],
               roots: list[np.ndarray], records: list[dict], meta: dict, max_bytes: int,
               rng_states: list[dict] | None = None) -> dict:
    if path.exists(): raise FileExistsError("trace directory exists; never overwrite a freeze")
    required = sum(a.size * 4 for a in nodes) + 8 * (len(nodes) + 1) + pairs.nbytes + labels.nbytes + 16 * len(nodes)
    rng_arrays = {}
    if rng_states is not None:
        if len(rng_states) != len(pairs): raise ValueError("captured RNG record count mismatch")
        rng_arrays, schema = pack_rng(rng_states)
        required += sum(a.nbytes for a in rng_arrays.values())
        meta = {**meta, "post_raw_rng_schema": schema}
    if required > max_bytes: raise GuardError("ragged trace exceeds declared byte budget")
    if len(nodes) != len(pairs) or len(roots) != len(pairs) or len(records) != len(pairs):
        raise ValueError("trace record count mismatch")
    ptr = np.r_[0, np.cumsum([len(a) for a in nodes], dtype=np.int64)]
    ids = np.concatenate(nodes).astype(np.int32)
    root_positions = np.asarray(roots, dtype=np.int64)
    for i, values in enumerate(nodes):
        if np.any(values[1:] <= values[:-1]): raise ValueError("PENCIL trace requires literal sorted unique S")
        if root_positions.shape != (len(nodes), 2) or np.any(root_positions[i] < 0) or np.any(root_positions[i] >= len(values)):
            raise ValueError("invalid trace root positions")
        if not np.array_equal(values[root_positions[i]], pairs[i]): raise ValueError("root positions disagree with query")
    path.mkdir(parents=True)
    arrays = {"pairs.npy": pairs, "labels.npy": labels, "node_ptr.npy": ptr,
              "nodes.npy": ids, "root_positions.npy": root_positions}
    arrays.update(rng_arrays)
    for name, values in arrays.items(): np.save(path / name, values, allow_pickle=False)
    atomic_json(path / "records.json", {"records": records})
    manifest = {"schema_version": 1, "status": "complete", "metadata": meta,
                "files": [{"name": name, "bytes": (path / name).stat().st_size,
                           "sha256": sha_file(path / name)} for name in [*arrays, "records.json"]]}
    atomic_json(path / "manifest.json", manifest)
    return manifest


class Trace:
    def __init__(self, path: Path):
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        if self.manifest.get("status") != "complete": raise ValueError("trace is not complete")
        if sum(f["bytes"] for f in self.manifest["files"]) > 512 * 2**20:
            raise GuardError("trace exceeds512MiB admitted payload")
        required_names = {"pairs.npy", "labels.npy", "node_ptr.npy", "nodes.npy", "root_positions.npy", "records.json"}
        self.rng_arrays = None
        schema = self.manifest["metadata"].get("post_raw_rng_schema")
        if schema is not None:
            required_names.update(("rng_python_words.npy", "rng_numpy_words.npy", "rng_torch_cpu.npy", "rng_scalars.npy"))
        if {f["name"] for f in self.manifest["files"]} != required_names or len(self.manifest["files"]) != len(required_names):
            raise ValueError("trace manifest file set mismatch")
        for file in self.manifest["files"]:
            p = path / file["name"]
            if p.stat().st_size != file["bytes"] or sha_file(p) != file["sha256"]:
                raise ValueError("trace content hash mismatch: " + file["name"])
        for name, file in (("pairs", "pairs.npy"), ("labels", "labels.npy"), ("ptr", "node_ptr.npy"),
                           ("ids", "nodes.npy"), ("roots", "root_positions.npy")):
            setattr(self, name, np.load(path / file, mmap_mode="r", allow_pickle=False))
        if (path / "records.json").stat().st_size > 32 * 2**20: raise GuardError("trace JSON record budget exceeded")
        self.records = json.loads((path / "records.json").read_text())["records"]
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2 or self.pairs.dtype != np.int64:
            raise ValueError("invalid trace pairs")
        q = len(self.pairs)
        if schema is not None:
            self.rng_arrays = {name: np.load(path / name, mmap_mode="r", allow_pickle=False) for name in
                 ("rng_python_words.npy", "rng_numpy_words.npy", "rng_torch_cpu.npy", "rng_scalars.npy")}
            expected = {"rng_python_words.npy": ((q, 625), np.uint32), "rng_numpy_words.npy": ((q, 624), np.uint32),
                        "rng_torch_cpu.npy": ((q, schema["torch_cpu_bytes"]), np.uint8), "rng_scalars.npy": ((q, 5), np.float64)}
            for name, (shape, dtype) in expected.items():
                if self.rng_arrays[name].shape != shape or self.rng_arrays[name].dtype != dtype:
                    raise ValueError("captured RNG trace array shape/dtype mismatch")
        if self.ptr.shape != (q + 1,) or self.ptr.dtype != np.int64 or int(self.ptr[0]) != 0 or int(self.ptr[-1]) != len(self.ids):
            raise ValueError("invalid ragged trace pointers")
        if self.ids.dtype != np.int32 or self.roots.shape != (q, 2) or self.roots.dtype != np.int64 or self.labels.shape != (q,) or len(self.records) != q:
            raise ValueError("invalid trace data shapes")
        for i in range(q):
            nodes = self.nodes(i)
            if not 1 <= len(nodes) <= 82 or np.any(nodes[1:] <= nodes[:-1]): raise ValueError("invalid literal sorted trace S")
            if np.any(self.roots[i] < 0) or np.any(self.roots[i] >= len(nodes)) or not np.array_equal(nodes[self.roots[i]], self.pairs[i]):
                raise ValueError("trace root/query mismatch")

    def __len__(self): return len(self.pairs)
    def nodes(self, i): return np.array(self.ids[int(self.ptr[i]):int(self.ptr[i + 1])], dtype=np.int64)
    def rng(self, i):
        if self.rng_arrays is None: raise ValueError("trace has no captured post-native-raw CPU RNG states")
        a = self.rng_arrays; s = a["rng_scalars.npy"][i]
        return {"py_version": 3, "py_words": np.array(a["rng_python_words.npy"][i]),
                "py_gauss": float(s[1]) if s[0] else None,
                "np_algorithm": "MT19937", "np_words": np.array(a["rng_numpy_words.npy"][i]),
                "np_position": int(s[2]), "np_has_gauss": int(s[3]), "np_cached_gauss": float(s[4]),
                "torch_cpu": np.array(a["rng_torch_cpu.npy"][i])}


def tiny_graph() -> tuple[Graph, np.ndarray, np.ndarray, np.ndarray]:
    # Loop-free symmetric, bridge, cold vertices; includes self-query and zero-edge query.
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (6, 7)]
    directed = sorted(edges + [(v, u) for u, v in edges])
    ptr = np.zeros(10, dtype=np.int64)
    for u, v in directed: ptr[u + 1] += 1
    np.cumsum(ptr, out=ptr)
    col = np.array([v for u, v in directed], dtype=np.int32)
    pairs = np.array([[3, 0], [0, 1], [8, 5], [8, 0], [0, 0], [7, 6], [8, 8]], dtype=np.int64)
    labels = np.array([0, 1, 0, 0, 0, 1, 0], dtype=np.int64)
    features = (np.arange(9 * 128, dtype=np.float32).reshape(9, 128) % 97) / 97
    return Graph(ptr, col, {"fixture": True, "n": 9, "m": len(col)}), features, pairs, labels


def summary(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean_s": float(a.mean()), "p50_s": float(np.percentile(a, 50)),
            "p95_s": float(np.percentile(a, 95)), "min_s": float(a.min()), "max_s": float(a.max())} if len(a) else {"n": 0}
