"""Convert the Graphalytics / SNAP com-Friendster edge list into our GraphCSR layout.

Input (Zenodo record 5146230, "Com-Friendster Dataset Densely Relabelled", CC-BY-4.0):
  com-friendster-dense.ez          zlib stream of little-endian uint64 (u, v) records (undirected, each edge once)
  com-friendster-dense.vz          zlib stream of little-endian uint64 vertex ids (dense 0..N-1)
  (a text "u v" variant is supported with --format text)
  com-friendster-dense.properties  vertices = 65,608,366, edges = 1,806,067,135, directed = false

Output (<out>/):
  edge_starts.bin   raw int64 [num_nodes]            (little-endian by default, like the VK / citation2 prepared data)
  edge_ends.bin     8-byte size header + int32 [E2]  (both directions, rows sorted ascending, deduplicated)
  timestamps.bin    8-byte size header + int32 [E2]  (all zeros — the source has no timestamps)
  num_nodes.txt, meta.json

Pipeline: streaming zlib decompression -> newline-aligned text chunks -> pandas C parser in a process pool ->
int32 src/dst arrays -> numba counting sort into CSR (both directions) -> per-row sort + dedup (numba, parallel)
-> GraphCSRSerializer. Peak memory ≈ 2 × 4 B × E (pairs) + 2 × 4 B × 2E (ends + scratch) ≈ 45 GB for Friendster.

Usage:
  python -m scripts.prepare_friendster_csr --edges com-friendster-dense.ez --vertices com-friendster-dense.vz \
      --out /home/jovyan/friendster/csr --workers 24
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_csr.serializer import GraphCSRSerializer  # noqa: E402
from scripts.prepare_ogb_dataset import _write_graph_csr  # noqa: E402

LOGGER = logging.getLogger("prepare_friendster_csr")


# ----------------------------------------------------------------------------------------------------------------------
# streaming decompression + parsing
# ----------------------------------------------------------------------------------------------------------------------
def iter_zlib_text_chunks(path: Path, chunk_bytes: int = 256 << 20, read_bytes: int = 32 << 20) -> Iterator[bytes]:
    """Yield newline-aligned chunks (≈chunk_bytes) of the decompressed text."""
    dec = zlib.decompressobj()
    pending = bytearray()
    with open(path, "rb") as f:
        while True:
            raw = f.read(read_bytes)
            if not raw:
                break
            pending += dec.decompress(raw)
            while len(pending) >= chunk_bytes:
                cut = pending.rfind(b"\n", 0, chunk_bytes)
                if cut < 0:
                    cut = pending.find(b"\n")
                    if cut < 0:
                        break
                yield bytes(pending[: cut + 1])
                del pending[: cut + 1]
    pending += dec.flush()
    if pending:
        if not pending.endswith(b"\n"):
            pending += b"\n"
        yield bytes(pending)


def iter_zlib_binary_chunks(path: Path, record_bytes: int, chunk_bytes: int = 256 << 20, read_bytes: int = 32 << 20) -> Iterator[bytes]:
    """Yield record-aligned chunks (≈chunk_bytes) of a zlib-compressed binary stream."""
    dec = zlib.decompressobj()
    pending = bytearray()
    with open(path, "rb") as f:
        while True:
            raw = f.read(read_bytes)
            if not raw:
                break
            pending += dec.decompress(raw)
            while len(pending) >= chunk_bytes:
                cut = (chunk_bytes // record_bytes) * record_bytes
                yield bytes(pending[:cut])
                del pending[:cut]
    pending += dec.flush()
    if len(pending) % record_bytes:
        raise ValueError(f"trailing {len(pending) % record_bytes} bytes: not a multiple of the record size {record_bytes}")
    if pending:
        yield bytes(pending)


def load_pairs_u64le(path: Path, expected_edges: int, chunk_bytes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Edge list as little-endian uint64 (u, v) records (vtxremap / Graphalytics `.ez` layout)."""
    cap = int(expected_edges * 1.02) + 1_000_000
    src = np.empty(cap, dtype=np.int32)
    dst = np.empty(cap, dtype=np.int32)
    n = 0
    t0 = time.time()
    for k, chunk in enumerate(iter_zlib_binary_chunks(path, 16, chunk_bytes=chunk_bytes)):
        arr = np.frombuffer(chunk, dtype="<u8").reshape(-1, 2)
        if arr.max() >= 2**31:
            raise ValueError("vertex id outside int32 range")
        m = arr.shape[0]
        if n + m > cap:
            raise RuntimeError(f"more edges than expected (cap={cap:,})")
        src[n : n + m] = arr[:, 0]
        dst[n : n + m] = arr[:, 1]
        n += m
        if (k + 1) % 10 == 0:
            LOGGER.info("decoded %d chunks, %s edges, %.0f s", k + 1, f"{n:,}", time.time() - t0)
    LOGGER.info("decoded %s edges in %.0f s", f"{n:,}", time.time() - t0)
    return src[:n], dst[:n]


def load_vertices_u64le(path: Path, chunk_bytes: int) -> np.ndarray:
    parts = [np.frombuffer(c, dtype="<u8").astype(np.int64) for c in iter_zlib_binary_chunks(path, 8, chunk_bytes=chunk_bytes)]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _parse_pairs(chunk: bytes) -> np.ndarray:
    df = pd.read_csv(
        io.BytesIO(chunk),
        sep=r"\s+",
        header=None,
        comment="#",
        dtype=np.int64,
        engine="c",
        names=["u", "v"],
        usecols=[0, 1],
    )
    return np.ascontiguousarray(df.to_numpy(dtype=np.int64))


def _parse_ints(chunk: bytes) -> np.ndarray:
    df = pd.read_csv(io.BytesIO(chunk), sep=r"\s+", header=None, comment="#", dtype=np.int64, engine="c", names=["v"], usecols=[0])
    return np.ascontiguousarray(df["v"].to_numpy(dtype=np.int64))


def load_pairs(path: Path, expected_edges: int, workers: int, chunk_bytes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse the whole edge list into int32 src/dst arrays (ids must fit int32; checked)."""
    cap = int(expected_edges * 1.02) + 1_000_000
    src = np.empty(cap, dtype=np.int32)
    dst = np.empty(cap, dtype=np.int32)
    n = 0
    t0 = time.time()
    done_chunks = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        inflight = []
        for chunk in iter_zlib_text_chunks(path, chunk_bytes=chunk_bytes):
            inflight.append(pool.submit(_parse_pairs, chunk))
            if len(inflight) >= workers * 2:
                arr = inflight.pop(0).result()
                n = _append_pairs(src, dst, n, arr, cap)
                done_chunks += 1
                if done_chunks % 10 == 0:
                    LOGGER.info("parsed %d chunks, %s edges, %.0f s", done_chunks, f"{n:,}", time.time() - t0)
        for fut in inflight:
            arr = fut.result()
            n = _append_pairs(src, dst, n, arr, cap)
            done_chunks += 1
    LOGGER.info("parsed %d chunks, %s edges in %.0f s", done_chunks, f"{n:,}", time.time() - t0)
    return src[:n], dst[:n]


def _append_pairs(src: np.ndarray, dst: np.ndarray, n: int, arr: np.ndarray, cap: int) -> int:
    m = arr.shape[0]
    if n + m > cap:
        raise RuntimeError(f"more edges than expected (cap={cap:,})")
    if arr.size and (arr.min() < 0 or arr.max() >= 2**31):
        raise ValueError("vertex id outside int32 range")
    src[n : n + m] = arr[:, 0]
    dst[n : n + m] = arr[:, 1]
    return n + m


def load_vertices(path: Path, workers: int, chunk_bytes: int) -> np.ndarray:
    parts = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        for arr in pool.map(_parse_ints, iter_zlib_text_chunks(path, chunk_bytes=chunk_bytes)):
            parts.append(arr)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# CSR construction (numba)
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _fill_both_directions(src: np.ndarray, dst: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> int:
    """Counting-sort fill of both directions; self loops are skipped. Returns number of skipped self loops."""
    n = starts.shape[0]
    cursor = starts.copy()
    skipped = 0
    for i in range(src.shape[0]):
        u = src[i]
        v = dst[i]
        if u == v:
            skipped += 1
            continue
        ends[cursor[u]] = v
        cursor[u] += 1
        ends[cursor[v]] = u
        cursor[v] += 1
    return skipped


@njit(parallel=True, cache=True)
def _sort_rows_count_unique(starts: np.ndarray, total: int, ends: np.ndarray, uniq: np.ndarray) -> None:
    n = starts.shape[0]
    for node in prange(n):
        s = starts[node]
        e = starts[node + 1] if node + 1 < n else total
        if e > s:
            row = ends[s:e]
            row.sort()
            c = 1
            for k in range(s + 1, e):
                if ends[k] != ends[k - 1]:
                    c += 1
            uniq[node] = c
        else:
            uniq[node] = 0


@njit(parallel=True, cache=True)
def _compact_unique(starts: np.ndarray, total: int, ends: np.ndarray, new_starts: np.ndarray, new_total: int, out: np.ndarray) -> None:
    n = starts.shape[0]
    for node in prange(n):
        s = starts[node]
        e = starts[node + 1] if node + 1 < n else total
        o = new_starts[node]
        if e > s:
            out[o] = ends[s]
            o += 1
            for k in range(s + 1, e):
                if ends[k] != ends[k - 1]:
                    out[o] = ends[k]
                    o += 1


def build_csr(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    t0 = time.time()
    deg = np.bincount(src, minlength=num_nodes).astype(np.int64)
    deg += np.bincount(dst, minlength=num_nodes)
    loops = int(np.count_nonzero(src == dst))
    starts = np.zeros(num_nodes, dtype=np.int64)
    np.cumsum(deg[:-1], out=starts[1:])
    total = int(deg.sum())
    ends = np.empty(total, dtype=np.int32)
    LOGGER.info("degrees + offsets in %.0f s; filling %s entries ...", time.time() - t0, f"{total:,}")
    skipped = _fill_both_directions(src, dst, starts, ends)
    assert skipped == loops, (skipped, loops)
    # self loops left holes at the end of the affected rows: rebuild offsets from the actual counts
    if loops:
        deg_eff = deg - 2 * (np.bincount(src[src == dst], minlength=num_nodes))
        starts_eff = np.zeros(num_nodes, dtype=np.int64)
        np.cumsum(deg_eff[:-1], out=starts_eff[1:])
        total_eff = int(deg_eff.sum())
        ends_eff = np.empty(total_eff, dtype=np.int32)
        _gather_rows(starts, deg_eff, ends, starts_eff, ends_eff)
        starts, ends, total = starts_eff, ends_eff, total_eff
    LOGGER.info("filled in %.0f s (self loops skipped: %d); sorting rows ...", time.time() - t0, loops)
    uniq = np.empty(num_nodes, dtype=np.int64)
    _sort_rows_count_unique(starts, total, ends, uniq)
    dups = total - int(uniq.sum())
    LOGGER.info("rows sorted in %.0f s; duplicate entries: %d", time.time() - t0, dups)
    if dups:
        new_starts = np.zeros(num_nodes, dtype=np.int64)
        np.cumsum(uniq[:-1], out=new_starts[1:])
        new_total = int(uniq.sum())
        out = np.empty(new_total, dtype=np.int32)
        _compact_unique(starts, total, ends, new_starts, new_total, out)
        starts, ends, total = new_starts, out, new_total
        LOGGER.info("deduplicated in %.0f s", time.time() - t0)
    degrees = np.diff(np.append(starts, total))
    stats = {
        "stored_entries": int(total),
        "undirected_edges": int(total // 2),
        "self_loops_dropped": int(loops),
        "duplicate_entries_dropped": int(dups),
        "isolated_nodes": int(np.count_nonzero(degrees == 0)),
        "degree_mean": float(degrees.mean()),
        "degree_max": int(degrees.max()),
        "degree_p50": int(np.percentile(degrees, 50)),
        "degree_p99": int(np.percentile(degrees, 99)),
        "build_seconds": round(time.time() - t0, 1),
    }
    return starts, ends, stats


@njit(parallel=True, cache=True)
def _gather_rows(starts: np.ndarray, deg_eff: np.ndarray, ends: np.ndarray, starts_eff: np.ndarray, ends_eff: np.ndarray) -> None:
    n = starts.shape[0]
    for node in prange(n):
        s = starts[node]
        o = starts_eff[node]
        for k in range(deg_eff[node]):
            ends_eff[o + k] = ends[s + k]


# ----------------------------------------------------------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _check_symmetric_sample(starts: np.ndarray, total: int, ends: np.ndarray, sample_pos: np.ndarray) -> int:
    """For sampled positions p (u = row of p, v = ends[p]) check that u is in row v (binary search). Returns #failures."""
    n = starts.shape[0]
    bad = 0
    for i in range(sample_pos.shape[0]):
        p = sample_pos[i]
        # row of p
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= p:
                lo = mid
            else:
                hi = mid - 1
        u = lo
        v = ends[p]
        s = starts[v]
        e = starts[v + 1] if v + 1 < n else total
        a, b = s, e - 1
        found = False
        while a <= b:
            m = (a + b) // 2
            if ends[m] == u:
                found = True
                break
            if ends[m] < u:
                a = m + 1
            else:
                b = m - 1
        if not found:
            bad += 1
    return bad


def verify_written(out: Path, file_endian: str, num_nodes: int, total: int, sample: int = 200_000) -> dict:
    g = GraphCSRSerializer.deserialize(out, use_mmap=True, file_endian=file_endian)
    starts = np.ascontiguousarray(g.edge_starts.numpy(), dtype=np.int64)
    ends = np.ascontiguousarray(g.edge_ends.numpy(), dtype=np.int32)
    assert starts.shape[0] == num_nodes, (starts.shape, num_nodes)
    assert ends.shape[0] == total, (ends.shape, total)
    rng = np.random.default_rng(0)
    pos = np.sort(rng.integers(0, total, size=min(sample, total), dtype=np.int64))
    bad = _check_symmetric_sample(starts, total, ends, pos)
    # sortedness of a sample of rows
    nodes = rng.integers(0, num_nodes, size=min(20_000, num_nodes))
    unsorted = 0
    for u in nodes:
        s = int(starts[u]); e = int(starts[u + 1]) if u + 1 < num_nodes else total
        row = ends[s:e]
        if row.size > 1 and (np.any(row[1:] <= row[:-1])):
            unsorted += 1
    ts = g.timestamps
    ts_ok = ts is not None and int(ts.size) == total
    res = {"symmetry_failures": int(bad), "symmetry_sample": int(pos.size), "unsorted_rows_in_sample": int(unsorted), "timestamps_present": bool(ts_ok)}
    g.close()
    return res


# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", required=True, type=Path)
    ap.add_argument("--vertices", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expected-edges", type=int, default=1_806_067_135)
    ap.add_argument("--num-nodes", type=int, default=None, help="override; default = max id + 1 (checked against the vertex list)")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--file-endian", choices=["little", "big"], default="little")
    ap.add_argument("--source-note", default="Graphalytics com-friendster (SNAP com-Friendster), Zenodo 10.5281/zenodo.5146230, CC-BY-4.0")
    ap.add_argument("--plain-text", action="store_true", help="inputs are plain text, not zlib streams")
    ap.add_argument("--format", choices=["u64le", "text"], default="u64le",
                    help="u64le: zlib stream of little-endian uint64 records (vtxremap .ez/.vz, the Zenodo release); text: zlib (or plain) 'u v' lines")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()

    global iter_zlib_text_chunks
    if args.plain_text:
        _orig = iter_zlib_text_chunks

        def iter_plain(path: Path, chunk_bytes: int = 256 << 20, read_bytes: int = 32 << 20) -> Iterator[bytes]:
            pending = bytearray()
            with open(path, "rb") as f:
                while True:
                    raw = f.read(read_bytes)
                    if not raw:
                        break
                    pending += raw
                    while len(pending) >= chunk_bytes:
                        cut = pending.rfind(b"\n", 0, chunk_bytes)
                        if cut < 0:
                            break
                        yield bytes(pending[: cut + 1]); del pending[: cut + 1]
            if pending:
                if not pending.endswith(b"\n"):
                    pending += b"\n"
                yield bytes(pending)

        iter_zlib_text_chunks = iter_plain  # type: ignore[assignment]

    chunk_bytes = args.chunk_mb << 20
    vertices = None
    if args.vertices is not None:
        vertices = load_vertices_u64le(args.vertices, chunk_bytes) if args.format == "u64le" else load_vertices(args.vertices, args.workers, chunk_bytes)
        LOGGER.info("vertex list: %s ids, min %d, max %d", f"{vertices.size:,}", int(vertices.min()), int(vertices.max()))

    src, dst = load_pairs_u64le(args.edges, args.expected_edges, chunk_bytes) if args.format == "u64le" else load_pairs(args.edges, args.expected_edges, args.workers, chunk_bytes)
    max_id = int(max(src.max(), dst.max()))
    num_nodes = args.num_nodes or (max_id + 1)
    if vertices is not None:
        dense = bool(vertices.size == num_nodes and int(vertices.min()) == 0 and int(vertices.max()) == num_nodes - 1)
        if not dense:
            raise SystemExit(f"vertex list is not dense 0..N-1 (size {vertices.size:,}, max {int(vertices.max())}, edges max {max_id}); remap needed")
    if num_nodes <= max_id:
        raise SystemExit(f"--num-nodes {num_nodes} <= max id {max_id}")
    LOGGER.info("num_nodes = %s, parsed edges = %s", f"{num_nodes:,}", f"{src.size:,}")

    starts, ends, stats = build_csr(src, dst, num_nodes)
    del src, dst
    total = int(ends.size)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    LOGGER.info("writing %s (%s entries) ...", out, f"{total:,}")
    _write_graph_csr(out_dir=out, edge_starts=starts, edge_ends=ends, timestamps=np.zeros(total, dtype=np.int32), file_endian=args.file_endian)
    (out / "num_nodes.txt").write_text(f"{num_nodes}\n")
    LOGGER.info("written in %.0f s total; verifying ...", time.time() - t0)
    checks = verify_written(out, args.file_endian, num_nodes, total)
    meta = {
        "graph": "com-friendster",
        "source": args.source_note,
        "input_edges_file": str(args.edges),
        "num_nodes": int(num_nodes),
        "vertex_ids": "dense 0..N-1 as in the Graphalytics relabelled release (rank of the SNAP id; stable-map=false)",
        "undirected": True,
        "both_directions_stored": True,
        "timestamps": "none in the source; timestamps.bin is all zeros (use a random edge split, not the temporal one)",
        "file_endian": args.file_endian,
        "layout": "edge_starts.bin raw int64[num_nodes]; edge_ends.bin / timestamps.bin = 8-byte size header + int32[]",
        "stats": stats,
        "verification": checks,
        "built_by": "scripts/prepare_friendster_csr.py",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    LOGGER.info("meta: %s", json.dumps(meta))
    if checks["symmetry_failures"] or checks["unsorted_rows_in_sample"]:
        raise SystemExit("VERIFICATION FAILED")
    LOGGER.info("done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main()
