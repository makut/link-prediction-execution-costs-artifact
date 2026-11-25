"""Build the transposed (in-edge) GraphCSR of a directed GraphCSR.

For a directed CSR whose row u lists v with u -> v, the output's row v lists u (sorted ascending),
with the timestamps carried over. Together (out, in) drive `seal.directed: true`.

Usage: python -m scripts.transpose_graph_csr --graph <prepared>/train_directed_csr --out <prepared>/train_directed_csr_in --file-endian little
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_csr.graph import GraphCSR
from graph_csr.large_int32 import LargeInt32Array
from graph_csr.large_int64_raw import LargeInt64RawArray
from graph_csr.serializer import GraphCSRSerializer


def transpose_graph_csr(graph_dir: Path, out_dir: Path, *, file_endian: str = "little") -> dict:
    t0 = time.time()
    g = GraphCSRSerializer.deserialize(str(graph_dir), use_mmap=True, file_endian=file_endian, writable=False, allow_non_native=True, chunk_bytes=256 * 1024 * 1024)
    starts = np.asarray(g.edge_starts.numpy(), dtype=np.int64)
    num_nodes = int(starts.size); num_edges = int(g.edge_ends.size)
    ends = np.asarray(g.edge_ends.numpy(), dtype=np.int64)
    ts = np.asarray(g.timestamps.numpy(), dtype=np.int32) if g.timestamps is not None else None
    counts = np.diff(np.append(starts, num_edges))
    src = np.repeat(np.arange(num_nodes, dtype=np.int64), counts)
    order = np.lexsort((src, ends))  # sort by (dst, src)
    in_starts = np.zeros(num_nodes, dtype=np.int64)
    np.cumsum(np.bincount(ends, minlength=num_nodes)[:-1], out=in_starts[1:])
    in_ends = src[order].astype(np.int32)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = GraphCSR(
        edge_ends=LargeInt32Array(size=int(in_ends.size), arr=np.ascontiguousarray(in_ends)),
        edge_starts=LargeInt64RawArray(arr=np.ascontiguousarray(in_starts)),
        timestamps=None if ts is None else LargeInt32Array(size=int(num_edges), arr=np.ascontiguousarray(ts[order])),
    )
    GraphCSRSerializer.serialize(graph, out_dir, file_endian=file_endian)
    return {"num_nodes": num_nodes, "num_edges": num_edges, "elapsed_s": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, type=Path); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--file-endian", default="little", choices=["little", "big"])
    a = ap.parse_args()
    print(transpose_graph_csr(a.graph, a.out, file_endian=a.file_endian))


if __name__ == "__main__":
    main()
