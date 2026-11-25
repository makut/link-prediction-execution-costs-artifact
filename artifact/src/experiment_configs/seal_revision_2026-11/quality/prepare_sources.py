"""Freeze uniformly sampled complete Citation2 source groups, retaining all negatives.

Only mmap views and bounded chunks of large negative files are held in memory.
No torch, pickle, training or model scores are needed to select these subsets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_edges(path: Path):
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.ndim != 2 or value.shape[1] != 2 or value.dtype.kind not in "iu":
        raise ValueError(f"Expected integer [N,2] edges: {path}, {value.shape}/{value.dtype}")
    return value


def _write_selected(edges, sources, path: Path, *, chunk_rows: int):
    # Count first, then write directly to a .npy mmap; never concatenate a full split.
    selected_count = 0
    for start in range(0, len(edges), chunk_rows):
        selected_count += int(np.isin(edges[start:start + chunk_rows, 0], sources).sum())
    target = np.lib.format.open_memmap(path, mode="w+", dtype=np.int64, shape=(selected_count, 2))
    counts = np.zeros(len(sources), dtype=np.int64)
    cursor = 0
    for start in range(0, len(edges), chunk_rows):
        block = edges[start:start + chunk_rows]
        selected = np.asarray(block[np.isin(block[:, 0], sources)], dtype=np.int64)
        target[cursor:cursor + len(selected)] = selected
        if len(selected):
            counts += np.bincount(np.searchsorted(sources, selected[:, 0]), minlength=len(sources))
        cursor += len(selected)
    target.flush()
    del target
    return selected_count, counts


def freeze_source_subset(*, pos_path: Path, neg_path: Path, output_dir: Path,
                         name: str, count: int, seed: int,
                         expected_negatives: int = 1000, chunk_rows: int = 250_000) -> dict:
    if count <= 0 or chunk_rows <= 0 or expected_negatives <= 0:
        raise ValueError("Source count, chunk_rows and expected_negatives must be positive")
    positive = _load_edges(pos_path)
    negative = _load_edges(neg_path)
    all_sources, positive_counts = np.unique(positive[:, 0], return_counts=True)
    if np.any(positive_counts != 1):
        raise ValueError("Citation2 protocol requires exactly one positive per original source")
    if count > len(all_sources):
        raise ValueError(f"Requested {count} groups, split contains only {len(all_sources)}")
    chosen = np.sort(np.random.default_rng(seed).choice(all_sources, size=count, replace=False))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"positive": output_dir / f"{name}_edge.npy", "negative": output_dir / f"{name}_edge_neg.npy",
             "sources": output_dir / f"{name}_sources.npy"}
    n_pos, pos_counts = _write_selected(positive, chosen, paths["positive"], chunk_rows=chunk_rows)
    n_neg, neg_counts = _write_selected(negative, chosen, paths["negative"], chunk_rows=chunk_rows)
    if np.any(pos_counts != 1) or np.any(neg_counts != expected_negatives):
        raise ValueError(f"Incomplete {name} groups: positive range {pos_counts.min()}/{pos_counts.max()}, "
                         f"negative range {neg_counts.min()}/{neg_counts.max()}; expected 1/{expected_negatives}")
    np.save(paths["sources"], chosen, allow_pickle=False)
    manifest = {"split": name, "subset_seed": int(seed), "selection": "uniform source IDs without replacement",
                "source_count": len(chosen), "original_source_count": len(all_sources),
                "positive_rows": n_pos, "negative_rows": n_neg,
                "negatives_per_source": expected_negatives, "all_provided_negatives_retained": True,
                "inputs": {key: {"path": str(path.resolve()), "sha256": sha256(path)}
                           for key, path in (("positive", pos_path), ("negative", neg_path))},
                "outputs": {key: {"path": str(path.resolve()), "sha256": sha256(path)}
                            for key, path in paths.items()}}
    (output_dir / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
