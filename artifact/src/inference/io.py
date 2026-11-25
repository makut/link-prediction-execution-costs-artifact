from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SCORES_FILENAME = "scores.npy"
MANIFEST_FILENAME = "manifest.json"


def load_pairs(path: str | Path, *, use_mmap: bool = True) -> np.ndarray:
    """Load the unified sample format: an ``[N, 2]`` int64 ``.npy`` of ``(src, dst)``.

    This mirrors ``dataset_utils.eval_data._load_edges`` — the same on-disk format
    every edge array in the repo already uses (val/test splits, negatives).
    """
    pairs_path = Path(path)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    array = np.load(pairs_path, mmap_mode="r" if use_mmap else None)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Pairs array at {pairs_path} must have shape [N, 2], got {array.shape}")
    if np.asarray(array).dtype == np.int64:
        return array
    return np.asarray(array, dtype=np.int64)


def write_scores(
    output_dir: str | Path,
    *,
    scores: np.ndarray,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist ``scores.npy`` (``[N]`` float32) and ``manifest.json`` to a directory.

    Returns the ``(scores_path, manifest_path)`` written.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    scores_path = out_dir / SCORES_FILENAME
    np.save(scores_path, scores_arr)
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return scores_path, manifest_path
