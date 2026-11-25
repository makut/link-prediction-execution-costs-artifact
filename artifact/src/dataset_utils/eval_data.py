from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .positive_edges import GraphCSRPositiveEdgeDataset

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EdgeSplits:
    val_pos: np.ndarray
    val_neg: np.ndarray
    test_pos: np.ndarray
    test_neg: np.ndarray


@dataclass(slots=True)
class EvalGroupStats:
    num_rows: int
    num_positive_rows: int
    num_negative_rows: int
    num_sources: int
    num_sources_with_positives: int
    num_sources_with_negatives: int
    num_sources_with_pos_and_neg: int
    max_candidates_per_source: int
    mean_candidates_per_source: float

    @property
    def positive_sources_without_negatives(self) -> int:
        return int(self.num_sources_with_positives - self.num_sources_with_pos_and_neg)


def _load_edges(path: str | None, *, use_mmap: bool) -> np.ndarray | None:
    if path is None:
        return None
    if not Path(path).exists():
        return None
    array = np.load(path, mmap_mode="r" if use_mmap else None)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Edge array at {path} must have shape [N, 2], got {array.shape}")
    if np.asarray(array).dtype == np.int64:
        return array
    return np.asarray(array, dtype=np.int64)


def _splitmix64_vec(values: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 — deterministic per-element hash for sort keys."""
    v = values.astype(np.uint64, copy=True)
    _splitmix64_inplace(v)
    return v


def _splitmix64_inplace(values: np.ndarray) -> None:
    values += np.uint64(0x9E3779B97F4A7C15)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)


def _row_tiebreak_keys(num_rows: int) -> np.ndarray:
    v = np.arange(int(num_rows), dtype=np.uint64)
    _splitmix64_inplace(v)
    return v


def _compute_group_stats(edges: np.ndarray, labels: np.ndarray, *, chunk_rows: int = 1_000_000) -> EvalGroupStats:
    labels_arr = np.asarray(labels, dtype=np.float32).reshape(-1)
    num_rows = int(labels_arr.shape[0])
    chunk_rows = max(1, int(chunk_rows))
    num_positive_rows = 0
    for start in range(0, num_rows, chunk_rows):
        end = min(start + chunk_rows, num_rows)
        num_positive_rows += int(np.count_nonzero(labels_arr[start:end] > 0.5))
    num_negative_rows = int(num_rows - num_positive_rows)
    if num_rows == 0:
        return EvalGroupStats(
            num_rows=0,
            num_positive_rows=0,
            num_negative_rows=0,
            num_sources=0,
            num_sources_with_positives=0,
            num_sources_with_negatives=0,
            num_sources_with_pos_and_neg=0,
            max_candidates_per_source=0,
            mean_candidates_per_source=0.0,
        )

    num_sources = 0
    num_sources_with_positives = 0
    num_sources_with_negatives = 0
    num_sources_with_pos_and_neg = 0
    max_candidates_per_source = 0
    current_src: int | None = None
    current_count = 0
    current_pos = 0

    def finalize_current() -> None:
        nonlocal num_sources
        nonlocal num_sources_with_positives
        nonlocal num_sources_with_negatives
        nonlocal num_sources_with_pos_and_neg
        nonlocal max_candidates_per_source
        nonlocal current_src
        nonlocal current_count
        nonlocal current_pos
        if current_src is None:
            return
        current_neg = int(current_count - current_pos)
        num_sources += 1
        max_candidates_per_source = max(max_candidates_per_source, int(current_count))
        if current_pos > 0:
            num_sources_with_positives += 1
        if current_neg > 0:
            num_sources_with_negatives += 1
        if current_pos > 0 and current_neg > 0:
            num_sources_with_pos_and_neg += 1
        current_src = None
        current_count = 0
        current_pos = 0

    for start in range(0, num_rows, chunk_rows):
        end = min(start + chunk_rows, num_rows)
        srcs = np.asarray(edges[start:end, 0], dtype=np.int64)
        positives = labels_arr[start:end] > 0.5
        group_starts = np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(srcs[1:] != srcs[:-1]).astype(np.int64) + 1,
            ]
        )
        group_ends = np.concatenate([group_starts[1:], np.asarray([end - start], dtype=np.int64)])
        pos_counts = np.add.reduceat(positives.astype(np.int64, copy=False), group_starts)

        for local_start, local_end, pos_count in zip(group_starts, group_ends, pos_counts, strict=False):
            src = int(srcs[int(local_start)])
            group_count = int(local_end - local_start)
            group_pos = int(pos_count)
            if current_src is None:
                current_src = src
                current_count = group_count
                current_pos = group_pos
            elif src == current_src:
                current_count += group_count
                current_pos += group_pos
            else:
                finalize_current()
                current_src = src
                current_count = group_count
                current_pos = group_pos

    finalize_current()
    return EvalGroupStats(
        num_rows=num_rows,
        num_positive_rows=num_positive_rows,
        num_negative_rows=num_negative_rows,
        num_sources=num_sources,
        num_sources_with_positives=num_sources_with_positives,
        num_sources_with_negatives=num_sources_with_negatives,
        num_sources_with_pos_and_neg=num_sources_with_pos_and_neg,
        max_candidates_per_source=max_candidates_per_source,
        mean_candidates_per_source=float(num_rows / num_sources) if num_sources > 0 else 0.0,
    )


def _log_group_stats(split_name: str, stats: EvalGroupStats) -> None:
    LOGGER.info(
        "%s eval groups: rows=%d pos_rows=%d neg_rows=%d sources=%d "
        "pos_sources=%d neg_sources=%d pos_and_neg_sources=%d "
        "pos_sources_without_neg=%d mean_candidates_per_source=%.3f max_candidates_per_source=%d",
        split_name,
        stats.num_rows,
        stats.num_positive_rows,
        stats.num_negative_rows,
        stats.num_sources,
        stats.num_sources_with_positives,
        stats.num_sources_with_negatives,
        stats.num_sources_with_pos_and_neg,
        stats.positive_sources_without_negatives,
        stats.mean_candidates_per_source,
        stats.max_candidates_per_source,
    )
    if stats.num_positive_rows > 0 and stats.num_negative_rows == 0:
        LOGGER.warning(
            "%s eval split has positives but no negatives; source-local ranking metrics "
            "such as MRR/Hits/NDCG will be trivially perfect.",
            split_name,
        )
    elif stats.positive_sources_without_negatives > 0:
        LOGGER.warning(
            "%s eval split has %d/%d positive source groups without any negative candidates; "
            "those groups make source-local MRR/Hits/NDCG artificially high.",
            split_name,
            stats.positive_sources_without_negatives,
            stats.num_sources_with_positives,
        )


class LabeledEdgeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        pos_edges: np.ndarray,
        neg_edges: np.ndarray,
        *,
        split_name: str = "eval",
    ) -> None:
        pos = np.asarray(pos_edges, dtype=np.int64)
        neg = np.asarray(neg_edges, dtype=np.int64)
        all_edges = np.concatenate([pos, neg], axis=0)
        all_labels = np.concatenate(
            [np.ones(pos.shape[0], dtype=np.float32), np.zeros(neg.shape[0], dtype=np.float32)],
            axis=0,
        )
        # Sort by a randomized source key, then by src. Keep every source
        # contiguous for streaming per-source eval, but randomize rows inside
        # each source group so score ties do not inherit pos-then-neg order.
        order = np.lexsort(
            (
                _row_tiebreak_keys(int(all_edges.shape[0])),
                all_edges[:, 0],
                _splitmix64_vec(all_edges[:, 0]),
            )
        )
        self.edges = all_edges[order]
        self.labels = all_labels[order]
        self.group_stats = _compute_group_stats(self.edges, self.labels)
        _log_group_stats(split_name, self.group_stats)

    def __len__(self) -> int:
        return int(self.edges.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(idx)
        return torch.from_numpy(self.edges[index]), torch.tensor(self.labels[index], dtype=torch.float32)


class ContiguousShardSampler(Sampler[int]):
    def __init__(self, dataset_len: int, *, rank: int, world_size: int) -> None:
        self.dataset_len = int(dataset_len)
        shard_bounds = np.linspace(0, self.dataset_len, int(world_size) + 1, dtype=np.int64)
        self.start = int(shard_bounds[int(rank)])
        self.end = int(shard_bounds[int(rank) + 1])

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self) -> int:
        return max(0, self.end - self.start)


def build_eval_loader(
    *,
    dataset: Dataset[Any],
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    device_type: str,
    collate_fn=None,
) -> DataLoader:
    dataset_len = len(dataset)  # type: ignore[arg-type]
    sampler = ContiguousShardSampler(dataset_len, rank=rank, world_size=world_size) if world_size > 1 else None
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=device_type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )


def build_eval_dataset(
    *,
    dataset_config: Any,
    split_name: str,
    pos_edges: np.ndarray,
    neg_edges: np.ndarray,
) -> Dataset[Any]:
    del dataset_config
    return LabeledEdgeDataset(pos_edges, neg_edges, split_name=split_name)


def load_edge_splits(config: Any) -> EdgeSplits:
    """Load val/test positive and negative edges from npy files.

    Required config attributes: `valid_edge_path`, `valid_edge_neg_path`,
    `test_edge_path`, `test_edge_neg_path`. Optional `split_use_mmap`
    (default True).
    """
    use_mmap = bool(getattr(config, "split_use_mmap", True))
    paths = {
        "valid_edge_path": getattr(config, "valid_edge_path", None),
        "valid_edge_neg_path": getattr(config, "valid_edge_neg_path", None),
        "test_edge_path": getattr(config, "test_edge_path", None),
        "test_edge_neg_path": getattr(config, "test_edge_neg_path", None),
    }
    loaded = {name: _load_edges(path, use_mmap=use_mmap) for name, path in paths.items()}
    missing = [name for name, value in loaded.items() if value is None]
    if missing:
        raise ValueError(f"Missing required dataset paths: {missing}")
    return EdgeSplits(
        val_pos=np.asarray(loaded["valid_edge_path"], dtype=np.int64),
        val_neg=np.asarray(loaded["valid_edge_neg_path"], dtype=np.int64),
        test_pos=np.asarray(loaded["test_edge_path"], dtype=np.int64),
        test_neg=np.asarray(loaded["test_edge_neg_path"], dtype=np.int64),
    )


def build_train_positive_dataset(*, dataset_config: Any) -> Dataset[torch.Tensor]:
    return GraphCSRPositiveEdgeDataset(dataset_config)
