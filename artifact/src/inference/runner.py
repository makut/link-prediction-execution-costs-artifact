from __future__ import annotations

import numpy as np

from .edge_scorer import EdgeScorer


def per_rank_count(num_pairs: int, world_size: int) -> int:
    """Padded per-rank pair count so every rank runs the same number of batches.

    Sharded embedding lookups inside ``score_batch`` are collectives: every rank
    must issue the same number of calls or the all-to-all deadlocks. We give each
    rank ``ceil(N / world_size)`` rows (padding the tail) so batch counts match.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return (int(num_pairs) + int(world_size) - 1) // int(world_size)


def shard_pairs_for_rank(pairs: np.ndarray, *, rank: int, world_size: int) -> np.ndarray:
    """Return ``rank``'s contiguous, padded ``[per_rank, 2]`` slice of ``pairs``.

    Rank ``r`` owns padded global indices ``[r*per_rank, (r+1)*per_rank)``; indices
    past the real end are filled with row 0 (scored, then discarded after gather).
    Concatenating every rank's slice in rank order reproduces the original pair
    order, so the gathered scores need only be truncated to ``N``.
    """
    pairs_arr = np.asarray(pairs)
    if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
        raise ValueError(f"pairs must have shape [N, 2], got {pairs_arr.shape}")
    num_pairs = int(pairs_arr.shape[0])
    per_rank = per_rank_count(num_pairs, world_size)
    start = int(rank) * per_rank
    idx = np.arange(start, start + per_rank)
    safe_idx = np.where(idx < num_pairs, idx, 0)
    return np.asarray(pairs_arr[safe_idx], dtype=np.int64)


def run_scoring(
    scorer: EdgeScorer,
    pairs: np.ndarray,
    *,
    batch_size: int = 262_144,
    show_progress: bool = True,
) -> np.ndarray:
    """Score every pair, returning a ``[N]`` float32 array aligned to ``pairs``.

    Delegates to ``scorer.score_pairs``: the default implementation chunks into
    ``batch_size`` ``score_batch`` calls, while methods that own a worker pool
    (e.g. SEAL) override it to run a single long-lived DataLoader over all pairs.
    """
    return scorer.score_pairs(pairs, batch_size=batch_size, show_progress=show_progress)


def run_scoring_distributed(
    scorer: EdgeScorer,
    pairs: np.ndarray,
    *,
    rank: int,
    world_size: int,
    batch_size: int = 262_144,
    show_progress: bool = True,
) -> np.ndarray:
    """Score ``pairs`` across ``world_size`` ranks holding a sharded embedding table.

    Each rank scores its own padded contiguous slice (see ``shard_pairs_for_rank``)
    so all ranks score the same padded row count — required because sharded
    embedding lookups inside the scorer can be collective all-to-all operations.
    Per-rank scores are gathered to rank 0, which returns the full ``[N]`` float32
    array in the original ``pairs`` order; every other rank returns an empty array.

    For ``world_size == 1`` this is exactly ``run_scoring``.
    """
    if world_size <= 1:
        return run_scoring(scorer, pairs, batch_size=batch_size, show_progress=show_progress)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Imported lazily so the single-process path never needs torch.distributed.
    from dataset_utils import gather_eval_arrays

    pairs_arr = np.asarray(pairs)
    if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
        raise ValueError(f"pairs must have shape [N, 2], got {pairs_arr.shape}")
    num_pairs = int(pairs_arr.shape[0])

    local_pairs = shard_pairs_for_rank(pairs_arr, rank=rank, world_size=world_size)
    per_rank = int(local_pairs.shape[0])
    local_scores = np.asarray(
        scorer.score_pairs(
            local_pairs,
            batch_size=int(batch_size),
            show_progress=show_progress and rank == 0,
        ),
        dtype=np.float32,
    ).reshape(-1)
    if local_scores.shape[0] != per_rank:
        raise ValueError(f"score_pairs returned {local_scores.shape[0]} scores for {per_rank} edges")

    # gather_eval_arrays concatenates rank-0..rank-(W-1) in rank order, which is
    # exactly the padded global order; truncate the padding back to N.
    (gathered,) = gather_eval_arrays(local_scores)
    if rank != 0:
        return np.empty(0, dtype=np.float32)
    full = np.asarray(gathered, dtype=np.float32).reshape(-1)[:num_pairs]
    if full.shape[0] != num_pairs:
        raise ValueError(
            f"gathered {full.shape[0]} scores after truncation, expected {num_pairs}"
        )
    return full
