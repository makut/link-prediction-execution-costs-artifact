from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore


class EdgeScorer(ABC):
    """Method-agnostic edge-scoring interface.

    A scorer wraps a single trained model (loaded once) and turns a batch of
    ``(src, dst)`` pairs into link-existence probabilities. Every method's
    ``scoring`` module ships a thin subclass whose ``score_batch`` reuses the
    method's existing forward pass — no scoring math is reimplemented here.
    """

    @abstractmethod
    def score_batch(self, edges: np.ndarray) -> np.ndarray:
        """Score one batch of edges.

        Args:
            edges: ``[B, 2]`` int64 array of ``(src, dst)`` vertex ids.

        Returns:
            ``[B]`` float32 array of probabilities in ``[0, 1]``, aligned to the
            input row order.
        """
        raise NotImplementedError

    def score_pairs(
        self,
        pairs: np.ndarray,
        *,
        batch_size: int = 262_144,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Score an entire ``[N, 2]`` pairs array, returning ``[N]`` float32 in order.

        The default implementation chunks ``pairs`` into ``batch_size`` slices and
        delegates each to ``score_batch`` — fully method-agnostic. Methods with
        heavy per-pair CPU work (e.g. SEAL subgraph extraction) override this to
        drive a single long-lived DataLoader over all pairs, so the worker pool is
        created once instead of per outer batch.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        pairs_arr = np.asarray(pairs)
        if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
            raise ValueError(f"pairs must have shape [N, 2], got {pairs_arr.shape}")
        num_pairs = int(pairs_arr.shape[0])
        scores = np.empty(num_pairs, dtype=np.float32)

        starts = range(0, num_pairs, int(batch_size))
        if show_progress and tqdm is not None and num_pairs > 0:
            starts = tqdm(starts, desc="scoring", unit="batch", leave=False)

        for start in starts:
            end = min(start + int(batch_size), num_pairs)
            batch = np.asarray(pairs_arr[start:end], dtype=np.int64)
            batch_scores = np.asarray(self.score_batch(batch), dtype=np.float32).reshape(-1)
            if batch_scores.shape[0] != end - start:
                raise ValueError(
                    f"score_batch returned {batch_scores.shape[0]} scores for {end - start} edges"
                )
            scores[start:end] = batch_scores
        return scores
