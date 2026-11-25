"""Streaming Citation2 ranks, including OGB's midpoint handling of tied scores."""
from __future__ import annotations

import numpy as np


class Citation2MetricAccumulator:
    def __init__(self, expected_negatives: int = 1000) -> None:
        self.expected_negatives = int(expected_negatives)
        self._source = None
        self._labels = []
        self._scores = []
        self._count = 0
        self._mrr = 0.0
        self._hits = {k: 0 for k in (1, 10, 50, 100)}
        self._closed_sources = set()

    def _finalize(self) -> None:
        if self._source is None:
            return
        labels = np.concatenate(self._labels)
        scores = np.concatenate(self._scores)
        positive = scores[labels == 1]
        negative = scores[labels == 0]
        if positive.size != 1 or negative.size != self.expected_negatives or labels.size != positive.size + negative.size:
            raise ValueError(f"Citation2 source {self._source}: expected 1 positive and {self.expected_negatives} negatives; got {positive.size}/{negative.size}")
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite Citation2 score for source {self._source}")
        rank = 1.0 + 0.5 * (int(np.sum(negative > positive[0])) + int(np.sum(negative >= positive[0])))
        self._mrr += 1.0 / rank
        self._count += 1
        for k in self._hits:
            self._hits[k] += int(rank <= k)
        self._closed_sources.add(self._source)
        self._labels, self._scores = [], []
        self._source = None

    def update(self, *, srcs, labels, scores) -> None:
        srcs, labels, scores = np.asarray(srcs), np.asarray(labels), np.asarray(scores)
        if not (len(srcs) == len(labels) == len(scores)):
            raise ValueError("Citation2 metric arrays differ in length")
        starts = np.r_[0, np.flatnonzero(srcs[1:] != srcs[:-1]) + 1]
        for start, end in zip(starts, np.r_[starts[1:], len(srcs)]):
            if start == end:
                continue
            source = int(srcs[start])
            if self._source != source:
                self._finalize()
                if source in self._closed_sources:
                    raise ValueError("Citation2 sources must be contiguous complete groups")
                self._source = source
            self._labels.append(labels[start:end].copy())
            self._scores.append(scores[start:end].copy())

    def compute(self) -> dict[str, float]:
        self._finalize()
        if not self._count:
            raise ValueError("No complete Citation2 source groups scored")
        return {"ogb_mrr": self._mrr / self._count, "ogb_num_sources": float(self._count),
                **{f"ogb_hits@{k}": value / self._count for k, value in self._hits.items()}}
