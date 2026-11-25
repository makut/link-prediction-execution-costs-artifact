"""Unified link-prediction inference (scoring).

Given a file of ``(src, dst)`` vertex-id pairs and a trained checkpoint, score the
pairs with any of the five LP methods (verse, node2vec, mf, graphsage, seal) and
persist the scores. The scoring math lives in each method's package; this package
only provides the method-agnostic engine (I/O, batching, dispatch).
"""
from __future__ import annotations

from .edge_scorer import EdgeScorer
from .io import load_pairs, write_scores
from .runner import run_scoring, run_scoring_distributed

__all__ = [
    "EdgeScorer",
    "load_pairs",
    "run_scoring",
    "run_scoring_distributed",
    "write_scores",
]
