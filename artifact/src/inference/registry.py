from __future__ import annotations

from typing import Callable

import torch

from .edge_scorer import EdgeScorer


# Methods whose scoring is pure embedding lookup + a head: everything needed is
# inside the checkpoint, so `--config` is optional. Graph methods (graphsage,
# seal) need the training YAML for the graph + sampling/extractor params.
EMBEDDING_METHODS = frozenset({"verse", "node2vec", "mf"})
GRAPH_METHODS = frozenset({"graphsage", "seal"})
METHODS = tuple(sorted(EMBEDDING_METHODS | GRAPH_METHODS))


def _build_embedding_lp(
    *,
    checkpoint: str,
    config_path: str | None,
    device: torch.device,
    embedding_checkpoint_dir: str | None,
    num_workers: int | None = None,
) -> EdgeScorer:
    from embedding_lp.scoring import EmbeddingLPScorer

    return EmbeddingLPScorer.from_checkpoint(
        checkpoint,
        device=device,
        embedding_checkpoint_dir=embedding_checkpoint_dir,
    )


def _build_mf(
    *,
    checkpoint: str,
    config_path: str | None,
    device: torch.device,
    embedding_checkpoint_dir: str | None,
    num_workers: int | None = None,
) -> EdgeScorer:
    from mf_torchrec.scoring import MFScorer

    return MFScorer.from_checkpoint(checkpoint, device=device)


def _build_graphsage(
    *,
    checkpoint: str,
    config_path: str | None,
    device: torch.device,
    embedding_checkpoint_dir: str | None,
    num_workers: int | None = None,
) -> EdgeScorer:
    if config_path is None:
        raise ValueError("graphsage inference requires --config (training YAML) for the graph + sampling params")
    from graphsage_torchrec.scoring import GraphSAGEScorer

    return GraphSAGEScorer.from_checkpoint(
        checkpoint,
        config_path=config_path,
        device=device,
        num_workers=num_workers,
    )


def _build_seal(
    *,
    checkpoint: str,
    config_path: str | None,
    device: torch.device,
    embedding_checkpoint_dir: str | None,
    num_workers: int | None = None,
) -> EdgeScorer:
    if config_path is None:
        raise ValueError("seal inference requires --config (training YAML) for the graph + extractor params")
    from seal.scoring import SEALScorer

    return SEALScorer.from_checkpoint(
        checkpoint, config_path=config_path, device=device, num_workers=num_workers
    )


_Builder = Callable[..., EdgeScorer]

BUILDERS: dict[str, _Builder] = {
    "verse": _build_embedding_lp,
    "node2vec": _build_embedding_lp,
    "mf": _build_mf,
    "graphsage": _build_graphsage,
    "seal": _build_seal,
}


def build_scorer(
    method: str,
    *,
    checkpoint: str,
    config_path: str | None,
    device: torch.device,
    embedding_checkpoint_dir: str | None = None,
    num_workers: int | None = None,
) -> EdgeScorer:
    """Construct the EdgeScorer for ``method``, lazily importing its adapter.

    ``num_workers`` parallelises CPU-side work (currently SEAL subgraph
    extraction) across DataLoader workers; other methods ignore it.
    """
    if method not in BUILDERS:
        raise ValueError(f"Unknown method {method!r}; supported: {', '.join(METHODS)}")
    return BUILDERS[method](
        checkpoint=checkpoint,
        config_path=config_path,
        device=device,
        embedding_checkpoint_dir=embedding_checkpoint_dir,
        num_workers=num_workers,
    )
