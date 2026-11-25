"""Inference adapter for SEAL LP.

Rebuilds the trained SEAL model and reuses the existing per-pair k-hop subgraph
extraction (`SEALSubgraphExtractor` / `SEALEvalCollator`) and the model factory
(`build_model`) + checkpoint loader (`_load_seal_checkpoint`) from
`seal.seal_link_pred`. The SortPooling `k` recomputes deterministically from the
train split (seed=0), matching training, so existing `best_model.pth` checkpoints
load without any format change. Needs the training YAML for the graph + extractor
parameters.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset_utils import build_negative_sampler, build_train_positive_dataset
from inference import EdgeScorer

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore

from .config import SEALConfig, load_config, resolve_train_loader_config
from .data_bridge import SEALEvalCollator, SEALSubgraphExtractor, SEALTrainDatasetFacade
from .graph_csr_dataset import load_graph_csr_data


def make_subgraph_extractor(config: SEALConfig, data) -> SEALSubgraphExtractor:
    seal = config.seal
    return SEALSubgraphExtractor(
        graph_data=data,
        num_hops=int(seal.num_hops),
        node_label=str(seal.node_label),
        ratio_per_hop=float(seal.ratio_per_hop),
        max_nodes_per_hop=(None if seal.max_nodes_per_hop is None else int(seal.max_nodes_per_hop)),
        directed=bool(getattr(seal, "directed", False)),
        per_vertex_oversample=float(seal.per_vertex_oversample),
        graph_csr_use_per_vertex_sampling=bool(seal.graph_csr_use_per_vertex_sampling),
        graph_csr_use_pairwise_subgraph=seal.graph_csr_use_pairwise_subgraph,
        pairwise_auto_ratio=float(getattr(seal, "pairwise_auto_ratio", 1.0)),
    )


class _PairDataset(Dataset):
    """Order-preserving view over a ``[B, 2]`` pairs array for the eval collator.

    Yields ``(edge, dummy_label)`` tuples in the exact input row order so the
    DataLoader (``shuffle=False``, no sampler) returns scores aligned to ``pairs``.
    The label is a placeholder — ``SEALEvalCollator`` only stashes it on the graph
    and it does not affect scoring.
    """

    def __init__(self, edges: np.ndarray) -> None:
        self.edges = np.ascontiguousarray(edges, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.edges.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.edges[int(idx)]), torch.tensor(0.0, dtype=torch.float32)


def _num_features(data) -> int:
    x_attr = getattr(data, "x", None)
    if x_attr is None:
        return 0
    return int(x_attr.shape[-1]) if getattr(x_attr, "ndim", 1) > 1 else 1


class SEALScorer(EdgeScorer):
    def __init__(
        self,
        *,
        model,
        node_embedding,
        extractor: SEALSubgraphExtractor,
        use_feature: bool,
        use_edge_weight: bool,
        device: torch.device,
        num_workers: int = 0,
        eval_batch_size: int = 1024,
    ) -> None:
        self.model = model.eval()
        self.node_embedding = node_embedding
        self.collator = SEALEvalCollator(extractor)
        self.use_feature = bool(use_feature)
        self.use_edge_weight = bool(use_edge_weight)
        self.device = device
        # Subgraph extraction (the CPU bottleneck) is parallelised across these
        # DataLoader workers, mirroring training-time eval; the model forward
        # still runs on this process. eval_batch_size bounds the per-forward
        # subgraph count, independent of the outer inference --batch-size.
        self.num_workers = max(0, int(num_workers))
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.pin_memory = device.type == "cuda"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        config_path: str,
        device: torch.device,
        num_workers: int | None = None,
    ) -> "SEALScorer":
        # Reuse the model factory + checkpoint loader from the trainer module.
        from .seal_link_pred import _load_seal_checkpoint, build_model

        config = load_config(config_path)
        data, _ = load_graph_csr_data(
            config.dataset.graph_csr_root,
            file_endian=config.dataset.graph_csr_file_endian,
            use_mmap=config.dataset.graph_csr_use_mmap,
            allow_non_native=config.dataset.graph_csr_allow_non_native,
            chunk_bytes=config.dataset.graph_csr_chunk_bytes,
            node_features_path=config.dataset.node_features_path,
            directed_out_root=config.dataset.directed_out_csr_root,
            directed_in_root=config.dataset.directed_in_csr_root,
        )
        config.train_loader = resolve_train_loader_config(
            dataset=config.dataset,
            train_loader=config.train_loader,
            num_nodes=int(data.num_nodes),
        )
        extractor = make_subgraph_extractor(config, data)

        # Reproduce the train-dataset facade so DGCNN's SortPooling k matches
        # the value computed during training (deterministic, seed=0).
        positive_dataset = build_train_positive_dataset(dataset_config=config.train_loader.positive_edges)
        negative_sampler = build_negative_sampler(
            config.train_loader.negative_sampling,
            config.train_loader.positive_edges,
        )
        train_dataset = SEALTrainDatasetFacade(
            positive_dataset=positive_dataset,
            negative_sampler=negative_sampler,
            extractor=extractor,
            train_loader=None,
            num_features=_num_features(data),
        )

        node_embedding = None
        if config.model.train_node_embedding:
            node_embedding = torch.nn.Embedding(data.num_nodes, config.model.hidden_channels).to(device)

        model = build_model(
            config.model,
            train_dataset=train_dataset,
            node_embedding=node_embedding,
            device=device,
        )
        _load_seal_checkpoint(Path(checkpoint), model, node_embedding, device)

        # Default to the same worker count training uses for eval: explicit
        # val_num_workers, else the train loader's num_workers. An explicit
        # override (CLI) wins over both.
        resolved_workers = num_workers
        if resolved_workers is None:
            resolved_workers = config.runtime.val_num_workers
        if resolved_workers is None:
            resolved_workers = config.train_loader.num_workers

        return cls(
            model=model,
            node_embedding=node_embedding,
            extractor=extractor,
            use_feature=config.model.use_feature,
            use_edge_weight=config.model.use_edge_weight,
            device=device,
            num_workers=int(resolved_workers),
            eval_batch_size=int(config.train_loader.batch_size),
        )

    def _forward_batch(self, batch) -> np.ndarray:
        batch = batch.to(self.device)
        x = batch.x if self.use_feature else None
        edge_weight = batch.edge_weight if self.use_edge_weight else None
        node_id = batch.node_id if self.node_embedding is not None else None
        logits = self.model(batch.z, batch.edge_index, batch.batch, x, edge_weight, node_id).float()
        # Export RAW logits (no sigmoid): keeps ranking consistent with the eval
        # fix (apply_sigmoid=False) and avoids sigmoid saturating large logits to
        # 1.0 ties. sigmoid(logits) is recoverable downstream if probabilities are
        # ever needed; logits are strictly more informative.
        return logits.view(-1).detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def score_pairs(
        self,
        pairs: np.ndarray,
        *,
        batch_size: int = 262_144,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Score all pairs through a single long-lived multi-worker DataLoader.

        Mirrors training-time eval (``run_eval`` over one ``build_eval_loader``):
        the worker pool forks once and stays warm/prefetching for the whole pass,
        so k-hop extraction overlaps the GPU forward end-to-end. This overrides the
        base per-``batch_size``-chunk loop, which would otherwise spawn and tear
        down a worker pool (and re-warm numba in each fresh worker) per chunk. The
        engine's ``batch_size`` is intentionally ignored — the per-forward subgraph
        count is ``eval_batch_size`` (the training eval batch size).
        """
        pairs_arr = np.asarray(pairs, dtype=np.int64)
        if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
            raise ValueError(f"pairs must have shape [N, 2], got {pairs_arr.shape}")
        num_pairs = int(pairs_arr.shape[0])
        if num_pairs == 0:
            return np.empty(0, dtype=np.float32)

        # shuffle=False + no sampler: the DataLoader yields batches in index order
        # (and reassembles multi-worker results in order), so scores stay aligned
        # to the input pairs.
        loader = DataLoader(
            _PairDataset(pairs_arr),
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )
        iterator = loader
        if show_progress and tqdm is not None:
            iterator = tqdm(loader, desc="scoring", unit="batch", leave=False)

        scores = np.empty(num_pairs, dtype=np.float32)
        pos = 0
        for batch in iterator:
            chunk = self._forward_batch(batch)
            scores[pos : pos + chunk.shape[0]] = chunk
            pos += int(chunk.shape[0])
        if pos != num_pairs:
            raise ValueError(f"scored {pos} pairs, expected {num_pairs}")
        return scores

    @torch.no_grad()
    def score_batch(self, edges: np.ndarray) -> np.ndarray:
        # Single-batch entry point (kept for the EdgeScorer contract); reuses the
        # same extraction path as the full-pass score_pairs.
        return self.score_pairs(np.asarray(edges, dtype=np.int64), show_progress=False)
