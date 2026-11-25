"""Shared link-prediction training objectives.

These are method-agnostic: they consume per-edge embeddings (already produced by
an MF id-lookup, a GraphSAGE encoder, or any other embedding method) plus the
edge labels, so every embedding-based LP trainer can share one implementation of
the in-batch sampled-softmax objective.
"""
from __future__ import annotations

import torch
from torch import nn


class InBatchSoftmaxLoss(nn.Module):
    """Sampled-softmax over in-batch (and optional extra hard) destinations.

    For a group of ``B`` positive pairs ``(src_i, dst_i)``, the destinations of
    every other pair in the group — plus any supplied extra (hard) negatives —
    form a shared candidate pool. Row ``i`` is trained to concentrate its mass on
    its own destination (column ``i``) via cross-entropy. This gives ``O(B)``
    negatives per positive at near-zero extra cost, versus one sampled negative.

    Accidental hits are masked: if a candidate's node id equals the row's own
    positive destination id (a duplicate destination elsewhere in the batch), that
    column is set to ``-inf`` so a true positive is never used as a negative. The
    row's own target column is always kept.

    Scores are ``src · dst`` (``dot``) or cosine (``cos``, both sides L2-normalised),
    divided by ``temperature`` before the softmax.
    """

    def __init__(self, *, scorer: str, temperature: float = 1.0, mask_accidental_hits: bool = True) -> None:
        super().__init__()
        if scorer not in ("dot", "cos"):
            raise ValueError(f"Unsupported in-batch scorer: {scorer}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.scorer = str(scorer)
        self.temperature = float(temperature)
        self.mask_accidental_hits = bool(mask_accidental_hits)

    def forward(
        self,
        *,
        src_emb: torch.Tensor,
        pos_dst_emb: torch.Tensor,
        pos_dst_ids: torch.Tensor,
        neg_dst_emb: torch.Tensor | None = None,
        neg_dst_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        has_neg = neg_dst_emb is not None and neg_dst_emb.numel() > 0
        if self.scorer == "cos":
            src_emb = nn.functional.normalize(src_emb, p=2, dim=-1)
            pos_dst_emb = nn.functional.normalize(pos_dst_emb, p=2, dim=-1)
            if has_neg:
                neg_dst_emb = nn.functional.normalize(neg_dst_emb, p=2, dim=-1)

        pos_dst_ids = pos_dst_ids.reshape(-1)
        if has_neg:
            cand_emb = torch.cat([pos_dst_emb, neg_dst_emb], dim=0)
            cand_ids = torch.cat([pos_dst_ids, neg_dst_ids.reshape(-1)], dim=0)
        else:
            cand_emb = pos_dst_emb
            cand_ids = pos_dst_ids

        logits = (src_emb @ cand_emb.t()) / self.temperature  # [B, C]
        num_pos = int(src_emb.shape[0])
        targets = torch.arange(num_pos, device=logits.device)
        if self.mask_accidental_hits:
            same = cand_ids.unsqueeze(0) == pos_dst_ids.unsqueeze(1)  # [B, C]
            is_target = torch.zeros_like(same)
            is_target[targets, targets] = True
            logits = logits.masked_fill(same & ~is_target, float("-inf"))
        return nn.functional.cross_entropy(logits, targets)


def in_batch_softmax_loss_from_edges(
    loss_module: InBatchSoftmaxLoss,
    *,
    src_emb: torch.Tensor,
    dst_emb: torch.Tensor,
    dst_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Run the in-batch softmax over a single batch of labelled edges.

    Splits per-edge embeddings into positives/negatives by ``labels``; positives
    define the queries and target destinations, negatives become shared extra
    hard-negative columns. ``src_emb``/``dst_emb`` are ``[E, D]`` per-edge
    embeddings (gathered however the method produces them), ``dst_ids`` are the
    ``[E]`` global destination ids (for accidental-hit masking), and ``labels``
    are ``[E]`` in ``{0, 1}``. Returns a scalar loss over the whole batch as one
    in-batch group — suitable when the batch is small enough for a single ``[P, C]``
    score matrix; large-batch methods should chunk into groups themselves.
    """
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    has_neg = bool(neg_mask.any())
    return loss_module(
        src_emb=src_emb[pos_mask],
        pos_dst_emb=dst_emb[pos_mask],
        pos_dst_ids=dst_ids[pos_mask],
        neg_dst_emb=dst_emb[neg_mask] if has_neg else None,
        neg_dst_ids=dst_ids[neg_mask] if has_neg else None,
    )
