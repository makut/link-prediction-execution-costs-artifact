"""Unified CLI: score a file of (src, dst) pairs with a trained LP checkpoint.

Single process:
    python -m inference.run_inference \
        --method node2vec \
        --checkpoint results/node2vec-lp/checkpoints/best.pt \
        --pairs pairs/pairs.npy \
        --output-dir scores/node2vec

Multi-GPU, sharded (no checkpoint consolidation needed) — launch with the SAME
world size used for training so each rank loads its embedding-table shard and the
lookups become collective. Pairs are split evenly across ranks and the scores are
gathered to rank 0, which writes scores.npy / manifest.json:
    torchrun --nproc_per_node=4 -m inference.run_inference \
        --method node2vec \
        --checkpoint results/node2vec-lp/checkpoints/best.pt \
        --pairs pairs/pairs.npy \
        --output-dir scores/node2vec

Embedding methods (verse, node2vec, mf) need only the checkpoint. Graph methods
(graphsage, seal) additionally need --config (the training YAML) for the graph
and sampling/extractor parameters. Do not pass --device under torchrun: each rank
auto-selects cuda:LOCAL_RANK.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

from dataset_utils import distributed_is_initialized, infer_device, maybe_init_distributed

from .io import load_pairs, write_scores
from .registry import METHODS, build_scorer
from .runner import run_scoring_distributed

LOGGER = logging.getLogger("inference")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score (src, dst) pairs with a trained LP model")
    parser.add_argument("--method", required=True, choices=METHODS, help="LP method")
    parser.add_argument("--checkpoint", required=True, type=str, help="Checkpoint file or dir for the method")
    parser.add_argument("--pairs", required=True, type=Path, help="Input pairs.npy ([N, 2] int64)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where to write scores.npy + manifest.json")
    parser.add_argument("--config", type=str, default=None, help="Training YAML (required for graphsage/seal)")
    parser.add_argument(
        "--embedding-checkpoint-dir",
        type=str,
        default=None,
        help="Override the embedding-table checkpoint dir referenced by an embedding_lp classifier checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=262_144, help="Edges scored per forward pass")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers for CPU-side work (GraphSAGE sampling / SEAL subgraph extraction); "
        "defaults to the training config's val_num_workers / num_workers when supported",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device (default: auto from LOCAL_RANK)")
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="torch.distributed backend used when launched under torchrun (WORLD_SIZE > 1)",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Under torchrun (WORLD_SIZE > 1) this initialises the process group so the
    # embedding store loads its per-rank shard and lookups become collective; in a
    # single process it is a no-op returning (0, 1).
    rank, world_size = maybe_init_distributed(args.backend)
    device = infer_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    LOGGER.info(
        "Scoring method=%s checkpoint=%s device=%s rank=%d world_size=%d",
        args.method,
        args.checkpoint,
        device,
        rank,
        world_size,
    )

    pairs = load_pairs(args.pairs)
    LOGGER.info("Loaded %d pairs from %s", int(pairs.shape[0]), args.pairs)

    # Built after distributed init so torchrec shards the table over the WORLD group.
    scorer = build_scorer(
        args.method,
        checkpoint=str(args.checkpoint),
        config_path=args.config,
        device=device,
        embedding_checkpoint_dir=args.embedding_checkpoint_dir,
        num_workers=args.num_workers,
    )

    t0 = time.time()
    scores = run_scoring_distributed(
        scorer,
        pairs,
        rank=rank,
        world_size=world_size,
        batch_size=int(args.batch_size),
        show_progress=not args.no_progress and rank == 0,
    )
    elapsed = time.time() - t0

    if rank == 0:
        manifest = {
            "method": args.method,
            "checkpoint": str(args.checkpoint),
            "config": args.config,
            "embedding_checkpoint_dir": args.embedding_checkpoint_dir,
            "pairs_path": str(args.pairs),
            "num_pairs": int(pairs.shape[0]),
            "batch_size": int(args.batch_size),
            "num_workers": args.num_workers,
            "device": str(device),
            "world_size": world_size,
            "score_type": "probability",
            "elapsed_seconds": elapsed,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        scores_path, manifest_path = write_scores(args.output_dir, scores=scores, manifest=manifest)
        LOGGER.info(
            "Wrote %d scores to %s (manifest: %s) in %.1fs",
            int(scores.shape[0]),
            scores_path,
            manifest_path,
            elapsed,
        )
        print(json.dumps(manifest, indent=2))

    if distributed_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
