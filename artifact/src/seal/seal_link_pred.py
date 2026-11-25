"""SEAL link prediction trainer (config-driven, graph_csr only).

Single entry point: `python -m seal.seal_link_pred --config path/to/config.yaml`.
Everything else (model, training, eval cadence, TB) comes from YAML.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import resource
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import SparseEfficiencyWarning
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", SparseEfficiencyWarning)

THIS_DIR = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from dataset_utils import (
    EarlyStopping,
    PerSourceMetricAccumulator,
    build_eval_dataset,
    build_eval_loader,
    build_negative_sampler,
    build_train_loader,
    build_train_positive_dataset,
    load_edge_splits,
    metric_improved,
)

try:
    from .config import SEALConfig, load_config, resolve_train_loader_config
    from .data_bridge import (
        SEALBatchTransform,
        SEALEvalCollator,
        SEALSubgraphExtractor,
        SEALTrainDatasetFacade,
    )
    from .graph_csr_dataset import load_graph_csr_data
    from .models import DGCNN, GCN, GIN, SAGE
    from .ogb_metrics import Citation2MetricAccumulator
    from .reproducibility import configure_seeded_loader, model_state_sha256, seed_everything
except ImportError:  # script-style invocation
    from config import SEALConfig, load_config, resolve_train_loader_config
    from data_bridge import (
        SEALBatchTransform,
        SEALEvalCollator,
        SEALSubgraphExtractor,
        SEALTrainDatasetFacade,
    )
    from graph_csr_dataset import load_graph_csr_data
    from models import DGCNN, GCN, GIN, SAGE
    from ogb_metrics import Citation2MetricAccumulator
    from reproducibility import configure_seeded_loader, model_state_sha256, seed_everything


# ---------------------------------------------------------------------------
# Diagnostic logging helpers
# ---------------------------------------------------------------------------

def _rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    return rss_bytes / (1024 ** 3)


def _linux_proc_mem_gb(field: str) -> float | None:
    status_path = "/proc/self/status"
    if not osp.exists(status_path):
        return None
    with open(status_path, "r") as f:
        for line in f:
            if line.startswith(field + ":"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024 / (1024 ** 3)
    return None


def _format_mem() -> str:
    rss_gb = _linux_proc_mem_gb("VmRSS")
    hwm_gb = _linux_proc_mem_gb("VmHWM")
    if rss_gb is not None:
        return f"vmrss={rss_gb:.2f} GB hwm={hwm_gb:.2f} GB"
    return f"rss={_rss_gb():.2f} GB"


def _format_cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "cuda=off"
    try:
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        return (
            f"cuda_alloc={alloc:.2f} GB cuda_reserved={reserved:.2f} GB "
            f"cuda_max_alloc={max_alloc:.2f} GB cuda_max_reserved={max_reserved:.2f} GB"
        )
    except Exception as exc:
        return f"cuda_mem_unavailable({exc})"


def log_preprocessing_step(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PREPROC {timestamp}] {message} | {_format_mem()}", flush=True)


def log_runtime_step(tag: str, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{tag} {timestamp}] {message} | {_format_mem()} | {_format_cuda_mem()}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Debug env flags (off by default)
# ---------------------------------------------------------------------------

TRAIN_DEBUG = os.environ.get("SEAL_TRAIN_DEBUG", "0") == "1"
TRAIN_LOG_FIRST_N = int(os.environ.get("SEAL_TRAIN_LOG_FIRST_N", "3"))
TRAIN_SLOW_BATCH_SEC = float(os.environ.get("SEAL_TRAIN_SLOW_BATCH_SEC", "10.0"))
EVAL_DEBUG = os.environ.get("SEAL_TEST_DEBUG", "0") == "1"
EVAL_LOG_FIRST_N = int(os.environ.get("SEAL_TEST_LOG_FIRST_N", "2"))
EVAL_SLOW_BATCH_SEC = float(os.environ.get("SEAL_TEST_SLOW_BATCH_SEC", "10.0"))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    model_cfg,
    *,
    train_dataset,
    node_embedding,
    device: torch.device,
):
    name = model_cfg.name
    common = dict(
        hidden_channels=model_cfg.hidden_channels,
        num_layers=model_cfg.num_layers,
        max_z=model_cfg.max_z,
    )
    if name == "DGCNN":
        model = DGCNN(
            **common,
            k=model_cfg.sortpool_k,
            train_dataset=train_dataset,
            dynamic_train=True,
            use_feature=model_cfg.use_feature,
            node_embedding=node_embedding,
        )
    elif name == "SAGE":
        model = SAGE(
            **common,
            train_dataset=train_dataset,
            use_feature=model_cfg.use_feature,
            node_embedding=node_embedding,
        )
    elif name == "GCN":
        model = GCN(
            **common,
            train_dataset=train_dataset,
            use_feature=model_cfg.use_feature,
            node_embedding=node_embedding,
        )
    elif name == "GIN":
        model = GIN(
            **common,
            train_dataset=train_dataset,
            use_feature=model_cfg.use_feature,
            node_embedding=node_embedding,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)


def _seal_checkpoint_payload(model, node_embedding) -> dict:
    payload = {
        "format": "seal_checkpoint_v2",
        "model": model.state_dict(),
    }
    if node_embedding is not None:
        payload["node_embedding"] = node_embedding.state_dict()
    return payload


def _load_seal_checkpoint(path: Path, model, node_embedding, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        model.load_state_dict(payload["model"])
        node_embedding_state = payload.get("node_embedding")
        if node_embedding is not None and node_embedding_state is not None:
            node_embedding.load_state_dict(node_embedding_state)
        return payload
    model.load_state_dict(payload)
    return {"format": "legacy_model_state_dict"}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(
    model,
    loader,
    *,
    device: torch.device,
    use_feature: bool,
    use_edge_weight: bool,
    node_embedding,
    max_batches: int | None,
    tag: str,
    metrics: tuple[str, ...] = (
        "roc_auc@k",
        "hits@k",
        "recall@k",
        "ndcg@k",
        "mrr",
        "mrr@k",
        "num_sources",
        "num_sources_with_positives",
    ),
    metrics_at_k: tuple[int, ...] = (10, 50, 100),
    dump_path: Path | None = None,
    citation2_expected_negatives: int | None = None,
) -> dict[str, float]:
    """Eval `model` over `loader`, return metrics dict.

    If `dump_path` is given, every scored row (src, dst, label, raw logit) is
    kept and written to `dump_path` (npz) at the end.
    """
    model.eval()
    if citation2_expected_negatives is not None and max_batches is not None:
        raise ValueError("Citation2 OGB metrics require complete source groups; prepare a subset instead of max_batches")
    ogb_accumulator = (Citation2MetricAccumulator(citation2_expected_negatives)
                       if citation2_expected_negatives is not None else None)
    dump_src: list[np.ndarray] = []
    dump_dst: list[np.ndarray] = []
    dump_lab: list[np.ndarray] = []
    dump_sco: list[np.ndarray] = []
    # Rank on raw logits, NOT clipped sigmoid. All metrics here are order-only
    # and sigmoid is monotonic, so this is identical when logits are small. But
    # the model trains with BCEWithLogitsLoss and emits unbounded logits; once
    # they grow (>~37) sigmoid() (clipped to +/-40 in metrics.py) rounds them all
    # to exactly 1.0 -> spurious ties -> ranking collapses to chance and hides
    # real signal. Ranking on logits avoids that saturation entirely.
    accumulator = PerSourceMetricAccumulator(
        metrics=metrics,
        metrics_at_k=metrics_at_k,
        apply_sigmoid=False,
    )

    last_t = time.perf_counter()
    saw_examples = False
    truncated = False
    for batch_idx, data in enumerate(tqdm(loader, ncols=70, desc=tag)):
        if max_batches is not None and batch_idx >= max_batches:
            log_runtime_step(tag, f"Reached max_batches={max_batches}, stopping eval loop")
            truncated = True
            break

        t0 = time.perf_counter()
        fetch_sec = t0 - last_t
        data = data.to(device)
        x = data.x if use_feature else None
        edge_weight = data.edge_weight if use_edge_weight else None
        node_id = data.node_id if node_embedding is not None else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        if hasattr(data, "edge_src") and data.edge_src is not None:
            srcs_np = data.edge_src.view(-1).cpu().numpy().astype(np.int64, copy=False)
            labels_np = data.y.view(-1).cpu().numpy().astype(np.int64, copy=False)
            scores_np = logits.view(-1).detach().cpu().numpy().astype(np.float32, copy=False)
            accumulator.update(srcs=srcs_np, labels=labels_np, scores=scores_np)
            if ogb_accumulator is not None:
                ogb_accumulator.update(srcs=srcs_np, labels=labels_np, scores=scores_np)
            if dump_path is not None:
                dump_src.append(srcs_np.copy())
                dump_dst.append(data.edge_dst.view(-1).cpu().numpy().astype(np.int64, copy=True))
                dump_lab.append(labels_np.astype(np.int8, copy=True))
                dump_sco.append(scores_np.copy())
            saw_examples = True

        batch_sec = time.perf_counter() - t0
        last_t = time.perf_counter()
        if EVAL_DEBUG or batch_idx < EVAL_LOG_FIRST_N or batch_sec >= EVAL_SLOW_BATCH_SEC:
            log_runtime_step(
                tag,
                f"Batch {batch_idx} done total={batch_sec:.3f}s fetch={fetch_sec:.3f}s "
                f"graphs={data.num_graphs} nodes={data.num_nodes} edges={data.edge_index.size(1)}",
            )

    if not saw_examples:
        return {}
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dump_path,
            src=np.concatenate(dump_src),
            dst=np.concatenate(dump_dst),
            label=np.concatenate(dump_lab),
            logit=np.concatenate(dump_sco),
        )
        log_runtime_step(tag, f"Dumped {sum(a.size for a in dump_src)} scored rows to {dump_path}")
    if truncated:
        accumulator.drop_last_src = True
    result = accumulator.compute()
    if ogb_accumulator is not None:
        result.update(ogb_accumulator.compute())
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_run(
    *,
    run: int,
    config: SEALConfig,
    train_dataset,
    train_loader,
    val_loader,
    test_loader,
    data,
    device: torch.device,
    output_dir: Path,
    writer,
    log_file: Path,
) -> dict[str, float]:
    """Train one run, return final test metrics dict."""
    run_dir = output_dir / f"run{run + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_dir / "best_model.pth"
    run_seed = None if config.training.seed is None else int(config.training.seed) + run
    if run_seed is not None:
        seed_everything(run_seed)
        for offset, loader in enumerate((train_loader, val_loader, test_loader), start=1):
            configure_seeded_loader(loader, run_seed + offset * 100_003)
        log_runtime_step("SEED", f"Run {run + 1} seed={run_seed}; CUDA bitwise determinism is not guaranteed")
        (run_dir / "seed_manifest.json").write_text(json.dumps({
            "seed": run_seed, "train_order_seed": config.train_loader.seed,
            "negative_sampling_seed": config.train_loader.negative_sampling.seed,
            "eval_pair_seed": config.runtime.eval_seed,
            "loader_workers": config.train_loader.num_workers,
            "cuda_bitwise_deterministic": False,
        }, indent=2) + "\n")

    node_embedding = None
    if config.model.train_node_embedding:
        node_embedding = torch.nn.Embedding(data.num_nodes, config.model.hidden_channels).to(device)
        torch.nn.init.xavier_uniform_(node_embedding.weight)

    model = build_model(
        config.model,
        train_dataset=train_dataset,
        node_embedding=node_embedding,
        device=device,
    )
    if run_seed is not None:
        seed_path = run_dir / "seed_manifest.json"
        seed_manifest = json.loads(seed_path.read_text())
        seed_manifest["initial_model_sha256"] = model_state_sha256(model)
        seed_path.write_text(json.dumps(seed_manifest, indent=2) + "\n")
    parameters = list(model.parameters())
    if node_embedding is not None:
        parameters += list(node_embedding.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.training.lr)

    total_params = sum(p.numel() for p in parameters)
    log_runtime_step(
        "TRAIN",
        f"Run {run + 1} model={config.model.name} total_params={total_params}",
    )
    if config.model.name == "DGCNN":
        log_runtime_step("TRAIN", f"Run {run + 1} SortPooling k={model.k}")
    with open(log_file, "a") as f:
        f.write(f"Run {run + 1} model={config.model.name} total_params={total_params}\n")
        if config.model.name == "DGCNN":
            f.write(f"Run {run + 1} SortPooling k={model.k}\n")

    test_max_batches = config.runtime.max_test_batches
    citation2_expected_negatives = (config.runtime.citation2_negatives_per_source
                                    if config.runtime.citation2_ogb_metrics else None)

    if config.training.only_test:
        if best_ckpt_path.exists():
            _load_seal_checkpoint(best_ckpt_path, model, node_embedding, device)
            log_runtime_step("TEST", f"Run {run + 1} loaded {best_ckpt_path}")
        else:
            if config.training.require_test_checkpoint:
                raise FileNotFoundError(f"Refusing held-out test without frozen best checkpoint: {best_ckpt_path}")
            log_runtime_step("TEST", f"Run {run + 1} no best checkpoint, testing fresh init")
        if config.training.require_test_checkpoint:
            (run_dir / "test_access.json").write_text(json.dumps({
                "started_unix": time.time(), "checkpoint": str(best_ckpt_path),
                "mode": "only_test",
            }, indent=2) + "\n")
        test_metrics = run_eval(
            model,
            test_loader,
            device=device,
            use_feature=config.model.use_feature,
            use_edge_weight=config.model.use_edge_weight,
            node_embedding=node_embedding,
            max_batches=test_max_batches,
            tag="TEST",
            dump_path=(run_dir / "test_scores.npz") if config.runtime.dump_test_scores else None,
            citation2_expected_negatives=citation2_expected_negatives,
        )
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
        return test_metrics

    early_stopper = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode=config.training.checkpoint_metric_mode,
        min_delta=float(config.training.early_stopping_min_delta),
    )
    monitored_key = config.training.checkpoint_metric
    monitored_mode = config.training.checkpoint_metric_mode
    best_val: float | None = None
    global_step = 0
    stop_training = False

    eval_every = int(config.training.eval_every_steps or 0)  # <= 0 disables step-based eval
    log_every = max(1, int(config.training.log_loss_every_steps))
    timing_path = run_dir / "timing.jsonl"

    def _validate(step: int, epoch: int) -> bool:
        """Run val eval + checkpoint + early-stop bookkeeping. Returns True to stop."""
        nonlocal best_val
        t_eval = time.perf_counter()
        val_metrics = run_eval(
            model,
            val_loader,
            device=device,
            use_feature=config.model.use_feature,
            use_edge_weight=config.model.use_edge_weight,
            node_embedding=node_embedding,
            max_batches=config.runtime.max_val_batches,
            tag="VAL",
            citation2_expected_negatives=citation2_expected_negatives,
        )
        eval_sec = time.perf_counter() - t_eval
        log_runtime_step(
            "VAL",
            f"Run {run + 1} epoch {epoch} step {step} eval_sec={eval_sec:.1f} val_metrics={_fmt_metrics(val_metrics)}",
        )
        with open(log_file, "a") as f:
            f.write(
                f"Run {run + 1} epoch {epoch} step {step} eval_sec={eval_sec:.1f} val_metrics={_fmt_metrics(val_metrics)}\n"
            )
        with open(timing_path, "a") as f:
            f.write(json.dumps({
                "kind": "val", "run": run + 1, "epoch": epoch, "step": step,
                "eval_sec": eval_sec, "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                "vmrss_gb": _linux_proc_mem_gb("VmRSS"), "vmhwm_gb": _linux_proc_mem_gb("VmHWM"),
                "cuda_max_alloc_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0,
                "wall": time.strftime("%Y-%m-%d %H:%M:%S"),
            }) + "\n")
        if writer is not None:
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", float(v), step)

        monitored = val_metrics.get(monitored_key)
        if monitored is None:
            available = list(val_metrics.keys())
            log_runtime_step(
                "VAL",
                f"Run {run + 1} checkpoint_metric={monitored_key!r} not in eval output {available}; "
                f"checkpoint+early_stop disabled this step",
            )
            return False
        monitored_f = float(monitored)
        if metric_improved(monitored_f, best=best_val, mode=monitored_mode):
            best_val = monitored_f
            torch.save(_seal_checkpoint_payload(model, node_embedding), best_ckpt_path)
            (run_dir / "best_selection.json").write_text(json.dumps({
                "selection_split": "validation", "metric": monitored_key,
                "value": best_val, "step": step, "epoch": epoch, "seed": run_seed,
            }, indent=2) + "\n")
            log_runtime_step(
                "VAL",
                f"Run {run + 1} step {step} new best {monitored_key}={best_val:.4f}, ckpt saved",
            )
            if writer is not None:
                writer.add_scalar(f"val/best_{monitored_key}", best_val, step)
        if config.training.save_epoch_checkpoints:
            torch.save(_seal_checkpoint_payload(model, node_embedding), run_dir / f"model_epoch{epoch}_step{step}.pth")
        if early_stopper.update(monitored_f):
            log_runtime_step(
                "VAL",
                f"Run {run + 1} step {step} {early_stopper.reason()}",
            )
            return True
        return False

    epoch = 0
    for epoch in range(1, config.training.epochs + 1):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch - 1)

        epoch_start = time.perf_counter()
        log_runtime_step("TRAIN", f"Run {run + 1} epoch {epoch} start")

        epoch_loss_sum = 0.0
        epoch_examples = 0
        last_t = time.perf_counter()

        pbar = tqdm(total=len(train_loader), ncols=70, desc=f"epoch {epoch}")
        batch_idx_in_epoch = 0
        for batch in train_loader:
            model.train()
            t0 = time.perf_counter()
            fetch_sec = t0 - last_t

            batch = batch.to(device)
            optimizer.zero_grad()
            x = batch.x if config.model.use_feature else None
            edge_weight = batch.edge_weight if config.model.use_edge_weight else None
            node_id = batch.node_id if node_embedding is not None else None
            logits = model(batch.z, batch.edge_index, batch.batch, x, edge_weight, node_id)
            loss = BCEWithLogitsLoss()(logits.view(-1), batch.y.to(torch.float))
            loss.backward()
            optimizer.step()

            global_step += 1
            batch_loss = float(loss.item())
            epoch_loss_sum += batch_loss * batch.num_graphs
            epoch_examples += batch.num_graphs

            batch_sec = time.perf_counter() - t0
            last_t = time.perf_counter()
            should_log = (
                TRAIN_DEBUG
                or batch_idx_in_epoch < TRAIN_LOG_FIRST_N
                or batch_sec >= TRAIN_SLOW_BATCH_SEC
            )
            if should_log:
                log_runtime_step(
                    "TRAIN",
                    f"Run {run + 1} epoch {epoch} step {global_step} "
                    f"total={batch_sec:.3f}s fetch={fetch_sec:.3f}s "
                    f"graphs={batch.num_graphs} loss={batch_loss:.6f}",
                )
            batch_idx_in_epoch += 1

            if writer is not None and global_step % log_every == 0:
                writer.add_scalar("train/loss", batch_loss, global_step)

            if eval_every > 0 and global_step % eval_every == 0:
                if _validate(global_step, epoch):
                    stop_training = True
                    break

            if config.training.max_steps is not None and global_step >= int(config.training.max_steps):
                log_runtime_step("TRAIN", f"Run {run + 1} reached max_steps={config.training.max_steps}, stopping")
                stop_training = True
                break

            pbar.update(1)
        pbar.close()

        avg_loss = epoch_loss_sum / max(1, epoch_examples)
        epoch_sec = time.perf_counter() - epoch_start
        log_runtime_step(
            "TRAIN",
            f"Run {run + 1} epoch {epoch} done avg_loss={avg_loss:.6f} time={epoch_sec:.1f}s "
            f"graphs={epoch_examples} graphs_per_sec={epoch_examples / max(epoch_sec, 1e-9):.1f}",
        )
        with open(timing_path, "a") as f:
            f.write(json.dumps({
                "kind": "train_epoch", "run": run + 1, "epoch": epoch, "step": global_step,
                "avg_loss": avg_loss, "train_sec": epoch_sec, "graphs": int(epoch_examples),
                "graphs_per_sec": epoch_examples / max(epoch_sec, 1e-9),
                "vmrss_gb": _linux_proc_mem_gb("VmRSS"), "vmhwm_gb": _linux_proc_mem_gb("VmHWM"),
                "cuda_max_alloc_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0,
                "wall": time.strftime("%Y-%m-%d %H:%M:%S"),
            }) + "\n")
        if writer is not None:
            writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        if stop_training:
            break

        if config.training.eval_every_epoch:
            if _validate(global_step, epoch):
                stop_training = True
                break

    (run_dir / "training_complete.json").write_text(json.dumps({
        "seed": run_seed, "steps": global_step, "last_epoch": epoch,
        "best_validation_metric": monitored_key, "best_validation_value": best_val,
        "test_scored": False,
    }, indent=2) + "\n")
    if config.training.skip_final_test:
        if not best_ckpt_path.exists():
            raise RuntimeError("Validation-only training finished without a best-validation checkpoint")
        log_runtime_step("TRAIN", f"Run {run + 1} complete; held-out test deferred until checkpoint freeze")
        return {}

    # ----- Final test (on best-val checkpoint if available) -----
    if best_ckpt_path.exists():
        _load_seal_checkpoint(best_ckpt_path, model, node_embedding, device)
        log_runtime_step(
            "TEST",
            f"Run {run + 1} loaded best checkpoint best_val_{monitored_key}={best_val} "
            f"running final test",
        )
    else:
        log_runtime_step(
            "TEST",
            f"Run {run + 1} no best checkpoint found, testing current model state",
        )

    t_test = time.perf_counter()
    test_metrics = run_eval(
        model,
        test_loader,
        device=device,
        use_feature=config.model.use_feature,
        use_edge_weight=config.model.use_edge_weight,
        node_embedding=node_embedding,
        max_batches=test_max_batches,
        tag="TEST",
        dump_path=(run_dir / "test_scores.npz") if config.runtime.dump_test_scores else None,
        citation2_expected_negatives=citation2_expected_negatives,
    )
    test_sec = time.perf_counter() - t_test
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
    log_runtime_step(
        "TEST",
        f"Run {run + 1} final test_sec={test_sec:.1f} test_metrics={_fmt_metrics(test_metrics)}",
    )
    with open(timing_path, "a") as f:
        f.write(json.dumps({
            "kind": "test", "run": run + 1, "test_sec": test_sec,
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "best_val": best_val, "wall": time.strftime("%Y-%m-%d %H:%M:%S"),
        }) + "\n")
    with open(log_file, "a") as f:
        f.write(f"Run {run + 1} final test_metrics={_fmt_metrics(test_metrics)}\n")
    if writer is not None:
        for k, v in test_metrics.items():
            writer.add_scalar(f"test/{k}", float(v), 0)
    return test_metrics


def _fmt_metrics(d: dict[str, float]) -> str:
    return "{" + ", ".join(f"{k}={float(v):.4f}" for k, v in d.items()) + "}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SEAL link prediction (graph_csr, config-driven)")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve output dir
    output_dir = Path(
        config.output.dir
        or f"results/seal-graph_csr_{time.strftime('%Y%m%d%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "log.txt"
    cmd = "python " + " ".join(sys.argv) + "\n"
    with open(output_dir / "cmd_input.txt", "a") as f:
        f.write(cmd)
    with open(log_file, "a") as f:
        f.write("\n" + cmd)
    print(f"Results will be saved in {output_dir}", flush=True)

    # Backup source files for reproducibility
    for fname in (
        "seal_link_pred.py",
        "utils.py",
        "models.py",
        "config.py",
        "data_bridge.py",
        "graph_csr_dataset.py",
        "graph_csr_extraction.py",
        "graph_csr_kernels.py",
        "reproducibility.py",
        "ogb_metrics.py",
    ):
        src = Path(THIS_DIR) / fname
        if src.exists():
            shutil.copy(src, output_dir)

    # TensorBoard writer
    writer = None
    if SummaryWriter is not None:
        tb_log_dir = config.training.tensorboard_log_dir or str(output_dir / "tensorboard")
        writer = SummaryWriter(log_dir=tb_log_dir)
        log_preprocessing_step(f"TensorBoard writing to {tb_log_dir}")
    else:
        log_preprocessing_step("TensorBoard not available (install `tensorboard` to enable)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_preprocessing_step(f"Using device={device}")

    # Load train graph (CSR)
    log_preprocessing_step(f"Loading graph_csr from {config.dataset.graph_csr_root}")
    data, split_meta = load_graph_csr_data(
        config.dataset.graph_csr_root,
        file_endian=config.dataset.graph_csr_file_endian,
        use_mmap=config.dataset.graph_csr_use_mmap,
        allow_non_native=config.dataset.graph_csr_allow_non_native,
        chunk_bytes=config.dataset.graph_csr_chunk_bytes,
        node_features_path=config.dataset.node_features_path,
        directed_out_root=config.dataset.directed_out_csr_root,
        directed_in_root=config.dataset.directed_in_csr_root,
    )
    log_preprocessing_step(
        f"Loaded graph num_nodes={data.num_nodes} train_edges={split_meta['num_positive_edges']}"
    )
    if config.model.use_feature and getattr(data, "x", None) is None:
        raise ValueError(
            "model.use_feature=true but graph_csr has no node features (data.x is None)"
        )

    # Resolve train loader config (needs num_nodes)
    config.train_loader = resolve_train_loader_config(
        dataset=config.dataset,
        train_loader=config.train_loader,
        num_nodes=int(data.num_nodes),
    )

    # Build train extractor + loader
    log_preprocessing_step(
        "SEAL GraphCSR extraction "
        f"per_vertex_sampling={config.seal.graph_csr_use_per_vertex_sampling} "
        f"pairwise_subgraph={config.seal.graph_csr_use_pairwise_subgraph} directed={config.seal.directed}"
    )

    def _make_extractor() -> SEALSubgraphExtractor:
        return SEALSubgraphExtractor(
            graph_data=data,
            num_hops=int(config.seal.num_hops),
            node_label=str(config.seal.node_label),
            ratio_per_hop=float(config.seal.ratio_per_hop),
            max_nodes_per_hop=(
                None if config.seal.max_nodes_per_hop is None else int(config.seal.max_nodes_per_hop)
            ),
            directed=bool(config.seal.directed),
            per_vertex_oversample=float(config.seal.per_vertex_oversample),
            graph_csr_use_per_vertex_sampling=bool(config.seal.graph_csr_use_per_vertex_sampling),
            graph_csr_use_pairwise_subgraph=config.seal.graph_csr_use_pairwise_subgraph,
            pairwise_auto_ratio=float(config.seal.pairwise_auto_ratio),
        )

    train_extractor = _make_extractor()
    train_loader = build_train_loader(
        train_loader_config=config.train_loader,
        batch_transform=SEALBatchTransform(train_extractor),
        rank=0,
        world_size=1,
        device_type=device.type,
    )

    positive_dataset = build_train_positive_dataset(
        dataset_config=config.train_loader.positive_edges,
    )
    negative_sampler = build_negative_sampler(
        config.train_loader.negative_sampling,
        config.train_loader.positive_edges,
    )

    num_features = 0
    x_attr = getattr(data, "x", None)
    if x_attr is not None:
        num_features = int(x_attr.shape[-1]) if getattr(x_attr, "ndim", 1) > 1 else 1
    train_dataset = SEALTrainDatasetFacade(
        positive_dataset=positive_dataset,
        negative_sampler=negative_sampler,
        extractor=train_extractor,
        train_loader=train_loader,
        num_features=num_features,
    )
    log_preprocessing_step(f"Train dataset ready len={len(train_dataset)}")

    # Eval datasets/loaders
    edge_splits = load_edge_splits(config.dataset)
    val_dataset = build_eval_dataset(
        dataset_config=config.dataset,
        split_name="val",
        pos_edges=edge_splits.val_pos,
        neg_edges=edge_splits.val_neg,
    )
    test_dataset = build_eval_dataset(
        dataset_config=config.dataset,
        split_name="test",
        pos_edges=edge_splits.test_pos,
        neg_edges=edge_splits.test_neg,
    )
    val_extractor = _make_extractor()
    test_extractor = _make_extractor()

    val_num_workers = config.runtime.val_num_workers
    if val_num_workers is None:
        val_num_workers = config.train_loader.num_workers

    val_loader = build_eval_loader(
        dataset=val_dataset,
        batch_size=config.train_loader.batch_size,
        num_workers=val_num_workers,
        rank=0,
        world_size=1,
        device_type=device.type,
        collate_fn=SEALEvalCollator(val_extractor, seed=config.runtime.eval_seed),
    )
    test_loader = build_eval_loader(
        dataset=test_dataset,
        batch_size=config.train_loader.batch_size,
        num_workers=val_num_workers,
        rank=0,
        world_size=1,
        device_type=device.type,
        collate_fn=SEALEvalCollator(test_extractor, seed=config.runtime.eval_seed),
    )
    val_len = len(val_dataset)  # type: ignore[arg-type]
    test_len = len(test_dataset)  # type: ignore[arg-type]
    log_preprocessing_step(
        f"val_len={val_len} test_len={test_len} val_num_workers={val_num_workers}"
    )

    log_preprocessing_step("Preprocessing complete, entering model setup/training")

    all_test_metrics: list[dict[str, float]] = []
    for run in range(int(config.training.runs)):
        test_metrics = train_one_run(
            run=run,
            config=config,
            train_dataset=train_dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            data=data,
            device=device,
            output_dir=output_dir,
            writer=writer,
            log_file=log_file,
        )
        all_test_metrics.append(test_metrics)

    if writer is not None:
        writer.close()

    # Aggregate across runs
    if all_test_metrics:
        keys = sorted(all_test_metrics[0].keys())
        summary_lines = [f"=== Aggregate over {len(all_test_metrics)} runs ==="]
        for key in keys:
            values = np.array([m[key] for m in all_test_metrics if key in m], dtype=np.float64)
            if values.size == 0:
                continue
            mean = float(values.mean())
            std = float(values.std()) if values.size > 1 else 0.0
            summary_lines.append(f"  test/{key}: {mean:.4f} ± {std:.4f}")
        summary = "\n".join(summary_lines)
        print(summary, flush=True)
        with open(log_file, "a") as f:
            f.write("\n" + summary + "\n")

    print(f"Results saved in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
