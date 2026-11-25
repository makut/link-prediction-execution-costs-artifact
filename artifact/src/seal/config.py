from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from dataset_utils import DatasetConfig, NegativeSamplingConfig, PositiveEdgesConfig, TrainLoaderConfig, load_yaml


ModelName = Literal["DGCNN", "SAGE", "GCN", "GIN"]


@dataclass(slots=True)
class SEALExtractionConfig:
    num_hops: int = 1
    node_label: str = "drnl"
    ratio_per_hop: float = 1.0
    max_nodes_per_hop: int | None = None
    per_vertex_oversample: float = 1.5
    graph_csr_use_per_vertex_sampling: bool = True
    # True | False | "auto" (auto: pairwise iff |S|(|S|-1)/2 <= pairwise_auto_ratio * sum deg(S))
    graph_csr_use_pairwise_subgraph: bool | str = True
    pairwise_auto_ratio: float = 1.0
    directed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelConfig:
    name: ModelName = "DGCNN"
    hidden_channels: int = 32
    num_layers: int = 3
    sortpool_k: float = 0.6
    max_z: int = 1000
    use_feature: bool = False
    use_edge_weight: bool = False
    train_node_embedding: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingConfig:
    # None preserves historical behavior. An explicit seed covers model init,
    # Python/NumPy/Torch/CUDA and independent DataLoader worker RNGs.
    seed: int | None = None
    lr: float = 1e-4
    epochs: int = 50
    runs: int = 1
    eval_every_steps: int = 500
    log_loss_every_steps: int = 50
    continue_from: int | None = None
    only_test: bool = False
    checkpoint_metric: str = "roc_auc@100"
    checkpoint_metric_mode: Literal["min", "max"] = "max"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    tensorboard_log_dir: str | None = None
    # Also validate (+checkpoint) at the end of every epoch (SEAL-OGB style);
    # set eval_every_steps: 0 to validate only at epoch ends.
    eval_every_epoch: bool = False
    # Save a checkpoint after every validation (model_epoch{E}_step{S}.pth).
    save_epoch_checkpoints: bool = False
    # Stop training after this many optimizer steps (None = no limit). Used for
    # equal-step ablations and throughput measurements.
    max_steps: int | None = None
    # Validation-only training, with held-out test invoked later after freezing.
    skip_final_test: bool = False
    require_test_checkpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeConfig:
    # Optional per-edge extraction seed, stable across validation passes/workers.
    eval_seed: int | None = None
    citation2_ogb_metrics: bool = False
    citation2_negatives_per_source: int = 1000
    val_num_workers: int | None = None
    max_val_batches: int | None = None
    max_test_batches: int | None = None
    # Dump the final test (src, dst, label, logit) rows to <run_dir>/test_scores.npz
    # so the official OGB Evaluator can be applied offline (scripts/ogb_mrr_from_scores.py).
    dump_test_scores: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OutputConfig:
    dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SEALConfig:
    dataset: DatasetConfig
    train_loader: TrainLoaderConfig
    seal: SEALExtractionConfig = field(default_factory=SEALExtractionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "train_loader": asdict(self.train_loader),
            "seal": self.seal.to_dict(),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "runtime": self.runtime.to_dict(),
            "output": self.output.to_dict(),
        }


def _load_train_loader_config(raw: dict[str, Any], dataset: DatasetConfig) -> TrainLoaderConfig:
    data = dict(raw.get("train_loader", {}))
    positive_defaults = asdict(dataset.to_positive_edges_config())
    positive_defaults.update(data.get("positive_edges", {}))
    data["positive_edges"] = positive_defaults

    negative_sampling = data.get("negative_sampling")
    if negative_sampling is None:
        data["negative_sampling"] = asdict(NegativeSamplingConfig(seed=int(data.get("seed", 42))))
    return TrainLoaderConfig.from_dict(data)


def resolve_train_loader_config(
    *,
    dataset: DatasetConfig,
    train_loader: TrainLoaderConfig,
    num_nodes: int | None = None,
) -> TrainLoaderConfig:
    # `train_loader.positive_edges` already merges the dataset defaults with
    # any YAML overrides (subset_fraction, pairs_are_canonical, ...); only the
    # real graph num_nodes is injected here.
    del dataset
    positive_edges = train_loader.positive_edges
    if num_nodes is not None:
        positive_edges = replace(positive_edges, num_nodes=int(num_nodes))
    return replace(
        train_loader,
        positive_edges=positive_edges,
    )


def load_config(path: str | Path) -> SEALConfig:
    raw = load_yaml(Path(path))
    dataset = DatasetConfig(**raw.get("dataset", {})).resolve()
    if dataset.graph_csr_root is None:
        raise ValueError("dataset.graph_csr_root must be set for SEAL")
    train_loader = _load_train_loader_config(raw, dataset)
    seal = SEALExtractionConfig(**raw.get("seal", {}))
    model = ModelConfig(**raw.get("model", {}))
    training = TrainingConfig(**raw.get("training", {}))
    runtime = RuntimeConfig(**raw.get("runtime", {}))
    output = OutputConfig(**raw.get("output", {}))
    return SEALConfig(
        dataset=dataset,
        train_loader=train_loader,
        seal=seal,
        model=model,
        training=training,
        runtime=runtime,
        output=output,
    )
