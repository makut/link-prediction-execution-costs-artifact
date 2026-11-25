"""Prepare -> train all controls -> freeze all best-validation weights -> test.

This launcher creates no pods and changes no input graph. Execute it inside an
already authorized compute environment, with explicit dataset/output paths.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

try:
    from .prepare_sources import freeze_source_subset, sha256
except ImportError:
    from prepare_sources import freeze_source_subset, sha256

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _prepare(args, out: Path, *, smoke=False):
    if (out / "plan.json").exists():
        raise FileExistsError(f"Frozen plan already exists: {out / 'plan.json'}; select train/freeze/test")
    protocol = yaml.safe_load(Path(args.protocol).read_text())
    if smoke:
        protocol.update(seeds=[42], validation_source_count=2, test_source_count=2,
                        epochs=1, num_workers=2, smoke_max_steps=2)
    prepared = Path(args.prepared_root).resolve()
    graph = Path(args.graph_root).resolve() if args.graph_root else prepared / "train_csr"
    pairs = Path(args.pairs_root).resolve() if args.pairs_root else prepared / "train_directed_csr"
    features = Path(args.features).resolve() if args.features else prepared / "node_feat.npy"
    for path in (graph, pairs, features):
        if not path.exists():
            raise FileNotFoundError(path)
    out.mkdir(parents=True, exist_ok=True)
    subsets = {}
    for split, prefix, count_key, seed_key in (
            ("val", "valid", "validation_source_count", "validation_subset_seed"),
            ("test", "test", "test_source_count", "test_subset_seed")):
        pos_arg, neg_arg = getattr(args, f"{split}_pos"), getattr(args, f"{split}_neg")
        subsets[split] = freeze_source_subset(
            pos_path=Path(pos_arg).resolve() if pos_arg else prepared / f"{prefix}_edge.npy",
            neg_path=Path(neg_arg).resolve() if neg_arg else prepared / f"{prefix}_edge_neg.npy",
            output_dir=out / "subsets", name=prefix, count=int(protocol[count_key]),
            seed=int(protocol[seed_key]), expected_negatives=int(protocol["negatives_per_source"]))
    configs = out / "configs"
    configs.mkdir()
    jobs = []
    for seed in protocol["seeds"]:
        # Alternating order avoids always measuring one variant first.
        variants = ("union", "pv") if int(seed) % 2 == 0 else ("pv", "union")
        for variant in variants:
            name = f"{variant}_seed{seed}"
            cfg = {
                "dataset": {"num_nodes": 2927963, "graph_csr_root": str(graph),
                    "pairs_graph_csr_root": str(pairs), "split_root": str(out / "subsets"),
                    "valid_edge_path": subsets["val"]["outputs"]["positive"]["path"],
                    "valid_edge_neg_path": subsets["val"]["outputs"]["negative"]["path"],
                    "test_edge_path": subsets["test"]["outputs"]["positive"]["path"],
                    "test_edge_neg_path": subsets["test"]["outputs"]["negative"]["path"],
                    "node_features_path": str(features), "graph_csr_file_endian": "little",
                    "graph_csr_use_mmap": True, "graph_csr_allow_non_native": False,
                    "split_use_mmap": True, "has_self_loops": False},
                "train_loader": {"batch_size": protocol["batch_size"], "seed": seed,
                    "num_workers": protocol["num_workers"],
                    "positive_edges": {"pairs_are_canonical": False,
                        "subset_fraction": protocol["train_subset_fraction"],
                        "subset_seed": protocol["train_subset_seed"]},
                    "negative_sampling": {"name": "uniform", "neg_per_pos": 1, "seed": seed,
                        "reject_self_loops": True, "extra_kwargs": {"mutate": "dst"}}},
                "seal": {"num_hops": protocol["num_hops"], "node_label": "drnl",
                    "ratio_per_hop": 1.0, "max_nodes_per_hop": protocol["max_nodes_per_hop"],
                    "per_vertex_oversample": protocol["per_vertex_oversample"],
                    "graph_csr_use_per_vertex_sampling": variant == "pv",
                    "graph_csr_use_pairwise_subgraph": True, "directed": False},
                "model": {"name": "DGCNN", "hidden_channels": 32, "num_layers": 3,
                    "sortpool_k": protocol["sortpool_k"], "max_z": 1000,
                    "use_feature": True, "use_edge_weight": False, "train_node_embedding": False},
                "training": {"seed": seed, "lr": protocol["learning_rate"],
                    "epochs": protocol["epochs"], "runs": 1, "eval_every_steps": 0,
                    "eval_every_epoch": True, "checkpoint_metric": protocol["selection_metric"],
                    "checkpoint_metric_mode": "max", "early_stopping_patience": None,
                    "skip_final_test": True, "require_test_checkpoint": True},
                "runtime": {"val_num_workers": protocol["num_workers"],
                    "max_val_batches": None, "max_test_batches": None,
                    "eval_seed": protocol["eval_extraction_seed"], "citation2_ogb_metrics": True,
                    "citation2_negatives_per_source": protocol["negatives_per_source"],
                    "dump_test_scores": False},
                "output": {"dir": str(out / "runs" / name)}}
            if smoke:
                cfg["training"].update(max_steps=protocol["smoke_max_steps"],
                                       eval_every_steps=protocol["smoke_max_steps"], eval_every_epoch=False)
            path = configs / f"{name}.json"
            _json(path, cfg)
            jobs.append({"name": name, "seed": seed, "variant": variant,
                         "config": str(path), "config_sha256": sha256(path)})
    provenance = None
    if args.dataset_manifest:
        path = Path(args.dataset_manifest).resolve()
        provenance = {"path": str(path), "sha256": sha256(path), "content": json.loads(path.read_text())}
    code_paths = sorted(path for package in ("seal", "dataset_utils", "graph_csr")
                        for path in (REPO_ROOT / package).glob("*.py"))
    _json(out / "plan.json", {"created_unix": time.time(), "protocol": protocol,
                              "graph_root": str(graph), "pairs_root": str(pairs),
                              "dataset_provenance": provenance, "subsets": subsets, "jobs": jobs,
                              "runner_sha256": sha256(Path(__file__)),
                              "code": {str(path): sha256(path) for path in code_paths},
                              "test_access_policy": "score only after all validation-selected checkpoints are frozen"})
    print(f"Prepared {len(jobs)} controls at {out}; no model was trained or tested", flush=True)


def _verify_plan(out: Path):
    plan = json.loads((out / "plan.json").read_text())
    for path, digest in plan.get("code", {}).items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f"Training source changed after experiment preparation: {path}")
    for subset in plan["subsets"].values():
        for item in subset["outputs"].values():
            if sha256(Path(item["path"])) != item["sha256"]:
                raise RuntimeError(f"Frozen subset changed: {item['path']}")
    for job in plan["jobs"]:
        if sha256(Path(job["config"])) != job["config_sha256"]:
            raise RuntimeError(f"Frozen training config changed: {job['name']}")
    return plan


def _invoke(args, cfg: dict, config_path: Path, log_path: Path):
    _json(config_path, cfg)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(cfg["training"]["seed"])
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env.setdefault(key, "1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as log:
        subprocess.run([args.python, "-m", "seal.seal_link_pred", "--config", str(config_path)],
                       cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def _train(args, out: Path):
    plan = _verify_plan(out)
    selected = set(args.jobs or [job["name"] for job in plan["jobs"]])
    if selected - {job["name"] for job in plan["jobs"]}:
        raise ValueError(f"Unknown jobs: {selected}")
    for job in plan["jobs"]:
        if job["name"] not in selected:
            continue
        cfg = json.loads(Path(job["config"]).read_text())
        run = Path(cfg["output"]["dir"]) / "run1"
        if run.exists():
            raise FileExistsError(f"Refusing to overwrite or silently resume run: {run}")
        print(f"Training {job['name']}; held-out test disabled", flush=True)
        _invoke(args, cfg, out / "configs" / f"{job['name']}_executed.json",
                out / "logs" / f"{job['name']}_train.log")


def _freeze(out: Path):
    plan = _verify_plan(out)
    frozen_path = out / "FROZEN_CHECKPOINTS.json"
    if frozen_path.exists():
        raise FileExistsError(frozen_path)
    frozen = []
    initial_hashes = {}
    for job in plan["jobs"]:
        cfg = json.loads(Path(job["config"]).read_text())
        run = Path(cfg["output"]["dir"]) / "run1"
        complete = json.loads((run / "training_complete.json").read_text())
        selection = json.loads((run / "best_selection.json").read_text())
        initial_hash = json.loads((run / "seed_manifest.json").read_text())["initial_model_sha256"]
        if job["seed"] in initial_hashes and initial_hashes[job["seed"]] != initial_hash:
            raise RuntimeError(f"Model initialization differs between paired variants for seed {job['seed']}")
        initial_hashes[job["seed"]] = initial_hash
        if selection["selection_split"] != "validation" or selection["metric"] != "ogb_mrr":
            raise RuntimeError(f"Non-validation selection in {job['name']}")
        if any((run / name).exists() for name in ("test_access.json", "test_metrics.json", "test_scores.npz")):
            raise RuntimeError(f"Test was accessed before joint checkpoint freeze: {job['name']}")
        expected_steps = cfg["training"].get("max_steps")
        if expected_steps is not None and complete["steps"] != expected_steps:
            raise RuntimeError(f"Incomplete training budget: {job['name']}")
        if expected_steps is None and complete["last_epoch"] != plan["protocol"]["epochs"]:
            raise RuntimeError(f"Incomplete epoch budget: {job['name']}")
        checkpoint = run / "best_model.pth"
        frozen.append({**job, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                       "training": complete, "selection": selection, "initial_model_sha256": initial_hash})
    _json(frozen_path, {"created_unix": time.time(), "plan_sha256": sha256(out / "plan.json"),
                        "jobs": frozen, "selection_uses_test": False})
    print(f"Froze all {len(frozen)} validation-selected checkpoints before any test scoring", flush=True)


def _test(args, out: Path):
    _verify_plan(out)
    frozen = json.loads((out / "FROZEN_CHECKPOINTS.json").read_text())
    if sha256(out / "plan.json") != frozen["plan_sha256"]:
        raise RuntimeError("Plan changed after checkpoint freeze")
    selected = set(args.jobs or [job["name"] for job in frozen["jobs"]])
    if selected - {job["name"] for job in frozen["jobs"]}:
        raise ValueError(f"Unknown test jobs: {selected}")
    results = []
    for job in frozen["jobs"]:
        if job["name"] not in selected:
            continue
        if sha256(Path(job["checkpoint"])) != job["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen checkpoint changed: {job['name']}")
        cfg = copy.deepcopy(json.loads(Path(job["config"]).read_text()))
        cfg["training"].update(only_test=True, skip_final_test=False, require_test_checkpoint=True)
        cfg["runtime"]["dump_test_scores"] = True
        print(f"Scoring frozen checkpoint {job['name']}", flush=True)
        _invoke(args, cfg, out / "configs" / f"{job['name']}_test.json",
                out / "logs" / f"{job['name']}_test.log")
        metrics = json.loads((Path(job["checkpoint"]).parent / "test_metrics.json").read_text())
        results.append({"name": job["name"], "seed": job["seed"], "variant": job["variant"],
                        "selection": job["selection"], "test_metrics": metrics,
                        "checkpoint_sha256": job["checkpoint_sha256"]})
    _json(out / ("test_results.json" if not args.jobs else "test_results_" + "_".join(sorted(selected)) + ".json"),
          {"frozen_manifest_sha256": sha256(out / "FROZEN_CHECKPOINTS.json"), "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "smoke", "train", "freeze", "test", "all"), required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", default=str(HERE / "protocol.yaml"))
    for flag in ("graph-root", "pairs-root", "features", "val-pos", "val-neg", "test-pos", "test-neg", "dataset-manifest"):
        parser.add_argument("--" + flag)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES; same device budget for paired runs")
    parser.add_argument("--jobs", nargs="+", help="Specific prepared job names; freeze always requires every job")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if args.phase == "smoke":
        out = out / "smoke"
        _prepare(args, out, smoke=True)
        _train(args, out)
        return
    if args.phase in ("prepare", "all"):
        _prepare(args, out)
    if args.phase in ("train", "all"):
        _train(args, out)
    if args.phase in ("freeze", "all"):
        _freeze(out)
    if args.phase in ("test", "all"):
        _test(args, out)


if __name__ == "__main__":
    main()
