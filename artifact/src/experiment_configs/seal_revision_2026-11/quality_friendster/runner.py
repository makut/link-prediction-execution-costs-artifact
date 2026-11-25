"""Friendster query-row-aware adapter; frozen Citation2/production sources unchanged."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
COMMON_DIR = HERE.parent / "quality"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
SPEC = importlib.util.spec_from_file_location("seal_revision_frozen_common_runner", COMMON_DIR / "runner.py")
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)

from query_data import candidate_order, freeze_query_subset, sha256


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _graph_header(path: Path) -> tuple[int, int]:
    n_bytes = (path / "edge_starts.bin").stat().st_size
    if n_bytes % 8:
        raise ValueError(f"Invalid CSR offset file: {path}")
    with (path / "edge_ends.bin").open("rb") as stream:
        m = struct.unpack("<q", stream.read(8))[0]
    if m < 0 or (path / "edge_ends.bin").stat().st_size != 8 + 4 * m:
        raise ValueError(f"Invalid CSR neighbor file: {path}")
    return n_bytes // 8, m


def validate_dataset_manifest(content: dict, protocol: dict, manifest_path: Path) -> None:
    if protocol["dataset"] == "synthetic_fixture" and content.get("fixture") is True:
        return
    expected = {"dataset": "friendster", "variant": "exact_colleague_80_10_10",
                "num_nodes": protocol["num_nodes"], "seed": protocol["split_seed"],
                "threshold_train": protocol["threshold_train"],
                "threshold_validation": protocol["threshold_validation"],
                "train_stored_entries": protocol["expected_train_entries"],
                "train_undirected_edges": protocol["expected_canonical_positive_entries"]}
    for key, value in expected.items():
        if content.get(key) != value:
            raise ValueError(f"Dataset manifest differs from pinned Friendster protocol: {key}")
    if not (content.get("source_graph", {}).get("historical_source_revisions_equal")
            or content.get("source_graph", {}).get("public_numeric_payload_hashes_equal_verified")):
        raise ValueError("Historical full-graph source identity has not been verified")
    evaluation = content.get("evaluation", {})
    if evaluation.get("layout") != "original_query_row" or not evaluation.get("latest_series_arrays_sha256_equal"):
        raise ValueError("Latest colleague query-row identity has not been verified")
    if not evaluation.get("source_group_evaluator_forbidden"):
        raise ValueError("Expected duplicate-source query protocol was not recorded")
    for name in ("valid", "test"):
        stats = evaluation["group_statistics"][name]
        if stats["query_rows"] != protocol["original_queries_per_split"] or stats["negatives_per_query"] != protocol["negatives_per_query"]:
            raise ValueError(f"Wrong original evaluation budget: {name}")
    marker = json.loads(manifest_path.with_name("PREPARED_COMPLETE.json").read_text())
    if marker["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Prepared Friendster completion-manifest hash mismatch")


def _verify_friendster_inputs(out: Path):
    plan = json.loads((out / "plan.json").read_text())
    provenance = plan["dataset_provenance"]
    if sha256(Path(provenance["path"])) != provenance["sha256"]:
        raise RuntimeError("Dataset provenance changed after quality preparation")
    for path, expected in plan["graph_input_stats"].items():
        stat = Path(path).stat()
        if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise RuntimeError(f"Training topology changed after preparation: {path}")


def _invoke(args, cfg, config_path, log_path):
    # Reuse common train/freeze/test orchestration, substituting only the explicit
    # Friendster entry point. It installs its evaluator in that child process.
    _json(config_path, cfg)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(cfg["training"]["seed"])
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env.setdefault(name, "1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as stream:
        subprocess.run([args.python, str(HERE / "train_entry.py"), "--config", str(config_path)],
                       cwd=REPO_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


common._invoke = _invoke


def _prepare(args, out: Path, *, smoke=False):
    if (out / "plan.json").exists():
        raise FileExistsError(f"Plan already frozen: {out}")
    protocol = yaml.safe_load(Path(args.protocol).read_text())
    if smoke:
        protocol.update(seeds=[42], validation_query_count=2, test_query_count=2,
                        num_workers=2, max_steps=2, eval_every_steps=2)
    prepared = Path(args.prepared_root).resolve()
    graph, pairs = prepared / "train_csr", prepared / "train_pairs_csr"
    n, m = _graph_header(graph)
    pair_n, pair_m = _graph_header(pairs)
    if (n, m) != (protocol["num_nodes"], protocol["expected_train_entries"]):
        raise ValueError(f"Training graph differs from frozen protocol: {n}/{m}")
    if (pair_n, pair_m) != (n, protocol["expected_canonical_positive_entries"]):
        raise ValueError(f"Canonical positive graph differs from frozen protocol: {pair_n}/{pair_m}")
    manifest_path = Path(args.dataset_manifest).resolve()
    provenance = {"path": str(manifest_path), "sha256": sha256(manifest_path),
                  "content": json.loads(manifest_path.read_text())}
    validate_dataset_manifest(provenance["content"], protocol, manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    evaluation = Path(args.evaluation_root).resolve() if args.evaluation_root else prepared / "evaluation"
    subsets = {}
    for split, prefix, count_key, seed_key in (
            ("val", "valid", "validation_query_count", "validation_subset_seed"),
            ("test", "test", "test_query_count", "test_subset_seed")):
        subsets[split] = freeze_query_subset(
            input_dir=evaluation / prefix, output_dir=out / "subsets", name=prefix,
            count=int(protocol[count_key]), seed=int(protocol[seed_key]),
            candidate_order_seed=int(protocol["eval_candidate_order_seed"]),
            expected_queries=int(protocol["original_queries_per_split"]),
            expected_negatives=int(protocol["negatives_per_query"]))
        if protocol["dataset"] != "synthetic_fixture":
            for key, item in subsets[split]["inputs"].items():
                expected_file = provenance["content"]["files"][f"evaluation/{prefix}/{key}.npy"]
                if item["sha256"] != expected_file["sha256"]:
                    raise ValueError(f"Query input differs from verified colleague data: {prefix}/{key}")
    (out / "configs").mkdir()
    jobs = []
    for seed in protocol["seeds"]:
        for variant in (("union", "pv") if seed % 2 == 0 else ("pv", "union")):
            name = f"{variant}_seed{seed}"
            cfg = {
                "dataset": {"num_nodes": n, "graph_csr_root": str(graph),
                    "pairs_graph_csr_root": str(pairs), "split_root": str(out / "subsets"),
                    "valid_edge_path": subsets["val"]["outputs"]["positive"]["path"],
                    "valid_edge_neg_path": subsets["val"]["outputs"]["negative"]["path"],
                    "test_edge_path": subsets["test"]["outputs"]["positive"]["path"],
                    "test_edge_neg_path": subsets["test"]["outputs"]["negative"]["path"],
                    "graph_csr_file_endian": "little", "graph_csr_allow_non_native": False,
                    "graph_csr_use_mmap": True, "split_use_mmap": True, "has_self_loops": False},
                "train_loader": {"batch_size": protocol["batch_size"], "seed": seed,
                    "num_workers": protocol["num_workers"],
                    "positive_edges": {"pairs_are_canonical": True},
                    "negative_sampling": {"name": "uniform", "neg_per_pos": 1, "seed": seed,
                        "reject_self_loops": True, "extra_kwargs": {"mutate": "dst"}}},
                "seal": {"num_hops": protocol["num_hops"], "node_label": "drnl",
                    "ratio_per_hop": 1.0, "max_nodes_per_hop": protocol["max_nodes_per_hop"],
                    "per_vertex_oversample": protocol["per_vertex_oversample"],
                    "graph_csr_use_per_vertex_sampling": variant == "pv",
                    "graph_csr_use_pairwise_subgraph": True, "directed": False},
                "model": {"name": "DGCNN", "hidden_channels": 32, "num_layers": 3,
                    "sortpool_k": protocol["sortpool_k"], "max_z": 1000,
                    "use_feature": False, "use_edge_weight": False, "train_node_embedding": False},
                "training": {"seed": seed, "lr": protocol["learning_rate"], "runs": 1,
                    "epochs": protocol["epochs"], "max_steps": protocol["max_steps"],
                    "eval_every_steps": protocol["eval_every_steps"], "eval_every_epoch": False,
                    "checkpoint_metric": protocol["implementation_metric_alias"],
                    "checkpoint_metric_mode": "max", "early_stopping_patience": None,
                    "skip_final_test": True, "require_test_checkpoint": True},
                "runtime": {"val_num_workers": protocol["num_workers"], "max_val_batches": None,
                    "max_test_batches": None, "eval_seed": protocol["eval_extraction_seed"],
                    "citation2_ogb_metrics": True,
                    "citation2_negatives_per_source": protocol["negatives_per_query"],
                    "dump_test_scores": False},
                "output": {"dir": str(out / "runs" / name)}}
            path = out / "configs" / f"{name}.json"
            _json(path, cfg)
            jobs.append({"name": name, "seed": seed, "variant": variant,
                         "config": str(path), "config_sha256": sha256(path)})
    code_paths = [path for package in ("seal", "dataset_utils", "graph_csr") for path in (REPO_ROOT / package).glob("*.py")]
    code_paths += list(HERE.glob("*.py")) + list(COMMON_DIR.glob("*.py"))
    _json(out / "plan.json", {
        "created_unix": time.time(), "protocol": protocol, "graph_root": str(graph), "pairs_root": str(pairs),
        "graph_input_stats": {str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                              for directory in (graph, pairs) for path in (directory / "edge_starts.bin", directory / "edge_ends.bin")},
        "dataset_provenance": provenance, "subsets": subsets, "jobs": jobs,
        "code": {str(path): sha256(path) for path in sorted(set(code_paths))},
        "metric_name": "custom_sampled_mrr", "implementation_metric_alias": "ogb_mrr",
        "is_official_ogb_benchmark": False,
        "ranking_unit": "original query row; repeated source IDs remain separate",
        "positive_sampling": {"source": "train_pairs_csr canonical u<v", "subset_fraction": None,
            "epoch": 0, "order": "existing seeded coprime-stride iterator, same paired prefix",
            "positive_examples_per_full_batch": protocol["batch_size"] // 2,
            "expected_graph_examples_per_job": protocol["max_steps"] * protocol["batch_size"],
            "expected_positive_examples_per_job": protocol["max_steps"] * (protocol["batch_size"] // 2)},
        "test_access_policy": "all six validation-selected checkpoints jointly frozen before test model scoring"})
    print(f"Prepared {len(jobs)} Friendster controls; original query identities retained", flush=True)


def _normalize_test_results(out: Path, args):
    names = sorted(args.jobs) if args.jobs else None
    result_path = out / ("test_results_" + "_".join(names) + ".json" if names else "test_results.json")
    result = json.loads(result_path.read_text())
    plan = json.loads((out / "plan.json").read_text())
    subset = plan["subsets"]["test"]
    query_ids = np.load(subset["outputs"]["query_ids"]["path"], allow_pickle=False)
    positive = np.load(subset["outputs"]["positive"]["path"], mmap_mode="r", allow_pickle=False)
    negative = np.load(subset["outputs"]["negative"]["path"], mmap_mode="r", allow_pickle=False)
    k = subset["negatives_per_query"]
    for row in result["results"]:
        run = out / "runs" / row["name"] / "run1"
        with np.load(run / "test_scores.npz", allow_pickle=False) as data:
            ids, destinations, labels, scores = [data[key] for key in ("src", "dst", "label", "logit")]
        expected_ids = np.repeat(query_ids, k + 1)
        if not np.array_equal(ids, expected_ids):
            raise RuntimeError("Test dump does not retain exactly the frozen original query groups")
        slots = np.concatenate([candidate_order(int(q), k, subset["candidate_order_seed"]) for q in query_ids])
        if not np.array_equal(labels, (slots == 0).astype(labels.dtype)):
            raise RuntimeError("Test dump candidate slots/labels differ from frozen query order")
        candidates = np.column_stack((positive[:, 1], negative[:, 1].reshape(len(query_ids), k)))
        expected_destinations = np.take_along_axis(candidates, slots.reshape(len(query_ids), k + 1), axis=1).reshape(-1)
        if not np.array_equal(destinations, expected_destinations):
            raise RuntimeError("Scored targets differ from the provided candidates for an original query row")
        grouped = scores.reshape(len(query_ids), k + 1)
        positive_scores = grouped[(slots == 0).reshape(grouped.shape)]
        negative_scores = grouped[(slots != 0).reshape(grouped.shape)].reshape(len(query_ids), k)
        ranks = 1 + 0.5 * ((negative_scores > positive_scores[:, None]).sum(1)
                           + (negative_scores >= positive_scores[:, None]).sum(1))
        reciprocal = 1.0 / ranks
        raw = row["test_metrics"]
        if int(raw["ogb_num_sources"]) != len(query_ids) or not np.isclose(reciprocal.mean(), raw["ogb_mrr"], rtol=0, atol=1e-12):
            raise RuntimeError("Query-row dump and streaming midpoint MRR disagree")
        np.save(run / "query_rr.npy", reciprocal, allow_pickle=False)
        np.save(run / "query_ids.npy", query_ids, allow_pickle=False)
        np.savez(run / "query_scores.npz", query_id=ids, src=np.repeat(positive[:, 0], k + 1),
                 dst=destinations, candidate_slot=slots, label=labels, logit=scores)
        row["raw_implementation_metrics"] = raw
        row["test_metrics"] = {"custom_sampled_mrr": raw["ogb_mrr"], "num_queries": raw["ogb_num_sources"],
                               **{f"custom_sampled_hits@{cutoff}": raw[f"ogb_hits@{cutoff}"] for cutoff in (1, 10, 50, 100)}}
        row["selection"]["implementation_metric_alias"] = row["selection"]["metric"]
        row["selection"]["metric"] = "custom_sampled_mrr"
        row["is_official_ogb_benchmark"] = False
        row["original_query_ids_sha256"] = sha256(run / "query_ids.npy")
    result.update(metric_name="custom_sampled_mrr", is_official_ogb_benchmark=False,
                  ranking_unit="original query row", preserves_repeated_sources=True)
    _json(result_path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "smoke", "train", "freeze", "test", "all"), required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--evaluation-root")
    parser.add_argument("--protocol", default=str(HERE / "protocol.yaml"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--jobs", nargs="+")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if args.phase == "smoke":
        _prepare(args, out / "smoke", smoke=True)
        _verify_friendster_inputs(out / "smoke")
        common._train(args, out / "smoke")
        return
    if args.phase in ("prepare", "all"):
        _prepare(args, out)
    if args.phase in ("train", "all"):
        _verify_friendster_inputs(out)
        common._train(args, out)
    if args.phase in ("freeze", "all"):
        _verify_friendster_inputs(out)
        common._freeze(out)
    if args.phase in ("test", "all"):
        _verify_friendster_inputs(out)
        common._test(args, out)
        _normalize_test_results(out, args)


if __name__ == "__main__":
    main()
