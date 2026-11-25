#!/usr/bin/env python3
"""Unattended sequential H100 pipeline controls, gated on completed quality work.

No cluster management, downloads, backups, cache eviction or process killing.
Launch this on the existing first H100 pod. See pipeline_suite/README.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

DEFAULT_BASE = Path("/home/jovyan/seal_revision_20261124")
ARMS = ("reference_union_stock", "reference_union_corrected", "production_union", "production_pv", "reference_union_compiled")
SOURCE_FILES = (
    "experiment_configs/seal_revision_2026-11/run_pipeline_suite_v4.py",
    "scripts/seal_revision/pipeline_benchmark_v4.py",
    "scripts/seal_revision/exact_benchmark.py",
    "scripts/seal_revision/memory_benchmark.py",
    "seal/graph_csr_kernels.py", "seal/graph_csr_extraction.py", "seal/models.py", "seal/utils.py", "seal/reproducibility.py",
)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def reserve_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing existing pipeline outputs; no automatic resume/overwrite: {path}")
    path.mkdir(parents=True, exist_ok=True)


def source_hashes(repo: Path) -> dict:
    return {name: digest_file(repo / name) for name in SOURCE_FILES}


def assert_frozen_sources(repo: Path, frozen: dict) -> None:
    if source_hashes(repo) != frozen:
        raise RuntimeError("Frozen runtime sources changed; refusing to launch another job")


def build_plan(base: Path, repo: Path, python: str) -> dict:
    output = base / "results/pipeline"
    benchmark = repo / "scripts/seal_revision/pipeline_benchmark_v4.py"
    protocol = {
        "protocol_version": 4,
        "routing_version": "production_matched_vectorized_selected_degree_once_v4_reference_bridge",
        "reference_bridge_arm": "reference_union_compiled",
        "seed": 20261124, "workers": [1, 8], "batch_size": 32, "steps": 100,
        "warmup_steps": 8, "prefetch_factor": 1, "n_positives": 4000,
        "manifest_rows": 8000, "negative_kind": "uniform", "hops": 2, "cap": 200,
        "max_neighbors": 20_000_000, "cpu_memory_budget_gib": 180,
        "gpu_memory_budget_gib": 24, "scenario_timeout_seconds": 3600,
        "stock_index_dtype": "auto",
        "arms": list(ARMS), "model": "DGCNN k281 hidden32 layers3 max_z1000 Adam1e-4 features=false",
        "gpu_policy": "first permitted whole H100 UUID mapped to cuda:0; all permitted GPUs idle before each owned GPU job",
        "failure_policy": "retain failed/guarded arms and continue independent real-data arms; fatal preconditions stop suite",
    }
    jobs = []
    smoke_common = [python, "-u", str(benchmark), "--fixture", "--device", "cuda", "--workers", "1",
                    "--batch-sizes", "2", "--steps", "2", "--warmup-steps", "1", "--cap", "3",
                    "--n-positives", "64", "--seed", str(protocol["seed"]), "--memory-budget-gib", "8"]
    for name, kind, gpu_budget, arms in (
            ("guard_refusal", "expected_gpu_refusal", "1000000", [ARMS[1]]),
            ("gpu_fixture", "gpu_smoke", "24", list(ARMS))):
        directory = output / "jobs" / name
        jobs.append({"name": name, "kind": kind, "gpu": True, "dataset": None,
                     "out": str(directory), "argv": smoke_common + ["--gpu-memory-budget-gib", gpu_budget,
                     "--arms", *arms, "--out", str(directory)]})
    datasets = [
        ("citation2", base / "data/citation2/prepared/train_csr", base / "data/citation2/prepared/train_pairs_csr", None),
        ("friendster", base / "data/friendster/full_csr", None, base / "data/friendster_DOWNLOAD_COMPLETE.json"),
        ("vk", base / "data/vk/train_csr", None, base / "data/vk_DOWNLOAD_COMPLETE.json"),
    ]
    inputs = {}
    for dataset, graph, positive_graph, marker in datasets:
        inputs[dataset] = {"graph": str(graph), "positive_graph": str(positive_graph) if positive_graph else None,
                           "download_marker": str(marker) if marker else None}
        pairs = output / "manifests" / (dataset + "_pairs.npy")
        common = [python, "-u", str(benchmark), "--graph-path", str(graph), "--file-endian", "little",
                  "--pairs-path", str(pairs), "--n-positives", "4000", "--negative-kind", "uniform",
                  "--batch-sizes", "32", "--steps", "100", "--warmup-steps", "8", "--prefetch-factor", "1",
                  "--hops", "2", "--cap", "200", "--max-neighbors", "20000000", "--seed", "20261124",
                  "--memory-budget-gib", "180", "--gpu-memory-budget-gib", "24",
                  "--stock-index-dtype", "auto",
                  "--timeout-seconds", "3600", "--device", "cuda"]
        if positive_graph:
            common += ["--train-pairs-csr", str(positive_graph)]
        directory = output / "manifests" / (dataset + "_prepare")
        jobs.append({"name": dataset + "_manifest", "kind": "manifest", "gpu": False, "dataset": dataset,
                     "out": str(directory), "pairs": str(pairs),
                     "argv": common + ["--workers", "1", "8", "--manifest-only", "--out", str(directory)]})
        for workers in protocol["workers"]:
            for arm in ARMS:
                name = f"{dataset}_{arm}_workers{workers}"
                directory = output / "jobs" / name
                jobs.append({"name": name, "kind": "pipeline", "gpu": True, "dataset": dataset,
                             "workers": workers, "arm": arm, "out": str(directory), "pairs": str(pairs),
                             "argv": common + ["--workers", str(workers), "--arms", arm, "--out", str(directory)]})
    return {"schema_version": 1, "created_unix": time.time(), "base": str(base), "repo": str(repo),
            "protocol": protocol, "protocol_sha256": digest_json(protocol), "inputs": inputs, "jobs": jobs}


def quality_gate(results: Path) -> dict | None:
    marker = results / "GPU_SUITE_COMPLETE.json"
    if not marker.exists():
        status = results / "gpu_status.json"
        if status.exists() and json.loads(status.read_text()).get("stage") == "failed":
            raise RuntimeError("Quality suite failed; pipeline GPU work is blocked")
        return None
    complete = json.loads(marker.read_text())
    if complete.get("quality_complete") is not True:
        raise RuntimeError("GPU_SUITE_COMPLETE does not attest completed quality work")
    summary_path = results / "QUALITY_COMPLETE.json"
    summary = json.loads(summary_path.read_text())
    rows = summary.get("results", [])
    if (len(rows) != 6 or len({row["name"] for row in rows}) != 6
            or sorted(row["variant"] for row in rows) != ["pv"] * 3 + ["union"] * 3
            or {(row["variant"], row.get("seed")) for row in rows}
            != {(variant, seed) for variant in ("union", "pv") for seed in (42, 43, 44)}):
        raise RuntimeError("Quality gate requires six unique results: three union and three PV")
    # The quality controller removes this lock in its finalizer, after writing
    # the completion marker. Wait for that finalizer rather than racing it.
    if (results / "GPU_SUITE_RUNNING.lock").exists():
        return None
    return {"marker_sha256": digest_file(marker), "summary_sha256": digest_file(summary_path),
            "quality_results": 6, "checked_unix": time.time()}


def parse_inventory(text: str) -> list[dict]:
    inventory = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 5 or not row[0].strip().isdigit() or not row[1].strip().startswith("GPU-"):
            raise RuntimeError("Unrecognized whole-GPU nvidia-smi inventory; refusing an ambiguous device")
        inventory.append({"index": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip(),
                          "memory_total_mib": int(float(row[3])), "driver_version": row[4].strip()})
    if not inventory:
        raise RuntimeError("No whole GPUs reported by nvidia-smi")
    return sorted(inventory, key=lambda row: row["index"])


def permitted_inventory(inventory: list[dict], environment: dict) -> list[dict]:
    tokens = environment.get("NVIDIA_VISIBLE_DEVICES", "all").strip()
    if tokens and tokens != "all":
        requested = [token.strip() for token in tokens.split(",")]
        if any(token.startswith("MIG-") for token in requested):
            raise RuntimeError("MIG selection is outside this whole-H100 protocol")
        selected = [row for row in inventory if row["uuid"] in requested or str(row["index"]) in requested]
        if len(selected) != len(set(requested)):
            raise RuntimeError("NVIDIA_VISIBLE_DEVICES does not match inventory")
        inventory = selected
    if not inventory:
        raise RuntimeError("No permitted NVIDIA devices")
    return inventory


def select_target(inventory: list[dict], environment: dict) -> dict:
    cuda = environment.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda and cuda != "all":
        first = cuda.split(",")[0].strip()
        if first.startswith("GPU-"):
            selected = next((row for row in inventory if row["uuid"] == first), None)
        elif first.isdigit() and int(first) < len(inventory):
            selected = inventory[int(first)]
        else:
            selected = None
        if selected is None:
            raise RuntimeError("Cannot resolve existing CUDA_VISIBLE_DEVICES to a permitted whole GPU")
    else:
        selected = inventory[0]
    if "H100" not in selected["name"]:
        raise RuntimeError(f"This frozen protocol requires H100, found {selected['name']}")
    return selected


def parse_compute_apps(text: str) -> list[dict]:
    apps = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 4 or not row[0].strip().startswith("GPU-") or not row[1].strip().isdigit():
            raise RuntimeError("Unrecognized compute-app output; refusing to assume the GPU is idle")
        apps.append({"gpu_uuid": row[0].strip(), "pid": int(row[1]),
                     "process_name": row[2].strip(), "used_memory_mib": row[3].strip()})
    return apps


def nvidia_query(query: str) -> str:
    result = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False, timeout=30)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed ({result.returncode}): {result.stderr.strip()[:1000]}")
    return result.stdout


def snapshot_input(settings: dict, base: Path) -> dict:
    paths = [Path(settings["graph"])]
    if settings["positive_graph"]:
        paths.append(Path(settings["positive_graph"]))
    files, graphs, small_metadata = {}, [], {}
    for graph in paths:
        indices, starts = graph / "edge_ends.bin", graph / "edge_starts.bin"
        with indices.open("rb") as stream:
            header = stream.read(8)
        if len(header) != 8:
            raise RuntimeError(f"Short GraphCSR header: {indices}")
        count = struct.unpack("<q", header)[0]
        if count < 0 or indices.stat().st_size != 8 + 4 * count or starts.stat().st_size % 8:
            raise RuntimeError(f"GraphCSR byte sizes/endian mismatch: {graph}")
        graphs.append({"path": str(graph), "nodes": starts.stat().st_size // 8, "adjacency_entries": count})
        for file in (indices, starts):
            stat = file.stat()
            files[str(file)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        for file in (graph / "revision_input.json", graph / "meta.json", graph.parent / "manifest.json"):
            if file.exists() and file.stat().st_size <= 8 << 20:
                small_metadata[str(file)] = digest_file(file)
    download = base / "data/input_download_manifest.json"
    provenance = json.loads(download.read_text()) if download.exists() else {}
    relevant = {name: provenance[name] for name in files if name in provenance}
    return {"files": files, "graphs": graphs, "small_metadata_sha256": small_metadata,
            "download_records": relevant, "proof_scope": "header/stat plus immutable download/preparation metadata; no rehash of billion-edge payloads"}


class Suite:
    def __init__(self, args, plan):
        self.args, self.plan = args, plan
        self.base, self.repo = Path(plan["base"]), Path(plan["repo"])
        self.results, self.output = self.base / "results", self.base / "results/pipeline"
        self.records, self.input_freezes, self.pair_freezes = [], {}, {}
        self.hardware = None
        self.environment = os.environ.copy()
        self.environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                NUMBA_NUM_THREADS="1", PYTHONUNBUFFERED="1", OUTDATED_IGNORE="1")

    def state(self, stage, **fields):
        value = {"stage": stage, "updated_unix": time.time(), "controller_pid": os.getpid(), **fields}
        atomic_json(self.output / "status.json", value)
        print(json.dumps(value), flush=True)

    def wait_quality(self):
        deadline = time.monotonic() + self.args.quality_wait_hours * 3600
        while True:
            gate = quality_gate(self.results)
            if gate is not None:
                atomic_json(self.output / "quality_gate.json", gate)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for completed quality suite")
            self.state("waiting_for_quality")
            time.sleep(self.args.poll_seconds)

    def wait_input(self, dataset):
        settings = self.plan["inputs"][dataset]
        marker = settings["download_marker"]
        deadline = time.monotonic() + self.args.input_wait_hours * 3600
        while marker and not Path(marker).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for {dataset} download marker")
            self.state("waiting_for_input", dataset=dataset, marker=marker)
            time.sleep(self.args.poll_seconds)
        if marker:
            if json.loads(Path(marker).read_text()).get("dataset") != dataset:
                raise RuntimeError(f"Download marker is for a different dataset: {marker}")
        snapshot = snapshot_input(settings, self.base)
        if marker:
            snapshot["download_marker_sha256"] = digest_file(Path(marker))
        self.input_freezes[dataset] = snapshot
        atomic_json(self.output / "inputs" / (dataset + ".json"), snapshot)

    def assert_input(self, dataset):
        expected = dict(self.input_freezes[dataset])
        expected.pop("download_marker_sha256", None)
        if snapshot_input(self.plan["inputs"][dataset], self.base) != expected:
            raise RuntimeError(f"Frozen {dataset} topology/provenance changed")
        if dataset in self.pair_freezes:
            path, frozen = self.pair_freezes[dataset]
            if digest_file(Path(path)) != frozen:
                raise RuntimeError(f"Frozen {dataset} pair manifest changed")

    def wait_idle(self, job):
        deadline = time.monotonic() + self.args.gpu_idle_wait_minutes * 60
        consecutive = 0
        while True:
            inventory = permitted_inventory(parse_inventory(nvidia_query(
                "--query-gpu=index,uuid,name,memory.total,driver_version")), self.environment)
            if self.hardware is None:
                target = select_target(inventory, self.environment)
                self.hardware = {"permitted_inventory": inventory, "target": target,
                                 "original_cuda_visible_devices": self.environment.get("CUDA_VISIBLE_DEVICES"),
                                 "gpu_mapping": "target UUID exposed as the sole CUDA device, cuda:0"}
                self.environment["CUDA_VISIBLE_DEVICES"] = target["uuid"]
                atomic_json(self.output / "hardware.json", self.hardware)
            if inventory != self.hardware["permitted_inventory"]:
                raise RuntimeError("GPU inventory/driver changed after hardware freeze")
            apps = parse_compute_apps(nvidia_query("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"))
            visible_uuids = {row["uuid"] for row in inventory}
            busy = [app for app in apps if app["gpu_uuid"] in visible_uuids]
            observation = {"job": job["name"], "checked_unix": time.time(), "busy_apps": busy}
            with (self.output / "gpu_idle_observations.jsonl").open("a") as stream:
                stream.write(json.dumps(observation) + "\n")
            consecutive = consecutive + 1 if not busy else 0
            if consecutive >= 2:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"GPUs stayed busy before {job['name']}; no process was killed")
            self.state("waiting_for_idle_gpu", job=job["name"], busy_apps=busy, clean_samples=consecutive)
            time.sleep(self.args.poll_seconds)

    def execute(self, job):
        assert_frozen_sources(self.repo, self.plan["source_hashes"])
        if job["dataset"]:
            self.assert_input(job["dataset"])
        directory = Path(job["out"])
        if directory.exists():
            raise RuntimeError(f"Refusing existing job output: {directory}")
        if job["gpu"]:
            self.wait_idle(job)
        self.state("running", job=job["name"], argv=job["argv"])
        log_path = self.output / "logs" / (job["name"] + ".log")
        log_path.parent.mkdir(exist_ok=True)
        started = time.time()
        with log_path.open("x") as stream:
            process = subprocess.Popen(job["argv"], cwd=self.repo, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, env=self.environment, start_new_session=True)
            while process.poll() is None:
                self.state("running", job=job["name"], child_pid=process.pid,
                           elapsed_seconds=time.time() - started, log=str(log_path))
                time.sleep(min(15, self.args.poll_seconds))
            code = process.wait()
        record = {"name": job["name"], "kind": job["kind"], "dataset": job["dataset"],
                  "returncode": code, "started_unix": started, "finished_unix": time.time(),
                  "log": str(log_path), "out": job["out"], "status": "failed"}
        report_path = directory / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            record["report_sha256"] = digest_file(report_path)
            runs = report.get("runs", [])
            if job["kind"] == "expected_gpu_refusal":
                messages = "\n".join(path.read_text(errors="replace") for path in directory.rglob("process.log"))
                if code != 0 and len(runs) == 1 and runs[0].get("status") == "failed" and "GPU preflight refused" in messages:
                    record["status"] = "expected_guard_refusal"
            elif job["kind"] == "manifest":
                manifest = report["manifest"]
                if (code == 0 and manifest["rows"] == 8000 and manifest["positive_occurrences"] == 4000
                        and manifest["sha256"] == digest_file(Path(job["pairs"]))):
                    record["status"] = "complete"
                    self.pair_freezes[job["dataset"]] = (job["pairs"], manifest["sha256"])
                    atomic_json(self.output / "manifests" / (job["dataset"] + "_frozen.json"), manifest)
            elif code == 0 and runs and all(run.get("status") == "complete" for run in runs):
                try:
                    self.validate_gpu_report(job, report)
                    record["status"] = "complete"
                except (KeyError, ValueError, RuntimeError) as error:
                    record["protocol_error"] = type(error).__name__ + ": " + str(error)
        self.records.append(record)
        atomic_json(self.output / "jobs" / (job["name"] + "_exit.json"), record)
        atomic_json(self.output / "job_records.json", self.records)
        return record

    def validate_gpu_report(self, job, report):
        if (report.get("benchmark_version") != 4
                or report.get("routing_contract", {}).get("version") != self.plan["protocol"]["routing_version"]
                or report.get("routing_contract", {}).get("bridge_arm") != "reference_union_compiled"):
            raise RuntimeError("Production-matched V4 routing metadata missing or changed")
        for run in report["runs"]:
            guard = run["gpu_memory"]["guard"]
            if (run["device"]["type"] != "cuda" or guard["status"] != "configured_before_model_allocation"
                    or guard["allocator_budget_bytes"] != 24 * 2**30 or not 0 < guard["allocator_fraction"] <= 1):
                raise RuntimeError(f"Actual GPU allocator guard smoke/protocol check failed: {job['name']}")
            if run["gpu_memory"]["reserved_peak_gib"] > 24:
                raise RuntimeError("Torch allocator peak exceeded the declared GPU budget")
            if run["model"]["sortpool_k"] != 281 or run["model"]["use_feature"] is not False:
                raise RuntimeError("Fixed DGCNN model protocol changed")
            if job["kind"] == "pipeline":
                cfg, timing = run["config"], run["timing"]
                if (len(report["runs"]) != 1 or cfg["arm"] != job["arm"] or cfg["num_workers"] != job["workers"]
                        or cfg["stock_index_dtype"] != "auto"
                        or timing["measured_steps"] != 100 or timing["measured_graphs"] != 3200
                        or timing["warmup_graphs_excluded"] != 256
                        or cfg["manifest_sha256"] != self.pair_freezes[job["dataset"]][1]):
                    raise RuntimeError("Frozen real-data steps/pairs/worker protocol changed")
                if job["arm"] == "reference_union_stock" and run["stock_physical_route"]["measured_temporary_row_dtype_counts"] != {"int32": 3200}:
                    raise RuntimeError("Native-auto stock row width is not int32 on the pinned three workloads")
        if job["kind"] == "gpu_smoke":
            from itertools import combinations
            refs = ("reference_union_stock", "reference_union_corrected", "reference_union_compiled")
            audits = report.get("equality_audits", [])
            if (len(report["runs"]) != 5 or {r["config"]["arm"] for r in report["runs"]} != set(ARMS)
                    or {(a["reference_arm"], a["candidate_arm"]) for a in audits} != set(combinations(refs, 2))
                    or len(audits) != 3 or not all(a["same_nodes_edges_drnl_and_initial_model"] for a in audits)):
                raise RuntimeError("GPU fixture did not establish five-arm/reference-bridge equality control")

    def comparisons(self):
        from itertools import combinations
        checks = []
        refs = ("reference_union_stock", "reference_union_corrected", "reference_union_compiled")
        for dataset in self.plan["inputs"]:
            for workers in (1, 8):
                reports = {}
                for arm in refs:
                    name = f"{dataset}_{arm}_workers{workers}"
                    record = next((r for r in self.records if r["name"] == name), None)
                    if record is not None and record["status"] == "complete":
                        reports[arm] = json.loads((Path(record["out"]) / "report.json").read_text())["runs"][0]
                for left_arm, right_arm in combinations(refs, 2):
                    item = {"dataset": dataset, "workers": workers, "reference_arm": left_arm, "candidate_arm": right_arm}
                    if left_arm in reports and right_arm in reports:
                        left, right = reports[left_arm], reports[right_arm]
                        equal = (left["audits"] == right["audits"]
                                 and left["model"]["initial_state_sha256"] == right["model"]["initial_state_sha256"]
                                 and left["config"]["manifest_sha256"] == right["config"]["manifest_sha256"])
                        item.update(status="matched" if equal else "mismatch", audited_pairs=len(left["audits"]),
                                    scope="S/edges/DRNL and initial model on frozen audited prefix; V4 compiled reference bridge")
                    else:
                        item["status"] = "unavailable_due_to_failed_or_unrun_arm"
                    checks.append(item)
        return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--quality-wait-hours", type=float, default=48)
    parser.add_argument("--input-wait-hours", type=float, default=48)
    parser.add_argument("--gpu-idle-wait-minutes", type=float, default=120)
    args = parser.parse_args(argv)
    if min(args.poll_seconds, args.quality_wait_hours, args.input_wait_hours, args.gpu_idle_wait_minutes) <= 0:
        parser.error("poll and wait limits must be positive")
    base = args.base.resolve()
    repo = args.repo.resolve() if args.repo else base / "code"
    results, output = base / "results", base / "results/pipeline"
    results.mkdir(parents=True, exist_ok=True)
    # Reject prior results before creating any status/plan/terminal marker.
    if (results / "PIPELINE_SUITE_COMPLETE.json").exists():
        raise RuntimeError("Existing terminal pipeline marker; no automatic rerun")
    lock = results / "PIPELINE_SUITE_RUNNING.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    suite = None
    try:
        reserve_output(output)
        plan = build_plan(base, repo, sys.executable)
        suite = Suite(args, plan)
        plan["source_hashes"] = source_hashes(repo)
        atomic_json(output / "plan.json", plan)
        suite.wait_quality()
        entered_datasets = set()
        for job in plan["jobs"]:
            dataset = job["dataset"]
            if dataset and dataset not in entered_datasets:
                suite.wait_input(dataset)
                entered_datasets.add(dataset)
            record = suite.execute(job)
            if record.get("protocol_error"):
                raise RuntimeError(f"Completed job violated the frozen protocol: {record['protocol_error']}")
            if job["kind"] in ("expected_gpu_refusal", "gpu_smoke", "manifest") and record["status"] not in ("complete", "expected_guard_refusal"):
                raise RuntimeError(f"Required {job['kind']} failed: {job['name']}; outputs retained")
        assert_frozen_sources(repo, plan["source_hashes"])
        checks = suite.comparisons()
        success = (all(record["status"] in ("complete", "expected_guard_refusal") for record in suite.records)
                   and not any(check["status"] == "mismatch" for check in checks))
        terminal = {"controller_finished": True, "success": success,
                    "terminal_status": "complete" if success else "completed_with_failures",
                    "completed_unix": time.time(), "protocol_sha256": plan["protocol_sha256"],
                    "records": suite.records, "reference_correction_audits": checks}
        atomic_json(output / "summary.json", terminal)
        suite.state(terminal["terminal_status"], summary=str(output / "summary.json"))
        atomic_json(results / "PIPELINE_SUITE_COMPLETE.json", terminal)
        return 0 if success else 1
    except BaseException as error:
        if suite is not None:
            suite.state("failed_precondition", error=type(error).__name__ + ": " + str(error))
            terminal = {"controller_finished": True, "success": False, "terminal_status": "failed_precondition",
                        "completed_unix": time.time(), "error": type(error).__name__ + ": " + str(error), "records": suite.records}
            atomic_json(output / "summary.json", terminal)
            atomic_json(results / "PIPELINE_SUITE_COMPLETE.json", terminal)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
