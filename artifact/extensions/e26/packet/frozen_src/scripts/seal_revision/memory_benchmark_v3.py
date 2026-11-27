#!/usr/bin/env python3
"""Guard-only V3: never budget shared arrays using non-unique summed RSS.

Isolated, CPU-only GraphCSR memory measurements; see memory/README.md.

No torch import, training, cache eviction, or graph-sized temporary buffers.
The coordinator samples fresh process trees; only summed PSS is a proportional
footprint. Summed RSS is explicitly non-unique when pages are shared.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import selectors
import signal
import statistics
import struct
import subprocess
import sys
import time

GIB = 1 << 30
REPRESENTATIONS = ("mixed-mmap", "mixed-resident", "scipy-int64", "conversion-peak")
PROCESS_RESERVE = 192 << 20
IMPLEMENTATION_VERSION = "conversion_offset_stable_tree_budget_guard_v3"
ORIGINAL_SOURCE_SHA256 = "34176d3b751cf4a9e2c79cb1326c4d5515d41622faae097f7cb3c6273b4b3251"
PREVIOUS_V2_SHA256 = "8f0a3c1ff2b8f3ea844325eba2568eff86a9a0c78e087c7cfe1b90bd5ed86a84"
GUARD_POLICY = {
    "version": "stable_pss_individual_rss_cgroup90_v3",
    "primary": "complete PSS with stable PID/start-time membership; non-atomic estimate",
    "individual_rss": "only max single-process RSS can independently exceed the budget; never sum shared RSS",
    "unknown_pss": "own footprint remains unknown; use cgroup charged memory at 90% of finite limit for safety",
    "cgroup_scope": "whole container/pod, including baseline, unrelated processes and reclaimable file cache; no baseline/cache subtraction or eviction",
    "guarantee": "sampled conservative policy; not a strict OOM-prevention or unique-footprint guarantee",
}


def inspect_graph(path: Path, endian: str) -> dict:
    """Read headers/stat only. edge_starts has N entries, not N+1."""
    path = path.resolve()
    prefix = ">" if endian == "big" else "<"

    def int32_file(name: str, optional: bool = False) -> int:
        file = path / name
        if optional and (not file.exists() or file.stat().st_size == 0):
            return 0
        with file.open("rb") as stream:
            header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"{name}: missing int64 header")
        count = struct.unpack(prefix + "q", header)[0]
        if count < 0 or file.stat().st_size != 8 + count * 4:
            raise ValueError(f"{name}: count/size/endian mismatch")
        return count

    entries = int32_file("edge_ends.bin")
    starts = path / "edge_starts.bin"
    size = starts.stat().st_size
    if size % 8:
        raise ValueError("edge_starts.bin: raw int64 length must be divisible by 8")
    nodes = size // 8
    if not nodes and entries:
        raise ValueError("nonempty edge array requires nodes")
    if nodes:
        with starts.open("rb") as stream:
            first = struct.unpack(prefix + "q", stream.read(8))[0]
            stream.seek(-8, os.SEEK_END)
            last = struct.unpack(prefix + "q", stream.read(8))[0]
        if first != 0 or not 0 <= last <= entries:
            raise ValueError("edge_starts.bin: invalid first/last offset or wrong endian")
    timestamps = int32_file("timestamps.bin", optional=True)
    if timestamps not in (0, entries):
        raise ValueError("timestamp count must equal adjacency-entry count")
    return {
        "path": str(path), "file_endian": endian, "nodes": nodes,
        "adjacency_entries": entries, "timestamp_entries": timestamps,
        "input_file_bytes": {p.name: p.stat().st_size for p in
                             (path / "edge_ends.bin", starts, path / "timestamps.bin")
                             if p.exists()},
        "validation": "header lengths and first/last offset only; no full semantic graph audit",
    }


def logical_bytes(meta: dict, representation: str, include_timestamps: bool) -> dict:
    n, m = meta["nodes"], meta["adjacency_entries"]
    scipy = representation in ("scipy-int64", "conversion-peak")
    parts = {"indices": m * (8 if scipy else 4),
             "offsets": (n + int(scipy)) * 8,
             "values": m if scipy else 0,
             "timestamps": meta["timestamp_entries"] * 4 if include_timestamps else 0}
    return {"components_bytes": parts, "total_bytes": sum(parts.values()),
            "total_gib": sum(parts.values()) / GIB,
            "timestamp_bytes_on_disk": meta["timestamp_entries"] * 4,
            "timestamps_loaded": include_timestamps and bool(meta["timestamp_entries"]),
            "headers_excluded_bytes": 8 + (8 if meta["timestamp_entries"] else 0),
            "indices_dtype": "int64" if scipy else "int32",
            "offsets_dtype": "int64", "values_dtype": "int8" if scipy else None,
            "terminal_offset_added": scipy}


def estimate_peak_bytes(meta: dict, representation: str, workers: int,
                        include_timestamps: bool, chunk_bytes: int) -> int:
    target = logical_bytes(meta, representation, include_timestamps)["total_bytes"]
    # Conservative full-source residency during conversion, even though mmap
    # pages are reclaimable. Worker consumers share arrays read-only via fork.
    source = (logical_bytes(meta, "mixed-mmap", include_timestamps)["total_bytes"]
              if representation == "conversion-peak" else 0)
    return target + source + workers * PROCESS_RESERVE + 2 * chunk_bytes


def available_memory_bytes() -> int | None:
    if not Path("/proc/meminfo").exists():
        return None
    info = parse_kb_file(Path("/proc/meminfo"))
    available = info.get("MemAvailable")
    # Respect the current cgroup v2 limit where available; no cgroup mutation.
    try:
        group = next(line.split(":", 2)[2] for line in
                     Path("/proc/self/cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        root = Path("/sys/fs/cgroup") / group.lstrip("/")
        maximum = (root / "memory.max").read_text().strip()
        if maximum != "max":
            remaining = max(0, int(maximum) - int((root / "memory.current").read_text()))
            available = min(available, remaining) if available is not None else remaining
    except (OSError, StopIteration, ValueError):
        pass
    return available


def parse_kb_file(path: Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        key, sep, rest = line.partition(":")
        fields = rest.split()
        if sep and fields and fields[0].isdigit():
            result[key] = int(fields[0]) * (1024 if len(fields) > 1 and fields[1] == "kB" else 1)
    return result


def cgroup_memory(root: Path = Path('/sys/fs/cgroup')) -> dict:
    result = {"current_bytes": None, "max_bytes": None, "anon_bytes": None, "file_bytes": None,
              "threshold_fraction": .9, "threshold_bytes": None, "backend": "unavailable"}
    for backend, current_name, maximum_name, stat_name in (
        ("cgroup_v2", "memory.current", "memory.max", "memory.stat"),
        ("cgroup_v1", "memory/memory.usage_in_bytes", "memory/memory.limit_in_bytes", "memory/memory.stat")):
        try:
            current = int((root / current_name).read_text().strip())
            maximum_raw = (root / maximum_name).read_text().strip()
            maximum = int(maximum_raw) if maximum_raw != 'max' else None
            # v1 often represents an unlimited limit with a near-int64 sentinel.
            if maximum is not None and (maximum <= 0 or maximum >= 2**60):
                maximum = None
            if current < 0:
                raise ValueError('negative cgroup charge')
            result.update(current_bytes=current, max_bytes=maximum, backend=backend,
                          threshold_bytes=maximum * 9 // 10 if maximum is not None else None)
            try:
                values = dict(line.split() for line in (root / stat_name).read_text().splitlines())
                for output, keys in (("anon_bytes", ("anon", "total_rss", "rss")),
                                     ("file_bytes", ("file", "total_cache", "cache"))):
                    for key in keys:
                        if key in values:
                            result[output] = int(values[key]); break
            except (OSError, ValueError):
                pass
            break
        except (OSError, ValueError):
            continue
    return result


def budget_decision(sample: dict, limit: int, charged: dict) -> dict:
    """No cross-PID RSS sum is ever interpreted as unique/allocated memory."""
    def nonnegative(value):
        return value if type(value) is int and value >= 0 else None
    pss = nonnegative(sample.get('tree_pss_bytes'))
    individual = nonnegative(sample.get('max_process_rss_bytes'))
    stable_pss = bool(sample.get('pss_complete') is True and sample.get('membership_stable') is True and pss is not None)
    current, maximum = nonnegative(charged.get('current_bytes')), nonnegative(charged.get('max_bytes'))
    threshold = maximum * 9 // 10 if maximum else None
    reason, basis = None, 'stable_complete_tree_pss' if stable_pss else 'unknown_own_footprint'
    if stable_pss and pss > limit:
        reason = 'stable complete PSS tree estimate exceeded memory budget'
    elif individual is not None and individual > limit:
        reason, basis = 'single-process RSS exceeded memory budget', 'individual_rss_lower_bound'
    elif not stable_pss and current is not None and threshold is not None and current >= threshold:
        reason, basis = 'cgroup charged memory reached 90% of finite limit while own PSS was unknown', 'cgroup_safety_threshold'
    return {'abort_reason': reason, 'decision': 'abort' if reason else 'continue', 'basis': basis,
            'own_footprint_status': 'estimated_nonatomic' if stable_pss else 'unknown',
            'own_pss_estimate_bytes': pss if stable_pss else None, 'memory_budget_bytes': limit,
            'max_individual_rss_bytes': individual,
            'nonunique_rss_sum_diagnostic_only_bytes': sample.get('tree_rss_sum_nonunique_bytes'),
            'cgroup': charged, 'cgroup_threshold_bytes': threshold}


def completion_status(returncode: int, result_event_complete: bool, abort_reason) -> str:
    return 'complete' if returncode == 0 and result_event_complete and abort_reason is None else 'failed'


def tree_members(root_pid: int) -> tuple[dict[int, int], bool]:
    """PID -> /proc start ticks, detecting missing members and PID reuse."""
    pending, members, complete = [root_pid], {}, True
    while pending:
        pid = pending.pop()
        if pid in members:
            continue
        try:
            # comm may contain spaces/parentheses; the suffix begins at field3.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            members[pid] = int(fields[19])  # field22: process start ticks
            pending.extend(map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
        except (OSError, ValueError, IndexError):
            complete = False
    return members, complete and root_pid in members


def process_memory(pid: int) -> dict | None:
    try:
        smaps = parse_kb_file(Path(f"/proc/{pid}/smaps_rollup"))
        return {"pid": pid, "rss_bytes": smaps["Rss"], "pss_bytes": smaps["Pss"]}
    except (OSError, KeyError):
        try:
            status = parse_kb_file(Path(f"/proc/{pid}/status"))
            if "VmRSS" in status:
                return {"pid": pid, "rss_bytes": status["VmRSS"], "pss_bytes": None}
        except OSError:
            pass
    return None


def sample_tree(root_pid: int) -> dict:
    """Non-atomic PSS scan bracketed by complete PID/start-time tree snapshots."""
    if platform.system() != "Linux":
        return {"backend": "logical-metadata-only", "process_count": None,
                "tree_pss_bytes": None, "tree_rss_sum_nonunique_bytes": None,
                "max_process_rss_bytes": None, "pss_complete": False,
                "membership_stable": False, "publication_eligible": False,
                "membership_before": [], "membership_after": []}
    before, before_complete = tree_members(root_pid)
    records = [record for pid in sorted(before) if (record := process_memory(pid)) is not None]
    after, after_complete = tree_members(root_pid)
    membership_stable = before_complete and after_complete and before == after
    complete = bool(records) and len(records) == len(before) and all(r["pss_bytes"] is not None for r in records)
    return {"backend": "linux-proc-smaps_rollup", "process_count": len(records),
            "tree_pss_bytes": sum(r["pss_bytes"] for r in records) if complete else None,
            "tree_rss_sum_nonunique_bytes": sum(r["rss_bytes"] for r in records),
            "max_process_rss_bytes": max((r["rss_bytes"] for r in records), default=0),
            "pss_complete": complete, "processes": records,
            "membership_stable": membership_stable,
            "membership_before_complete": before_complete, "membership_after_complete": after_complete,
            "membership_before": [{"pid": pid, "start_ticks": ticks} for pid, ticks in sorted(before.items())],
            "membership_after": [{"pid": pid, "start_ticks": ticks} for pid, ticks in sorted(after.items())],
            "publication_eligible": bool(membership_stable and complete)}


def label_sample(sample: dict, phase_before: str, phase_after: str, consumers: int) -> dict:
    """Reject scans crossing phase boundaries or observing shared-tree teardown."""
    reasons = []
    if not sample.get("membership_stable"):
        reasons.append("tree_membership_changed_or_incomplete")
    if not sample.get("pss_complete"):
        reasons.append("incomplete_pss")
    if phase_before != phase_after:
        reasons.append("phase_changed_during_scan")
    if phase_before == "releasing_consumers" or phase_after == "releasing_consumers":
        reasons.append("teardown")
    if phase_before == "warm_shared_consumers" and sample.get("process_count") != consumers:
        reasons.append("unexpected_consumer_count")
    sample.update(phase=phase_before, phase_after_scan=phase_after,
                  phase_stable=phase_before == phase_after,
                  publication_eligible=not reasons, invalid_reasons=reasons)
    sample["steady_eligible"] = bool(not reasons and phase_before == "warm_shared_consumers"
                                     and sample.get("process_count") == consumers)
    return sample


def stream_into(path: Path, destination, file_dtype, offset: int, chunk_bytes: int) -> None:
    """Bounded temporary read buffer; assignment converts endian/width in chunks."""
    import numpy as np
    per_chunk = max(1, chunk_bytes // file_dtype.itemsize)
    with path.open("rb") as stream:
        stream.seek(offset)
        for start in range(0, destination.size, per_chunk):
            count = min(per_chunk, destination.size - start)
            raw = stream.read(count * file_dtype.itemsize)
            if len(raw) != count * file_dtype.itemsize:
                raise EOFError(f"short read: {path}")
            destination[start:start + count] = np.frombuffer(raw, dtype=file_dtype, count=count)


def load_mixed(meta: dict, resident: bool, include_timestamps: bool, chunk_bytes: int):
    import numpy as np
    root = Path(meta["path"])
    prefix = ">" if meta["file_endian"] == "big" else "<"
    definitions = [("indices", "edge_ends.bin", meta["adjacency_entries"], "i4", 8),
                   ("offsets", "edge_starts.bin", meta["nodes"], "i8", 0)]
    if include_timestamps and meta["timestamp_entries"]:
        definitions.append(("timestamps", "timestamps.bin", meta["timestamp_entries"], "i4", 8))
    arrays = {}
    for key, name, count, code, offset in definitions:
        dtype = np.dtype(prefix + code)
        if resident:
            array = np.empty(count, dtype=np.dtype(code))
            stream_into(root / name, array, dtype, offset, chunk_bytes)
        else:
            array = (np.memmap(root / name, mode="r", dtype=dtype, offset=offset, shape=(count,))
                     if count else np.empty(0, dtype=dtype))
        arrays[key] = array
    return arrays


def close_mixed(arrays: dict) -> None:
    # No exported views survive this call.
    for array in arrays.values():
        mm = getattr(array, "_mmap", None)
        if mm is not None:
            mm.close()
    arrays.clear()


def load_scipy(meta: dict, include_timestamps: bool, chunk_bytes: int, source=None):
    import numpy as np
    from scipy.sparse import csr_matrix
    n, m = meta["nodes"], meta["adjacency_entries"]
    arrays = {"indices": np.empty(m, dtype=np.int64),
              "offsets": np.empty(n + 1, dtype=np.int64), "values": np.empty(m, dtype=np.int8)}
    if source is None:
        prefix = ">" if meta["file_endian"] == "big" else "<"
        root = Path(meta["path"])
        stream_into(root / "edge_ends.bin", arrays["indices"], np.dtype(prefix + "i4"), 8, chunk_bytes)
        stream_into(root / "edge_starts.bin", arrays["offsets"][:n], np.dtype(prefix + "i8"), 0, chunk_bytes)
    else:
        for key in ("indices", "offsets"):
            per_chunk = max(1, chunk_bytes // arrays[key].dtype.itemsize)
            for start in range(0, source[key].size, per_chunk):
                # Mixed offsets omit the terminal sentinel; the destination has
                # N+1 entries. Cap BOTH slices at the source length so the final
                # partial chunk never includes that extra destination entry.
                stop = min(start + per_chunk, source[key].size)
                arrays[key][start:stop] = source[key][start:stop]
    arrays["offsets"][n] = m
    arrays["values"].fill(1)
    if include_timestamps and meta["timestamp_entries"]:
        arrays["timestamps"] = np.empty(m, dtype=np.int32)
        if source is None:
            dtype = np.dtype((">" if meta["file_endian"] == "big" else "<") + "i4")
            stream_into(Path(meta["path"]) / "timestamps.bin", arrays["timestamps"], dtype, 8, chunk_bytes)
        else:
            per_chunk = max(1, chunk_bytes // 4)
            for start in range(0, m, per_chunk):
                arrays["timestamps"][start:start + per_chunk] = source["timestamps"][start:start + per_chunk]
    # The public tuple constructor can silently downcast indices on tiny graphs
    # and allocate another whole index array. Attach already native int64 arrays
    # to an empty CSR instead, then assert identity and the actual dtype.
    matrix = csr_matrix((0, 0), dtype=np.int8)
    matrix._shape = (n, n)
    matrix.indices, matrix.indptr, matrix.data = arrays["indices"], arrays["offsets"], arrays["values"]
    assert matrix.indices is arrays["indices"] and matrix.indptr is arrays["offsets"]
    assert matrix.indices.dtype == matrix.indptr.dtype == np.dtype(np.int64)
    return arrays, matrix


def touch_arrays(arrays: dict) -> int:
    """Read one byte from each OS page (plus the tail); bounded reduction."""
    import numpy as np
    checksum = 0
    page = os.sysconf("SC_PAGE_SIZE")
    for array in arrays.values():
        byte_view = array.view(np.uint8)
        # Bound even the reduction's working set for billion-edge inputs.
        for start in range(0, byte_view.size, 8 << 20):
            checksum += int(byte_view[start:start + (8 << 20):page].sum(dtype=np.uint64))
        if byte_view.size:
            checksum += int(byte_view[-1])
    return checksum


def emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


def child_main(args) -> int:
    import numpy as np
    if args.representation in ("scipy-int64", "conversion-peak"):
        import scipy.sparse  # noqa: F401
    meta = inspect_graph(args.graph_path, args.file_endian)
    emit("phase", phase="baseline_imported")
    time.sleep(args.hold_seconds)
    source, matrix = None, None
    if args.representation.startswith("mixed-"):
        arrays = load_mixed(meta, args.representation == "mixed-resident", args.include_timestamps, args.chunk_bytes)
    else:
        if args.representation == "conversion-peak":
            source = load_mixed(meta, False, args.include_timestamps, args.chunk_bytes)
            emit("phase", phase="conversion_source_mapped")
        emit("phase", phase="allocating_or_converting")
        arrays, matrix = load_scipy(meta, args.include_timestamps, args.chunk_bytes, source=source)
        if source is not None:
            emit("phase", phase="conversion_source_and_destination_live")
            time.sleep(args.hold_seconds)
            close_mixed(source)
            source = None
            gc.collect()
    for array in arrays.values():
        array.flags.writeable = False
    emit("phase", phase="loaded_unfaulted" if args.representation == "mixed-mmap" else "loaded_resident",
         source_mappings_live=False, array_dtypes={k: str(a.dtype) for k, a in arrays.items()},
         array_nbytes={k: int(a.nbytes) for k, a in arrays.items()}, scipy_int64=matrix is not None)
    time.sleep(args.hold_seconds)
    checksum = touch_arrays(arrays)
    emit("phase", phase="warm_single_consumer", checksum=checksum)
    time.sleep(args.hold_seconds)
    children, pipes = [], []
    try:
        for _ in range(args.consumer_workers - 1):
            read_fd, write_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                try:
                    touch_arrays(arrays)
                    os.write(write_fd, b"1")
                    os.close(write_fd)
                    while True:
                        time.sleep(60)
                except BaseException:
                    os._exit(1)
            os.close(write_fd)
            children.append(pid)
            pipes.append(read_fd)
        for fd in pipes:
            ready = os.read(fd, 1)
            os.close(fd)
            if ready != b"1":
                raise RuntimeError("read-only sharing worker failed")
        emit("phase", phase="warm_shared_consumers", consumers=args.consumer_workers,
             sharing="read-only fork; no copy-on-write mutation")
        time.sleep(args.hold_seconds)
    finally:
        # Flush this boundary BEFORE killing/freeing any consumer, including the
        # single-consumer case; coordinator also drains events after each scan.
        emit("phase", phase="releasing_consumers", consumers=args.consumer_workers)
        for pid in children:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for pid in children:
            os.waitpid(pid, 0)
    emit("result", status="complete", numpy_version=np.__version__,
         source_mappings_live=False, gpu_memory_bytes=None,
         torch_imported="torch" in sys.modules,
         gpu_memory_note="No GPU framework imported; no GPU allocation measured or attributed to graph.")
    return 0


def maxima(samples: list[dict]) -> dict:
    eligible = [s for s in samples if s.get("publication_eligible") is True]
    result = {}
    for key in ("tree_pss_bytes", "tree_rss_sum_nonunique_bytes", "max_process_rss_bytes"):
        values = [s[key] for s in eligible if s.get(key) is not None]
        result[key] = max(values) if values else None
        result[key.replace("_bytes", "_gib")] = result[key] / GIB if values else None
        result["median_" + key] = statistics.median(values) if values else None
        result["median_" + key.replace("_bytes", "_gib")] = statistics.median(values) / GIB if values else None
    result["samples"] = len(eligible)
    result["total_samples"] = len(samples)
    result["invalid_samples"] = len(samples) - len(eligible)
    starts = [s["scan_started_seconds"] for s in eligible if "scan_started_seconds" in s]
    ends = [s["scan_finished_seconds"] for s in eligible if "scan_finished_seconds" in s]
    result["first_valid_scan_started_seconds"] = min(starts) if starts else None
    result["last_valid_scan_finished_seconds"] = max(ends) if ends else None
    return result


def steady_summary(samples: list[dict], consumers: int) -> dict:
    valid = [s for s in samples if s.get("steady_eligible") is True
             and s.get("publication_eligible") is True and s.get("pss_complete") is True
             and s.get("membership_stable") is True and s.get("phase_stable") is True
             and s.get("phase") == "warm_shared_consumers" and s.get("process_count") == consumers]
    result = maxima(valid)
    result.update(total_samples=len(samples), valid_samples=len(valid), invalid_samples=len(samples) - len(valid),
                  rejected_samples=len(samples) - len(valid),
                  minimum_valid_samples=2, expected_consumers=consumers,
                  steady_status="complete" if len(valid) >= 2 else "missing_insufficient_valid_samples")
    if len(valid) < 2:
        for key in result:
            if key.startswith("median_"):
                result[key] = None
    return result


def kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_isolated(args, representation: str, workers: int, meta: dict) -> dict:
    estimate = estimate_peak_bytes(meta, representation, workers, args.include_timestamps, args.chunk_bytes)
    available = available_memory_bytes()
    limit = int(args.memory_budget_gib * GIB)
    result = {"representation": representation, "consumers": workers,
              "guard_policy": GUARD_POLICY, "baseline_cgroup_memory": cgroup_memory(),
              "logical": logical_bytes(meta, representation, args.include_timestamps),
              "preflight": {"estimated_peak_bytes": estimate, "estimated_peak_gib": estimate / GIB,
                            "memory_budget_bytes": limit, "available_bytes": available}}
    if estimate > limit or (available is not None and estimate > available * .8):
        return {**result, "status": "skipped_preflight", "reason": "conservative peak exceeds budget or 80% of current available/cgroup memory"}
    if workers > 1 and not hasattr(os, "fork"):
        return {**result, "status": "unsupported", "reason": "read-only worker sharing requires fork"}
    command = [sys.executable, str(Path(__file__).resolve()), "--_child",
               "--graph-path", str(args.graph_path), "--representation", representation,
               "--consumer-workers", str(workers), "--file-endian", args.file_endian,
               "--hold-seconds", str(args.hold_seconds), "--chunk-bytes", str(args.chunk_bytes),
               "--sample-interval", str(args.sample_interval)]
    if args.include_timestamps:
        command.append("--include-timestamps")
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    samples, events, phase, started = [], [], "startup", time.monotonic()
    stderr_path = args.out / f"{representation}.workers{workers}.stderr.log"
    with stderr_path.open("w") as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors,
                                   bufsize=0, start_new_session=True, env=env)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        aborted = None
        stdout_buffer = b""
        exit_cleanup_done = False

        def drain_events(wait):
            nonlocal stdout_buffer, phase
            ready = selector.select(timeout=wait)
            while ready:
                for key, _ in ready:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    stdout_buffer += chunk
                    while b"\n" in stdout_buffer:
                        line, stdout_buffer = stdout_buffer.split(b"\n", 1)
                        event = json.loads(line)
                        events.append(event)
                        if event["event"] == "phase":
                            phase = event["phase"]
                ready = selector.select(timeout=0)

        try:
            while True:
                drain_events(args.sample_interval)
                phase_before = phase
                scan_start = time.monotonic() - started
                sample = sample_tree(process.pid)
                scan_end = time.monotonic() - started
                # A phase marker is flushed before teardown. Read pending events
                # AGAIN so a long PSS scan cannot be mislabeled as steady.
                drain_events(0)
                label_sample(sample, phase_before, phase, workers)
                sample.update(elapsed_seconds=scan_end, scan_started_seconds=scan_start,
                              scan_finished_seconds=scan_end)
                sample['budget_guard'] = budget_decision(sample, limit, cgroup_memory())
                samples.append(sample)
                if sample['budget_guard']['abort_reason']:
                    aborted = sample['budget_guard']['abort_reason']
                if time.monotonic() - started > args.timeout_seconds:
                    aborted = "scenario timeout"
                if aborted:
                    kill_process_group(process.pid)
                    break
                # A loader that crashes after forking must not leave consumers
                # holding stdout open until the full scenario timeout expires.
                if process.poll() is not None and not exit_cleanup_done:
                    kill_process_group(process.pid)
                    exit_cleanup_done = True
                if process.poll() is not None and not selector.get_map():
                    break
            process.wait()
        finally:
            selector.close()
            if process.poll() is None:
                kill_process_group(process.pid)
                process.wait()
            process.stdout.close()
    samples_path = args.out / f"{representation}.workers{workers}.samples.jsonl"
    with samples_path.open("w") as stream:
        for sample in samples:
            stream.write(json.dumps(sample) + "\n")
    phases = {}
    for name in dict.fromkeys(s["phase"] for s in samples):
        selected = [s for s in samples if s["phase"] == name]
        phases[name] = steady_summary(selected, workers) if name == "warm_shared_consumers" else maxima(selected)
    steady = phases.get("warm_shared_consumers", steady_summary([], workers))
    complete = any(e.get("status") == "complete" for e in events if e["event"] == "result")
    return {**result, "status": completion_status(process.returncode, complete, aborted),
            "abort_reason": aborted, "returncode": process.returncode,
            "elapsed_seconds": time.monotonic() - started, "peak": maxima(samples),
            "phases": phases, "steady_shared": steady, "events": events, "samples_file": str(samples_path),
            "stderr_file": str(stderr_path), "child_pid": process.pid}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-path", type=Path, required=True)
    p.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    p.add_argument("--memory-budget-gib", type=float)
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, nargs="+", default=[1], help="Total read-only consumers, e.g. 1 4 16")
    p.add_argument("--file-endian", choices=("little", "big"), default="little")
    p.add_argument("--include-timestamps", action="store_true")
    p.add_argument("--chunk-bytes", type=int, default=8 << 20)
    p.add_argument("--sample-interval", type=float, default=.1)
    p.add_argument("--hold-seconds", type=float, default=2)
    p.add_argument("--timeout-seconds", type=float, default=3600)
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--representation", choices=REPRESENTATIONS, help=argparse.SUPPRESS)
    p.add_argument("--consumer-workers", type=int, default=1, help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.chunk_bytes < 8 or args.sample_interval <= 0 or args.hold_seconds < args.sample_interval * 2:
        p.error("chunk>=8, interval>0, hold>=2*interval required")
    if min(args.workers) < 1 or args.consumer_workers < 1 or args.timeout_seconds <= 0:
        p.error("workers and timeout must be positive")
    if args._child:
        return child_main(args)
    if args.out is None or args.memory_budget_gib is None or args.memory_budget_gib <= 0:
        p.error("--out and a positive --memory-budget-gib are required")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        p.error("V3 requires a new separate supplement directory; existing V2/raw outputs are never overwritten")
    args.graph_path = args.graph_path.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = inspect_graph(args.graph_path, args.file_endian)
    report = {"schema_version": 1,
              "implementation": {"version": IMPLEMENTATION_VERSION, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "frozen_original_sha256": ORIGINAL_SOURCE_SHA256,
                                 "previous_v2_sha256": PREVIOUS_V2_SHA256},
              "supplement_policy": "Separate V3 measurements; retain V2 raw rows and failures without relabeling or merging them here",
              "units": "bytes and GiB (2**30)", "graph": meta,
              "platform": platform.platform(), "python": sys.version,
              "measurement_contract": {
                  "footprint": "sum process-tree PSS; proportional sharing with external processes",
                  "rss": "sum RSS is non-unique and double counts shared pages; diagnostic only",
                  "steady": "warm_shared median requires >=2 samples with stable phase and PID/start-time membership, complete PSS and exactly the declared consumers; teardown/cross-boundary scans excluded. Other phase medians are descriptive. Absolute process footprint, not baseline-subtracted",
                  "peak": "maxima use only complete-PSS, stable-membership/phase scans, excluding teardown; non-atomic sampled maxima, not allocator high-water marks or a strict lower bound",
                  "sample_validity": "PID/start ticks and tree-discovery completeness checked before and after each PSS scan; pending phase events drained after scan. Endpoint consistency cannot exclude an unobserved transient process between snapshots. Invalid raw samples retained for diagnosis/safety guards only.",
                  "cache": "No cache eviction. loaded_unfaulted means this process has not touched the arrays; OS page-cache state is uncontrolled. warm means each page was read, not pinned residency.",
                  "workers": "total consumers including loader; read-only fork, not independent copies or a training DataLoader",
                  "timestamp_policy": "excluded by default; on-disk bytes always separate",
                  "gpu": "not used; GPU/framework memory is not attributed to graph",
                  "fallback": "non-Linux: logical metadata only; RSS/PSS null"},
              "budget_guard_policy": GUARD_POLICY,
              "runs": []}
    for representation in args.representations:
        for workers in args.workers:
            run = run_isolated(args, representation, workers, meta)
            report["runs"].append(run)
            temp = args.out / "report.json.tmp"
            temp.write_text(json.dumps(report, indent=2) + "\n")
            temp.replace(args.out / "report.json")
            print(json.dumps({"representation": representation, "consumers": workers, "status": run["status"]}), flush=True)
    return int(any(run["status"] != "complete" for run in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
