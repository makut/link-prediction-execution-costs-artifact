# Corrected native association-map control, v1

This is a local source package for a separately scheduled CPU control. No full-graph replay has been executed by this package's author. The passing tests execute small literal graphs, including two fresh worker processes with 128 repetitions of a literal fixture. They are correctness/admission evidence, not a Friendster/VK speed measurement.

## Compared implementations

`reference_full_clear` allocates and initializes an int64 association array of length N on every invocation. `persistent_touched_reset` initializes the same 8N-byte array once in each worker, touches only S entries, and resets those entries on normal and exceptional exits. Each call in both arms creates fresh C++ vectors and fresh cloned buffers for local rows, local columns and original global CSR edge IDs. No persistent output-vector capacity is reused. Both arms scan selected CSR rows in the same order and use identical extra safety checks (including O(|S| log |S|) uniqueness validation).

These are **matched standalone C++17 controls**, not an unmodified Torch extension. The optional third arm, `upstream_torch_sparse_direct`, invokes the actual installed `torch.ops.torch_sparse.saint_subgraph`, through the hash-bound frozen external helper. With `--include-upstream`, an actual upstream fixture must pass before graph conversion starts. This arm is required for a comparison to the installed upstream implementation. It does not prove that a published PENCIL experiment used that version.

Source pin: pytorch_sparse 0.6.18, commit `7d22892fb47626d2c5b0171769d44bdc8edd606c`, CPU Git blob `21a13be8067e66ef4d6f9ae57f98407332f31929`. The exact MIT license, exact pinned CPU source bytes, and restoration receipt for one cache-added trailing newline are under `third_party/`.

## Input and timing contract

The four supplied plans select 128 or all 768 already archived S records for each graph. The 128 selection balances the 12 `(sampler, cap, workload)` strata, uses seed 20261125 for hash ordering, then preserves archive order. It does not resample graph vertices or edges, change any S bytes/order, or use virtual graph padding. The int32 and int64 archives contain repeated S, not independent datasets; one int32 archive supplies each replay.

The native primitive returns ordered raw induced entries and original CSR edge IDs, retaining the target edge. Identical packaging removes both target directions, builds local CSR, and computes DRNL. Every repeated call checks raw ordered results against an independent Python oracle and local edges/DRNL against archived hashes. The selected-row contract (sorted, unique, loop-free rows) is checked before timing. The Python oracle scans selected neighbors outside timers and can dominate wall time.

`edge_seconds` includes invocation, native fresh output clones, NumPy views, target removal and local CSR packaging. `raw_operator_seconds` is a narrower diagnostic. DRNL is separate. Conversion, file hashing, validation, oracle work, fixture admission and persistent-map initialization are outside query timers. Initialization time, 8N bytes per worker, build/source/library bindings, setup time and absolute process memory diagnostics are reported separately. Repeated cached timings do not supply independent-seed uncertainty or model-quality evidence. No speed advantage is inferred by the scripts.

The adapter requires contiguous native int64 arrays and never widens a whole graph implicitly. The explicit streaming converter reuses the unmodified frozen external helper. It writes fresh `rowptr.i8`, `col.i8` and a manifest, and verifies actual content. Workers rehash those files before mapping. Direct CSR invocation avoids an E-length global COO row array.

## Memory, disk and time

All arms use separate sequential fresh processes, one CPU thread and CPU-only environment. A creator-PID guard prevents a parent-created map from hiding worker initialization through fork. The native state rejects concurrent calls. Python interruption terminates and reaps only the worker's newly created session; an outer supervisor must own cleanup if the parent is abruptly killed. No unrelated process group is signaled.

For the current full physical graphs, GiB = 2^30:

| Graph | int64 CSR files | map per worker | conversion peak estimate | 128-case worker peak estimate | conversion free-disk minimum |
|---|---:|---:|---:|---:|---:|
| Friendster | 27.4013 GiB | 0.4888 GiB | 42.8576 GiB | 28.8907 GiB | 29.4013 GiB |
| VK | 96.8965 GiB | 1.3308 GiB | 148.3633 GiB | 99.2327 GiB | 98.8965 GiB |

These are conservative admission estimates, not observed peaks. Conversion includes source/native working sets, map allowance, chunk allowance and 1 GiB; its disk reserve is native files plus 2 GiB. The parent additionally requires 1 GiB free for result/log files. Native files may be reused only with their intact content-bound manifest. Do not convert both graphs unnecessarily. Current host and cgroup headroom must admit the estimate at 85%; charged reclaimable file cache can conservatively cause a skip. Do not bypass this guard. Default budgets intentionally reject large-graph execution.

The recommended initial replay is **128 cases, three arms, one warmup plus three timed repeats**, not all 768. The selected maximum neighbor scans are 375735 (Friendster) and 880110 (VK), below the explicit 20000000 guard. Default per-arm timeout is 3600 s (maximum accepted 14400 s); conversion defaults to 1800 s. With three arms and default conversion/admission limits, the explicit worker/conversion ceiling is 3*3600+1800+120 = 12720 s, plus parent setup, hashing and build overhead. It is not an ETA. Timeout/failure retains independent evidence and cannot produce success. Run the second graph only after inspecting the first result.

## Local verification

Requires Python, NumPy, SciPy, pytest and a C++17 compiler. The upstream arm additionally requires an actual compatible torch/torch-sparse installation; the package installs nothing.

```sh
python -m pytest -q work/vldb_corrected_native/test_native_v1.py work/vldb_corrected_native/test_replay_v1.py --basetemp work/vldb_corrected_native/test_runs/new_run
python -m py_compile work/vldb_corrected_native/*.py
```

Builds are platform-specific. Build fresh on the execution host; a local macOS `.dylib` is not a Linux binary. `test_runs`, bytecode caches, graph data and compiled binaries are excluded from the source archive.

## Recommended execution, scheduled by root

First make the pipeline, persistence and other benchmark work quiescent. The admission marker must be an explicit scientifically terminal JSON from that schedule; the script also rejects nearby quality/pipeline locks. It cannot discover every foreign process and cannot establish hardware comparability with a different GPU pipeline. A generic idle-CPU terminal marker is accepted for an independently scheduled CPU control.

Use fresh output, build and conversion directories. Below, paths are relative to the deployed project root; set the original GraphCSR and terminal marker to their actual host paths. The frozen external helper SHA is exact. The Friendster graph is the physical full graph, not its quality train split.

```sh
python work/vldb_corrected_native/run_parent_replay_v1.py \
  --archive outputs/seal_revision_20261124/cpu/extracted/results/cpu/friendster_exact_int32.json \
  --count 128 --output-dir results/corrected_native_v1/friendster128 \
  --execute --build-dir results/corrected_native_v1/build_friendster128 \
  --native-dir data/corrected_native_v1/friendster_full_i64 \
  --prepare-native --graph-path /home/jovyan/seal_revision_20261124/data/friendster/full_csr \
  --admission-marker /ABSOLUTE/PATH/TO/POST_PIPELINE_COMPLETE.json \
  --include-upstream --external-helper work/vldb_literature/external_operator_benchmark.py \
  --external-helper-sha256 525e88c904fd03e10fc617853c8d733e88af90487cbbdd9122ced97962df112c \
  --conversion-max-native-gib 28 --max-assoc-gib 0.5 --max-peak-gib 48 \
  --max-query-neighbors 20000000 --warmup 1 --repeats 3 \
  --timeout-seconds 3600 --conversion-timeout-seconds 1800
```

For VK use `vk_exact_int32.json`, fresh `vk128`, `build_vk128`, `vk_i64` directories, original `/home/jovyan/seal_revision_20261124/data/vk/train_csr`, and budgets `--conversion-max-native-gib 98 --max-assoc-gib 1.4 --max-peak-gib 160`. Keep the live-headroom guard: these declared budgets do not grant capacity. Existing int64 conversion can be reused by omitting `--prepare-native` and `--graph-path`; do not overwrite or automatically delete a partial conversion. A prepare-only command omits `--execute`, and yields no performance result.

`COMPLETE.json` is scientifically eligible only when `success` is true and each intended arm has a clean exit, complete report, expected record count and exact-output flag. Preserve partial failures and the original exact archives; this control does not replace them.

## Adapter consumer lifetime

```python
with NativeCSR(rowptr, col, library_path,
               mode="persistent_touched_reset", max_assoc_bytes=budget) as state:
    init = state.initialization
    with state.induce(S) as result:
        rows, cols, original_eids = result.copy_arrays()
```

`result.rows`, `result.cols`, and `result.edge_ids` are views valid only until `result.close()`. Copy before retaining arrays or constructing Torch tensors that outlive the result. `pack_local` and `drnl` accept arbitrary distinct `root_positions`; primitive outputs contain no root-specific filtering. Every persistent worker must charge its own initialization and 8N storage.
