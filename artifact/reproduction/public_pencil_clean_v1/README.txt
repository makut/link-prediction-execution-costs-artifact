Public portable PENCIL clean consumer v1 (NEW44-job experiment)

This local runner performs a fresh complete clean-consumer experiment around the
unchanged clean bridge. Default is PLAN ONLY: no data opened, output created,
package installation or GPU/native imports. It is a portable reproduction adapter,
not the private internal controller and not a relabelled old48-job V3 run.

Required user inputs, all supplied as ordinary local paths:
  --prepared-root: public strict-clean citation2 prepared directory, manifest.json
    and all19 numeric payloads. Full5,593,015,516 numeric bytes are streamed and
    checked against metadata/citation2_prepared_expected.json. This hashes all19
    public files, including valid/test split payloads, solely for integrity; the
    experiment does not fit a model or compute full1000-candidate ranking/MRR.
  --validation-root: exactly two original512-source validation subset NPY files,
    valid_edge.npy and valid_edge_neg.npy. Each contains original1000 negatives
    per source; exact hashes are pinned in bridge/input_contract.json. The1024
    timed queries are512 positives followed by candidate0 negative per source.
    See the top-level artifact's public preparation/subset command table; no fake
    quality-plan receipt, hostname, private download path or old mtime is needed.
  --out: fresh directory outside all source/data roots; never resume/overwrite.
  --python: supplied exact LinuxPython3.10 environment. Its venv interpreter
    symlink is preserved lexically. No packages or wheels are installed here.
  --coordination-dir: THE SAME writable local directory used by the portable CPU
    runner and every systems timing run. Shared persistent flock file
    PUBLIC_SYSTEMS_RUNNING.lock plus O_EXCL PUBLIC_SYSTEMS_ACTIVE.json prevent
    overlapping CPU/PENCIL measurement. Unknown/unquiescent/replaced markers are
    retained and refused; no automatic stale-marker deletion.
  --gpu-uuid: observed actual whole GPU UUID, GPU-<canonical UUID>. The same UUID
    is preserved as CUDA_VISIBLE_DEVICES for GPU children; CPU children see none.
    Other GPU processes on that UUID cause refusal. Foreign jobs are never killed.
  --source-manifest-sha256: independent release pin supplied in artifact catalog;
    mandatory for --execute. Every listed source is rehashed before each phase.

Documented exact runtime:
  Linux x86_64, Python3.10, Torch2.5.0+cu121, CUDA runtime12.1, NumPy1.24.4,
  SciPy1.10.1, Numba0.58.0, PyG2.7.0, torch-sparse0.6.18 (matching pt25/cu121
  wheel), Transformers4.46.2, wandb0.18.7, rich14.2.0, tqdm4.67.1.
  bridge/constraints-linux-cp310.txt and requirements-extras.txt document pinned
  optional extras. Consult the top-level environment/wheel-asset instructions.
  This runner installs nothing and never contacts package mirrors or model hubs.
  Native saint.py source/git-blob and original CPU extension SHA are independently
  pinned; all extension/numerical/HF origins/hashes are freshly observed before
  and after the full matrix. Graph-only HF policy is configured before BERT imports.

Default hardware admission is one actual H100 with >=79GiB visible device memory.
Supply --gpu-name-contains and --minimum-gpu-memory-gib to document another actual
whole CUDA GPU; all fresh observations are saved. Such a rerun is a separate new
platform experiment and does not reproduce a historical H100 wall-time number by
assertion. CPU model/affinity/quota are observed; dedicated CPU ownership is not
inferred. Default explicit owned-tree guard is96GiB, finite cgroup limit256GiB;
--memory-guard-gib/--memory-limit-gib may be supplied consistently for another
container. The actual finite cgroup limit must equal the supplied limit and guard
must be <=90percent. A bare host with an unbounded cgroup is refused; provide a
properly constrained Linux CUDA container. Resource choices are recorded uniformly
for all arms; no favorable configuration is selected.

Generic command template (replace variables with actual local paths/UUID/release
pin; these are supplied values, not assertions that a run already happened):

  "$PENCIL_PY" "$REPRO_ROOT/public_pencil_clean_v1/run.py" \
    --prepared-root "$C2_PREPARED" --validation-root "$C2_VALID512" \
    --out "$PENCIL_OUT" --python "$PENCIL_PY" \
    --coordination-dir "$TIMING_COORDINATION" --gpu-uuid "$GPU_UUID" \
    --source-manifest-sha256 "$PENCIL_SOURCE_PIN"

Review the complete printed plan, then use the exact same command plus --execute.
The environment must already be available; out must still be fresh. Whole archive
is created beside out as <out>.tgz. No YT/cloud storage or publication is performed.

Exact scientific command matrix:
  tiny_capture + tiny_audit (7 queries, tinyBERT CPU, both wrapper modes/all8arms),
  real_preflight + real_capture (1024 fixed queries),
  8 real forward semantic audits = FP32/BF16 x batch128/512 x workers0/1,
  32 ONLINE BF16 benchmarks = all8arms x batch128/512 x workers0/1.
  Total44 science jobs;9semantic audits; no REPLAY benchmark. All job argv append
  --query-memory disabled. Every benchmark requires its exact fresh matching BF16
  audit receipt from the same new code/memory mode, model/input/source/native stack.

Arms, fixed before new results:
  torch_sparse_public, torch_sparse_direct, compiled_scan_int64,
  compiled_pairwise_int64, compiled_scan_int32, compiled_pairwise_int32,
  cpp_full_clear, cpp_touched_reset.

Unchanged seed42 random BERT3x512/8heads/FFN2048, features128early, adjacency_row;
extraction seed20261126; one excluded full warmup + three measured full sweeps;
all1024 ordered full model scores checked bitwise within each of the4 separate
batch/worker groups. No pooling shapes, favorable-arm selection or trained-quality
claim. The original native sampler/wrapper/model/upstream/physical kernels remain
whole-byte identical to the frozen clean bridge. Fresh local C++ build uses pinned
source/flags; library and receipt are archived with raw data.

Memory policy: per-query smaps disabled. Worker PSS/maxRSS are null/unobserved,
never zero. Parent Linux RSS/PSS/high-water observations occur before and after
wall. Mandatory S verification, wrapper/IPC, collation, H2D, model and score copy
remain inside wall. External owned-tree memory guard is unchanged and sampled,
not a true peak. All native graph representations remain resident; no PENCIL
memory-saving result follows. Raw stages nest sampler and induction; stage sums
are not additive wall components with overlapping workers. GPU high water includes
warmup. No old-to-new or different-host speed ratio is implied by these commands.

Finite ownership/persistence behavior:
  Max total4hours,10minute final reserve, max science phase1800s; bounded logs,
  guards,512MiB output/cache,384MiB whole archive and2GiB initial scratch reserve.
  SIGINT/SIGTERM stop science, reap confirmed owned child identities only, and
  allow bounded local finalization; repeated stop interrupts. Marker removed only
  after known owned quiescence and exact own-token readback. Partial failed/guarded
  raw evidence is retained. Whole local archive includes SHA manifest, phase
  logs/guards/exits, traces, sources/input/platform/build/scientific receipts.
  It excludes only top-level cache/. No truncated success archive is invented.
  LOCAL_ARCHIVE.json/COMPLETE.json are later local receipts outside the tar; outer
  final controller PID exit is explicitly UNKNOWN. Inspect scientific_success and
  completed_jobs44 separately from archive integrity; failed science is not a
  performance result. This public producer validates all44 jobs internally.
  A standalone independent numerical auditor for FRESH public reruns is not
  included. The frozen independent44 auditor verifies the historical internal
  archive with its actual root anchors; it cannot directly consume this public
  runner's local archive. Do not fabricate root/pod anchors or claim that a
  separate fresh-public independent audit has already happened.

Tiny wrapper/path/ownership tests (no actual GPU/full graph/model execution):
  python -m pytest -q -p no:cacheprovider public_pencil_clean_v1/test_portable.py
Run this suite separately from other adapters to isolate generic module names.
The suite tests arbitrary spaces/Unicode paths, lexical venv preservation, exact44
matrix, source/full tiny19+2 hashes, shared marker/flock/unknown/replacement/init
cleanup, exact CUDA UUID preservation, local archive/nested cache, interrupted
finalization and explicit partial failure retention. It does not claim fresh Linux
CUDA/full-dataset numerical results. See artifact catalog for actual evidence.
