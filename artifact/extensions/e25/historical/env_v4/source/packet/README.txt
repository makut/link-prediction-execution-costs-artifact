E25 ENVv3: Friendster DGCNN lifecycle, separate fresh24-app cohort

ENVv3 amendment after the separate v2 cohort aborted3/24: the fourth app
finished training but its worker finalizer rejected one non-stable task scan.
All14 resource predicates and observed masks were valid. V2 omitted after
brackets, so the exact task transition is not asserted retrospectively.

The NEW observer stores before/after PID/start, cgroup, task lists/identities and
per-TID affinity brackets with named instability reasons. It checks observed
leader, PID/TID and cgroup negatives even when the overall task scan is unstable.
Any observed wrong quota/mask/identity/cgroup fails immediately and cannot be
hidden by a later PASS. Boundary guards may rescan ONLY ancillary membership or
observation churn while the own PID/start, leader mask and cgroup are known and
exact, and all hard predicates hold. Limits frozen before the new cohort:
8 total attempts, 0.5s retry window, 10ms pauses capped by remaining time. A scan
returning after the deadline cannot PASS; OS I/O overrun is reported as failure,
not a hard real-time cancellation guarantee. Every attempt is retained under
boundary_rescan. A final fully stable PASS within both caps is mandatory.
Failure artifacts are written only for terminal violation or exhaustion.

All24 jobs restart in fresh output/cache; neither the successful3 nor the failed
fourth v2 app is pooled. Native/core/supervisor/model/sampler/workload/checkpoints
and timing definitions are unchanged. Read-only guard overhead stays inside the
same application wall clock. Legacy wrapper kind/error text may still say V2;
source/protocol hashes and environment_policy=fixed27_env_v3 identify this cohort.

The previous v1 attempt aborted after15/24 apps when the ambient shared CPU
pool changed. Its complete partial outputs remain separate diagnostic evidence.
ENVv3 restarts all24 apps in a fresh output and shared-Numba-cache namespace;
no old timing, partial block, report or cache is appended or pooled.

CPU selection is declared before any build/qualification/full-graph admission:
choose the highest27 currently allowed online logical CPU IDs, or require the
same exact root-bound list in E25_EXPECTED_CPU_AFFINITY (JSON sorted27 integers).
The latter must be exported identically for standalone H100 qualification and
the controller. No host CPU IDs are hardcoded or selected using timing results.
The controller applies os.sched_setaffinity once while still single-threaded;
all apps and spawn workers inherit it and only read/fail thereafter. Standalone
qualification pins its own still-single-threaded process to that SAME list, or
validates an inherited controller E25_ENV_CONTRACT without another pin.

Target gates require exact cpu.max2700000/100000; unchanged host/boot/visible
cgroup identity, visible quota hierarchy, memory limit, cpu.weight and mems;
all selected CPUs still online and in effective cpuset. Main process leaders
must have the exact chosen mask. Ancillary runtime threads may have nonempty
subsets, which are recorded, but any widening beyond chosen fails. All mask
checks are read-only after the one admission pin. Changing the broader ambient
pool alone is recorded and allowed when all chosen CPUs and invariants survive.

ENVIRONMENT_ADMISSION.json binds the actual list, topology and before/after
state; plan/configs bind the same contract. environment_events.jsonl records
boundary/periodic controller guards; every external CPU/PSS sample carries
stable PID/start and TID/start task-mask evidence when observable. An unstable
transient process/task scan is explicitly incomplete, never affirmative proof.
Application/worker start and normal-close receipts carry their own guards.
Failures preserve named predicates and observations in ENVIRONMENT_FAILURE.json.
The same roughly2s observer remains sampled; no exclusive physical cores,
absence of interference, bandwidth or continuous placement is guaranteed by
this source. Guard work performed during apps stays inside wall time.

Mac CPU1 qualification skips unsupported Linux affinity/cgroup checks with
NOT_APPLICABLE_PORTABLE_CPU1 receipts. It is portability evidence only and
cannot satisfy mandatory actual Linux/H1001/8 ENVv3 qualification.

No actual full-graph E25 result is supplied by this source packet. Frozen E23
and native operator files are retained unchanged. This is a new instrumented
protocol, not a reuse of historical E23 timing values or full upstream SEAL.

Grid fixed before measurements: Friendster only; reference_union_compiled and
standalone reference_union_native_persistent; workers1/8;3 fresh process runs;
two genuinely terminated horizons. Short has8 warmup +10 measured batches,
576 total trained graphs. Long has8 warmup +200 measured batches,6656 total.
Batch32, original8000-row Friendster workload, seed20261124,2hops/cap200,
featureless original DGCNN32/3/k281/max_z1000, Adam1e-4 and one numerical thread.
There are24 timing apps, no stock/local/Citation2 expansion or new sampling law.

Plan only (stdlib; no graph array/model imports, jobs, locks or output):
python3 -B run_lifecycle.py --friendster-root FULL_CSR --out NEW_OUT --coordination-dir COORD

Actual Linuxx86_64Python3.10/Torch2.5.0+cu121/PyG2.7.0 execution additionally
requires the printed SOURCE_MANIFEST.json SHA and --execute. Numerical stack
NumPy1.24.4/SciPy1.10.1/Numba0.58.0 matches E23. One idle whole H100 is required.
No installation, download, YT or Kubernetes API is used by this packet.

Controller builds the unchanged C++17 standalone core once as installed
software qualification, then requires actual tiny CUDA consumer qualification
at both1/8workers before any full native allocation. Standalone tiny gate:
python3 -B qualify.py --library BUILT_LIBRARY --out NEW_TINY --device cuda --workers 1 8
Portable local CPU smoke is intentionally restricted to --device cpu --workers 1.
CUDA8 tiny warmup delivers a batch from every worker, proving actual init/close.

Application start is the parent's monotonic_ns immediately BEFORE Popen, passed
through the environment. The child's first-line timestamp uses the same kernel
clock. Imports, model construction, required representation preparation, worker
spawn, and real first-use setup are inside cumulative application wall time.
Worker_init sets only torch.set_num_threads(1). Compiled kernels JIT/cache-load
naturally on first needed invocation; no unused sampler or kernel prewarm is
charged to another arm. Shared disk Numba/OS caches remain, with metadata-only
before/after inventories. Qualification may populate caches; no cold-cache or
cold-software claim, extra prefault, page-cache flush or baseline subtraction.

Each native app creates a fresh29,421,941,096-byte file-backed int64 rowptr/column
representation from the same compact input, including fsync, inside its clock.
Each spawn worker maps it read-only and creates one524,866,928-byte private8N
association map lazily before its first query. The unchanged NativeCSR.induce
and pack_local remove the target edge with roots0/1; then the unchanged V4
reference sampler, ordered S, original DRNL and PyG constructor are used.
The original V4 Dataset.__getitem__ is invoked via subclass super(), not copied.
No PENCIL NativeGraph/Facade, sorted-S substitution, values/transpose/COO global
caches, per-query global reset or per-worker full29GB integrity scan is used.

Every actually consumed CPU batch is hashed before H2D: typed/shaped ordered
query indices, node_id, edge_index, edge_weight, z, y, ptr and batch. All consumed
inputs must agree across arms/workers/repeats within a horizon. Short's final
SHA must equal long's semantic prefix576 SHA. That long576 snapshot adds NO
CUDA synchronization or timing checkpoint. Hash cost stays inside wall time;
its separately reported cost is not subtracted. Initial model SHA is a hard
gate; final CUDA model hashes/losses are descriptive and do not gate semantics.

Exactly three short CUDA sync checkpoints: trained32(first batch),256(warmup),
576(final measured320). Exactly five long checkpoints:32,256,1280(measured1024),
3456(measured3200),6656(measured6400). Their clocks directly measure cumulative
launch-to-completion time. Interior checkpoints are continuous-run prefixes,
not additional independently terminated lifetimes. No overlapping worker-stage
service sums are added to wall time. This synchronized E25 measured window may
include natural JIT/cache-load activity and is not the original E23 window.

After final training synchronization, explicit DataLoader shutdown waits for
normal worker finalizers. Each worker proves touched state reset, closes native
state and records completed extractions, including any unused prefetch. The
dataset bounds total extractions to consumed +workers*32: short608/832 and long
6688/6912 for workers1/8. The parent records actual wait returncode, start_ticks,
whole owned-tree quiescence and process wall time including teardown. Nonzero,
timeout, guard abort, signaled group or orphan cleanup cannot produce success.

Full original input hashes are scientific admission outside application clocks.
Fresh native content hashes are streamed after the app's actual wait/quiescence
and compared to independently validated E20 native hashes for the same compact
Friendster files. Then only the three known owned fresh native scratch files
are removed. These integrity/experiment-cleanup intervals are not hidden in
required application setup or represented as overlapping service-time sums.
Failed/partial outputs are preserved; no automatic resume or retry is provided.

External complete process-tree PSS sampling runs about every2s with PID/start
membership brackets and before/after phase labels. Only stable complete same-
phase samples are publishable. Raw non-unique summed RSS is diagnostic only;
it never guards unique memory. Safety uses the unchanged V3 complete PSS /
max individual RSS / unknown-PSS finite-cgroup90% policy. Observed PSS includes
partial file residency and worker-private state; it is not full-resident array
storage, exact allocation, new graph-only memory or a strict unsampled peak.

Limits:180GiB sampled CPU guard,24GiB Torch allocator,4h producer total,
2400s/app,2GiB non-runtime evidence,4GiB live disk reserve; each full native app
needs native bytes plus6GiB free initially. One derived full copy exists at a
time. Source graph is never overwritten/copied into evidence. Controller's own
COMPLETE precedes its future OS exit and lock.close; root must independently
collect actual controller exit/quiescence/lock release. No STS self-release.
