# E25 ENV v4: Friendster application lifetimes

**Source-only addition. This directory contains no completed 24-job result or
rendered figure.** ENV v4 starts a fresh cohort and cache. All earlier v1, ENV v2 and interrupted ENV v3 attempts remain separate diagnostics;
their timing observations are never pooled. No partial ENV v3 metrics or actual
outer OS exit0 were recovered or inferred.
This staging preserves v0.6, its tags and all six immutable v0.5 companion assets.

`source/packet` contains the exact 41-payload producer plus its manifest:

| Identity | SHA256 |
|---|---|
| `SOURCE_MANIFEST.json` | `82f93337ba4b4333cc319b610f6bad1ad24c5f9edf1c79731ae6127eed94c005` |
| Canonical semantic protocol | `ff42b44bf17ff54d9994731543fbb3d73caaabb56c7150f576e208ad6c3f1fc2` |
| Source transport archive | `992c1c742ba2b4f336f4f8c4ecadb94e57e0c22dba6a923f36577d8cc4711bf6` |

The canonical protocol hash differs from the on-disk `PROTOCOL.json` hash.
`source_archives/e25_native_lifecycle_source_env_v4.tgz` is 184,956 bytes with
prefix `e25_native_lifecycle_packet_v4/`. Licenses and third-party notices remain.

## Input preparation

Use v0.6's recipe for Per Fuchs's densely relabelled Friendster, Zenodo 5146230,
DOI <https://doi.org/10.5281/zenodo.5146230>, CC-BY-4.0. A different SNAP vertex
mapping changes identity. Set absolute `ARTIFACT`, `DATA`, `RESULTS` and `E25_PY`
paths on a Linux host. From the artifact's `src/`:

```sh
cd "$ARTIFACT/src"
"$E25_PY" -m artifact_public.download friendster \
  --out "$DATA/friendster_inputs" --execute
"$E25_PY" -m artifact_public.prepare_friendster_full \
  --inputs "$DATA/friendster_inputs" --out "$DATA/friendster/full_csr" \
  --workers 8 --memory-budget-gib 64 --execute
```

Omit `--execute` to inspect a plan. The preparation environment may be separate
from the model environment. The three public inputs total 6,672,243,067 bytes;
provider size/MD5 and full converted numeric SHA256 identities are checked.
Preparation admits at least 56 GiB available RAM and 32 GiB free disk; the
existing public recipe recommends 128 GiB RAM and 100 GiB scratch.
Use `full_csr`; E25 needs no quality split or evaluation arrays.

The producer independently hashes these full inputs:

| File | Bytes | SHA256 |
|---|---:|---|
| `num_nodes.txt` | 9 | `8551541f6f2ef955481af83ef5e6f92c5f206c968d5d9db6a4437efecf767753` |
| `edge_starts.bin` | 524866928 | `b6ebf026fa54498f241af834b96b5496b2a1bcbf8be3edea2f4b3a1cf0a674e9` |
| `edge_ends.bin` | 14448537088 | `0f74897efa3172252a8371bb0d2c8e0129602475ebb84a863ec1210f9c5e5973` |

There are 65,608,366 vertices and 3,612,134,270 stored entries. Starts are
little-endian int64[N] without a terminal sentinel; ends contain an int64 count
and int32 neighbors. The included 8,000-row workload SHA256 is
`340e61cdd1eb8d329d52be164f919480e6a41feb4c620852df0393c2e9a192f0`.

## Fresh replay

Provide Linux x86_64 Python 3.10.12, NumPy 1.24.4, SciPy 1.10.1, Numba 0.58.0,
Torch 2.5.0+cu121, PyG 2.7.0 and v0.6's declared public environment extras.
A C++17 compiler builds the unchanged standalone native library locally.
Use one idle whole H100 with compatible CUDA/driver. The producer has no CPU
fallback, MIG or DDP path and installs or downloads no packages.

The target container/cgroup must expose readable own cgroup-v2 CPU, memory and
cpuset files plus visible enclosing limits. Required CPU quota is exactly
`2700000 100000`, or 27 CPU-equivalents. The authors used a 256 GiB memory limit.
These allocation values do not establish a minimum memory requirement or
exclusive cores. A global hierarchy root lacking own limit files fails closed.

Set `E25_CPU_MASK_JSON` to a predeclared sorted JSON list of exactly 27 logical
CPUs available on your host. Use the same mask for qualification and every app;
chosen CPU loss fails rather than selecting another mask. Author CPU IDs are
not portable.

```sh
export E25_EXPECTED_CPU_AFFINITY="$E25_CPU_MASK_JSON"
"$E25_PY" -B "$ARTIFACT/extensions/e25/source/packet/run_lifecycle.py" \
  --friendster-root "$DATA/friendster/full_csr" \
  --out "$RESULTS/e25_new_run" --coordination-dir "$RESULTS/timing_coordination" \
  --max-hours 4
```

This prints a plan without reading graph arrays, creating output or launching
jobs. For execution, use fresh paths:

```sh
"$E25_PY" -B "$ARTIFACT/extensions/e25/source/packet/run_lifecycle.py" \
  --friendster-root "$DATA/friendster/full_csr" \
  --out "$RESULTS/e25_new_run" --coordination-dir "$RESULTS/timing_coordination" \
  --max-hours 4 \
  --source-manifest-sha256 82f93337ba4b4333cc319b610f6bad1ad24c5f9edf1c79731ae6127eed94c005 \
  --execute
```

Output must be absent and disjoint from inputs and source; coordination must
be writable and outside those trees. The shared lock coordinates cooperating
benchmarks; run no overlapping timing jobs. The producer performs no YT or
Kubernetes actions. Observe actual controller exit, owned-tree quiescence and
lock release separately because its own terminal predates its exit.

The grid has 24 apps: two induction arms, workers 1/8, three fresh repetitions,
and two genuinely terminated horizons. Each consumes 256 warmup graphs followed
by 320 or 6,400 measured graphs, for 576/6,656 total. Batch size is 32; reference
sampling, original featureless DGCNN and numerical thread limits are unchanged.
Build and actual tiny CUDA 1/8 qualification precede full apps. Limits remain
four hours overall and 2,400 seconds per app. No resume or automatic retry exists.

## ENV v4 platform; retained `fixed27_env_v3` CPU policy

The author allocation was Guaranteed with CPU request=limit 27 and memory
request=limit 256 GiB. Its fixed mask `26-38,137-150` spans 27 logical CPUs on
14 physical cores: 13 SMT pairs and one sibling of the fourteenth core.
NUMA memory nodes were `0`, CPU weight `1055`. Preflight admission and all four
actual ENV v4 H100 tiny-consumer qualifications are source/qualification facts;
they do not certify a completed 24-app timing result.

The controller pins once before graph admission, build and qualification.
Children inherit that contract. Before/Popen/after-wait guards, app/worker
boundaries and roughly two-second samples check fixed leader masks, ancillary
thread subsets, cgroup identity, boot, quota, visible limits, weight and mems.
Changes to ambient CPUs outside the chosen mask are recorded and allowed.

The unchanged ENV v3 CPU observer retains full before/after PID, TID and cgroup brackets. Any observed
negative resource, mask, PID/TID or cgroup predicate fails immediately.
A boundary can rescan only ancillary task churn with known unchanged own
PID/start, exact leader mask and cgroup, and all hard predicates true:
at most eight attempts, a 0.5-second window and 10ms pauses capped by remaining
time. Every attempt is recorded; a final fully stable PASS is required.
A scan returning after the deadline fails. OS I/O may overrun before failure
is reported; this is no hard real-time cancellation guarantee.

The auditor independently reconstructs bracket consistency and every attempt.
Absolute boundary start and actual sleep durations were not serialized, so
their exact timing remains source-pinned; declared parameters/elapsed and raw
span consistency are checked. All observations remain sampled and non-atomic,
without continuous-placement, exclusive-core or bandwidth-isolation guarantees.
Guard work stays inside app wall time. CPU-unobserved scans never filter PSS.

`schema_reference/outer_source_v4/` preserves the complete 16-payload bound
author envelope and manifest, with its source peer. Its author UID, ingress,
host and YT contract provide provenance; they are not public replay commands.

## Persistent storage and hard guard amendment

`schema_reference/outer_source_v4/ROOT_STORAGE_AMENDMENT.json` is an exact,
unchanged 14,041-byte copy, SHA256
`27b1cd73151fbe27592bdde318532f38593d877477478e2ddf1b066ef7b3b668`.
Its nested original admission is preserved separately as
`ROOT_STORAGE_ACTUAL_ADMISSION.json`, SHA256
`d3e1d4feb0c519eaf6a7f21d5f9847df5a3523efc8840cd1a3fb00c6b82da026`.
These are recorded pre-science author platform facts, not current liveness or
completed timing results. The full exact storage proof includes the author
image digest, CPU/NUMA projection, owned write/fsync/readback/unlink tests and
PVC claim UID/volumeName/project subPath. Direct PV claimRef access was unavailable.

The authors used the admitted ext4 project PVC for source, compact graph,
results, fresh native files, temporary files and caches. Initial observed
available bytes were 86,092,836,864; initial admission required at least 80 GiB
and 65,536 inodes where exposed. This is actual available space, distinct from
the nominal 200 GiB claim capacity. Runtime guards require at least 4 GiB and
positive exposed inode availability, and retain a 2 GiB sampled non-runtime
output cap. Neither allocation nor observation reserves capacity or guarantees
exclusive bandwidth or continuous health. Storage placement can affect timing.
The author image was
`registry-gitlab.corp.mail.ru/platforma-adm/ml/registry/ucp-research/gnn-verse@sha256:2c0d776d47eb7d3b763f5a42c70d51bff2ba6b1f9cb1f01b4adbad76b658cbb1`;
this historical internal reference is provenance, not a reader download command.

The v4 controller fixes a known guard propagation error: a hard deadline,
resource, readonly, disk or output-cap negative latches before throttling and
causes owned-process abort, independently of transient PSS errors. A later
sample or unchecked throttle return cannot clear it. Actual disk checks append
bounded fsynced observations of byte/inode availability, device and readonly
flag. The output scan uses one lstat per entry; a normal atomic rename may
make an entry vanish, which is skipped and counted. This sampled non-atomic
inventory is not an instantaneous quota. A real observed cap negative still
latches even if a neighbouring entry vanished. Full/readonly storage can prevent
writing a failure receipt; the in-memory latch remains and successful terminal
requires it to be null.

The resource investigation identified that code defect. The cause of the earlier
external replica-zero stop remains unknown; disk-full/scaler causation is not
claimed. An historical node readonly condition is not a present global-health
verdict. The admitted owned PVC and root-layer writes passed their recorded
preflight tests.

For public replay, provision a suitable writable filesystem, keep source/data,
outputs and temp/cache locations on that intended storage, record your own
mount/device/capacity and platform, and disclose differences. Author device/CPU
IDs and cluster contracts are not portable. The exact author outer source is
schema/provenance, and the provided offline auditor expects that author-envelope
archive schema. A replay on another host needs its own provenance rather than
claiming the preserved author storage proof applies to it.

## Measurement and resource scope

CPU safety uses the unchanged 180 GiB sampled process-tree guard and finite
cgroup fallback. Torch allocator budget is 24 GiB and must fit preflight free
device memory; it does not cap every CUDA allocation. Each native app creates
one fresh 29,421,941,096-byte int64 representation, including flush/fsync inside
its clock. Require that size plus 6 GiB free initially and 4 GiB live reserve.
Copies are sequential. After actual successful wait, full derived hashes
precede removal of only known owned scratch. Failed scratch stays preserved.
Each native worker initializes a private 524,866,928-byte association map:
524,866,928 bytes at W=1 or 4,198,935,424 bytes at W=8, apart from graph mappings,
interpreter/model/batch state. These are logical sizes, not measured PSS.

Primary instrumented application wall runs from immediately before Popen through
actual wait, including setup, reporting and teardown. First trained batch is the synchronized total-32
checkpoint, distinct from CPU-batch readiness. Short syncs: 32/256/576. Long
syncs: 32/256/1280/3456/6656. Interior prefixes are continuous-run checkpoints;
the long semantic prefix576 adds no GPU synchronization. Typed/shaped/ordered
inputs and occurrence IDs are hashed before transfer; observer cost stays in
wall. Natural JIT/cache loading and native setup remain included. OS/Numba
caches remain, with no cold-cache claim or causal E23 cross-run speed ratio.
Common whole-input admission audits and post-exit native verification are outside
applications. Full consumed-input hashing, resource/ENV/disk observations and metadata fsync remain
inside instrumented wall; it does not establish production throughput or
time to quality.

Both arms preserve consumed workload and DataLoader policy. Executed prefetch
counts may differ. Unconsumed prefetch can overlap training and shutdown;
it cannot be a separate additive stack segment. Executed extractions are
reported separately and bounded by consumed graphs plus `workers * 32`.
Input SHA equality certifies consumed inputs, not equal executed prefetch work.
E10/E23 supply the 32-query structural audit; E25 binds every consumed CPU input
stream and the cross-horizon576 prefix.

PSS headline follows the frozen pre-outcome rule: each run's maximum across
**all** complete stable-member same-phase samples, followed by the range of
three observed run maxima per configuration. Retain phase and sample counts.
Missing samples are unobserved, never zero. Call this a sampled app-tree PSS
maximum: partial file residency and worker-private state are included; it is
not a full-resident footprint, minimum requirement, cgroup charge or atomic peak.

## Offline audit and future evidence

All three files `independent/audit_e25_env_v4.py`, adjacent `independent_env.py`
and `independent_disk.py` are required. After collection, supply the genuine completed raw archive and an
independently obtained whole SHA to a fresh output:

```sh
python3 -B "$ARTIFACT/extensions/e25/independent/audit_e25_env_v4.py" \
  --archive "$E25_RAW_ARCHIVE" --archive-sha256 "$E25_RAW_SHA256" \
  --output "$RESULTS/e25_new_offline_audit.json"
```

The bounded stdlib helper streams and rehashes all members and the disk journal without extraction,
producer imports or recorded-path access. It requires the new source/protocol,
all 24 apps and internal H100 1/8 qualification, and rederives raw clocks,
paired descriptive statistics, PSS eligibility and full ENV brackets/rescans. It independently validates original
PVC admission/device/readonly/free-byte/exposed-inode observations, the latched
guard state and latest non-atomic output inventory. Raw statvfs block counters
were not serialized, so their multiplication is source-pinned rather than
independently recomputed. Disk/CPU unknown does not select PSS observations.
It does not recompute full graph/PyG bytes or check live PID/lock state or
future outer/YT completion. A fresh public replay directory alone is not the
author-envelope archive schema accepted by this helper.

The only prior capsule admitted in a future raw archive is
`e25/prior_aborted/E25_ENV_V3_EXTERNAL_STOP_DIAGNOSTIC_v1.tgz`: 1,168,934 bytes, SHA256
`34d1dbc0aba48f3ce72b766a655f48ee42b857edf70a7d19415d12aa90490d39`,
plus its pinned preservation receipt. The failed ENV v2 capsule remains nested once, with v1 nested inside v2.
The latest capsule preserves available local sources/observations only: it is
not a completed cohort, recovered partial job metrics or actual OS-exit proof. The helper
checks whole identity without expansion or reading/merging old timing values.
This source staging does not duplicate either diagnostic archive.

After successful collection and independent verification of every fresh app, place genuine raw
evidence and audits under `recorded/` and update the catalog with exact hashes.
Keep future outer COMPLETE, upload pointer and actual outer exit/lock proofs
separate from the raw archive that precedes them. Render the plot only from a
whole-SHA-bound successful ENV v4 audit, then inspect the actual layout before
manuscript use. No results, URLs, figures or future publication proofs are
invented by this source-only stage. Root owns the separate methods draft; it is not copied here.
