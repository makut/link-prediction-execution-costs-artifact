E25 v6: prospective three-allocation repetition blocks (source only)

Global protocol: the unchanged two arms x workers 1/8 x short320/long6400 x
three original global repetitions gives 24 applications. Each invocation selects
one prospective block of 8 applications with --repetition-block (1, 2, or 3). It builds
the original full 24 randomized order, then filters the selected repetition;
original r1/r2/r3 job names and within-block order/pairs are preserved.
All three independently complete blocks are required for a 24-app aggregate.
Failed/timed-out/partial blocks are excluded, never salvaged or pooled with
failed v1/v2/v3/v4 or unused v5 attempts; no automatic retry exists.

The scheduling change addresses the user-confirmed approximately 90-minute
allocation autostop. Producer maximum 3900 s and outer maximum 4200 s are finite
failure limits, not a completion/speed guarantee. No earlier failed timings
enter this prospective experiment. Fresh output and Numba cache are required
per block; qualification and applications share that block's cache. OS/PVC
caches remain, without clearing, prefault, eviction or cold-cache claims.

Required CLI (plan-only without --execute):
python3 -B run_lifecycle.py --repetition-block 1 \
  --friendster-root ABSOLUTE_COMPACT_FRIENDSTER_PATH --out FRESH_BLOCK_OUTPUT \
  --coordination-dir ABSOLUTE_SHARED_TIMING_COORDINATION \
  --platform-admission-json ACTUAL_ROOT_PLATFORM_JSON \
  --platform-admission-sha256 INDEPENDENT_WHOLE_PLATFORM_SHA

For execution add --source-manifest-sha256 EXACT_FROZEN_SOURCE_SHA --execute.
The default/cap --max-hours is 1.0833333333333333 (3900 seconds). Output must be absent and disjoint
from immutable source/graph. Shared coordination remains outside those trees.

The root supplies one actual E25_V6_ACTUAL_BLOCK_PLATFORM_ADMISSION proof for
this exact block, after fresh API/Pod/PVC/image observation and local probes.
Its bytes and SHA are bound into plan.json and copied exactly as
PLATFORM_ADMISSION.json. The global protocol contains only schema/policy, not
a stale actual Pod/GPU/CPU/storage hash. Proof includes block, Pod/STS/host/boot/
node/image, GPU uuid/name/memory/driver, actual declared 27 CPU pool/quota/memory/
topology and PVC/subPath/device/mount/probe facts. Source checks the available
local identities/mount/CPU/GPU facts; root API identity/image/PVC binding is
recorded as root provenance, not invented by container introspection.

Source, graph, results, proof, coordination and temporary/cache paths must use
the actual admitted project mount. Admission requires writable 40 GiB free and
positive exposed inodes, plus recorded owned-write/fsync/readback/unlink probes
and writable-layer 4 GiB reserve. Runtime 4 GiB, readonly/inode/hard-negative latch
and sampled non-atomic output cap remain unchanged. Capacity is not reserved;
there is no continuous/global node-health or exclusive-bandwidth guarantee.

CPU policy is the unchanged fixed27_env_v5: external observed leaders/tasks
may use recorded nonempty subsets of the declared 27 pool. Empty/outside masks
remain hard failures. Own controller/app/worker boundary leaders remain exact 27
with full PID/TID/cgroup brackets and bounded ancillary-only rescans. No repin
or PSS filtering occurs. Actual Pod/GPU/CPU pool is fixed within each block;
allocations/platforms may differ between blocks and must be disclosed.
Paired ratios are computed only within each block, not across allocations.

Source numerical stack/model/sampler/workload/native/supervisor/worker/PSS and
application clock/checkpoint/hash definitions are unchanged from v5. Each
native application still copies/flushes/fsyncs the complete 29421941096-byte
int64 representation inside its clock and owns one private 8N association map
per worker. Common full-input audit, C++ build and post-exit native integrity
verification remain outside individual application clocks. Observer/metadata
fsync cost is included; there is no causal CPU-placement/old-E23 ratio claim.

Per-block COMPLETE/SCIENTIFIC_TERMINAL records repetition_block, jobs 8/global 24,
actual proof SHA/file and actual app exit/guard evidence. Summary has four
horizon/worker rows, each holding the two arms' raw single observations, four
within-block paired-ratio rows and eight phase-PSS records. It does not invent
three-repetition SD/means for a single block. Four pair-stream comparisons and
two worker-specific cross-horizon 576 comparisons retain original consumed-
input equality, checkpoints, extraction/prefetch bounds and normal worker
close/wait/quiescence requirements. A block terminal predates its own OS exit;
root must retain actual outer/child wait, quiescence/lock and local archive
proofs separately. Later independent aggregate audit requires all three full
blocks and their separate immutable platform/provenance records.
