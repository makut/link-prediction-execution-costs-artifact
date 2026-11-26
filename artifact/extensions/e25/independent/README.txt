Independent E25 ENVv4 helper — source only; no completed ENVv4 archive audited.

After the root collector obtains and independently hashes a genuine completed
ENVv4 raw archive, use a fresh output path:

python3 -B work/vldb_native_lifecycle_20261126/audit_env_v4/audit_e25_env_v4.py \
  --archive ACTUAL_COMPLETED_ENV_V4_TGZ \
  --archive-sha256 ROOT_COLLECTED_WHOLE_SHA256 \
  --output NEW_ACTUAL_ENV_V4_AUDIT.json

Only stdlib and adjacent independent_env.py/independent_disk.py are imported.
Producer code and recorded paths are never imported/opened. Archive reading
remains bounded, streaming and without extraction. Every member and the whole
archive are rehashed. ENVv1/v2/v3 helpers remain unchanged.

All24 grid/order/clock/statistics/input/PSS rules are retained. The grid, stats,
raw PSS eligibility, command-boundary labels and tiny H100 qualification
functions are AST-identical to v3. independent_env.py is byte-identical to v3.
The CPU policy remains fixed27_env_v3. Raw PSS before/after membership and phase
eligibility are independently rederived. The frozen headline is each run's
maximum across ALL eligible phases, then the range of all three observed run
maxima per configuration. CPU/disk incompleteness never filters PSS samples;
missing evidence remains explicitly unobserved. A hard failure prevents a
successful complete-cohort audit rather than selecting a favorable subset.

The v4 CPU predeclaration contains observed package/core/NUMA fields but does
not serialize complete sibling lists. These available fields are compared to
the admission contract; no independent full-sibling comparison is claimed.

ENVv3 independently reconstructs PID/TID/cgroup stability and named reasons
from full before/after identity tables and per-TID affinity brackets. Every
boundary attempt and all original command/app/worker and tiny qualification
gates are checked. Exact absolute boundary start and actual sleep durations
were not serialized; declared elapsed/parameters remain source-pinned receipts
bounded by raw attempt-span consistency. Scans are sampled and non-atomic,
without continuous/exclusive CPU placement or interference guarantees.

ENVv4 adds independent original/nested storage-admission hashes, actual PVC
UID/volumeName plus Pod mount/subPath/owned-file probes, Linux dev_t decoding,
runtime recorded-device/readwrite/available-byte/inode checks, latest sampled
output inventory and independently regenerated outer controller_guard_proof.
Direct PV claimRef was not visible to root. Successful project/root-layer
owned-file write/fsync/readback/unlink probes do not establish global node
health or explain the historical stop/readonly episode.

disk_observations.jsonl is streamed with an8MiB file and4096-byte complete-line
cap. Every raw row must retain the same admitted output path/device, ordered
clocks, at least4GiB available bytes, writable mount and positive available
inodes where totals are exposed. Initial80GiB/64K thresholds apply only to the
separate initial admission. Raw statvfs block components were not serialized,
so recorded byte availability is threshold-checked, not independently rebuilt
from block arithmetic. The latest <=4096-byte inventory must bind the final
disk row, retain vanished-entry counts, and observe at most2GiB nonruntime
output. This sampled non-atomic cap is not an instantaneous quota/reservation.
Any new raw controller failure receipt, or nonnull COMPLETE guard latch, fails
the complete audit. The protocol, original full disk history and latest
inventory hashes are retained in the independent result.

The prior1168934-byte ENVv3 capsule is independently identified by whole SHA
and its pinned local-preservation receipt. It contains v2 once and v1 nested
inside v2. It is never expanded, and no old timing becomes a new24 metric
input. The two locally observed completions are not recovered raw job metrics
or proof of actual OS exit0.

Report keys environment_v4 and storage_v4 identify the new cohort. No results,
mock scientific observations or archive audits were generated during source
preparation. This helper does not check live PID/lock/YT state or final outer/
YT publication, recompute full graph/PyG bytes, launch experiments, or infer
population speedups. New times are instrumented application wall, including
input/resource/ENV/disk observation and native promotion/fsync costs; common
whole-input hashing, native build and post-exit verification remain outside.
