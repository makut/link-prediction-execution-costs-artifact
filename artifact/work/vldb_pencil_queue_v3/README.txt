PENCIL recovery queue V3 — root deployment only

Old bridge V1, queue V2 and failed persisted results stay frozen. This NEW queue
uses bridge V2 with explicit graph-only Transformers4.46.2 optional-vision policy.
It fixes the reproduced unused torchvision import error without modifying base
packages, Torch operators,53 upstream source files or scientific computation.

Root reconstructs the original strict-clean sources/data and corrected native
module/library/build receipt at the existing BASE. Native library SHA must equal
2a8b892a63a84061ec03fc8e7d6c5d616ecb351720a1e4f236cae0374999f967.
If a rebuild differs, fail closed and review; never forge an old build receipt.
Reuse the exact matched58MB wheel archive (SHA885ba0...ee30) already stored at
//home/ucp-research/projects/graph_baselines/seal_revision_20261124/pencil_wheelhouse_input_v1.tgz
The archive member prefix remains pencil_bridge_wheelhouse_linux_cp310_matched;
extraction is fresh pencil_bridge_wheelhouse_linux_cp310_matched_v3.

Admission: root supplies BASE/RECOVERY_ADMISSION.json and its explicit SHA via
--recovery-admission-sha256. Schema1/kind pencil_v3_recovery_admission, exact BASE,
actual replacement_hostname, old_environment {pod_absent:true,
statefulset_replicas:0,root_confirmed:true}, three prior_archives roles pipeline,
native, pencil_failed_v2, each exact yt_path/bytes/sha256 and
all_internal_hashes_verified:true; scientific_success true,true,false; old
pencil_failed_v2 completed_jobs must remain0. This is explicit root attestation
from immutable archives plus deleted original pod; it does not fabricate legacy
COMPLETE/PID files or infer a clean old final controller/lock state.
Actual new-pod active-process scan and all original GPU_SUITE_RUNNING,
PIPELINE_SUITE_RUNNING, ADDITIONAL_TIMING_RUNNING locks are checked.

Default CLI is preflight only, no output or launch. Root execute example:
/opt/python/bin/python3 -u BASE/pencil_queue_v3_code/pencil_queue_v3.py \
  --base BASE --execute --recovery-admission-sha256 ROOT_VERIFIED_SHA \
  --max-hours 12 --admission-hours 8
Launch detached with stdin /dev/null; save actual PID/start/source/argv receipt.
No automatic retry or overwrite; choose reviewed fresh version on failure.

Unchanged48 jobs: tinycapture7, tiny8-arm eval/train wrapper+forward audit, real
preflight/capture1024,8 FP32/BF16 x b128/512 x declaredworkers0/1 full semantic
forward audits,32 primary ONLINE BF16 jobs,4 supplemental REPLAY jobs.
Every benchmark:1 excluded full warmup +3 measured full sweeps. Ordered1024
scores must match across all arms/repeats within each separate shape/mode group.
Audit itself is sequential; worker field binds subsequent loader admission.
No training, HPO, test, MRR, ranking quality, SOTA or memory-saving claim.

Plain existing base .pth routing from V2 preserved; new env pencil_env_v3 and
route pencil_base_site_v3.pth. Original numeric/native origins and pre/post
base+isolated snapshots must match. Loaded HF import_utils origin and SHA plus
policy source are runtime-bound. Every execute job's compatibility receipt must
exactly equal the isolated probe. No full58MB hash/extraction in admission wait;
full wheel reads happen after safe admission and exclusive timing lock.

Budgets retained: <=12h total including <=8h admission, <=1800s/job, external
reviewed sampled180GiB V3 guard,20min final persistence reserve,128MiB final
archive,32MiB whole member cap, no truncation. Shared timing lock held through
whole final persistence. One final readback-verified YT output:
//home/ucp-research/projects/graph_baselines/seal_revision_20261124/pencil_bridge_v3_20261125.tgz
Fresh results/pencil_bridge_v3_20261125, pencil_bridge_v3_20261125.tgz.
Archive includes full failed/successful receipts, logs, guard logs, traces,
source/binary/protocol bindings; not full graph/env/wheel/checkpoint trees.

Author prepared local sources/tests only. Root owns remote deployment and launch.
MIT new author code; preserve all third-party licenses and notices.
