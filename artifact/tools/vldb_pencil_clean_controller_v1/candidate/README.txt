NEW clean-consumer control candidate — no execution or allocation

This is a separate protocol and source candidate for root review. The original
frozen PENCIL V3 sources,247-file final result export, first failed independent
audit and explicit corrected audit remain untouched. No new scientific run,
remote call, pod lifecycle action, installation, wheel write or YT write has
been performed by preparing this directory. plan_only.py cannot execute jobs.

Reason for the control
The audited instrumented V3 results retain exact model scores but include
per-query bridge_core.memory()/smaps_rollup inside wall time. For workers0,
unattributed wall time accounts for roughly86–95percent of measured sweeps.
This residual includes RNG, validation, loader and other overhead: it is not
a pure smaps measurement or a causal estimate. It motivates a separate explicit
instrumentation control; old sweep times are never corrected or subtracted.

Source change
Only a NEW copy of pencil_bridge.py changes operational memory instrumentation.
All runtime/sampler/model/wrapper/upstream/C++/Numba sources remain byte-exact
from packet56954a16a9d45d66783a82961fe82cffe4144aad5b3bef382ce2d05ed724d26c.
The original bridge manifest and original bridge source are preserved under
metadata for source and AST comparison. The changed functions are
PipelineDataset.__getitem__,benchmark,verify_audit_receipt and parser; three
new memory-policy helpers are explicit. Capture,audit,model forward,semantic
gates,target masking,RNG,feature validation and mandatory per-queryS checking
retain their original implementations.

The NEW CLI --query-memory {inline_smaps,disabled} defaults to the original
inline behavior for comparability. The clean protocol explicitly passes
--query-memory disabled to every new job. This returns None for per-query
memory, not fake zeros. Worker RSS/PSS fields are None and explicitly labeled
unobserved. Actual parent memory is observed before/after each sweep, outside
the wall timer. These observations and the external sampled guard are not
true peaks. With workers1,the parent snapshots do not claim worker memory.

The separate external owned-tree guard algorithm is unchanged. Its budget is
an explicit planner parameter and must fit below90percent of the pod limit.
For the documented1A10080Single candidate row, root may choose17requestedCPU,
98GiB requested/122GiB limit and96GiB owned-tree guard. Root decides the actual
hardware after reviewing valid placement,availability and the user's stop
intent. An180GiB guard must not be used with a122GiB pod limit. Per-job process,
quiescence,guard and resource receipts remain mandatory; the driver records
this requirement and does not falsely claim to enforce the external guard.
Exact copied supervision/memory helpers are included for root review.

Full fixed protocol
44scientific jobs: new tiny capture/audit,real preflight/capture,all8 real
FP32/BF16×batch128/512×workers0/1 semantic audits,and32 ONLINE BF16 benchmarks.
All8physical arms and all4batch/worker settings are required. No favorable
configuration is selected after seeing results. Each benchmark uses1024fixed
queries,one complete excluded warmup and3complete measured sweeps. The model
is the same random frozen seed42 BERT and extraction seed20261126. No model
training,HPO,ranking quality or REPLAY benchmark is added. The native source,
strict-clean fullgraph/features and512-source validation payload contract are
unchanged. C++ is built from the pinned source; wheels may only be read.

Fresh semantic admission
Every benchmark needs a fresh new-source matching audit receipt, including
the explicit query-memory mode. Old V3 receipts fail the source/mode check.
All raw CSR slots/stored values,S/mask/features,eval/train wrapper/collator
inputs,and exact FP32/BF16 logits must pass for all8arms. All1024ordered
scores must also agree across all8arms and all4sweeps within each of4shape/
worker groups. Do not pool different shapes or relax zero logit tolerance.
Full payload hashing and NativeGraph setup remain outside sweep wall time.
All full native graph representations remain resident; no memory saving is
claimed. The same warmup,persistent-worker lifetime and3sweep protocol remain.

Interpretation and publication boundary
A new host or accelerator permits within-new-host arm comparisons only. It
does not permit a causal originalV3-to-clean wall-time ratio. The old and new
matrices must be reported separately; neither supports a quality or universal
native-performance claim. Source preparation and tiny unit tests are not a new
full-scale result. Root owns any actual controller/guard integration and audit,
one coarse final archive at most,and any future release. No new per-job YT
objects are allowed. There is no automatic launch,unapproved pod restart or
lifecycle cleanup in this candidate.

Inspect a plan without loading graph data or creating outputs:
python3 -B work/vldb_pencil_clean_consumer_v1/plan_only.py \
  --citation2-root /scratch/public-citation2-strict-clean \
  --validation-root /scratch/exact512-validation \
  --cpp-library /scratch/cpp/native_assoc_v1.so \
  --cpp-build-receipt /scratch/cpp/build_receipt_v1.json \
  --out /scratch/pencil-clean-new-output \
  --external-memory-guard-gib 96 --pod-memory-limit-gib 122

Root reviews the generated44-job plan and the exact source packet before any
actual execution. A controller is intentionally not supplied by this planner.
