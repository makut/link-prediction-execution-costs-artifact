Independent local PENCIL clean-consumer final auditor v1

This is a separate 44-job auditor, frozen before seeing the new actual H100
scientific results. Old V3 auditor, failed first actual attempt, replay addendum,
actual V3 numbers and all producer/controller packets remain unchanged.

The auditor uses NumPy and the standard library only. It imports no producer
parser, plan builder, science validator, Torch, Transformers or native extension.
It independently constructs the exact 44-job matrix and decodes every CLI/default.
The frozen producer default PLAN is used as a counterparty only by one synthetic
test, with an explicit macOS /home path normalization; not during actual auditing.

Derived checks:
- trace_audit.py is byte-identical to the original independently reviewed V3
  trace verifier, including literal 7-query tiny raw-slot/stored-value/root/edge
  oracle and all real1024 exported arrays, CPU RNG and cohort checks.
- science_core.py extracts17 original verifier functions. Thirteen are byte
  unchanged. Four explicit adaptations: guard96GiB; missing private quality-plan
  receipt must be false while exact payload proofs remain; new bridge path and
  query_memory CLI default; disabled memory/parent snapshots plus literal sweep
  counters. The original sampler-REPLAY==0 assertion is absent because all32
  benchmarks are ONLINE. DERIVATION.json records whole function source hashes.
- All source/runtime bindings use whole-file SHA256. No cross-version runtime
  AST hash is used. Source syntax is checked for Python3.10 only in tests.

Required collector export contract (read-only existing persisted data):
  final_snapshot.tgz                 whole original one-file archive
  extracted/<flat tar members>       all exact members including SHA256_MANIFEST
  extraction_manifest.json           {"files":[{"path":...,"bytes":...,"sha256":...}]}
  persistence_attribute.json         full decoded structured post-readback attr
  export.json with:
    status="verified_existing_persisted_archive"
    role="pencil_clean_consumer"
    all_internal_manifest_files_verified=true
    remote_archive_created=false
    backup_calls=YT_writes=pod_lifecycle_mutations=0 (literal integers)
    yt_path=<fixed controller target>, attribute_name="lpp_clean_consumer_persistence_v1"
    archive={"bytes":...,"sha256":...}
    attribute_canonical_json_sha256=<whole canonical ATTRIBUTE SHA>
    scientific_success=<literal boolean>

Canonical JSON EXACTLY matches the frozen publisher:
  json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
  Python DEFAULT ensure_ascii=True, followed by UTF-8 encoding for SHA256.
Do not substitute ensure_ascii=False, reserialize the embedded canonical receipt
string, coerce YSON integer/Boolean/null values, or bind the attribute inside the
archive hash. Only UTF-8 YSON byte strings may be decoded during collection.

Two independent root anchors are mandatory:
  --root-admission: the whole original fresh root admission bytes, identical to
    archived ROOT_ADMISSION.json. Fresh observed pod/StatefulSet UUID, hostname,
    image, oneH10080Single27CPU205GiB-request/256GiB-limit/96GiB guard and source
    pins are checked; launch must follow root observation within1hour.
  --root-launch: independently observed whole actual LAUNCH.json, or an object
    {"launch_receipt":<that exact object>}. PID/start ticks, exact source/packet/
    manifest/admission linkage and fixed code/output roots are mandatory.
A schema or old recovered pod/old PID is never a fresh root anchor.

Example (paths must be supplied by root collection):
  python audit_clean_v1.py --export /ABS/VERIFIED_EXPORT --output /ABS/NEW_AUDIT \
    --root-launch /ABS/ROOT_LAUNCH.json --root-admission /ABS/ROOT_ADMISSION.json
Missing export prints waiting and creates nothing. Existing output is refused.
No downloads, large source graph loading, extraction, lifecycle or network calls.
All tar hashes use4MiB streaming blocks; complete payload/archive capped384MiB,
member32MiB, count4096. Trace NPY arrays are memory-mapped, allow_pickle=False.

Actual target checks include full19 public numeric identities (5,593,015,516 bytes)
plus two exact original512-source validation hashes, full19 strict manifest,
unchanged pinned native saint source+CPU binary, stable all numerical/native/HF
module origins/hashes before/after, exactly one H10080 GPU and CPU/quota observation,
all phase exits/logs/96GiB sampled guards, archived freshly built C++ library and
actual source/flags/compiler/build receipt, random seed42 BERT configuration,
9 all-arm full native/wrapper/logit audits and all32 measured four-sweep receipts.

Each result/sweep must explicitly disable query memory. Worker maxrss/PSS must be
null and declared unobserved, never fake zero. Parent RSS/PSS/high-water snapshots
are actual integer Linux observations before/after, outside wall; no worker peak
or memory-saving claim follows. Mandatory S verification, IPC, wrapper, collation,
H2D, forward and score copy remain within wall. Raw stages nest sampler/induction.

All4 predeclared ONLINE BF16 shape/worker groups contain all8 arms; within each,
all1024 ordered scores are bitwise exact across all4 warmup/measured sweeps. Each
median/QPS/stage/ratio is recomputed. No favorable-arm/configuration selection,
REPLAY diagnostics, pooled batch scores, cross-host V3-to-clean speed ratio,
trained-quality/MRR/ranking/SOTA, or PENCIL memory-saving interpretation.

Transport integrity and scientific success are separate. Failed/guarded/incomplete
raw evidence and inventory remain explicit; no performance analysis is emitted.
Successful tar+post-readback attribute prove only the prepublication science
boundary. The publisher's later own exit and outer final controller PID exit are
outside tar and UNKNOWN; no recursive watcher/self-exit assertion is invented.

Validation consists of143 meaningful synthetic cases, including complete44-job
science+transport glue, inherited numerical/trace/model/score/unit mutations,
source/plan/root/admission/attribute/tar tampering, full19+2 identity and fresh
platform/native build checks, disabled/null memory and literal false-vs-zero
checks, failure withholding and missing-export behavior. They are synthetic
operational verifier tests, not actual Linux/GPU/native full-graph evidence.
