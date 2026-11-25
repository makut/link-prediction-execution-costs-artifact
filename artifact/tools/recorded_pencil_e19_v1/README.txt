Recorded E19 archival audit: exact-path relocation adapter

This tool audits the ORIGINAL recorded instrumented PENCIL V3 E19 archive.
It does not run a model, sampler, native kernel, producer or callspy; it does not
create new scientific observations, remeasure timings, edit recorded metrics,
install packages, contact a cluster or write remote storage. Fresh public rerun
validation is a separate task. The newer clean44 control uses a different frozen auditor.

Environment: OFFLINE audit Python3.12 with NumPy. Python3.10 is the historical
GPU science environment and is NOT accepted by this archival entrypoint, because
the original frozen REPLAY-callchain raw AST hashes were produced with Python3.12.
No attempt is made to reinterpret those AST pins across Python versions.

Place whole immutable sources under an arbitrary artifact root:
  work/vldb_pencil_final_audit_v3/        all8 manifest files + manifest
  work/vldb_pencil_replay_addendum_v2/    all6 manifest files + manifest
  work/vldb_pencil_queue_v3/             all98 manifest files + manifest
There are115 exact source dependencies, totaling less than1MiB. Include the whole
queue source tree, not only its2 runtime source pins. No historical wheel archive,
CUDA library, graph data or GPU environment is needed for this offline audit.
Every source file, source manifest, correction/callspy receipt and historical
anchor is verified by whole bytes/SHA before any frozen verifier import.

Genuine original evidence under ARTIFACT_ROOT/metadata/evidence/E19/:
  ORIGINAL_FIRST_FAILED_ATTEMPT.json   original unsuccessful first audit, exit2
  ORIGINAL_FIRST_FAILURE_STDERR.txt   original REPLAY-zero-timer rejection
  RECOVERY_ADMISSION.json             actual historical root recovery receipt
  ROOT_LAUNCH.json                    actual historical root launch receipt
The public materializer provides these whole original files and a full original
collector export (final_snapshot.tgz, extracted/,13 genuine unchanged sidecars).
Original file bytes and all old observation/path fields remain unchanged. They
are historical evidence; none is a new statement about the reader's machine.

Generic command:
  "$AUDIT_PY312" "$ARTIFACT_ROOT/tools/recorded_pencil_e19_v1/audit_recorded_e19.py" \
    --source-root "$ARTIFACT_ROOT" --export "$E19_MATERIALIZED_EXPORT" \
    --output "$FRESH_E19_ARCHIVAL_AUDIT" \
    --root-launch "$ARTIFACT_ROOT/metadata/evidence/E19/ROOT_LAUNCH.json"
--artifact-root is an exact alias for --source-root. --evidence-root can supply a
separate directory containing the four genuine historical anchors. Output must be
fresh and disjoint from source root, export, adapter and evidence directories.
The raw archive must be exactly9,966,534bytes with SHA256
7ab909e987af900f8ad5f9ea9a559e17e650f78891cb0bc99200a0fdb97fa36c.
The complete original collector sidecars are also whole-hash pinned, including
export.json; do not rewrite its historical absolute path metadata when relocating.

Exactly permitted transformations:
  1. Source-root globals locate the byte-unchanged core/addendum/queue trees.
  2. A MODULE-LOCAL Path constructor view maps THREE exact historical local file
     names: first failed attempt, original stderr, and root recovery admission.
     Each target is first whole-hash verified in the supplied evidence directory.
     No pathlib/global filesystem monkeypatch, broad prefix rewrite, fallback to
     author files, modified JSON read view or scientific predicate replacement is
     added by this adapter. Unlisted historical author paths are refused.
  3. The unchanged existing addendum performs its already-reviewed one-predicate
     correction: a nonzero REPLAY frozenS-materialization timer is permitted while
     real native sampler bypass remains required. All other original scientific,
     source, lifecycle, environment, trace/model/full-score, count/unit/median/QPS
     gates remain the original frozen functions. Original first-failure exit2 and
     stderr, correction SHA and one-shot callspy receipt are preserved verbatim.

The frozen report audit.json is written normally by the unchanged auditor. A
separate ARCHIVAL_RELOCATION.json records the real local audit Python/source root,
exact exercised alias mappings, report/source/archive hashes and scope. That new
receipt is a local archival verification record, not new scientific observations.
All48 historical jobs must pass, including32 ONLINE and4 supplemental REPLAY
benchmarks and9 semantic audits. No old48-to-clean44 relabelling is performed.
Original instrumented wall times retain their limitations and cannot be combined
with newer H100 measurements to infer a causal smaps speedup.

Meaningful corruption/path tests:
  E19_SOURCE_ROOT=<artifact-root-containing-whole-sources>
  E19_EXPORT=<genuine-full-materialized-E19-export>
  E19_EVIDENCE_ROOT=<genuine-four-historical-anchors>
  E19_ROOT_LAUNCH=<genuine-original-root-launch>
  [optional] E19_REFERENCE_AUDIT=<original corrected48-job audit.json>
  "$AUDIT_PY312" -m pytest -q -p no:cacheprovider \
    tools/recorded_pencil_e19_v1/test_relocation.py
Tests require the materialized archival assets for their integration cases; pure
path/version checks can run without them. Validation uses spaces/Unicode and a
wholly different temporary source/export/output root. Corruption tests unlink a
test hardlink before writing, preserving all shared frozen source/evidence inodes.
They reject changed core/producer/first-failure/stderr/recovery/launch/archive/
sidecar/extracted bytes, missing or symlinked anchors, changed dependency pins,
path traversal, unsupported Python and unsafe output locations. The genuine
archival roundtrip recomputes the report; an optional pinned original report is
compared only AFTER successful re-auditing, never copied into the result.
