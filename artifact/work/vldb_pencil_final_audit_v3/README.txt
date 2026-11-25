Independent LOCAL PENCIL recovery V3 auditor

This is separate from the frozen scientific queue/bridge. It reads only local
evidence, imports NumPy/stdlib and imports frozen queue code only to derive the
actual declared job plan. It never imports Torch/Numba/native extensions, calls
remote tools, changes sources/inputs, launches jobs or performs training/scoring.

Run:
  python3 audit_pencil_v3_v1.py
Optional --export DIR --output FRESH_DIR --root-launch LOCAL_ROOT_RECEIPT

Default absent export: prints waiting_for_verified_export, no output directory,
no scientific/performance parsing. A recovered raw archive without the external
final COMPLETE/launch receipts is NOT a verified final export. Its explicit
failure diagnostic is separate; no final controller/lock cleanup is invented.

Accepted full local export schema is the prior read-only collector layout,
adapted to V3 namespace/YT target and pencil_queue_v3_launch.json:
  export.json, final_snapshot.tgz, extraction_manifest.json,
  extracted/SHA256_MANIFEST.json,
  extracted/results/pencil_bridge_v3/whole queue results,
  remote_receipts/results/pencil_bridge_v3_20261125/
    COMPLETE.json, SCIENTIFIC_TERMINAL.json, plan.json,
    archive_pointer.json, archive_local_receipt.json,
  remote_receipts/pencil_queue_v3_launch.json
Status verified_existing_persisted_archive; role pencil; exact existing final YT
target/pointer/archive bytes+SHA; zero collector mutations. Every whole regular
tar member and actual extracted file is independently rehashed. No truncation.

FROZEN_CONTRACT.json binds NEW source packet56954a..., queue dda406..., all98
source files, eight driver source hashes, unchanged upstream53 files and original
strict-clean full citation2/cohort payloads, actual fresh Linux native build16d3...
and library2a8b..., exact35-wheel contract, root recovery receipt3589....
The independent root recovery anchor remains local and unchanged. It records
original pod deletion, prior pipeline/native scientific success, original PENCIL
V2 failure with0 completed jobs, and UNKNOWN old outer controller cleanup.

Checks:
- Exact saved48-job argv/defaults, original budgets and8-arm/precision/batch/
  declared-worker matrix; initial source bindings vs admitted wheel binding.
- Completed inventory is the exact admitted job prefix with output+exit hashes,
  whole bounded logs, clean owned process trees and external180GiB guard logs.
  Failed/guarded/not-started evidence stays explicit. Persisted scientific failure
  produces an inventory report without throughput headline or speedup analysis.
- Full base/isolated pre/post equality, all numeric/extra/native origins, plain
  existing base .pth/priority, all35 offline wheels, original saint wrapper and
  actual binaries, actual root-pinned replacement admission and policy bindings.
- Explicit graph-only optional vision policy version/source/hash/origin must
  agree with the loaded pinned Transformers4.46.2 import_utils and actual isolated
  probe; every executed driver's stable semantic environment must agree.
- Pickle-free7-case tiny and1024-query real trace files, arbitrary/equal roots,
  sorted literal S<=82, IDs/labels, CPU RNG payload hashes, directed raw CSR slots
  and separate stored values, postmask tensor metadata. Tiny graph raw order,
  reverse-edge stored IDs, target removal and features are independently rebuilt.
  Real edge/value arrays are not exported: their equality is checked through all
  full-cohort native receipts and source hashes, not a new graph oracle run.
- All8 arms x eval/train wrapper semantic gates, exact full batch partition,
  FP32/BF16 declared shapes and zero-tolerance actual model-logit fingerprints.
- Matching completed semantic audit bound by SHA/model/environment/trace before
  each benchmark; every full1024-query warmup+3 measured sweep, QPS and median
  recomputed, finite stage summaries checked in seconds, per-process C++ owner,
  library/map initialization costs, CUDA memory in bytes.
- All36 measured jobs, five separate precision/batch/worker/mode groups; complete
  ordered1024-score hash exact across every arm and all4 sweeps within its group.
  Different batch/mode shapes are never pooled. Online/replay ratio only if their
  same-arm score/model fingerprints also agree.

Outputs (fresh directory only): audit.json and summary.txt. Primary ONLINE BF16
rows separate from supplemental REPLAY. Ratios are ratios of median complete
sweep wall times; no paired per-query ratio or confidence interval is inferred.
Raw/induction stages are nested; worker overlap prevents summing CPU stages into
wall time. Induction includes value/SparseTensor packing. Wall includes smaps,
S checks, IPC, H2D, forward and score copy. GPU high-water includes warmup; PSS is
sampled. All native representations remain resident: no memory-saving claim.
Random frozen BERT seed42 is not a trained quality/MRR/ranking/SOTA result.
Historical different-host Numba/native timings are never used as speed ratios.

Meaningful regression: python3 -m pytest -q test_final_audit.py
110 tests cover archive/extraction/terminal/source/argv mutations; full failed
science vs transport; tiny/raw slot/stored-value/mask/RNG corruptions; real1024
group order; eval/train batch coverage and precision settings; missing arms/
scores; QPS/time units; C++ map/owner costs; owned guards/orphans; all numeric/
extra/HF/policy origins; .pth execution/priority; root recovery hash/old failure.
Full synthetic48-job success integration exercises the complete verified-export
to scientific-report path; Boolean exit/status/count and unknown PSS cases are
rejected or labelled explicitly. Remote SCI/plan bytes must equal archived bytes.
Synthetic inputs are labelled fixtures and are never scientific measurements.
