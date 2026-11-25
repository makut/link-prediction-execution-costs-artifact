PENCIL bridge V2 — explicit graph-only dependency admission

V1 remains frozen and unchanged. V2 changes only deployment compatibility:
- graph_only_transformers_v2.py explicitly disables unused optional torchvision
  discovery in Transformers4.46.2 BEFORE upstream/BERT import. The exact
  import_utils source SHA is pinned. No fake NMS operator, package install/edit,
  Torch registration, upstream source edit or graph/model approximation occurs.
- A stable compatibility receipt is exposed by load_upstream, every actual
  driver execution and audit_environment, and bound by recovery queue V3.
- source_files() includes the policy module, for eight driver source bindings.

Why: recovered PENCIL V2 failed in tiny_capture before NativeGraph allocation:
Torch2.5.0+cu121 shared a base image with torchvision0.17.2+cu121. BERT import
indirectly tried the unused vision package and raised:
  RuntimeError: operator torchvision::nms does not exist
The original persisted failure is retained; this version is a new experiment.

Scientific behavior is unchanged: actual pinned PENCIL upstream commit
2d32e29dbed533288d9d758138e07547a0a7d8a9; all53 upstream files unmodified;
strict-clean full citation2 source payload hashes; frozen validation512 positive
queries then candidate0 of each original1000-negative group, preserving equal
roots, duplicates and arbitrary root positions; sorted native S<=82.
Actual native7-case admission precedes full graph allocation.

All8 physical arms, ordered directed raw rows/cols/CSR slots and distinct stored
values, target removal, native Data/wrapper/collator and post-raw CPU RNG gates
are retained. Both eval and training wrappers are audited; measured throughput
uses eval. Real BERTLP is random frozen seed42, features128early, adjacency_row,
3x512/8heads/FFN2048. No quality/training/HPO/SOTA claim.

Use recovery queue V3 for reviewed execution; default driver remains preflight.
All native CSR/COO/value representations remain resident for every arm.
Instrumented wall throughput includes verification/smaps/IPC/H2D/forward/score
copy. Operator induction includes value packing. Nested stages cannot be summed.

Local verification: actual pinned BERT, wrappers/collator, Numba and corrected
C++ controls execute; local Torch-sparse is an independent sparse API simulator.
Linux Torch2.5/native/CUDA destination admission remains mandatory.

MIT for new author code; preserve THIRD_PARTY_NOTICES.txt and upstream licenses.
