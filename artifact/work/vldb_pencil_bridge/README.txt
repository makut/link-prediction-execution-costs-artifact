PENCIL strict-clean systems bridge v1
===================================

Purpose and claim boundary
--------------------------
This source-only package runs the actual unmodified pinned PENCIL dataset
__getitem__, native target mask, DatasetWrapper, Collator, STokenizer and BERTLP.
It replaces only the induced-subgraph physical operator behind the consumer.
PENCIL is pinned to quang-truong/pencil commit
2d32e29dbed533288d9d758138e07547a0a7d8a9 (Apache-2.0).
Reference paper: https://proceedings.mlr.press/v306/truong26a.html

The real systems cohort is predeclared: all 512 frozen citation2 validation
positive queries, followed by candidate index 0 from each corresponding original
1000-negative group. Every original group/source is checked before candidate 0
selection. This yields exactly 1024 queries; self/duplicate candidates are kept.
No test payload, training, HPO, MRR or full 1000-negative ranking is performed.
Random frozen model weights check functional equivalence, not predictive quality.

The driver preserves the actual sorted sampled S, arbitrary/equal root positions,
raw directed edge order, raw CSR edge-slot IDs, separate native stored edge-ID
values, x/id, native target masking, wrapper, collator and model inputs. Capture
freezes Python/NumPy/Torch CPU RNG after native sampling. The semantic audit uses
the actual ONLINE public sampler as reference, and restores that postraw state
before each candidate's unchanged wrapper, including training-time randperm.

Arms
----
torch_sparse_public       actual SparseTensor.saint_subgraph (primary control)
torch_sparse_direct       same installed registered native operator, direct call
compiled_scan_int64       new bounded sorted-S row-scan bridge kernel, int64 cols
compiled_pairwise_int64   new general directed pairwise bridge kernel, int64 cols
compiled_scan_int32       same new row scan, original compact int32 cols
compiled_pairwise_int32   same new general pairwise kernel, compact int32 cols
cpp_full_clear           corrected standalone C++ full N-sized assoc initialization
cpp_touched_reset        same corrected C++, persistent map and touched reset

The new compiled bridge kernels are NOT timings of the frozen production SEAL
kernel. The C++ persistent map charges the full 8*N bytes per process and its
initialization. Raw induction retains both target directions; unmodified PENCIL
removes both afterward. All arms return native idx-order CSR scan output.

All native graph representations remain resident for ALL measured arms. This
driver therefore DOES NOT measure memory savings. Timing includes native output
packaging, value gathering and downstream native transpose/mask/features/wrapper.
Throughput is instrumented: per-query smaps/memory recording, S checks, loader IPC
and score copy are in the denominator. Raw induction timing is separately recorded.

Environment
-----------
Read environment-target.txt and requirements-extras.txt. Root's prepared target is
Python 3.10.12, Torch 2.5.0+cu121, torch-sparse 0.6.18, PyG 2.7, NumPy 1.24.4.
Keep existing Torch/CUDA/NumPy/SciPy/Numba/PyG/torch-sparse. Do not install upstream
requirements or launch the upstream 8-GPU script. Only one visible GPU is admitted.
Transformers 4.46.2 is required because pinned PENCIL uses its private SDPA API.

Use a fresh venv with --system-site-packages for the extras if needed. An optional
separate Linux x86_64 CPython 3.10 wheelhouse is for offline staging, not part of
this public source archive. It contains NumPy 1.24.4 only to constrain resolution;
do not upgrade/reinstall the existing base stack. The wheelhouse's exact dependency
versions are not claimed to have been executed locally on Linux; native admission
and completed audit in the destination environment remain mandatory.

Example isolated offline install (after staging the MATCHED cp310 wheelhouse):
   python -m venv --system-site-packages ../pencil_env_v1
   ../pencil_env_v1/bin/python -m pip install --no-index \
     --find-links ../pencil_bridge_wheelhouse_linux_cp310_matched \
     -r requirements-extras.txt -c constraints-linux-cp310.txt
Use that interpreter for all subsequent commands. Wheelhouse dependency versions
match successful local imports, with the destination's existing NumPy1.24.4 pin;
shared binaries are specifically Linux x86_64 CPython3.10/compatible abi3 wheels.

Set WANDB_MODE=disabled and HF_HUB_OFFLINE=1. The included local BERT config is
570 bytes; no pretrained weights or network fetch is needed. See source.json.

Ordered execution (examples run FROM this package directory)
-----------------------------------------------------------
The commands below are examples, not an instruction to overlap running jobs.
Finish other timing jobs first. Use a FRESH results directory OUTSIDE this package,
all graph/input directories and the frozen trace directory. Every existing output
or output.partial is preserved, including failed/preflight output. Capture also
requires a fresh trace directory. Omit --execute for read-only preflight.

1. Build the exact reviewed C++ control locally (or use root's already matched
   library plus its build_receipt_v1.json). Never ship a Mac binary to Linux.

   python corrected_native/build_native_v1.py --output-dir ../pencil_cpp_build_v1

   Linux library: ../pencil_cpp_build_v1/native_assoc_v1.so
   Adapter: corrected_native/native_adapter_v1.py
   Default receipt: build_receipt_v1.json beside the supplied library.
   --cpp-build-receipt overrides that path if the receipt was staged separately.
   Adapter SHA and receipt source/library SHA are verified before full allocation.

2. Native tiny capture, before allocating the full graph. All seven cases include
   cold roots, cold self-query, equal warm roots, arbitrary root positions and
   target directions. This is an actual torch-sparse admission, not a simulator.

   python pencil_bridge.py capture --fixture --execute \
     --trace-dir ../pencil_results_v1/tiny_trace \
     --output ../pencil_results_v1/tiny_capture.json

3. Exact native tiny consumer/model audit, all eight arms, eval and train wrapper.

   ARMS=torch_sparse_public,torch_sparse_direct,compiled_scan_int64,compiled_pairwise_int64,compiled_scan_int32,compiled_pairwise_int32,cpp_full_clear,cpp_touched_reset
   python pencil_bridge.py audit --fixture --execute \
     --trace-dir ../pencil_results_v1/tiny_trace --arms "$ARMS" \
     --cpp-module corrected_native/native_adapter_v1.py \
     --cpp-library ../pencil_cpp_build_v1/native_assoc_v1.so \
     --forward --tiny-model --training-wrapper --batch-size 128 \
     --output ../pencil_results_v1/tiny_audit.json

4. Declare real inputs using the already frozen root paths. Shell arrays below
   use bash/zsh. No graph construction, heldout selection or quality plan is changed.

   REAL=(
     --graph-path /home/jovyan/seal_revision_20261124/data/citation2/prepared/train_csr
     --features /home/jovyan/seal_revision_20261124/data/citation2/prepared/node_feat.npy
     --valid-positive /home/jovyan/seal_revision_20261124/results/quality/subsets/valid_edge.npy
     --valid-negative /home/jovyan/seal_revision_20261124/results/quality/subsets/valid_edge_neg.npy
     --strict-manifest /home/jovyan/seal_revision_20261124/data/citation2/prepared/manifest.json
     --quality-plan /home/jovyan/seal_revision_20261124/results/quality/plan.json
   )
   python pencil_bridge.py preflight "${REAL[@]}" \
     --output ../pencil_results_v1/real_preflight.json

   The optional quality plan must be the exact original pinned plan. A recovered
   strict-clean manifest may differ in timestamps, but its critical graph/feature
   SHA/bytes and heldout audits must agree with the original contract. Every actual
   graph/feature/validation payload is streaming-hashed during setup, before use.
   No pickle is loaded from the dataset or trace.

5. Capture actual online native S and raw outputs on the fixed 1024 queries.

   python pencil_bridge.py capture --execute "${REAL[@]}" \
     --trace-dir ../pencil_results_v1/real_trace \
     --output ../pencil_results_v1/real_capture.json

6. Run the real semantic/input audit. A CPU audit without --forward is a useful
   first gate. For model throughput, audit with the exact device/precision/model
   and batch/worker settings intended for measurement. Here the default real BERT
   is 3 layers x512, 8 heads, FFN2048, adjacency_row, supported128-dim early features.
   This feature-enabled setup differs from the official feature-disabled quality
   configuration. The random model is frozen eval; there is no training claim.

   python pencil_bridge.py audit --execute "${REAL[@]}" \
     --trace-dir ../pencil_results_v1/real_trace --arms "$ARMS" \
     --cpp-module corrected_native/native_adapter_v1.py \
     --cpp-library ../pencil_cpp_build_v1/native_assoc_v1.so \
     --training-wrapper --forward --device cuda --precision fp32 \
     --batch-size 128 --num-workers 0 --max-logit-abs 0 \
     --output ../pencil_results_v1/real_audit_gpu_fp32_b128_w0.json

   Native/compiled raw edge order, slots, stored values, full Data, wrapper and
   collated tensors must agree exactly. Model logits use predeclared tolerance0;
   if a failure occurs, preserve the failed receipt and diagnose before changing
   tolerance. --precision bf16 needs its own forward audit. Different batch size,
   declared worker count, environment/library hashes or bridge sources also need
   a new audit. Maximum admitted batch is512; workers are0 or1 (Linux fork only).

7. Isolated one-arm throughput, one process/command at a time. Repeat with each arm
   and a FRESH output. No timing overlap. Replay bypasses only native sampling;
   --pipeline-mode online additionally runs actual sampler and checks S every query.

   python pencil_bridge.py benchmark --execute "${REAL[@]}" \
     --trace-dir ../pencil_results_v1/real_trace \
     --audit-receipt ../pencil_results_v1/real_audit_gpu_fp32_b128_w0.json \
     --arm torch_sparse_public --pipeline-mode replay \
     --forward --device cuda --precision fp32 --batch-size 128 --num-workers 0 \
     --warmup 1 --repeats 3 --output ../pencil_results_v1/bench_public_replay.json

   For C++ arms pass the same --cpp-module, --cpp-library and optional receipt.
   JIT and full C++ map initialization occur before steady-state sweeps; setup,
   initialization, worker-private memory and all model/device allocations are
   separately recorded. Persistent workers shut down on success or failure.

Verification provided with this package
---------------------------------------
32 local tests passed, plus Python3.10 syntax parsing of five bridge/test modules.
See local_validation.json for exact source hashes and scope. Local tests execute
actual pinned PENCIL source/model, Numba kernels and compiled C++ controls, but
use an explicit sparse API simulator because native torch-sparse is absent on Mac.
Actual Linux/CUDA/native compatibility, real 1024-query audit and timings remain
parent-owned execution tasks and are not represented as completed here.

Run local checks with:
   WANDB_MODE=disabled python -m unittest discover -s . -p 'test_*.py' -q
The optional C++ integration test uses PENCIL_BRIDGE_CPP_LIBRARY when set.
The local corrected_native/test_native_v1.py suite can be run separately.

Safety/capacity defaults
-----------------------
32GiB host guard, native public full-E COO budget1GiB, trace512MiB, per-query
selected-neighbor budget20M, C++ map budget1GiB; cgroup/available memory checked.
The full native64-bit transpose/values/COO setup is charged and warmed explicitly.
Default command behavior is preflight only; execution requires --execute.
