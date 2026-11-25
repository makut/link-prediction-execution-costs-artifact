Repaired public PENCIL entrypoint candidate
=========================================

This directory is a development candidate built on an unchanged copy of the
0.4 public artifact. It is not a released artifact or a scientific result.
`INHERITED_V0_4_MANIFEST_REFERENCE.json` describes the inherited 0.4 files only.
The new overlay inventory is separate. No recorded throughput is reproduced
merely by the presence of this code. A later release requires independent
review of the actual new results and source integration.

The new public entrypoint uses the exact bridge V2 from the frozen queue V3
source packet SHA256
`56954a16a9d45d66783a82961fe82cffe4144aad5b3bef382ce2d05ed724d26c`.
All 80 bridge files, the original runtime probe, and the protocol specification
are byte-identical copies. The graph-only optional-vision policy runs before
any BERT import. It verifies the pinned Transformers 4.46.2 import-utils source
and disables discovery of unused torchvision in that fresh worker. It changes
no package files, upstream sources, Torch operators, model weights, graph
computation, or precision settings.

From this candidate's `src` directory, the default is an offline plan:

```sh
python -B -m artifact_public.reproduce_pencil_v2 \
  --citation2-root /scratch/public-citation2-strict-clean \
  --out /scratch/pencil-v2-new-run
```

Plan mode verifies packaged sources and prints the plan. It does not import
NumPy, Torch, Transformers, native sparse libraries, load graph data, create an
output directory, install packages, access the network, or launch GPU work.
The two input/output paths in this example are user-supplied local paths.

For an actual rerun, provide an already prepared public strict-clean Citation2
directory and a fresh output directory outside both inputs and source. Use a
Linux Python 3.10 environment with Torch 2.5.0+cu121, NumPy 1.24.4, SciPy
1.10.1, Numba 0.58.0, torch-sparse 0.6.18 for that Torch/CUDA stack, PyG 2.7.0,
Transformers 4.46.2, wandb 0.18.7, rich 14.2.0 and tqdm 4.67.1, plus a C++17
compiler. The inherited public requirements and copied bridge constraints
document transitive dependencies; do not install the upstream PENCIL
requirements into the numerical environment. The adapter never installs or
repairs packages. Append `--execute` to the same command when the environment,
public input and compute allocation are ready.

The entrypoint checks that the strict-clean manifest declares the exact 19
numeric payload identities of the public reconstruction. It independently
streams hashes of the five graph/feature/full-validation payloads it uses.
It does not open the test candidate arrays or claim to independently rehash
all 19 payloads in this command. The public data-preparation workflow performs
the full reconstruction verification separately. The adapter regenerates only
the frozen 512-source validation cohort with seed 20261124 and requires both
output hashes to match the original bridge contract. An existing cohort may
be supplied with `--validation-root`; its hashes remain mandatory. No original
private quality-plan file, historical absolute data path, PID, predecessor
terminal marker, deployment, YT service, wheelhouse, or storage receipt is needed.

The scientific matrix retains all 48 jobs: tiny capture/audit, real preflight
and capture, eight real FP32/BF16 semantic audits, 32 ONLINE BF16 benchmarks,
and four supplemental REPLAY BF16 benchmarks. All eight physical arms,
batch sizes 128/512, worker counts 0/1, model seed 42, extraction seed 20261126,
one full excluded warmup and three measured sweeps are retained. The nine
semantic audits and all five separate measured score groups use the original
V3 validator function bodies. The only scientific command-line change is
omitting the optional private-path-bearing `--quality-plan` argument; the
unchanged content contract, strict-clean manifest, exact full graph/features
and exact 512-source cohort remain required.

Portable setup adds a numerical/native/HF environment probe before the jobs,
and another after them. It reuses the unchanged original probe in base mode,
checks the declared package target and saint wrapper, records native library
hashes and module origins, and configures the exact graph-only policy before
BERT. Each executed driver must match those native/version/policy bindings;
the complete before/after receipts must also agree. The C++ library is built
locally from the pinned source. Existing 0.4 local ownership, memory/disk
guards, timing locks, failure retention and single-visible-GPU controls apply.
No controller or environment from the historical experiment is reused.

This is an inference systems protocol using a random frozen BERT model. It
does not train, search hyperparameters, measure ranking quality, establish
time-to-quality, or reproduce a trained PENCIL checkpoint. REPLAY bypasses
sampling and is supplemental. Timed sweeps retain the original instrumentation,
including per-query memory observations. All full native representations
remain resident; this protocol supports no PENCIL memory-saving claim. Ratios
must use complete sweep medians within matching precision/batch/workers/mode
groups. Native setup and payload hashing occur outside those sweep timers.

Local tests
-----------

The public tests cover the exact job matrix, default offline CLI, source and
manifest corruption, private-plan independence, native/policy drift, fresh
output refusal, and a real pinned BERT forward under a deliberately failing
optional torchvision import. Copied bridge tests additionally exercise tiny
graph oracles, actual PENCIL classes and BERT with an explicitly labeled sparse
API simulator, and actual compiled C++ controls on tiny arrays. They are CPU
validation on the available local environment, not native torch-sparse/CUDA
admission or a full scientific rerun. Keep bytecode, Numba and HF caches outside
the source tree when invoking the tests.

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  artifact_public/test_pencil_public_v2.py
```

`metadata/PENCIL_V2_DERIVATION.json` records the new orchestration boundaries and
the exact AST identities of the extracted scientific validators. Frozen source
identities live in `metadata/PENCIL_V2_FROZEN_SOURCES.json`. Root coordinates
any future release and actual experimental use.
