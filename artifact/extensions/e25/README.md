# E25 v6: recorded Friendster application lifetimes

**Recorded PASS: 24 applications in three complete allocation blocks.** This
extension includes the original raw archives, post-archive root collection
sidecars, three independently reconstructed eight-application audits, the actual
complete-cohort aggregate and its figure. Source readiness, tiny qualifications
and synthetic checks remain separate historical preparation records.

| Recorded object | Bytes | SHA256 |
|---|---:|---|
| `recorded/RECORDED_IDENTITIES.json` | 1,913 | `931c6351e67e2454d53054503bcdc5bec93162b9935c7739bf950d9ff60dc6ce` |
| `recorded/ACTUAL_AGGREGATE.json` | 90,386 | `2a9bdfbbe4661bd41e1e5d1c1ade8046e40bcdd538bca44b5db1c2caf042e057` |

[EXTENSION_CATALOG.json](EXTENSION_CATALOG.json) indexes the current extension;
`SOURCE_STAGE_MANIFEST.json` records its payload identities.
The catalog pins each block's `block_raw.tgz`, `ROOT_COLLECTION.json` and
`ACTUAL_AUDIT.json`. The included aggregate reports
`PASS_COMPLETE_THREE_BLOCKS_24_APPS`; no failed or partial attempt supplies an
observation. Root collection records bind actual launch, subsequent wrapper
kernel exit 0, owned-tree quiescence, source/platform hashes and lock/marker
observations. The raw archives precede their wrapper's own exit and cannot alone
supply that later proof. These are recorded observations, not current liveness.

## One-command recorded verification

First verify the complete published code archive with its external release
SHA256. Its unchanged catalog is then the authority for the included evidence;
an unchecked archive's own manifest is not an independent trust anchor.
Set `ARTIFACT` to the absolute canonical artifact directory and `E25_FRESH_OUT`
to a new directory outside it, with an existing parent:

```sh
python3 -B -S "$ARTIFACT/extensions/e25/verify_recorded_v6.py" \
  --artifact "$ARTIFACT" --out "$E25_FRESH_OUT"
```

This Python 3.10+ standard-library entry point checks all input/source/PASS gates,
regenerates the three audits and aggregate using the unchanged included tools,
and requires byte-identical whole hashes before writing
`VERIFICATION_RECEIPT.json`. It performs no graph/model load, new measurement,
ML-library import, network or cluster action. Use a fresh output directory.
The exact schema and trust boundary are in [OFFLINE_VERIFICATION.txt](OFFLINE_VERIFICATION.txt).
The genuine all-three-block public-layout roundtrip passed with Python 3.12.10:
[helper receipt](recorded/OFFLINE_VERIFICATION_RECEIPT.json) and
[root context/argv/identity receipt](recorded/OFFLINE_ROUNDTRIP_ROOT_RECEIPT.json).
All three fresh audits and the aggregate were byte-identical to the originals.
The source-only entrypoint review remains a historical static check, distinct
from this actual validation.

## Plot the actual aggregate

The checked rendering uses Python 3.12.10 and Matplotlib 3.10.3. Set
`E25_PLOT_PY` to that plotting interpreter and `E25_FRESH_FIGURE` to a fresh output
directory; rendering is separate from the stdlib-only recorded verifier:

```sh
"$E25_PLOT_PY" -B "$ARTIFACT/extensions/e25/figure/plot_aggregate_v6.py" \
  --aggregate "$ARTIFACT/extensions/e25/recorded/ACTUAL_AGGREGATE.json" \
  --aggregate-sha256 "2a9bdfbbe4661bd41e1e5d1c1ade8046e40bcdd538bca44b5db1c2caf042e057" \
  --out "$E25_FRESH_FIGURE"
```

`figure/` includes the exact plot source, PNG/SVG/PDF, `plot_points.csv`,
`configuration_summary.csv`, `TABLEDATA.json`, a caption and TeX table.
`FIGURE_MANIFEST.json` pins the actual aggregate, source and output identities;
`ACTUAL_FIGURE_VISUAL_QA.json` records root's PASS inspection of the shipped
rendering. A new rendering starts with visual QA pending: the command does not
reproduce the separate recorded visual review. The current PNG SHA256 is
`769f56c27a2d6ab68840bea953fe62b2fc070413af036befe1edec42554b3f77`.

## Actual boundary finding

Compiled execution had lower launch-to-actual-wait time in all 12 within-block
pairs. For long applications with eight workers, the native measured-window
median was 35.22 s versus 38.23 s for compiled (median paired C/N 1.086), but
actual-wait medians were 332.47 s versus 52.30 s (median paired C/N 0.1573).
Warm-window ranking therefore did not determine terminated-application ranking
for this fresh file-backed path on the observed PVC. This is not a universal warm
native advantage: the long one-worker measured window favored compiled.
The [actual interpretation peer](recorded/E25_FINAL24_INTERPRETATION_PEER.json)
checks these scoped comparisons; it does not isolate promotion as the sole cause.

## Protocol and interpretation

Each original repetition is a fresh allocation block containing both physical
induction routes (`reference_union_compiled` and
`reference_union_native_persistent`), one/eight spawn workers and two genuinely
terminated horizons. Short applications train 576 graphs including 256 warmup
and 320 measured graphs; long applications train 6,656 including the same 256
warmup and 6,400 measured graphs. The sampler, ordered inputs, featureless DGCNN
and initial model are held fixed. Typed/shaped/ordered consumed CPU-input digest
and checkpoint-prefix equality are checked within and across blocks. This is
source-bound serialized equality, not independent regeneration of tensor/PyG or
full-graph bytes by the offline verifier.

Long curves use synchronized checkpoints at 32, 256, 1,280, 3,456 and 6,656
consumed graphs. The long run's digest at 576 is semantic-only: **there is no
long timed checkpoint at 576**. Separate markers show the actual waits of short
and long applications, including teardown and normal child completion. They are
not CUDA-checkpoint times; the short application is not a truncated long run.
Pointwise medians do not form one execution trajectory. Whiskers are descriptive
min/max across allocations, not confidence intervals. Ratios are formed between
arms within each block before descriptive aggregation; no E23 causal speedup,
fitted crossover, population uncertainty or time-to-quality claim follows.

The instrumented application clock includes imports/setup, worker spawn, natural
compiled JIT/cache load, hashing of every consumed model input, resource/PSS/disk
observations and metadata work. Each native application also creates and fsyncs
a fresh 29,421,941,096-byte file-backed representation from compact CSR inside
its clock. Whole-input admission hashes, C++ build and tiny qualification are
outside individual application lifetimes. The evaluated path is this fresh
file-backed persistent-native path; preconverted reuse or shared-memory promotion
would define different setup boundaries and were not timed here.

Outputs and Numba cache are fresh per allocation and shared within that block,
including qualification; OS/PVC filesystem caches are retained across blocks.
Source, graph, output, temp/cache and native scratch use the admitted writable
PVC mount. Its placement can affect setup and execution. These are not cold-cache,
exclusive-placement or best-possible-native-cost measurements.

PSS is the sampled application-process-tree maximum over every independently
eligible complete/stable/same-phase observation, including startup, spawn and
training. Missing eligible PSS remains unobserved, never zero; CPU/disk uncertainty
does not select PSS samples. File-backed pages are only partly resident and scans
are non-atomic: these values are not full-resident memory requirements, total RSS
sums or cache-subtracted private workspace.

## Observed platform and fresh replay boundary

The measured harness uses one whole 80 GB H100, a 256 GiB cgroup memory limit,
quota `2700000 100000`, and a chosen pool of 27 logical CPUs spanning 14 observed
physical-core identities. **The CPU marketing model/SKU was not recorded.**
Topology, quota, logical CPUs and physical cores are different facts; do not infer
an E25 CPU model from E23's separately observed host or from a node hostname.
Each block fixes its actual Pod/GPU/pool. All three observed allocations used
the same node; block 1 had a different logical pool from blocks 2/3, whose pools
match. These are not three independent machines or data/model seeds. The controls establish neither dedicated physical cores nor
continuous/exclusive placement. External observed masks may be nonempty subsets
of the pool; own controller/app/worker boundary leaders retain the exact pool.

A fresh execution needs Linux x86_64/Python 3.10.12, C++17, NumPy 1.24.4,
SciPy 1.10.1, Numba 0.58.0, Torch 2.5.0+cu121, PyG 2.7.0 and the declared extras,
readable proc/cgroup-v2/sysfs and the byte-pinned public full Friendster CSR.
There is no CPU/MIG/DDP fallback. The selected resources are measurement controls,
not established minimum hardware requirements. At least 40 GiB must remain free
above the prepared graph before a block; runtime PVC and writable-root reserves
are 4 GiB. Producer 3,900 s/envelope 4,200 s caps are failure limits, not speed or
completion guarantees.

The exact `source/packet/` and `source/outer/` document the author deployment.
The outer checks the authors' original PVC UID/subPath and internal image digest;
these are provenance, not a portable launcher or public image-access instructions.
`reference/admit_block_platform_v6.py.txt` is reference text. Another operator needs
a separately reviewed genuine admission/deployment binding preserving the
scientific protocol; no turnkey external binding is supplied or claimed tested.
Never fabricate identities or reuse the authors' platform proof. New-platform
measurements are new observations, not promises of the recorded latencies.

## History, exclusions and notices

`historical/env_v4/` preserves the previous 77-file source/provenance tree byte for
byte. Its old root-path examples, four-hour single-cohort CLI, v4 hashes,
`audit_e25_env_v4.py`/`independent_disk.py` and future collection/figure instructions
retain their original-layout meaning and are not active v6 commands.
`COPY_LINEAGE.json`, `RELOCATION_LINEAGE.json`, source-only status/readiness,
static peers and synthetic reviews describe preparation history. They do not
override the actual catalog/audits/aggregate or the later recorded-verification
receipt.

The latest failed capsule is included **once** under `recorded/excluded/`, with
v3/v2/v1 retained only inside it and status `FAILED_EXCLUDED_FROM_SCIENCE`.
Its whole SHA256 is
`5a470ea09dcfef2c754273dcb9d63288739b7c779799952e5f24a853b25c2365`.
No historical partial measurements enter the 24 successful applications.
Public release reviews are under `recorded/public_review/`.

Own code is MIT; original upstream MIT/Apache notices remain in the source tree.
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) covers every identical small public
Friendster workload copy, including archived ones: Per Fuchs, Zenodo 5146230,
CC-BY-4.0. Full graph/native binaries/caches/checkpoints are absent from the raw
archives. Historical author/container paths, UUIDs and inactive ownership fields
remain provenance; the offline tools never open those recorded paths.
The frozen `THIRD_PARTY.md` has artifact-wide references to `DATA_ACCESS.md`,
`PACKAGING_NOTES.md`, provenance metadata and PENCIL paths; resolve them at the
base artifact root, not as claims that unrelated PENCIL code is inside E25.
