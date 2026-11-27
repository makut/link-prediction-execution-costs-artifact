Native preparation reuse and two-graph application lifetimes
Public extension README — prospective draft, 27 November 2026

This directory is a source-only addition for a possible v0.8 revision. The
published v0.7 artifact, release catalog and manifests remain unchanged.
SOURCE_ONLY_STATUS.json and SOURCE_ADDITION_MANIFEST.json inventory this new
source addition; neither is a result catalog or publication receipt.

STATUS

This document describes the frozen design and how its eventual recorded
evidence is to be checked. It does not report completed experiments or an
observed winner. A successful tiny qualification, a source review, and a
complete independently audited 36-application cohort are different milestones.
Only the last can support the new application-lifetime result.

This extension keeps the earlier E25 experiment separate. E25 contains 24
Friendster applications with one/eight workers; these new controls require
36 new applications on two graphs with eight workers only. Do not pool them.

The examples assume an extension directory containing packet/,
audit_closure.py and the separate offline_verify_closure.py helper. Published
recorded input paths and hashes must come from a final immutable evidence
index. No complete recorded index is asserted by this draft. Placeholder
paths, pending examples and historical source-readiness records are not evidence.

CHOOSE THE CHECK YOU NEED

* Recorded verification: Python standard library only. Re-read the supplied
  graph-free raw archives, independently reconstruct six block audits and two
  dataset aggregates, and compare them with the recorded files. No GPU, graph
  download, numerical package, author account or cluster connection is needed.
* Tiny qualification: compile the standalone C++ control and run a small actual
  three-arm consumer. This checks integration, worker exits and input identity.
  It neither runs the full graph nor establishes timing or quality results.
* Fresh scientific execution: prepare the exact public graph bytes, then run
  six genuinely fresh admitted GPU allocations. This creates new observations;
  it does not promise the historical latencies or a turnkey external deployment.

Verify the published release/archive against its external checksum before
using its evidence index. An unchecked archive's own index cannot authenticate
that same archive. The offline helper also requires the expected whole SHA256
of its index; it does not derive trust from a self-reported PASS.

FROZEN DESIGN AND MEASUREMENT BOUNDARIES

Inputs are full dense Friendster and strict-clean Citation2 trainCSR. Each
dataset has three fresh allocation blocks. Each block runs the same two
horizons across three arms, six applications in total:

  reference_union_compiled
  reference_union_native_fresh
  reference_union_native_reuse

Every application uses eight spawn workers and the original fixed E23
reference-union queries, ordered selected vertices, DRNL and featureless DGCNN.
Short applications consume 256 warmup + 320 measured graphs (576 total);
long applications consume 256 + 6,400 (6,656 total). Both genuinely terminate.
The predeclared allocation order is Friendster1, Citation2_1, Friendster2,
Citation2_2, Friendster3, Citation2_3; arm order rotates within each dataset and
horizon so each arm occupies each position once. There is no result-based
selection of order or successful applications.

Native-fresh creates and fsyncs its full int64 representation inside each
application clock. Native-reuse opens one immutable, read-only representation
prepared once per dataset outside ALL application clocks and retained across
its three allocations. Both native arms still use compact CSR for sampling
and create a private mutable 8N-byte lookup in EACH worker inside that clock.
Read-only file pages may be shared; the mutable worker lookup is not shared.

The parent clock begins before process creation and ends at the actual
returned wait. Imports, setup, private state, natural JIT/cache loading,
training, complete consumed-input hashing, metadata/resource/PSS observations,
reporting and shutdown remain included. Full source/native integrity scans,
C++ build and tiny qualification are outside application clocks and may warm
pages. The Numba cache is fresh per allocation and shared within it, including
qualification; OS/filesystem caches are retained. No cold-I/O or uninstrumented
latency claim follows.

Typed, shaped, ordered consumed CPU inputs and query IDs are hashed before
transfer to the GPU. The short final digest must equal the long semantic
576-input prefix. That long digest is NOT an extra CUDA timing checkpoint.
Offline verification checks source-bound recorded digests and raw provenance;
it does not independently regenerate graph arrays or model tensors.

PSS maxima use complete, stable-membership, same-phase process-tree samples.
They are sampled, non-atomic and residency-dependent, not exact peaks or sums
of RSS. Missing observations stay unobserved. Ratios are paired within the
same dataset, horizon and actual allocation before descriptive medians/ranges
over three blocks. Allocations are not independent datasets, model seeds or
machines; these ranges are not population confidence intervals.

One-time preparation uses a CPU-only 16-CPU/64-GiB context. GPU applications
use 27 logical CPUs, a 256-GiB cgroup limit and one whole H100; these are
declared measurement resources, not established minimum requirements.
Preparation and applications use different hosts. Conversion/fsync and full
validation/readback are separately reported staging durations. Do not add CPU
preparation durations to GPU application medians or infer same-host amortized
break-even. Monotonic timestamps are comparable only within their host/boot.

DATA AND ENVIRONMENT FOR A NEW RUN

The packet deliberately contains no full graphs, native arrays, model
checkpoints or acquisition code. Use the base public artifact's DATA_ACCESS.md
and PUBLIC_COMMANDS.txt for the established public reconstruction recipes:

  https://github.com/makut/link-prediction-execution-costs-artifact/releases/tag/v0.5.0

Friendster's dense source is Per Fuchs, Zenodo 5146230 (CC-BY-4.0):
https://doi.org/10.5281/zenodo.5146230 . Citation2 is the official OGB release
(ODC-BY): https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2 . Preserve data
attribution as well as the separate upstream code notices. A generic OGB graph
or the similarly named prepared_original graph is not the strict-clean input.

Each graph root supplied below must contain the exact edge_starts.bin,
edge_ends.bin and num_nodes.txt bytes in packet/pins/*_graph_expected.json.
Friendster has 65,608,366 vertices and 3,612,134,270 stored entries; Citation2
has 2,927,963 and 60,703,502. The packet includes the small original query
workloads and verifies their whole identities. Generated metadata timestamps
or compressed container bytes are not interchangeable with numeric graph pins.

For the full frozen run, use Linux x86_64/Python 3.10, a C++17 compiler,
NumPy 1.24.4, SciPy 1.10.1, Numba 0.58.0, Torch 2.5.0+cu121 and
torch-geometric 2.7.0, plus the base artifact's declared runtime dependencies.
SOURCE_PINS.json is the version authority. Proc, cgroup-v2, sysfs and actual
GPU/resource observations must be readable. No CPU, MIG or DDP fallback is
provided for the full scientific control.

Use large remote storage for graph preparation and full execution. Account for
the compact graph, one retained native representation, additional native-fresh
scratch and caches together. A native Friendster representation is
29,421,941,096 bytes; Citation2 is 509,051,728 bytes. Preblock admission requires
additional free space of native-array bytes + 6 GiB, and the runtime reserve is
4 GiB. These are fail-closed observations, not capacity reservations. Keep
temporary files and caches on the admitted persistent storage. Do not download
full graphs merely to run the offline evidence check.

SOURCE PINS

Frozen packet SOURCE_MANIFEST.json whole SHA256:
  0bb01dfd0524340e5f3f9f08ab1e0f2fc604e47a726db498f59ee1b376149bb6
Canonical protocol-object SHA256 used by --protocol-sha256:
  ea487e898ae9dd07113408990bb48e55e37dafeef3d23e5b8ef0ab1d88462566
Frozen envelope/run_block.py whole SHA256:
  c5b90c729ca3956b9c306967161f8b8a676733d565149f35a55f11757e85a5d8
Frozen independent audit_closure.py whole SHA256:
  98ad346c6d6529878e859441df72ce9b33c226eb01229bf015adda1926b0834c

The canonical protocol pin hashes json.dumps(parsed_protocol, sort_keys=True),
not the pretty-printed PROTOCOL.json bytes. Do not substitute its file hash.
Do not edit frozen packet files or add output/__pycache__ files inside it:
complete source-inventory verification would fail. This external README
corrects an old frozen README command; the valid flag is --max-seconds 2400,
with a space (or --max-seconds=2400), not --max-seconds2400.

TINY ACTUAL-CONSUMER QUALIFICATION

Set CLOSURE_EXT to the extension's absolute path, CLOSURE_TINY_BUILD to a fresh
build directory outside the packet, and CLOSURE_TINY_OUT to a fresh output.
In the supplied numerical environment:

  python3 -B "$CLOSURE_EXT/packet/native_core/build_native_v1.py" \
    --output-dir "$CLOSURE_TINY_BUILD"

  python3 -B "$CLOSURE_EXT/packet/qualify.py" \
    --library "$CLOSURE_TINY_BUILD/native_assoc_v1.so" \
    --out "$CLOSURE_TINY_OUT" --device cpu --workers 1

On macOS the built library is native_assoc_v1.dylib. The macOS CPU1 path is a
portable integration check. On Linux, even CPU1 qualification retains the
fixed27 environment gates; it is not an unrestricted laptop/CPU smoke test.
On an actually admitted target H100 allocation, use --device cuda --workers 8
and a separate fresh output. The full controller performs that CUDA8 tiny
qualification itself; a prior CPU1 PASS does not replace it.

ONE-TIME NATIVE PREPARATION AND FULL CONTROLLER INTERFACE

The following is the source-correct interface, not a portable cluster launcher.
Paths must be canonical absolute paths. Graphs, reuse storage, packet, outputs,
proofs, temp/cache and coordination directories must satisfy the actual admitted
storage binding. Source, input, output and coordination trees must be disjoint
as required by the frozen wrapper. Never reuse or fabricate the authors'
platform identities to make admission pass.

Prepare each dataset once, before its first application, on the CPU helper:

  python3 -B "$CLOSURE_EXT/packet/prepare_native.py" \
    --dataset friendster --graph-root "$CLOSURE_FRIENDSTER_CSR" \
    --out "$CLOSURE_FRIENDSTER_REUSE" --max-seconds 2400

For Citation2 use --dataset citation2 and its own graph/reuse roots. The reuse
output must be new and disjoint from its compact graph. Keep rowptr.i8, col.i8
and READY.json read-only, retain the exact READY.json whole SHA256, and preserve
the genuine preparation launch/returned-wait receipts separately. Do not
reprepare this representation inside a later allocation's application clock.

Plan one six-application block, after genuine platform admission:

  python3 -B "$CLOSURE_EXT/packet/run_lifecycle.py" \
    --dataset friendster --graph-root "$CLOSURE_FRIENDSTER_CSR" \
    --native-reuse-root "$CLOSURE_FRIENDSTER_REUSE" \
    --native-reuse-receipt-sha256 "$CLOSURE_FRIENDSTER_READY_SHA" \
    --out "$CLOSURE_BLOCK_OUT" --coordination-dir "$CLOSURE_COORDINATION" \
    --repetition-block 1 --platform-admission-json "$CLOSURE_PLATFORM_JSON" \
    --platform-admission-sha256 "$CLOSURE_PLATFORM_SHA"

This prints a source/proof-bound plan without scientific execution. Even this
plan needs a genuine matching platform proof and prepared receipt. To execute
the admitted block, add:

    --source-manifest-sha256 \
    0bb01dfd0524340e5f3f9f08ab1e0f2fc604e47a726db498f59ee1b376149bb6 \
    --execute

Run blocks 1, 2 and 3 for EACH dataset in fresh allocations, with fresh outputs
and genuine per-allocation proofs. Three invocations in one allocation do not
satisfy this design. The 3,900-second controller / 4,200-second envelope caps
are failure limits, not completion guarantees. Failed/partial attempts remain
diagnostics; do not append or selectively salvage their applications.

The frozen packet and envelope describe the authors' deployment. They require
real Pod/PVC/image/resource bindings, declared temp/cache locations, the
preparation receipt layout, an actual outer OS wait, and post-exit collection.
A direct controller run alone does not produce the full archived provenance
required by audit_closure.py. An operator on another platform needs a reviewed
admission/deployment/collection adapter preserving the scientific protocol.
No turnkey external adapter or tested full external replay is claimed here.

OFFLINE RECORDED VERIFICATION — AFTER A COMPLETE RELEASE EXISTS

Use Python 3.10+ and the standard library only. The separate helper takes a
published immutable evidence index and its externally verified whole SHA256:

  python3 -B -S "$CLOSURE_EXT/offline_verify_closure.py" \
    --index "$CLOSURE_EVIDENCE_INDEX" \
    --index-sha256 "$CLOSURE_EVIDENCE_INDEX_SHA" \
    --out "$CLOSURE_FRESH_VERIFY_OUT"

offline_evidence_index.pending.example.json illustrates the exact schema;
its PENDING_NOT_A_RESULT status and placeholder pins intentionally fail.
Every input uses a {path, bytes, sha256} pin, with paths relative to the index's
directory and no symlinks or parent traversal. Keep the actual index at the
extension root, alongside the pinned auditor. Its kind must be
CLOSURE_V1_PUBLIC_OFFLINE_EVIDENCE_INDEX and its status
COMPLETE_36_APPLICATIONS, supported by all actual files rather than a renamed
pending example. It contains the fixed source_hashes above, auditor, blocks
in F1,C1,F2,C2,F3,C3 order, and aggregates in Friendster,Citation2 order.

The index supplies the six original graph-free block_raw archives, their
post-exit ROOT_COLLECTION sidecars and recorded ACTUAL_AUDIT files, plus both
recorded dataset aggregates and the frozen auditor. All input whole hashes
must match. A pending example cannot be verified as a complete cohort.
The helper independently reruns the frozen auditor for each block, then
aggregates exactly three complete blocks for Friendster and three for
Citation2. Fresh audit and aggregate bytes must equal their recorded pins.
Only a successful complete roundtrip writes OFFLINE_VERIFIED.json.
The output directory must be new and outside the directory containing the
index, with an existing parent; failed outputs are preserved for inspection.

For manual diagnosis, the actual auditor interfaces are:

  python3 -B -S "$CLOSURE_EXT/audit_closure.py" block \
    --archive "$CLOSURE_RAW" --archive-sha256 "$CLOSURE_RAW_SHA" \
    --root-collection "$CLOSURE_COLLECTION" \
    --root-collection-sha256 "$CLOSURE_COLLECTION_SHA" \
    --source-manifest-sha256 "$CLOSURE_PACKET_SHA" \
    --protocol-sha256 "$CLOSURE_CANONICAL_PROTOCOL_SHA" \
    --wrapper-sha256 "$CLOSURE_WRAPPER_SHA" \
    --output "$CLOSURE_FRESH_BLOCK_AUDIT"

  python3 -B -S "$CLOSURE_EXT/audit_closure.py" aggregate \
    --block-audit "$CLOSURE_BLOCK1_AUDIT" "$CLOSURE_BLOCK1_AUDIT_SHA" \
    --block-audit "$CLOSURE_BLOCK2_AUDIT" "$CLOSURE_BLOCK2_AUDIT_SHA" \
    --block-audit "$CLOSURE_BLOCK3_AUDIT" "$CLOSURE_BLOCK3_AUDIT_SHA" \
    --output "$CLOSURE_FRESH_DATASET_AGGREGATE"

Run the aggregate separately for each graph. The auditor has no combined
two-dataset aggregate subcommand; one dataset's complete18 does not establish
the complete36. Producer summaries cannot replace the original raw evidence.
In particular, an archive created before its wrapper exits cannot prove that
future exit; its independently collected post-exit sidecar is also mandatory.
The verifier never opens recorded historical filesystem paths or executes
code from a raw archive. It does not fetch inputs or perform new measurements.

PLOTTING — AFTER BOTH REAL AGGREGATES EXIST

The prospective design can be inspected without result files or rendering:

  python3 -B "$CLOSURE_EXT/plot_closure.py" describe

After successful independent audits, validate both actual aggregates with:

  python3 -B "$CLOSURE_EXT/plot_closure.py" build \
    --friendster-aggregate "$CLOSURE_FRIENDSTER_AGGREGATE" "$CLOSURE_FRIENDSTER_AGGREGATE_SHA" \
    --citation2-aggregate "$CLOSURE_CITATION2_AGGREGATE" "$CLOSURE_CITATION2_AGGREGATE_SHA" \
    --root-ready "$CLOSURE_EXT/provenance/root_SOURCE_READY.json" \
      eadd76ed4252672774f1bcfe8288dd4fbf021d0f5de6eb5598ed53fb29926eb5 \
    --output-dir "$CLOSURE_FRESH_FIGURE_OUT"

Without --execute this validates inputs and writes no figure. Rendering needs
Matplotlib and --execute; each actual exported figure still requires separate
visual review. No figure or synthetic result panel is included here.

The small provenance/ files are original CPU preparation/source-readiness
records. They describe conversion and validation outside application clocks;
they do not establish completion of any new scientific application. Historical
host/container paths are preserved as observations, not public paths to open.

LICENSES AND INTERPRETATION

Own code is MIT. Retain packet/LICENSE, THIRD_PARTY.md, copied upstream notices
and the dataset attribution above. Full graph/native binaries and caches stay
outside recorded evidence archives. These controls concern preparation/state
lifetime and execution under the fixed reference-union DGCNN workload. They
do not establish standard OGB quality, full upstream SEAL throughput,
time-to-quality, a causal effect of graph size alone, or a universal execution
ranking. Interpret graph-specific or mixed results using the complete matrix.
