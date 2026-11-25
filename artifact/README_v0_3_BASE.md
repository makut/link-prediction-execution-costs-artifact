# Public-data reproduction package — local staged version

This package contains the frozen SEAL extraction/training sources, exact
protocols, public input acquisition, deterministic Friendster split preparation,
small fixtures and payload identity checks. No private YT access is needed for
the public-data path. No complete graph is bundled.

**Release status:** version 0.3.0, local staging only. The authors' own code is
licensed under MIT (copyright 2026 LPP contributors), as approved by the user.
Upstream copyright and license notices are retained, including the Apache-2.0
notice for related VK-GNN prior work. Citation2 official-source reconstruction
is verified for all 19 numeric prepared payloads. Friendster full public
reconstruction remains pending; this is not a complete artifact reproduction.
See `LICENSE`, `THIRD_PARTY.md`,
`metadata/PUBLICATION_STATUS.json` and `metadata/SOURCE_SELECTION.json`.
No public repository URL has been created. The original versions 0.1.0 and
0.2.0 archives are preserved unchanged.

## Environment

Run the large workloads on Linux, Python 3.10.12, with enough CPU RAM and disk.
The original core environment used NumPy 1.24.4, SciPy 1.10.1, Numba 0.58.0,
pandas 2.2.1, PyTorch 2.5.0+cu121, PyG 2.7.0, torch-sparse 0.6.18,
torch-scatter 2.1.2 and OGB 1.3.6. `requirements-cpu.txt` covers numerical CPU
fixtures and Friendster preparation. Citation2 preparation additionally needs
a compatible PyTorch installation and OGB 1.3.6 with its dependencies, even
when no model is trained. `requirements-quality.txt` adds the quality dependencies.
They are dependency
recipes, not a previously verified container-image digest.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
```

For training, install the matching public PyTorch/PyG binary wheels, then the
quality requirements. CUDA 12.1 and a compatible driver are needed for this
recorded build. A CPU-only PyTorch build can be used for fixtures.

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
python -m pip install -r requirements-quality.txt
```

All commands below run from `src/`. Set `DATA` and `RESULTS` to absolute paths
on a large local scratch filesystem. Commands create fresh output directories;
retain failed attempts and use a new name for another attempt.

## Small checks

```bash
cd src
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest artifact_public/test_public_inputs.py scripts/seal_revision/test_exact_benchmark.py -q
python -m scripts.seal_revision.exact_benchmark --fixture --limit 8 --caps 50,200 --output "$RESULTS/exact_fixture.json"
```

The fixtures exercise the actual zlib binary parser, self-loop removal,
duplicate handling, sorted symmetric CSR, little-endian layout, input integrity,
safe bundle selection and exact local-edge/DRNL construction. They do not
establish full-scale runtime, peak memory or model quality.
Run the CPU geometry/dispatch tests in a fresh process separately from tests
that import Torch; the CPU tooling deliberately rejects a pre-imported Torch.

## Citation2: official source and exact prepared identity

Before transferring or extracting the large ZIP, install a compatible PyTorch
build and OGB 1.3.6 in the preparation environment. Verify the actual OGB loader
import; the CPU requirements alone do not provide it. Keep dependency installation
inside a separate environment. For an existing compatible PyTorch environment:

```bash
python -m pip install ogb==1.3.6
python -c "import importlib.metadata as m; import torch; from ogb.linkproppred import LinkPropPredDataset; assert m.version('ogb') == '1.3.6'; print(torch.__version__, 'LinkPropPredDataset ready')"
```

The loader must read citation2 release 1 from its installed
`ogb/linkproppred/master.csv`. A CPU preparation environment need not install
PyG, torch-sparse or torch-scatter. Keep the recorded model environment separate;
the final 19-payload hash check decides prepared-data identity.

The official OGB 1.3.6 metadata identifies `ogbl-citation2`, release 1,
`citation-v2.zip`. The public HTTPS URL currently serves 2,293,454,318 bytes.
The archive recorded in the original experiments is 3,754,138,412 bytes and has
a different archive identity. We therefore compare all **19 numeric prepared
payload SHA256 hashes**, not ZIP compression or generated timestamps. The
original SHA256 is an expectation about the recorded archive, not the official
download's checksum. OGB does not publish a SHA256 in the inspected metadata.

The single controller below checks four unchanged preparation-source hashes,
requires 64 GiB of free disk and 32 GiB of available RAM, downloads only the
official source, invokes the original offline preparer and verifies the 19
payloads. It creates a unique subdirectory, preserves logs and failures, and
never starts training, uses a GPU, or accesses YT. The initial command only
prints the plan; append `--execute` to perform the reproduction on a large host.

```bash
python -m artifact_public.reproduce_citation2 --work-parent "$DATA/public_reproduction" --threads 8
```

The successful output is `dataset/prepared` beneath the unique run directory,
plus `PUBLIC_19_PAYLOAD_IDENTITY.json`. The completed 2026-11-25 run matched
all 19 payload hashes to the recorded experiment. The fresh official ZIP has
SHA256 `f8502b935ea8f0a7bea5eb130991ca337cd26cad7f231cc9f5546fe076f4be53`.
This is a locally measured full-download checksum; OGB did not publish it.
See `metadata/CITATION2_PUBLIC_REPRODUCTION.json` and the unmodified receipts
under `metadata/evidence/citation2_public_v2/`. All 94 archived report files
(93 hashed payloads plus their manifest) passed local integrity verification.
The historical expectation JSON remains immutable; its old false verification
flag describes the earlier staging state, superseded by the proof sidecar.
The successful CPU preparation used Python 3.10.12, Torch 2.2.2+cu121,
NumPy 1.24.4, SciPy 1.10.1, Numba 0.58.0 and OGB 1.3.6 in an isolated
environment with base packages reused unchanged. Its environment summary
is included; this is separate from the recorded GPU quality environment. The
preparer keeps `prepared_original`, removes the 129 reverse held-out collisions
(258 symmetric entries), then checks both held-out directions. Both variants
and their audits are retained. Use the strict-clean `prepared` for quality.

On a host without Internet, first stream a fresh official ZIP to that host
without storing a graph on the laptop. Use `--archive-existing /remote/file.zip`
and `--official-download-receipt /remote/receipt.json` together. That mode
performs no network requests or source copying. The receipt must contain the
exact official `url`, `bytes=2293454318`, the actual streamed `sha256`, and
`source_kind="fresh_official_public_download"`. The controller recomputes the
remote archive SHA256 and rejects the old recorded archive identity.

The acquisition and verification stages can also be used separately:

```bash
python -m artifact_public.download citation2 --out "$DATA/citation2_input"
python -m artifact_public.download citation2 --out "$DATA/citation2_input" --execute
python experiment_configs/seal_revision_2026-11/prepare_citation2.py --zip "$DATA/citation2_input/citation-v2.zip" --root "$DATA/citation2"
python -m artifact_public.verify_payloads --root "$DATA/citation2/prepared" --expected ../metadata/citation2_prepared_expected.json --receipt "$DATA/citation2/public_payload_identity.json"
```

## Friendster: exact dense public release

**Verification status:** full public source reconstruction remains pending.
The bundled evaluation arrays and small-fixture checks do not establish
full-CSR or complete prepared-dataset identity.

Use Per Fuchs's **Com-Friendster Dataset Densely Relabelled**, Zenodo record
5146230, DOI [10.5281/zenodo.5146230](https://doi.org/10.5281/zenodo.5146230),
CC-BY-4.0. This release maps sorted original vertex IDs to their ranks
(`stable-map=false`), with 65,608,366 vertices and 1,806,067,135 undirected edges.
Do not substitute the ordinary SNAP text file with an arbitrary ID mapping.

The downloader verifies the provider's file sizes and MD5 checksums and records
SHA256. Inputs total 6,672,243,067 bytes. The converter retains the exact
little-endian binary interpretation and builder. Its public wrapper requires
56 GiB of available RAM, a budget of at least 56 GiB, and 32 GiB of free disk
for full CSR, including zero timestamps. These are conservative preflight
bounds, not a measured peak. A host with at least 128 GiB RAM and 100 GiB scratch
is recommended for the whole preparation; do not run it on a small laptop.

```bash
python -m artifact_public.download friendster --out "$DATA/friendster_inputs" --execute
python -m artifact_public.prepare_friendster_full --inputs "$DATA/friendster_inputs" --out "$DATA/friendster/full_csr" --workers 8 --memory-budget-gib 64
python -m artifact_public.prepare_friendster_full --inputs "$DATA/friendster_inputs" --out "$DATA/friendster/full_csr" --workers 8 --memory-budget-gib 64 --execute
NUMBA_NUM_THREADS=8 python -m artifact_public.prepare_friendster --full-graph "$DATA/friendster/full_csr" --out "$DATA/friendster/prepared" --threads 8 --memory-budget-gib 64
```

The full numeric CSR must match `metadata/friendster_full_expected.json` before
split preparation. Generated `meta.json` paths/times are excluded from byte
identity. CSR layout: `edge_starts.bin` is little-endian int64[N], without a
terminal sentinel; `edge_ends.bin` has an int64 count followed by int32 entries.
The zero timestamp array is irrelevant to these topology workloads.

Split preparation retains the exact seed 20261102 and archived 80/10/10 hash
thresholds, checks partition counts, excludes all held-out edges in both
directions, validates all evaluation negatives against the full graph and
regenerates the historical counter RNG for equality. The bundled eight small
public-data-derived evaluation arrays are byte-identical to the archived
reference, with 10,000 rows and 1,000 negatives per split. They have five repeated
physical source IDs per split. **Keep original query rows; do not merge by source.**
Before split preparation starts it requires 64 GiB of available RAM and 24 GiB
of free disk. The RAM budget is a conservative admission allowance, not a
process memory cap. All 23 prepared numeric payloads must also match the
archived original manifest before the final output directory is published.
The new public adapter verifies numeric payload hashes without claiming that
public downloads have private storage revisions. Training/sampling/metric
algorithms are unchanged. The two provenance-guard adaptations are recorded.

## Quality protocols

Citation2: 3 paired seeds (42,43,44), the same 2% training positives, 10 epochs,
2 hops/cap200, SortPooling k281, 512 validation and 4096 test source groups;
all 1000 official negatives. Friendster: the same three seeds, 20,000 steps,
batch128, k370, 512 validation and 4096 test **rows**. Both use fixed extraction
and subset seeds. Checkpoint selection uses validation only. The `freeze`
phase requires all six fits before any test scoring. Test data must not be used
for retuning. CUDA training is seeded but not promised bitwise deterministic.

```bash
python experiment_configs/seal_revision_2026-11/quality/runner.py --phase smoke --prepared-root "$DATA/citation2/prepared" --output-dir "$RESULTS/citation2_smoke" --gpu 0
python experiment_configs/seal_revision_2026-11/quality/runner.py --phase all --prepared-root "$DATA/citation2/prepared" --dataset-manifest "$DATA/citation2/prepared/manifest.json" --output-dir "$RESULTS/citation2_quality" --gpu 0
python experiment_configs/seal_revision_2026-11/quality_friendster/runner.py --phase smoke --prepared-root "$DATA/friendster/prepared" --dataset-manifest "$DATA/friendster/prepared/manifest.json" --output-dir "$RESULTS/friendster_smoke" --gpu 0
python experiment_configs/seal_revision_2026-11/quality_friendster/runner.py --phase all --prepared-root "$DATA/friendster/prepared" --dataset-manifest "$DATA/friendster/prepared/manifest.json" --output-dir "$RESULTS/friendster_quality" --gpu 0
```

See each runner's README and protocol.yaml for all settings. Use separate
output directories, run smoke first and inspect it. Expected durations and
end-to-end GPU memory for a fresh public installation have not been measured.

## Exact construction and memory

These timers measure local construction on frozen sampled sets. They are not
end-to-end SEAL or time-to-quality measurements. Report int32/int64 temporary
width separately. The native global-universe fixture can be run with both
SciPy 1.10.1 and 1.15.3; the larger universe allocates stock N-sized scratch.

```bash
python -m scripts.seal_revision.exact_benchmark --graph-path "$DATA/friendster/full_csr" --limit 128 --hops 2 --caps 50,200 --samplers reference_union,production_union,per_vertex --pair-sampling mixed_twohop --stock-index-dtype int32 --max-stock-workspace-mib 2048 --max-working-set-mib 4096 --output "$RESULTS/friendster_exact_int32.json"
python -m scripts.seal_revision.exact_benchmark --fixture --limit 8 --caps 50 --hops 2 --samplers reference_union --virtual-num-nodes 3000000,30000000,65000000,180000000 --repeats 5 --warmup 2 --seed 42 --stock-index-dtype int32 --max-stock-workspace-mib 2048 --max-working-set-mib 4096 --output "$RESULTS/native_int32_control.json"
python -m scripts.seal_revision.memory_benchmark_v3 --graph-path "$DATA/friendster/full_csr" --workers 1 4 16 --memory-budget-gib 120 --out "$RESULTS/friendster_memory_v3.json"
```

The V3 memory harness is the final implementation: Linux fork/read-only
consumers, process-tree PSS, stable complete scans, and guarded conversion.
The earlier memory harness is retained as historical source only. RSS sums are
not unique physical memory. A preflight skip is an unmeasured outcome.
Dispatch and degree-geometry tools are also included; follow their explicit
CLI plans. Ongoing native upstream-comparator and PENCIL adapters are separate
staged additions and are not claimed complete by this package version.

## Data access and publication

See `DATA_ACCESS.md` for the public/restricted boundary and attribution.
VK raw graph, raw node/query IDs, evaluation arrays, checkpoints and tokens
are excluded. Industrial results require an aggregate-only schema and a
restriction statement; no public access link is invented. See
`metadata/PUBLICATION_STATUS.json` for work still required before release.
