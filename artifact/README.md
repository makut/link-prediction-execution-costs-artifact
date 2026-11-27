# Reproducing the SEAL-style execution study

The v0.8.0 payload adds 36 matched application lifetimes on Friendster and strict-clean Citation2 (E26): three fresh allocations per graph, eight workers, compiled/native-fresh/native-reuse paths and two terminated horizons. All six raw audits and two separate aggregates passed real offline byte-identical replay on Python 3.12.10 and 3.13.12. [E26 instructions](extensions/e26/README.txt) explain the complete matrix, timing boundaries, sampled-memory coverage and the site-specific adapter needed for fresh GPU execution. E25's earlier 24 Friendster W1/W8 applications remain separate. Prior releases and the six immutable companion assets retain their bytes and URLs; code-archive identities are external.

This artifact accompanies **Execution Costs and Sampling Semantics in SEAL-Style Link Prediction**. It separates exact local induction, representation memory/sharing, consumer execution and sampling-law sensitivity.

The v0.6.0 payload includes verified E22 Citation2 CPU evidence, actual E23 repeated pipeline windows and the E24 descriptive matched-cohort reanalysis. Source-only checks remain separate modes. The six published v0.5.0 companion assets retain their exact bytes and URLs.

## Choose a workflow

| Goal | What it establishes | Resources |
|---|---|---|
| Recorded verification | Integrity and arithmetic of saved observations | Selected evidence; no full graph/GPU |
| E23 source check | Runner/source/workload identities | Python 3.10+ standard library |
| Public preparation | Declared numeric input identities | Provider downloads and suitable Linux RAM/disk |
| Fresh replay | New observations on your machine | Prepared inputs and the experiment's CPU/GPU stack |

Set absolute `ARTIFACT` (the `artifact/` directory in a clone, or the unpacked code-archive root), `ASSETS`, `DATA` and `RESULTS` paths. Create result directories outside source/input trees and use new receipt names. [PUBLIC_COMMANDS.txt](PUBLIC_COMMANDS.txt) supplies base recipes; extensions cover additions.

## Verify recorded results

E22 needs only Python 3.10+ standard library and its bundled evidence:

```sh
python3 -B "$ARTIFACT/extensions/e22/verify_e22.py" \
  --output "$RESULTS/E22_OFFLINE_VERIFICATION.json"
```

It streams Citation2 evidence, checks source/input/exit bindings and recomputes seven timing rows, without graph/model/native-library loads or external access.

For base results, select the needed archives from the [companion asset identities](metadata/COMPANION_ASSETS_V0_5.json), which lists all six immutable v0.5.0 URLs and complete SHA256 values. Check each downloaded file before use; for example:

```sh
printf '%s  %s\n' \
  'e17cdc9678d00a3f372fc8083741a200a63662226f730f47eeda79f89b4ad3b2' \
  "$ASSETS/lpp_public_quality_evidence_v1.tgz" | shasum -a256 -c -
```

Linux `sha256sum -c -` is an alternative. [PUBLIC_OBJECT_COMMANDS.json](PUBLIC_OBJECT_COMMANDS.json) maps the paper tables/figures to inputs and aggregation; [PUBLIC_COMMANDS.txt](PUBLIC_COMMANDS.txt) supplies their recorded systems, quality, CPU and PENCIL checks. Quality checks stream scores without loading models; recipes identify separate Python 3.12/NumPy analysis requirements.

### E26 prepared reuse and two-graph application lifetimes

Verify the complete code archive against its external release SHA256 first. Then,
with a new output directory outside the artifact and no full graph or GPU:

```sh
python3 -B -S "$ARTIFACT/extensions/e26/offline_verify_closure.py" --index "$ARTIFACT/extensions/e26/evidence_index.json" --index-sha256 "9218f6fcee5cf59c1c5a492c2d7d655a4e6fc35b315916bb451929b32c0ec903" --out "$RESULTS/e26_recorded_verified"
```

The helper independently reconstructs six block audits and two separate 18-app
aggregates. The actual recorded and fresh files must match byte for byte.
[The extension](extensions/e26/README.txt) documents all source/input pins,
separate one-time CPU staging, the timing figure and exact new-run interfaces.
Read-only representation reuse changes the observed winner in all 12 matched
comparisons; this is conditional on the fixed reference-union DGCNN workload,
retained caches and included observer costs. Application PSS misses phases and
supports no exact peak or reused-native memory ranking. Old E25 is not pooled.

### Historical E25 recorded application lifetimes

The actual catalog and all three block inputs are bundled. After checking the
complete code archive against its external release SHA256, run the existing
standard-library verifier with a fresh directory outside the artifact:

```sh
python3 -B -S "$ARTIFACT/extensions/e25/verify_recorded_v6.py" \
  --artifact "$ARTIFACT" --out "$RESULTS/e25_recorded_verified"
```

It regenerates all three audits and the aggregate and requires their exact
cataloged whole identities. The genuine Python 3.12.10 public-layout roundtrip passed with all three audits
and the aggregate byte-identical; its helper/root receipts are included.
The actual aggregate SHA256 is
`2a9bdfbbe4661bd41e1e5d1c1ade8046e40bcdd538bca44b5db1c2caf042e057`.
Rendering is separate; `E25_PLOT_PY` denotes Python 3.12.10 with Matplotlib 3.10.3:

```sh
"$E25_PLOT_PY" -B "$ARTIFACT/extensions/e25/figure/plot_aggregate_v6.py" \
  --aggregate "$ARTIFACT/extensions/e25/recorded/ACTUAL_AGGREGATE.json" \
  --aggregate-sha256 "2a9bdfbbe4661bd41e1e5d1c1ade8046e40bcdd538bca44b5db1c2caf042e057" \
  --out "$RESULTS/e25_figure"
```

The separate short/long applications and long synchronized checkpoints retain
different timing boundaries; there is no long timed checkpoint at 576. Min/max
across allocations is descriptive, with within-block paired ratios and retained
OS/PVC caches. Sampled all-phase PSS is not full-resident memory. The observed
27-logical-CPU pool spans 14 physical-core identities; CPU marketing SKU was not
recorded. [E25 details](extensions/e25/README.md) state the instrumentation,
file-backed native-promotion cost and honest external-replay limits.

### Matched cost regimes (E24)

This posthoc analysis uses the existing E20 Friendster archive (2.1 MB) and included E22 Citation2 archive (0.57 MB). It adds no experiment. Recompute with Python 3.10+ standard library:

```sh
python3 -B "$ARTIFACT/extensions/e24/publicanalyze_cost_regimes.py" \
  --friendster-archive "$ASSETS/lpp_public_friendster_cpu_matched_v2_raw.tgz" \
  --citation2-archive "$ARTIFACT/extensions/e22/recorded/citation2_cpu_matched_v1_20261126.tgz" \
  --output "$RESULTS/matched_cost_regimes.json"
```

Plot that output; the checked rendering uses Python 3.12.10 and Matplotlib 3.10.3:

```sh
python3 -B "$ARTIFACT/extensions/e24/plot_matched_cost_regimes.py" \
  --input "$RESULTS/matched_cost_regimes.json" \
  --out "$RESULTS/matched_cost_regimes_figure"
```

All 256 cases retain separate cohorts and the frozen unweighted proxy; no fitted threshold, CI or routing-cost measurement is added. [E24 instructions](extensions/e24/README.md) describe identities and numeric peer checks.

### E23 source-only check

```sh
python3 -B "$ARTIFACT/extensions/e23/verify_e23.py" \
  --output "$RESULTS/E23_SOURCE_ONLY.json"
```

Without an archive, this checks the packet, scientific pins, small workloads and declared job order. It produces **SOURCE_ONLY**, not an actual-result PASS.

### E23 recorded-result verification

The actual raw archive and separate final outer receipts are included. Run with Python 3.10+ standard library; the exact identities are recorded in `extensions/e23/recorded/RECORDED_RESULT.json`:

```sh
python3 -B "$ARTIFACT/extensions/e23/verify_e23.py" --archive "$ARTIFACT/extensions/e23/recorded/pipeline_repeats_e23_20261126_final_v1.tgz" --archive-sha256 "41def9a2735829c741fc9ddc490bda56dcd0780bf37cd9759bd0104e42f05073" --outer-source-sha256 "e8df35bb6ebacf5bc9c9f5b4ea4c2337e6fabb93b5e10fb3c2818c03eb025f5e" --outer-complete "$ARTIFACT/extensions/e23/recorded/outer_COMPLETE.json" --archive-pointer "$ARTIFACT/extensions/e23/recorded/archive_pointer.json" --output "$RESULTS/e23_recorded_verified.json"
```

This independently verifies all 40 declared gates, 36 real timing jobs, 12 rows of three windows, 12 groups of ratios paired by repetition block, and 36 comparisons of the 32-query excluded warmup prefix. Actual child wait/quiescence and post-exit flock receipts are checked. The included final outer and persistence receipts are separate from the raw archive. All three values in every row are retained. A separate Decimal numerical peer and its stdlib source are included. These audits do not independently regenerate every measured graph or execute live network/process observations.

## Prepare public data

Providers are [Friendster dense, Zenodo 5146230](https://zenodo.org/records/5146230) and [OGB Citation2](https://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip). Graphs are not bundled. Use [PUBLIC_COMMANDS.txt](PUBLIC_COMMANDS.txt) for acquisition, dependencies, preparation and numeric identity checks.

Citation2's `reproduce_citation2` command prints a plan by default; `--execute` downloads, prepares and verifies all nineteen numeric payloads. Use its unique run's `dataset/prepared` root. `prepared_original` is a different graph. Friendster acquisition/full-CSR commands have plan/execute modes; its legacy held-out-clean builder executes immediately, as marked in the recipe.

| Replay | Prepared inputs |
|---|---|
| Friendster CPU (E20) | Full Friendster node count and two CSR arrays |
| Citation2 CPU (E22) | Strict-clean Citation2 `train_csr` arrays |
| Pipeline repeats (E23) | Full Friendster and strict-clean Citation2 prepared root; eight files pinned |
| Application lifetimes (E25) | Full public Friendster compact CSR; no quality split |
| Clean PENCIL (E21) | Strict-clean Citation2 and declared validation-only subset |

## Run a fresh replay

Use fresh outputs and one shared `TIMING_COORDINATION` directory for cooperating timing workloads. Plans precede `--execute`; failures/partial outputs are retained.

| Experiment | Stack | Instructions |
|---|---|---|
| E20 Friendster / E22 Citation2 CPU | Linux x86_64 Python 3.10, Torch 2.2.2+cu121, exact native wheel packet, C++17 | [E20](reproduction/public_friendster_cpu_v1/README.txt), [E22](extensions/e22/README.md) |
| E21 clean PENCIL | Python 3.10, Torch 2.5.0+cu121, PyG 2.7.0, declared extras and GPU admission | [Consumer replay](reproduction/public_pencil_clean_v1/README.txt) |
| E23 pipeline repeats | Original Torch 2.5/PyG stack and one idle whole H100 | [Pipeline replay](extensions/e23/README.md) |
| E25 application lifetimes | Frozen author-bound deployment; another operator needs a reviewed actual-platform binding | [Recorded study and replay limits](extensions/e25/README.md) |

E22's practical plan command is:

```sh
"$CPU_PY" "$ARTIFACT/extensions/e22/fresh/run_public_cpu.py" \
  --python "$CPU_PY" --train-csr "$C2_TRAIN_CSR" \
  --wheel-packet "$ASSETS/pt22_native_wheels_v1.tgz" --out "$E22_OUT" \
  --coordination-dir "$TIMING_COORDINATION" \
  --source-manifest-sha256 "7b406e3e3909f64ebcdd2a3b1cb8b9dd2763b6c44ebc8e791b21d1699e62b2bb"
```

Add `--execute` for new timing measurements. It uses an isolated environment for the exact CPU wheels; the base stays unchanged. CPU Torch 2.2 and model Torch 2.5 stacks remain distinct.

E23 records three fresh benchmark/training-process repetitions of stock/local/compiled induction on each public graph with one/eight loader workers. The runtime/workload/model seed stays fixed; shared OS/Numba caches are retained. Each window measures 3,200 graphs after 256 excluded warmup graphs. Report descriptive process/window variation, not independent data/quality-seed uncertainty or a population confidence interval.

## Interpretation and terms

Operator time, setup/conversion, private workspace, process PSS, consumer sweeps and sampled MRR keep separate protocols. The PENCIL model is random and frozen; warmup-prefix equality does not verify every measured graph. These measurements do not establish time-to-quality or universal predictor superiority.

The six companions retain their bytes and v0.5.0 URLs in [companion asset identities](metadata/COMPANION_ASSETS_V0_5.json). Historical lineage remains in `metadata/lineage/`; E22/E23/E24/E25 retain separate catalogs. The code archive whole SHA is supplied by the external release checksum/catalog; it is not self-embedded here. Own code is [MIT licensed](LICENSE); third-party/data terms remain in their notices. [CITATION.cff](CITATION.cff) identifies authors, without implying conference acceptance.
