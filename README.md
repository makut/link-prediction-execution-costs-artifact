# SEAL-style link prediction: artifact v0.6.0

Research artifact for **Execution Costs and Sampling Semantics in SEAL-Style Link Prediction**.

Maksim Utushkin, Renat Aminov and Alexander D’yakonov — AI VK, Moscow, Russia. Corresponding author: Maksim Utushkin.

This study separates the cost of executing a fixed local-graph query from the effect of changing its sampler. It examines global versus local lookup work, state lifetime, representation memory, and sampled-ranking sensitivity on public Friendster and strict-clean Citation2. The PENCIL bridge checks a richer graph-input contract with a frozen random model.

This v0.6.0 revision includes independently verified Citation2 matched CPU results (E22), three fresh-process repetitions of the public pipeline control (E23), and a descriptive two-graph reanalysis of existing matched timings (E24). Recorded results, source qualification and fresh replay have separate commands. The published v0.5.0 tag and its six companion assets remain unchanged.

## Download the code

The [v0.6.0 source archive](https://github.com/makut/link-prediction-execution-costs-artifact/releases/download/v0.6.0/lpp_vldb_artifact_v0.6.0.tgz) is 85,260,158 bytes; SHA256:

`c9e26c90cffde59c34b1bb9c665e1f7ee3fb2904a0fd5aa3374b340c6bc087a2`

It contains the `artifact/` tree. To unpack it into a new directory, remove that one archive prefix:

```sh
mkdir seal-artifact-v0.6
# After downloading and checking the full SHA256 above:
tar -xzf lpp_vldb_artifact_v0.6.0.tgz --strip-components=1 -C seal-artifact-v0.6
export ARTIFACT="$PWD/seal-artifact-v0.6"
```

The source archive includes E22/E23 evidence. The six optional companions below retain their v0.5.0 URLs.

## Start here

Choose the operation you need:

| Goal | Entry point | Compute |
|---|---|---|
| Check recorded results | [Offline verification](artifact/README.md#verify-recorded-results) | No full graphs or GPU |
| Acquire public graph inputs | [Public data](artifact/README.md#prepare-public-data) | Linux host with the stated disk/RAM |
| Produce new measurements | [Fresh replay](artifact/README.md#run-a-fresh-replay) | CPU stack or one whole H100, by experiment |
| Inspect paper-object commands | [Paper object commands](artifact/PUBLIC_OBJECT_COMMANDS.json), [E22](artifact/extensions/e22/README.md), [E24 analysis/plot](artifact/README.md#matched-cost-regimes-e24), [E23 recorded results and replay](artifact/extensions/e23/README.md) | Depends on the selected operation |

Set `ARTIFACT` to the absolute path of `artifact/` after cloning. Recorded verification recomputes saved observations. A fresh replay generates new observations on your machine; matching input hashes does not promise identical latency.

## Companion assets

These six archives are reused from their immutable **v0.5.0 download URLs**. They are not uploaded again for this revision. Download only those needed for the operation you choose and check the complete SHA256 before use. Sizes and identities are also recorded in [ASSET_CATALOG.json](ASSET_CATALOG.json). Full graph datasets are acquired separately from their public providers.

| Asset | Purpose | SHA256 |
|---|---|---|
| [lpp_public_quality_evidence_v1.tgz](https://github.com/makut/link-prediction-execution-costs-artifact/releases/download/v0.5.0/lpp_public_quality_evidence_v1.tgz) | Frozen quality checkpoints and candidate scores | `e17cdc9678d00a3f372fc8083741a200a63662226f730f47eeda79f89b4ad3b2` |
| [lpp_public_systems_evidence_v1.tgz](https://github.com/makut/link-prediction-execution-costs-artifact/releases/download/v0.5.0/lpp_public_systems_evidence_v1.tgz) | Physical execution, memory and sampling evidence | `48232549089f3295c893bb33f6cea0d1aac4a0ee45b91a2669392c90ae1bfe40` |
| [lpp_public_friendster_cpu_matched_v2_raw.tgz](https://github.com/makut/link-prediction-execution-costs-artifact/releases/download/v0.5.0/lpp_public_friendster_cpu_matched_v2_raw.tgz) | Matched Friendster CPU results (E20) | `546cad6f6b14c9f56844dc3414804e97f0dd303028532910dd74bee9d10f9374` |
| [pt22_native_wheels_v1.tgz](https://github.com/makut/link-prediction-execution-costs-artifact/releases/download/v0.5.0/pt22_native_wheels_v1.tgz) | Exact Torch 2.2 native CPU wheel packet | `5e9edbdc067325cd0b41b5fb292e2cc7343cc535da87ff3bef720cbf85da38cf` |
| [lpp_public_pencil_instrumented48_raw.tgz](https://github.com/makut/link-prediction-execution-costs-artifact/releases/download/v0.5.0/lpp_public_pencil_instrumented48_raw.tgz) | Instrumented consumer diagnostic (E19) | `7ab909e987af900f8ad5f9ea9a559e17e650f78891cb0bc99200a0fdb97fa36c` |
| [lpp_public_pencil_clean44_raw.tgz](https://github.com/makut/link-prediction-execution-costs-artifact/releases/download/v0.5.0/lpp_public_pencil_clean44_raw.tgz) | Clean PENCIL consumer results (E21) | `708b6f32081d187dbe5e092b6782a36002333c3f8b0c5ff7dcbf20ec48c156b0` |

The E22 and E23 raw archives are included under their `artifact/extensions/*/recorded/` directories. E23 also includes its independently checked final outer-completion and persistence receipts, generated after the raw archive. E24 reuses the included E22 archive and the existing 2.1 MB E20 companion; its analysis and plot commands need no full graphs. Set `ARTIFACT` to `artifact/` in a clone or to the root of an unpacked code archive. Whole code-archive checksums are supplied externally by the release catalog.

## Interpretation and license

Fixed-set edge/label equality, short training windows, inference sweeps, representation memory and ranking quality answer different questions. The study does not combine them into measured time-to-quality or a universal predictor ranking. Source-only checks and tiny fixtures establish their stated source/portability gates.

Own project code uses the [MIT license](LICENSE). Third-party code, wheel and public-data notices retain their respective terms. Use [CITATION.cff](CITATION.cff) for attribution; this artifact does not imply conference acceptance.
