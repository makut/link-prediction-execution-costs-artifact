# Link prediction execution costs: artifact v0.5.0

Reproducibility artifact for **Execution Costs and Sampling Semantics in SEAL-Style Link Prediction**.

Maksim Utushkin, Renat Aminov, and Alexander D’yakonov — AI VK, Moscow, Russia. Corresponding author: Maksim Utushkin.

## Start here

- [Complete artifact and instructions](artifact/README.md)
- [Commands for all 13 paper tables and figures](artifact/PUBLIC_OBJECT_COMMANDS.json)
- [Public preparation, verification and fresh execution recipes](artifact/PUBLIC_COMMANDS.txt)
- [Companion asset catalog](artifact/metadata/EXTERNAL_ASSETS.json)
- [All seven release assets](ASSET_CATALOG.json) and [SHA-256 checksums](SHA256SUMS)
- [Independent package verification](verification/final_package_peer.json)

Set `ARTIFACT` to the absolute path of the `artifact/` directory after cloning. The same source tree is distributed as `lpp_vldb_artifact_v0.5.0.tgz`. Obtain the six companion archives listed in the catalog from this repository’s Releases page and verify their checksums. Full graph datasets are acquired and prepared using the public recipes.

The frozen v0.5.0 payload retains the earlier working title “Execution Costs and Sampling Semantics in Subgraph-Based Link Prediction” in archival metadata; it supports the same study.

## Contents and validation

The package retains the instrumented PENCIL diagnostic (E19, 48 jobs), the matched Friendster CPU study (E20, seven arms), and the clean PENCIL consumer control (E21, 44 jobs and 32 measured settings). Their raw reports, original provenance and independent verification commands are separate. All eight PENCIL arms and all four batch/worker settings are retained.

The quality asset contains twelve frozen checkpoints and complete candidate scores for public Friendster and strict-clean ogbl-citation2. Independent offline checks recompute ranks and metrics from all twelve score files. These checks stream archive contents without loading models or full graphs.

Recorded-result verification and fresh execution have separate commands. Compute recipes use documented Linux Python 3.10 stacks. The preserved E19 source audit uses a separate lightweight Python 3.12 analysis environment. Linux native/model fixture checks establish the stated small portability gates; they are distinct from fresh full experiment runs.

The PENCIL execution control uses random frozen parameters. Its timing results do not measure trained-model ranking quality. Query timings, setup costs, representation sizes and measured process memory have separate meanings, as documented with each experiment.

## License and attribution

Own project code is [MIT licensed](LICENSE). Original third-party code, wheel notices and dataset attribution are preserved in the artifact. Refer to the associated source and data notices for their respective terms.

The associated manuscript is a submission manuscript. This artifact does not imply acceptance by a venue.
