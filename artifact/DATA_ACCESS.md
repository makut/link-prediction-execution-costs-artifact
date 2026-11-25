# Data provenance, attribution and access

| Material | Access / license | Included |
|---|---|---|
| Friendster dense source | Per Fuchs, Com-Friendster Dataset Densely Relabelled (2021), DOI 10.5281/zenodo.5146230, CC-BY-4.0 | URLs/checksums; no full graph |
| Friendster evaluation | Derived by the study's authors from the same public dense graph; original query rows, positives, negatives and warm masks | Eight .npy arrays in the public bundle, approximately 80 MB uncompressed |
| Citation2 | Official OGB ogbl-citation2 release1, ODC-BY | Public URL and numeric payload expectations; no graph or features |
| VK graph and queries | Restricted industrial data; release requires the data owner's authorization | No raw graph, node/query IDs, evaluation arrays or checkpoints |
| Project code | MIT, copyright 2026 LPP contributors; user-approved on 2026-11-25 | Selected own code; upstream notices retained; local staging only |

Friendster source: https://zenodo.org/records/5146230 . The dense release was
prepared by sorting original Graphalytics/SNAP vertex IDs and storing their
ranks with vtxremap. Retain this source/creator/DOI attribution when reusing the
derived arrays. The package changes the preparation interface and packaging;
the bundled evaluation-array bytes are unchanged from the study reference.
CC-BY-4.0: https://creativecommons.org/licenses/by/4.0/ . Original network:
J. Yang and J. Leskovec, *Defining and Evaluating Network Communities based on
Ground-truth*, ICDM 2012; https://snap.stanford.edu/data/com-Friendster.html .

Citation2 source documentation: https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2 .
Dataset license ODC-BY: https://opendatacommons.org/licenses/by/1-0/ . OGB code
license MIT is a separate license and does not replace the dataset license.

Citation2 official-source reconstruction was verified on 2026-11-25: all 19
prepared numeric payloads match the original experiment. The proof and
unmodified receipts are linked from `metadata/CITATION2_PUBLIC_REPRODUCTION.json`.
Friendster public reconstruction was also verified on 2026-11-25: all3 full
numeric files and23 prepared split/query payloads match the study inputs.
See `metadata/FRIENDSTER_PUBLIC_REPRODUCTION.json`; historical expectation-file
status flags are preserved and superseded by that dated proof. Generated times, absolute paths and
compressed archive containers are not byte-identical targets. Neither input
identity receipt alone reproduces quality scores or timing measurements.

Industrial aggregate results should report schema fields such as dataset
version, graph dimensions, workload stratum, method, index width, count,
timing summary, PSS metric, guard status and source-version hash. No private
per-query keys are needed for this schema. This package does not claim that
private industrial results can be independently regenerated from public data.

The v0.4 candidate also contains the unchanged public Friendster fixed-S report
and raw E16 query/timing records, derived from the same CC-BY-4.0 public graph.
Its E17 evidence is filtered to Friendster. Private VK replay plans, raw
query IDs and fixed-S reports are excluded from this candidate.
