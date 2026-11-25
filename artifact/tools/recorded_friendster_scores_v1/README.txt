Independent recorded Friendster float-score reranking v1

This optional OFFLINE archival command checks the six complete ORIGINAL
Friendster test_scores.npz files in the pinned public quality asset. It performs
no training, prediction, graph or model loading, checkpoint deserialization,
package installation, download or remote operation. It is separate from the
existing saved-RR reaggregation recipe; neither frozen core nor that recipe is
changed. Python with NumPy is sufficient (validated on local Python3.12).

Generic command, fresh output JSON:
  "$AUDIT_PY312" "$ARTIFACT/tools/recorded_friendster_scores_v1/verify_friendster_scores.py" \
    --quality-asset "$ASSETS/lpp_public_quality_evidence_v1.tgz" \
    --manifest "$ARTIFACT/work/vldb_public_fresh_adapters_v1/QUALITY_MANIFEST.json" \
    --output "$RESULTS/friendster_full_score_rerank.json"
The archive must be404,147,012bytes, SHA256
e17cdc9678d00a3f372fc8083741a200a63662226f730f47eeda79f89b4ad3b2;
manifest SHA2569bf816acdce59b0905998adc47437092427cfa8d4c7154da69e5343a8eb15f7d.
Source inputs are unmodified. No NPZ or model is copied to disk. Output is an
exclusive new local receipt. The archive is streamed and whole-hash checked twice.

Each original NPZ is86,102,998bytes, containing four stored flat NPY arrays:
  src<int64>, dst<int64>, label<int8>, logit<float32>, each4,100,096 entries.
IMPORTANT: src here encodes ORIGINAL QUERY ROW ID, not the physical graph source
vertex. There are4,096 distinct queries and4,095 physical source vertices.
Collapsing by physical source or assuming a positive-first order is incorrect.

The checker identifies the positive from its stored label and independently
checks the declared candidate permutation: NumPy PCG64 default_rng with
SeedSequence([20261127, original_query_id]), permutation of1,001 candidate slots.
Observed positive positions span0..1000, with984 distinct positions in this
cohort. The seed/order rule is reconstructed from the frozen protocol; no
producer ranking or permutation function is imported.

Per query, rank=1+count(negative_score>positive_score)
                    +0.5*count(negative_score==positive_score).
All comparisons preserve stored float32 values. Every resulting float64 RR is
checked bitwise against the original saved4,096-vector and its whole NPY SHA.
All midpoint MRR and Hits@1/10/50/100 are checked against the original ogb_*
implementation aliases. The legacy mrr field uses different tie handling and is
explicitly excluded. This is custom sampled ranking, not an official OGB result.

The checker also binds candidate content/provenance:
- all original run/subset query IDs, selected positives and44cold flags;
- the exact sorted4,096-of10,000 query-row subset, seed20261125;
- inverse candidate permutation for each query, comparing positive targets and
  reconstructing ORIGINAL ordered negative(src,dst) pairs in64query blocks;
- streamed whole SHA of the original65,536,128-byte negative NPY, even though
  that redundant NPY is not present in this quality archive:
  d9a7d3dae79d2bf9d15c9fd47719b52c047de8bd6a960538bd51595eb7c70712;
- identical ordered query/candidate/label inputs across all six paired runs;
- genuine joint validation-freeze, selected checkpoint hashes, paired initial
  model hashes and later test-access records. Weight hashes are compared only
  through the pinned manifest/freeze; checkpoint members are only streamed/discarded in bounded chunks and are never
  interpreted, deserialized or materialized as model arrays.

Memory is explicitly bounded by one reused86MB NPZ buffer,64query blocks,
small original sidecars and vector headers. It never loads a complete NPY score
vector or65.5MB negative array. Non-target tar members are streamed/discarded in
1MiB chunks so tarfile does not allocate an86MB skip buffer. A monotonic measured
process high-water gate refuses a run reaching256MiB. The actual closure used
204,455,936bytes (194.984375MiB). No Torch import, model, graph or GPU operation.

Actual archived-score closure: PASS all6containers, all24,600,576float scores,
all24,576 original query-level RRs bitwise identical, and all six negative NPY
reconstruction hashes exact. Original seed means/sample SD and paired means/SD:
  union mean .7036321830343, SD .004175715677508995
  PV    mean .7566296818436684, SD .0022107871770645346
  difference mean .0529974988093684, SD .005544534327617058
See ACTUAL_FRIENDSTER_FLOAT_SCORE_RECEIPT_v3.json for all raw file pins, per-seed
MRR/Hits, schema, tie counts, identities, scope and measured memory. This receipt
is archival arithmetic; it adds no independent training seeds or new quality
measurements. Graph non-neighbor/leakage status remains based on the genuine
recorded preparation audits and is not re-executed by this tool.

Small meaningful tests (no real asset/model/graph needed):
  "$AUDIT_PY312" -m pytest -q -p no:cacheprovider \
    "$ARTIFACT/tools/recorded_friendster_scores_v1/test_score_check.py"
Nineteen cases cover positive positions/ties, repeated physical source with
separate query IDs, missing/duplicate labels, NaN, candidate permutation/order,
wrong positives/negative hash, wrong savedRR/metric, six-pair aggregation,
NPY whole-byte reconstruction and bounded memory-file reads.
