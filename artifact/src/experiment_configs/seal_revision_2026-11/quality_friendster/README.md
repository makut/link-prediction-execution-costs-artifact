# Friendster union/per-vertex quality control on the colleague's exact task

This package leaves the frozen Citation2 experiment and all production SEAL
sources unchanged. It reuses the frozen trainer, RNG helpers, midpoint-rank
accumulator and joint-freeze orchestration through an explicit child-process
adapter. Its only evaluation bridge groups by **original query-row ID** while
extracting subgraphs from the **real physical endpoints**.

## Why a query-aware adapter is required

The supplied validation and test sets each have 10,000 query rows but only
9,995 unique source vertices: five sources occur twice. Each row has its own
positive target and its own 1,000 provided negatives. Grouping these rows by
source would merge distinct tasks and invalidate the metric. We neither drop
queries nor combine candidate sets. `query_id` means the zero-based original
row in the colleague's `source_node.npy`, `target_node.npy` and
`target_node_neg.npy`. All candidate entries, including repeated destinations,
are kept exactly once per original candidate slot.

The fixed subsets select **512 validation query IDs** and **4,096 test query
IDs**, uniformly without replacement, with seeds20261124/20261125. They retain
cold queries too: supplied warm flags are preserved for audit, not used as a
filter. Original source/target/candidate arrays and selected query IDs are
hashed. Candidate scoring order is a fixed permutation per query (seed20261127),
independent of labels and model scores. Subgraph extraction uses the existing
fixed per-physical-pair seed20261126, so the same physical edge receives the
same sampled evaluation graph even when it appears in different queries.

The primary metric is **custom sampled MRR** with midpoint tie handling:
`rank = 1 + (count(neg > pos) + count(neg >= pos))/2`. This is the same rank
formula as Citation2, applied to the colleague's custom Friendster queries.
**This is not an official OGB benchmark.** The frozen trainer's internal alias
`ogb_mrr` is reused for checkpoint selection; final adapter reports rename it
`custom_sampled_mrr` and explicitly mark `is_official_ogb_benchmark: false`.

## Frozen quality design

- Input topology: colleague's **80/10/10 split**, hash seed20261102, with
  65,608,366 nodes and **2,889,707,416 stored training adjacency entries**.
  `prepare_friendster.py` owns exact split construction, source checksums and
  provenance. The exact archived thresholds are14757470836727281938 and
  16602285075010369541; a recomputed theoretical0.8/0.9 threshold is a different
  split. The runner requires matching counts/seed/thresholds, verified historical
  source identity, latest-query-array equality and a completed preparer manifest.
  It verifies selected input array checksums against that manifest and freezes
  training topology sizes/mtimes for all later phases.
- Positive iteration: canonical **u<v `train_pairs_csr`**, exactly
  **1,444,853,708** undirected training edges. No doubled symmetric positive
  iteration and no materialized billion-row subset. The existing seeded
  coprime-stride iterator supplies an identical deterministic prefix within
  each union/per-vertex pair, using the same worker count and source graph.
- Seeds42/43/44 cover model initialization, Python/NumPy/Torch/CUDA and NumPy/Numba
  in data workers. Paired initial model hashes must agree before the joint freeze.
  Bitwise deterministic CUDA training is not promised.
- DGCNN hidden32/layers3, **fixed k370**, no features, no trainable node embeddings.
  k370 is the fixed capacity from the historical Friendster training-only
  SortPooling estimate; it is not retuned from the new validation/test queries.
- Two hops, cap200 per hop, oversample1.5; same pairwise induced-edge backend for
  both variants. Only union/per-vertex neighbourhood sampling differs.
- Batch128 = **64 positives +64 uniform destination negatives** (one per positive,
  self loops rejected). Sampler and positive-order seeds match within each pair.
- Fixed **20,000 optimizer steps**: **2.56 million graphs /1.28 million positive
  examples per run**. This is a finite training budget, not convergence or a full
  pass over the billion-edge graph. Validation every2,000 steps; no early stop,
  no HPO and no budget adaptation from held-out test scores.
- All six jobs finish; the best-validation checkpoints are jointly frozen and
  hashed before any held-out test model scoring. Each frozen checkpoint is then
  tested once on the same4,096 original query IDs and all their provided negatives.

Training uses one nominal epoch, cut by the explicit step budget long before
exhausting canonical training edges. The batches are full because the selected
prefix is tiny compared with that epoch. Concurrent quality fits must not supply
isolated pipeline speedup claims; those benchmarks run separately.

## Commands inside the separately provisioned pod

The runner performs no cluster operations or installation. The preparer must
have created `train_csr/`, `train_pairs_csr/`, `manifest.json` and
`evaluation/{valid,test}/{source_node,target_node,target_node_neg,warm_query}.npy`.
Generated configs require the package's `train_entry.py`, which supplies original
query identity. The runner always invokes that entry point.

```bash
python experiment_configs/seal_revision_2026-11/quality_friendster/runner.py \
  --phase smoke --prepared-root /path/to/friendster/prepared \
  --dataset-manifest /path/to/friendster/prepared/manifest.json \
  --output-dir /path/to/results/quality_friendster --gpu 0
```

Smoke is separate under `quality_friendster/smoke/`: two variants, seed42,
two optimizer steps, two complete validation queries, no test scoring.

```bash
python experiment_configs/seal_revision_2026-11/quality_friendster/runner.py \
  --phase prepare --prepared-root /path/to/friendster/prepared \
  --dataset-manifest /path/to/friendster/prepared/manifest.json \
  --output-dir /path/to/results/quality_friendster

python experiment_configs/seal_revision_2026-11/quality_friendster/runner.py \
  --phase train --prepared-root /path/to/friendster/prepared \
  --dataset-manifest /path/to/friendster/prepared/manifest.json \
  --output-dir /path/to/results/quality_friendster --gpu 0 \
  --jobs union_seed42
```

Use `--jobs pv_seed42` on the other GPU, then waves43/44. Exactly16 data workers
per actual job; two simultaneous fits use32 workers. After all six finish, run
the same command with `--phase freeze` (no `--jobs`), followed by `--phase test`
and optional per-job GPU allocation. `--phase all` is a sequential alternative.
The protocol, source code, selected queries and generated configs are hashed;
existing fits/logs are never silently overwritten or resumed.

`--evaluation-root` overrides `prepared/evaluation`; `--protocol` is available
for explicit synthetic checks or a separately named future experiment.

## Outputs and comparison with node2vec

`test_results_*.json` contains normalized `test_metrics.custom_sampled_mrr` and
the raw implementation metrics for audit. Each `runs/<job>/run1/` also contains:

- `query_ids.npy`: original test-row IDs in reporting order;
- `query_rr.npy`: one reciprocal rank per corresponding original row;
- `query_scores.npz`: explicit `query_id`, real `src`/`dst`, original
  `candidate_slot` (0positive,1..1000provided negatives), label and raw logit;
- the frozen checkpoint, validation selection and initialization manifest.

The unmodified trainer's raw `test_scores.npz` uses its `src` channel for the
**query ID**, as required by its accumulator. Use the explicit canonical
`query_scores.npz` for physical endpoints. The adapter verifies every scored
target, slot, label and query ID against the frozen input candidate rows before
exporting normalized metrics.

To compare with an existing node2vec `test_rr.npy` in original10k row order,
select exactly `node2vec_rr[seal_query_ids]`. Do not compare to the mean on all
10k queries, silently discard repeated sources, or filter only warm rows.
Report the fixed-step SEAL budget explicitly rather than implying equal model
training cost or convergence across different learning objectives.

## Checks

```bash
python -m pytest \
  experiment_configs/seal_revision_2026-11/quality_friendster/test_query_protocol.py -q
```

Fixtures cover repeated physical sources with different positives/candidates,
cold-query preservation, original-row selection, complete candidate permutations,
actual endpoint extraction versus metric query identity, chunked midpoint MRR,
wrong candidate alignment and actual fixed-capacity DGCNN initialization.
