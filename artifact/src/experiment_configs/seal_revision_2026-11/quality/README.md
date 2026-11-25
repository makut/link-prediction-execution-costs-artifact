# SEAL sampling quality controls, November 2026

This is a new controlled experiment, not a continuation of old checkpoints and
not a reproduction of their numerical claims. It contrasts **union then sample**
with **per-vertex then union** on the same backend, graph and training budget.
Extraction algorithms themselves are unchanged.

## Frozen design

- Citation2, symmetric training graph for GNN extraction; original directed
  training positives from `train_directed_csr`. Features are the provided 128-D
  node features. Supply a dataset manifest after checking held-out overlaps in
  both directions; the runner records this provenance but does not repair input
  graphs. A cleaned graph constitutes a new dataset version.
- Seeds **42, 43, 44** are chosen as a conventional, explicit three-run sequence
  also used in earlier runs. These are entirely new fully initialized runs;
  results must not be pooled with historical partially seeded runs.
- Exactly the same 2% training-positive subset in every job, subset seed **123**.
  Same positive order and uniform destination-negative sampler seed within each
  paired model seed. Uniform negatives: one per positive, self loops rejected.
- Two hops, cap **200 per hop**, oversampling **1.5**, pairwise induced-edge
  reconstruction enabled for both variants. Only the per-vertex sampling flag
  differs within each pair.
- DGCNN hidden32/layers3, node features, no trainable node embeddings. Fixed
  SortPooling **k=281** for both variants and all seeds. This is the absolute k
  used for the September Citation2 union/per-vertex control, derived from its
  training-only union sample. It is a fixed capacity control, not recomputed
  using the new validation/test sets. This is not the 1-hop OGB leaderboard model.
- Ten full epochs over the fixed 2% positive subset, batch32 total graphs,
  Adam lr1e-4, no HPO, no early stopping. Validation after each epoch. Same epoch
  budget/positives, rather than equal wall time; the purpose is sampling quality.
- Freeze **512 validation source groups**, uniformly without replacement from
  all official validation source IDs, seed **20261124**. Freeze **4096 test source
  groups**, independently with seed **20261125**. Keep the one positive and **all
  1000 provided negatives** for every selected source. No first-N-batch cap and no
  resampling candidates. The helper streams negative files in bounded chunks.
- Checkpoint selection uses validation **`ogb_mrr`** only. Its rank formula is
  `1 + (count(negative > positive) + count(negative >= positive))/2`, matching OGB
  Citation2 midpoint tie semantics. Incomplete source groups and non-finite
  scores raise errors. Generic legacy `mrr` remains available but is not selected.
- All six training jobs finish, then their best-validation checkpoints and
  selection metadata are hashed in a joint **FROZEN_CHECKPOINTS.json**, before
  **any test model scoring**. Test runs load only those frozen checkpoints,
  produce `test_metrics.json` with OGB-compatible metrics and raw `test_scores.npz`.
  Future decisions must not retune the protocol from these test scores.

The evaluation subsets are fixed for all variants/seeds. Evaluation extraction
uses a fixed ordered-pair seed **20261126**, making its random subgraph sampling
independent of DataLoader assignment and earlier validation passes. Python,
NumPy, Torch CPU/CUDA model initialization and worker-local NumPy/Numba RNGs are
explicitly initialized. Initial model hashes must agree within each seed pair.
**Bitwise deterministic CUDA training is not promised.** Results are three
independent seeded runs with the stated software/hardware, not proof of formal
equivalence or a confidence interval for all graph/task regimes.

## Run inside an already provisioned pod

The runner never creates pods, invokes kubectl, downloads packages, or modifies
the input graph. Paths below are the agreed experiment locations; pass explicit
overrides if the prepared layout differs.

```bash
python experiment_configs/seal_revision_2026-11/quality/runner.py \
  --phase smoke \
  --prepared-root /home/jovyan/seal_revision_20261124/data/citation2/prepared \
  --output-dir /home/jovyan/seal_revision_20261124/results/quality \
  --gpu 0
```

Smoke lives under `quality/smoke/`: both variants, seed42, two training steps,
two whole validation groups, all1000 negatives, no test scoring. Inspect both
`training_complete.json` files and matching initial hashes before actual runs.

```bash
python experiment_configs/seal_revision_2026-11/quality/runner.py \
  --phase prepare \
  --prepared-root /home/jovyan/seal_revision_20261124/data/citation2/prepared \
  --output-dir /home/jovyan/seal_revision_20261124/results/quality \
  --dataset-manifest /path/to/verified_dataset_manifest.json
```

`--graph-root`, `--pairs-root`, `--features`, `--val-pos`, `--val-neg`,
`--test-pos`, `--test-neg` override standard prepared-layout paths. Input splits
must be full provided Citation2 `[N,2]` integer arrays, not old percent/prefix
files. The generated JSON files are also valid YAML trainer configs.

```bash
python experiment_configs/seal_revision_2026-11/quality/runner.py \
  --phase train --prepared-root /path/to/prepared \
  --output-dir /path/to/results/quality --gpu 0

python experiment_configs/seal_revision_2026-11/quality/runner.py \
  --phase freeze --prepared-root /path/to/prepared \
  --output-dir /path/to/results/quality

python experiment_configs/seal_revision_2026-11/quality/runner.py \
  --phase test --prepared-root /path/to/prepared \
  --output-dir /path/to/results/quality --gpu 0
```

By default jobs execute sequentially in balanced variant order on the chosen GPU.
`--jobs union_seed42 pv_seed42` selects prepared jobs for allocation to independent
GPUs. Keep CPU worker budgets explicit (16 per job); freeze still requires all
six jobs. Do not compare timings from concurrent and isolated fits as a speedup.
`--phase all` performs prepare/train/freeze/test in that order. Rerunning an
existing fit or test log fails instead of silently overwriting/resuming it.

The plan hashes generated subsets, configs, and training source files. Changed
source/config/subsets fail verification. Code changes after preparation require
a fresh experiment directory. `test_scores.npz` can additionally be verified
with an independently installed/patched official OGB Evaluator; the primary
streaming metric has the exact OGB formula and enforces all candidate groups.
Do not silently discard sources to make the evaluator run.

## Checks

```bash
python -m pytest seal/tests/test_quality_reproducibility.py \
  experiment_configs/seal_revision_2026-11/quality/test_prepare_sources.py -q
```

The fixtures exercise actual DGCNN initialization, two spawned DataLoader workers
with Python/NumPy/Torch and real Numba sampler draws, source-group preservation,
chunk boundaries, missing candidates and midpoint score ties. On this laptop,
use the existing pyenv Python3.12.10; the directory's Homebrew Python3.15 lacks
these dependencies. No packages need to be installed.
