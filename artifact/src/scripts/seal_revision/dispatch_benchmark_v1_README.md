# Frozen-S dispatch replay v1

This standalone CPU microbenchmark is prepared locally. The owner schedules it
**after the existing CPU and native queues**. It performs no queue operations,
downloads, YT writes, GPU work or changes to frozen quality/pipeline/exact sources.

The earlier ~18% `ours_auto` gap was measured in the **exact harness**, whose
`production_induced(..., "auto")` calls Python `graph.degree_sum`. Actual
`seal/graph_csr_extraction.py` already uses vectorized NumPy indexing and an int64
sum. That gap is not evidence of a production routing bottleneck.

Replay methods share the same frozen `exact_benchmark.production_induced` CSR
constructor and kernels:

- `fixed_pairwise`, `fixed_scan`;
- `exact_python_auto`, reproducing the original harness control;
- `production_vectorized_auto`, matching the production routing expression,
  including the safe terminal-row lookup for raw N-entry offsets;
- optional `precomputed_route_lower_bound`, a hypothetical bound with its route
  decision supplied outside the CSR timer, explicitly not an integrated backend.

The input is one completed Friendster exact JSON (int32 or int64), containing all
768 archived selected-node arrays. There is no resampling. Source hashes for
`exact_benchmark.py`, `graph_csr_kernels.py` and `graph_csr_extraction.py` must match
the supplied `--repo`, and they are checked again at completion. Importing uses
that repo's helper and rejects a conflicting cached kernel import.

Example for the owner, with existing pod-local files:

```bash
python /path/to/dispatch_benchmark_v1.py \
  --repo /home/jovyan/seal_revision_20261124/cpu_code \
  --input-json /home/jovyan/seal_revision_20261124/results/cpu/friendster_exact_int32.json \
  --graph-path /home/jovyan/seal_revision_20261124/data/friendster/full_csr \
  --out /home/jovyan/seal_revision_20261124/results/dispatch_v1
```

The default is all768 records, warmup1 and repeats5, with fresh randomized method
order for each warmup/measured repeat. All four methods execute once per warmup
before recorded timing, excluding Numba compilation. Every returned CSR,
including warmups, must match the archived canonical-edge and DRNL hashes.
Only CSR construction enters the edge timer. Validation/DRNL are outside it;
Python and vectorized routing also have separate route-only timers. This is not
full SEAL, sampling, DRNL throughput, or a GPU pipeline measurement.

The full replay targets a few minutes on the existing warm CPU node; its wall
time has **not** been measured locally on Friendster. If a bounded128-case probe
is preferred, `--limit 128` deterministically balances existing
sampler/cap/workload cells and preserves their archived node arrays. It does not
represent a new sample of graph pairs. Use a different fresh output directory
for any later full replay. Any existing `--out` path is refused; there is no
automatic overwrite or resume.

All graph arrays are read-only mmap. No full-graph SciPy object or O(N) stock
scratch is allocated. Defaults retain max20M selected neighbors, max20M pairwise
checks, max10k subgraph nodes and the original conservative4GiB local workspace
guard. This estimate excludes mmap page cache and runtime memory. NumPy/BLAS,
OpenMP and Numba are restricted to one thread. No Torch/PyG import is introduced.

Outputs are `records.jsonl`, `report.json`, `status.json` and an immutable
`COMPLETE.json` containing the final report hash. The report labels optional
precomputed routing as hypothetical. It records the raw input hash, frozen source
hashes, selected indices, route decisions, method order and all timings. A failure
retains completed records and a failed status; it does not publish COMPLETE.

Synthetic verification:

```bash
python -m pytest -q scripts/seal_revision/test_dispatch_benchmark_v1.py
```

The tests cover terminal N−1 with a nonempty row, isolated N=1 routing, int64
degree sums above2³¹, directed/undirected tie algebra, duplicate/hash/degree
rejection, full edge/DRNL agreement, balanced subset identity, mmap CLI success,
source mismatch and refusal to overwrite. They do not execute Friendster or
contact any service.
