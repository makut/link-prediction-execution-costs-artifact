# Public systems reproduction — locally staged, unpublished

This package adds runnable public entrypoints for the frozen V4, native and
PENCIL scientific workloads. It is unpublished; local tests and independent
critical review passed, while full Linux/GPU execution through these new entrypoints remains untested. Sources
and per-job arguments are checked locally; those checks do not measure speed.
Original recorded results and fresh rerun results must remain separate.

Use the public input acquisition and environment instructions in
`README_v0_3_BASE.md`. Set `DATA` and `RESULTS` to absolute paths on a Linux
compute host with suitable RAM and scratch space. Run all commands below from
`src/`. The printed commands are plan-only until `--execute` is added. No YT,
Kubernetes, historical pod paths or private credentials are needed.

Install the recorded Python 3.10/PyTorch 2.5.0+cu121/PyG 2.7.0/torch-sparse 0.6.18
stack and CPU numerical requirements. Native controls also need a C++17
compiler. PENCIL additionally needs the public packages in
`work/vldb_pencil_bridge/requirements-extras.txt`, constrained by its
`constraints-linux-cp310.txt`; use a separate environment and retain the base
numerical/native stack. No internal wheelhouse or compiled Mac binary is shipped.
The local BERT config is included; no pretrained weights are required.

Use the same `--coordination-dir` for every systems command and schedule an
otherwise idle timing window. The lock coordinates these entrypoints; it cannot
reserve a machine against unrelated CPU users. GPU jobs require two idle
observations15 seconds apart and run on one whole GPU. V4 requires H100 to match
its frozen protocol. Host, dependency, GPU, source and input identities are
recorded. Timings need not be identical on different machines.

Each execution requires fresh output outside the artifact and input trees.
Failure outputs are retained, not resumed or overwritten. Total execution is
bounded to at most12 hours, with per-job limits, a declared output budget and
the reviewed V3 sampled memory policy. PSS is a non-atomic proportional
estimate; the guard does not sum shared RSS as unique physical memory.

## V4: exact public projection

```bash
python -m artifact_public.reproduce_pipeline_v4 \
  --citation2-root "$DATA/citation2/prepared" \
  --friendster-root "$DATA/friendster/full_csr" \
  --out "$RESULTS/public_v4" --coordination-dir "$RESULTS/timing_coord"
```

Append `--execute` after inspecting the plan. The frozen benchmark/suite code is
unchanged. The public wrapper selects Citation2 and Friendster and rewrites
only local filesystem paths; it does not fabricate the original controller's
quality-completion or private download markers.

The public grid has24 jobs: two fixtures, two workload-manifest preparations,
and20 timings (two graphs × workers 1/8 × five arms). Each real timing measures
3,200 graphs/100 steps at batch32 after excluding256 warmup graphs. The common
seed is20261124; two hops, cap 200, oversample 1.5, native-auto stock width and
24 GiB Torch allocator budget are retained. The budget is refused when24 GiB
exceeds80%of then-free GPU memory; it is never silently reduced.

Twelve comparisons check the three reference arms pairwise on32 saved warmup
queries per graph/worker setting, plus the initial model and workload identity.
They do not independently audit all3,200 measured graphs. Worker service sums
overlap prefetch and GPU work and cannot be added to wall time. Production
union/PV are different sampler arms, not additional reference-equivalence
claims. This featureless, k281 short-training control does not establish MRR
or time-to-quality. See `metadata/protocols/V4_SOURCE_CONTRACT.json`.

## Native controls: public Friendster only

```bash
python -m artifact_public.reproduce_native_friendster \
  --friendster-root "$DATA/friendster/full_csr" \
  --out "$RESULTS/public_native" --coordination-dir "$RESULTS/timing_coord"
```

Append `--execute` for the three sequential native arms: matched standalone C++ full
association-map clearing, persistent touched-reset, and the actual installed
native torch-sparse primitive. All reuse the same128 deterministically selected
ordered sets from `data/friendster_exact_int32_reference.json`, with one warmup,
three repeats and one CPU thread. The original report SHA determines the
selection, so the packaged public report remains byte-identical.

The explicit int64 conversion uses about27.4 GiB of scratch; the default command
requires32 GiB free disk and keeps its fresh conversion beneath the output.
An existing content-bound conversion can be supplied with `--native-dir` to
avoid repeating it. The declared outer memory budget is64 GiB; the original
converter/worker preflights retain their48 GiB peak estimate bound and live
headroom checks. All setup, hashing, conversion and persistent-map initialization
remain outside query timers and are separately reported.

The generated admission file attests only this public coordinator's completed
exclusive-lock preflight. It does not attest any historical training or
pipeline result. Both target directions, canonical local edges and DRNL must
match the original public oracles. Private VK replay plans are excluded.

Two additional sequential Numba workers follow the successful native parent:
Friendster×32/64, using exactly these128 sets and the already created conversion.
They invoke the unchanged fixed `scan` and `pairwise` methods, the original
`exact.measure` timer, one warmup and three repeats. The64-bit worker opens the
existing hash-bound conversion; neither follow-up copies or converts a graph.
All selected-row,6 GiB workspace and20M pairwise guards are checked before graph
access. A failed, guarded or missing-conversion result cannot count as science.

All three coordinator jobs must share hostname, boot ID, CPU model and exact CPU
affinity. By default any one documented Linux host is accepted. Add
`--reference-hardware` to require the published8480+ CPU model. Hardware and
dependency differences are recorded; another CPU does not promise the published
timings. Each follow-up has a3,600-second limit inside the shared total deadline.
This public adapter is prepared and locally tested; no completed full-scale
same-host follow-up measurements or residual native/Numba ratios are asserted by
its presence. Older different-host timings cannot establish those ratios.

## PENCIL: unchanged consumer/model,48 scientific jobs

```bash
python -m artifact_public.reproduce_pencil \
  --citation2-root "$DATA/citation2/prepared" \
  --out "$RESULTS/public_pencil" --coordination-dir "$RESULTS/timing_coord"
```

Append `--execute`. Two setup jobs regenerate only the512 validation source
groups with the original subset seed20261124 and build the standalone C++
control locally. No test arrays are opened by this setup. Alternatively pass
an existing exact subset with `--validation-root`; its hashes must match.
The original metadata plan is retained unchanged for its optional identity gate;
actual arrays are read from the new public paths.

The fixed cohort is512 validation positives followed by original candidate0
from each complete1,000-negative group:1,024 queries, extraction seed20261126.
All eight physical arms use the unchanged pinned PENCIL consumer and random
frozen BERTLP weights, seed 42,3×512 layers,8 heads,FFN2048 and128 early features.
There is no training, HPO, MRR or pretrained-weight comparison.

The48 scientific jobs comprise two captures, one real preflight, nine semantic
audits including the tiny audit, and36 timings. The real audits cover fp32/bf16,
batch 128/512 and declared workers 0/1, with zero logit tolerance and both native
wrapper modes. Primary timing uses32 online configurations (eight arms × two
batches × two worker counts); four replay configurations use bf16/batch 128/
workers 0. One warmup and three measured1,024-query sweeps are retained. Score
fingerprints must agree across the complete arm/repeat set within each of the
five measured configuration groups. Groups are not pooled.

All native representations remain resident. This instrumented throughput
includes tracing/verification, memory observations, IPC and model score copies;
it is not a memory-saving or uninstrumented serving measurement.

## Existing public E16 and E17 evidence

The recorded V4 audit has a public Citation2/Friendster aggregate projection at
`metadata/evidence/E14/PUBLIC_TWO_GRAPH_NUMERIC_AUDIT.json`:20 timings and12
saved-prefix comparisons. The full original suite had35 jobs/30 timings/18
comparisons. This projection does not bundle the complete original raw archive.
The recovered native Friendster evidence is under
`metadata/evidence/native_friendster_v1/`, with15 unchanged public receipts and
reports plus a Friendster-only numerical audit. All384 case outputs from its
three int64 native arms were exact. See
`metadata/NATIVE_FRIENDSTER_REPRODUCTION.json` for archive/source bindings.
Both recovered studies retain the distinction between archived scientific
worker completion and the lost outer controller's final exit/lock release,
which remain UNKNOWN. Neither includes a completed matched Numba follow-up.

E16 raw report, JSONL records, completion receipt and summary are included under
`metadata/evidence/E16/`. Every member of the original18-file archive was
verified before this explicit public allowlist was copied. Recompute its
768 per-record timing medians without launching any experiment:

```bash
python -m artifact_public.verify_dispatch_e16
```

To rerun the actual E16 routing control later, on an idle large Linux CPU host:

```bash
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m scripts.seal_revision.dispatch_benchmark_v1 \
  --repo "$PWD" --input-json ../data/friendster_exact_int32_reference.json \
  --graph-path "$DATA/friendster/full_csr" --out "$RESULTS/public_dispatch" \
  --limit 0 --warmup 1 --repeats 5 --seed 20261124 --auto-ratio 1 \
  --max-neighbors 20000000 --max-working-set-mib 4096 \
  --max-pairwise-checks 20000000 --max-subgraph-nodes 10000
```

These are sums of per-record medians using identical constructors and two
routing algebras, not measured production or pipeline acceleration.

E17 has a public-only posthoc entrypoint, requiring only the packaged9 MB
Friendster report, NumPy and SciPy. It uses the original numerical functions
and independently checks sums and tie-aware Spearman with stdlib arithmetic:

```bash
python -m artifact_public.reproduce_posthoc_friendster --out "$RESULTS/public_cost_analysis"
```

Append `--execute` to recompute. No graph, new timing, HPO or private VK report
is needed. The original two-graph analysis/checker sources are retained for
provenance; use this public entrypoint rather than their historical main blocks.

## Local verification and release boundary

```bash
python -m pytest artifact_public/test_systems_reproduction.py -q
```

The tests verify exact public V4 projection, all PENCIL command arguments against
the frozen scientific builder, failure/guard rejection, candidate identity,
source integrity, same-host admission/selection/conversion guards, actual tiny
32/64-bit fixed-mode execution and independent raw E16 aggregation. They do not
simulate a completed large Linux/GPU rerun. Public Friendster reconstruction is
verified: the3 full numeric files and23 prepared split/query files match the
study inputs; see `metadata/FRIENDSTER_PUBLIC_REPRODUCTION.json`. This proves
input identity, not model quality or measured speed. Further independently audited
systems-result integration, full-scale execution through these public adapters,
a durable public release remain separate steps. Independent critical review of
the public adapters passed; its scope is recorded in metadata/SYSTEMS_VALIDATION.json.
