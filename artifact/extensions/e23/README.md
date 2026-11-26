# E23: independent-process repetitions of the public V4 pipeline control

**Recorded result: independently verified PASS.** The actual Linux/H100 E23 run completed the 40 declared gates and 36 real timing jobs. The exact raw archive, final external receipts and independent verifier output are under `recorded/`. The frozen runner packet and published v0.5.0 assets remain unchanged. New execution on another machine produces new observations.

## Source-only verification

Use Python 3.10 or newer, from this directory:

```sh
python3 -B verify_e23.py --output MY_SOURCE_CHECK.json
python3 -B -m unittest -v test_verify_e23.py
```

The first command verifies the complete immutable runner packet, original
scientific source pins, both small original public workload NPY files, the
semantic protocol hash, Python 3.10 grammar, and the declared job order. It
produces `SOURCE_ONLY`, which is not an actual-result PASS. Output receipt names
must be new. `SOURCE_ONLY_CHECK.json` records the actual preparation-time source
check. `SYNTHETIC_VALIDATOR_CHECK.json` records six focused validator tests.

The tests use one explicitly synthetic report in memory, elementary arithmetic
vectors, and a temporary byte-level archive without scientific result fields.
No fabricated full experiment or scientific summary is written or distributed.
The original source packet's larger test suite is not repeated here.

## Verify the included recorded E23 archive

The included raw archive is `pipeline_repeats_e23_20261126_final_v1.tgz`, 1496841 bytes, SHA256 `41def9a2735829c741fc9ddc490bda56dcd0780bf37cd9759bd0104e42f05073`. The actual outer source SHA256 is `e8df35bb6ebacf5bc9c9f5b4ea4c2337e6fabb93b5e10fb3c2818c03eb025f5e`. `recorded/RECORDED_RESULT.json` binds the original archived run and the separately generated final outer receipts to the independent PASS receipt. The artifact-wide command is:

```sh
python3 -B verify_e23.py \
  --archive "recorded/pipeline_repeats_e23_20261126_final_v1.tgz" \
  --archive-sha256 "41def9a2735829c741fc9ddc490bda56dcd0780bf37cd9759bd0104e42f05073" \
  --outer-source-sha256 "e8df35bb6ebacf5bc9c9f5b4ea4c2337e6fabb93b5e10fb3c2818c03eb025f5e" \
  --outer-complete recorded/outer_COMPLETE.json \
  --archive-pointer recorded/archive_pointer.json \
  --output MY_E23_OFFLINE_VERIFICATION.json
```

The archive namespace is `e23/results`, `e23/envelope`,
`e23/sources/{packet,outer,bootstrap}`, and `e23/bootstrap`, plus
`SHA256_MANIFEST.json`. The verifier does not import the runner, its validator,
its summary function, NumPy, Torch, or other scientific modules. It does not
extract the tar or read paths observed inside archived records.

It independently checks:

- Every archived whole-file identity, complete source inventories, exact
  seven-file scientific runtime bindings, both historical workload identities,
  and the eight used public graph input hash/stat attestations.
- The declared 40 ordered job gates: two manifests, one intentional impossible
  GPU-capacity refusal, one three-arm GPU fixture, and 36 real timing jobs.
  A `complete` string cannot override a nonzero scientific exit, timeout,
  resource guard, orphan cleanup, or a live owned process tree.
- Raw child configurations/results against parent reports, the fixed model
  initial-state hash, CUDA/H100 identity, the 24 GiB Torch allocator contract,
  excluded warmup, measured denominator, and throughput arithmetic.
- All 12 rows of three process/window observations, their descriptive
  mean/sample SD/median/range, and 12 groups of ratios paired by block before
  aggregation. These are recomputed with independent elementary arithmetic.
- All 36 cross-arm comparisons of the 32-query excluded warmup prefix, including
  workload pairs/labels and node/edge/DRNL hashes, plus initial model identity.
- Actual outer child `wait(0)`, PID/start-time quiescence and the post-exit
  exclusive-flock acquisition/release receipt. The runner's own `COMPLETE.json`
  is insufficient to establish these later lifecycle observations.

The raw tar is capped at 128 MiB compressed, 384 MiB payloads plus 16 MiB tar
header allowance, 4,096 regular members and 32 MiB per member. Retained JSON/NPY
bytes are capped at 96 MiB; JSONL is checked incrementally. No graph arrays,
models, native libraries, environments, credentials, checkpoints, or caches are
loaded. The verifier performs no network, cluster, YT, GPU, process, or flock
operation; it validates recorded observations of those events.

The archive is created after scientific child exit but before final upload and
outer `COMPLETE.json`. Those future receipts cannot be inside that same archive.
If independently collected, pass both separately:

```sh
python3 -B verify_e23.py \
  --archive "$E23_RAW_ARCHIVE" --archive-sha256 "$E23_RAW_SHA256" \
  --outer-source-sha256 "$E23_ACTUAL_OUTER_SOURCE_SHA256" \
  --outer-complete "$E23_OUTER_COMPLETE" \
  --archive-pointer "$E23_ARCHIVE_POINTER" \
  --output MY_E23_OFFLINE_WITH_PERSISTENCE.json
```

Without these files, an actual archive PASS explicitly leaves final outer
completion/YT persistence unchecked. Checking the external receipts still does
not constitute a live YT readback or a new outer process observation.

Graph-derived node/edge/DRNL hashes are cross-checked across matched arms and
bound to the exact sources/workloads. They are not regenerated by this offline
verifier because full graph payloads are not included. The 32 warmup audits do
not independently certify every measured graph.

## Exact public runner packet and data

`source/packet` is a byte-for-byte copy of the frozen E23 packet: 31 payloads
and `PACKET_MANIFEST.json`, SHA256
`16b14211782d5c7c765e52ec36a533131e3bf559633229da19a874cdfb28c68f`.
The canonical-JSON protocol SHA256 is
`6587e2055c5a77f4ffbe075fd7d70a6fcf648ad075e4d4cb8aa6574ae54061a1`;
it is deliberately distinct from the on-disk `PROTOCOL.json` file SHA256.
`SOURCE_PINS.json`, licenses and third-party notices are preserved unchanged.

The public input sources and deterministic preparation instructions are in
the [v0.5.0 artifact](https://github.com/makut/link-prediction-execution-costs-artifact/releases/tag/v0.5.0)
and its [public commands](https://github.com/makut/link-prediction-execution-costs-artifact/blob/v0.5.0/artifact/PUBLIC_COMMANDS.txt):

- [OGB Citation2 ZIP](https://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip),
  prepared with the existing strict-clean recipe. The runner uses five pinned
  files: `num_nodes.txt`, two `train_csr` arrays and two `train_pairs_csr` arrays.
  The similarly named `prepared_original` variant is not interchangeable.
- [Friendster dense public release, Zenodo 5146230](https://zenodo.org/records/5146230),
  prepared as the full graph with 65,608,366 vertices and 3,612,134,270 stored
  entries. The runner uses the pinned node count and two CSR arrays.

All eight used files are whole-hashed once in bounded reads, then their
size/mtime/inode/device bindings are checked between jobs. The two original
8,000-row workload NPY files are supplied (192,128 bytes each). Their entire
file SHA256 values and 4,000 positive occurrences are checked independently.
No node features or held-out negatives are needed for this finite-window
control. Graph arrays are not included in this extension.

## Fresh execution on the reader's Linux machine

Provide Linux x86_64 Python 3.10 and the original numerical/model stack:
NumPy 1.24.4, SciPy 1.10.1, Numba 0.58.0, Torch 2.5.0+cu121 and PyG 2.7.0.
The supplied requirements and v0.5.0 installation recipe retain the original
Torch/PyG extension versions (torch-scatter 2.1.2 and torch-sparse 0.6.18).
Use a separate prepared environment. This runner performs no downloading or
package installation. The E22 Torch 2.2 wheel packet is not the E23 model stack.

The runner requires one idle whole H100, with no MIG/DDP/CPU fallback. It checks
15-second separated idle observations before every GPU phase, uses one
numerical thread, and shares a local timing lock with cooperating benchmarks.
The output directory must be new and disjoint from input/source trees.
`TIMING_COORDINATION` must be writable and outside input/source trees.

Source-only plan; no graph read, output, job, or lock is created:

```sh
"$E23_PY" -B source/packet/run_pipeline_repeats.py \
  --citation2-root "$C2_STRICT_CLEAN_PREPARED" \
  --friendster-root "$FRIENDSTER_FULL_CSR" \
  --out "$E23_NEW_OUTPUT" --coordination-dir "$TIMING_COORDINATION"
```

Add `--execute` to run the declared full grid. There is no overwrite/resume or
automatic retry. The runner caps total duration at six hours, each phase at
3,600 seconds, output at 2 GiB, and outer CPU process-tree memory at 180 GiB,
with a finite cgroup 90% guard. Its Torch allocator budget is 24 GiB; CUDA
contexts, library/direct-CUDA allocations and external processes are outside
that allocator cap. It requires four GiB initial free scratch and retains a
one GiB reserve. Failures and guarded arms remain in the evidence.

The exact V4 measurement uses a featureless DGCNN, hidden width 32, three
layers, sortpool 281, batch 32, eight excluded warmup steps (256 graphs), 100
measured steps (3,200 graphs), and one continuous iterator. Three same-law
induction arms use workers 1/8 on both public graphs. Three fresh benchmark and
training-process repetitions use fixed runtime seed 20261124, with cyclic
Latin arm order and randomized blocks from order seed 20261126. Caches remain
shared and are not evicted; setup/JIT/warmup stay outside the window. Historical
inner resource sampling and the new outer guard consume CPU and are not
subtracted from the synchronized child timer.

The runner is portable source; the deployment envelope under
`schema_reference` is retained only to document the actual archive/lifecycle
schema. Its author-host/YT contract is intentionally not supplied as a public
launch command. A fresh run writes results directly to the reader's chosen
output; record its real external OS exit/quiescence separately when archiving.
Do not claim a runner-generated terminal as its own future outer process exit.

These are repeated-cache, fixed-workload process/window repetitions. They do
not estimate independent data or quality-seed uncertainty. No p-values,
population confidence intervals, cold-I/O speedup, full upstream SEAL
throughput, historical-to-new causal speedup or time-to-quality is claimed.


## Independent actual numerical audit

`recorded/ACTUAL_NUMERIC_AUDIT.json` is the separate actual numerical peer. Its byte-identical stdlib source is `independent/audit_e23_numbers.py`. All three measured wall times in each of the 12 descriptive rows and all three per-repetition ratios in each ratio group are retained; none are replaced by a synthetic mean-only observation. Reproduce the numerical audit with a new output:

```sh
python3 -B independent/audit_e23_numbers.py \
  --archive "recorded/pipeline_repeats_e23_20261126_final_v1.tgz" \
  --archive-sha256 "41def9a2735829c741fc9ddc490bda56dcd0780bf37cd9759bd0104e42f05073" \
  --output MY_E23_NUMERIC_AUDIT.json
```

This numerical audit complements the complete source/guard/bootstrap/lifecycle/persistence verifier; it does not perform fresh science or substitute for those gates.
