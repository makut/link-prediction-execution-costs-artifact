# New strict-clean Citation2 matched CPU extension (prospective E22)

This is a new experiment source candidate, not a result. Nothing here modifies
the public v0.5.0 artifact, old Friendster experiments, main manuscript, base
packages, a cluster resource, or YT. Root owns target staging, probes, launch,
collection and independent result review.

## Question and comparison

Do the state-lifetime and execution-plan observations from the full Friendster
control persist on the much smaller public strict-clean Citation2 graph?
All seven arms execute the same 128 ordered selected sets: unmodified upstream
torch-sparse direct induction, standalone full-clear C++, standalone persistent
touched-reset C++, and unchanged Numba scan/pairwise kernels with int32/int64
neighbors. Every arm has one excluded warmup and three measured repetitions.
The native three arms and two Numba workers run sequentially in fresh processes
with one numerical thread, the same hostname/boot/CPU model/affinity, and an
exclusive owned timing lock. Affinity is recorded; it is not called a dedicated
pinned core.

The scientific functions, native C++ and adapters, original converter, exact
samplers/oracles/timers and Numba kernels are copied byte for byte from the
reviewed Friendster package. Only orchestration, dataset admission and the new
pre-timing cohort builder change. Original full-file/portable AST pins remain
mandatory; only the already-reviewed empty PEP695 type-parameter normalization
is used across Python 3.10/3.12/3.13.

## Public input: exact existing strict-clean graph

N=2,927,963, M=60,703,502 stored entries. Only these two regular files are needed:

| File | Bytes | SHA256 |
|---|---:|---|
| edge_starts.bin | 23423704 | f06070adc368892f8441ea15e691dce400f9d44b6ec3e6e4505134c276c1072d |
| edge_ends.bin | 242814016 | d3d8d1e2016675321c1d3d9ae817d2c6dca4a9dd2d5cdbba3e638ea7201057a4 |

`--graph-path` points directly at the existing graph; no source copy or symlink
is necessary. The target's verified public-reproduction path is the default:

`BASE/public_source_repro_v1/execution_v2_runs/citation2_public_repro_20261125_145307_fbdd1d46/dataset/prepared/train_csr`

`prepared_original` is a different graph and will fail whole-file admission.
BASE is `/home/jovyan/seal_revision_20261124`. An override must be an absolute
path under BASE and still match both complete file hashes. No node features,
timestamps, labels, model weights, validation/test candidates or scores are used.

## New cohort, frozen before timing

There was no archived Citation2 exact128/768 cohort in the old CPU evidence;
the old archives are Friendster/VK. This extension does not relabel those files.
The immutable `COHORT_PROTOCOL.json` defines 128 seeded training-graph pairs
(64 existing edges, 64 two-hop nonedges), three original samplers and caps50/200.
Its 768 candidate combinations form 12 sampler/cap/workload strata.

The original deterministic `prepare.choose` round-robin selects 128 candidate
indices, salted by the exact protocol SHA, before generating their node sets
or reading any timing. Strata have ten or eleven cases; they are not population
weights or independent queries. The original sampler then constructs the
selected ordered S, with a declared per-case seed. The independent Python edge
oracle and DRNL compute and freeze node/edge/label hashes. No performance arm is
run in this stage. A guarded case aborts preparation; it is never replaced.

The resulting `cohort/citation2_exact_int32.json`, original-format replay plan,
and `COHORT_READY.json` are bound into the controller admission and all later
source/input bindings before the first physical timing arm. The source manifest
pins the preparation algorithm/protocol; the actual runtime cohort receipt
pins the generated S. Both are needed for a scientific result.

## Environment and accounting

Target: Linux x86_64 Python3.10.12, Torch2.2.2+cu121, NumPy1.24.4,
SciPy1.10.1, Numba0.58.0 and a C++17 compiler. The exact pt22 two-wheel packet
already used by Friendster must exist at `BASE/cpu_pt22_native_wheels_v1.tgz`:
15,064,339 bytes, SHA256
`5e9edbdc067325cd0b41b5fb292e2cc7343cc535da87ff3bef720cbf85da38cf`.

A fresh `citation2_cpu_matched_v1_env` exposes inherited base sites through a
plain path-only .pth and installs those exact wheels offline into its own
prefix. Before/after origin/version/binary probes prove the base is unchanged.
The old Friendster venv, build and caches are not reused or modified.
All CUDA visibility is disabled; no GPU API is called.

The original converter creates one fresh int64 graph: 509,051,728 bytes.
The independent stdlib streaming promotion of the pinned original arrays
establishes expected complete rowptr/col hashes before conversion. Converter
self-reported hashes alone cannot admit the representation. Native readback
and each runtime worker verify actual complete array bytes.

One native association map is 23,423,704 bytes per worker. Its initialization is
reported outside query timers. The unchanged converter's conservative estimate
is 1,896,454,976 bytes including source, native arrays, association, chunk and
reserve. Declared budgets are1GiB native,8GiB peak,.05GiB association, with8GiB
initial free disk and current-headroom admission. External owned-tree PSS/cgroup
guards remain active. Partial failures and outputs are preserved.

Setup, input hashing, conversion/readback, JIT, warmup, fixtures and persistent
initialization are excluded from the per-query timer and separately reported.
Native and Numba edge timers retain the original differing low-level output
materialization; DRNL is separate. Aggregate by summing 128 record-wise medians,
and combine edge+DRNL repeat pairs before taking their median. No significance,
language superiority, time-to-quality, fresh PSS or full-SEAL speedup follows.

## Root-only target steps

Stage the source-only packet under `BASE/citation2_matched_cpu_v1_code`.
Before any execution run, with the actual target interpreter:

```text
/opt/python/bin/python3 -B BASE/citation2_matched_cpu_v1_code/portable_ast_probe.py
/opt/python/bin/python3 -B BASE/citation2_matched_cpu_v1_code/cpu_coordinator_v1.py
```

Both must pass, the second as `plan_only`, without numerical imports, graph
content reads, output/env/lock creation. Then root may execute the concrete
reviewed plan with its exact source-manifest pin:

```text
/opt/python/bin/python3 -B -u BASE/citation2_matched_cpu_v1_code/cpu_coordinator_v1.py
  --execute --source-manifest-sha256 <frozen-manifest-SHA> --max-hours 6
  --graph-path <existing-verified-public-strict-clean-train_csr>
```

The genuine unchanged upstream and standalone tiny native admission must pass
on Linux before full derived graph allocation. Local tests cover real tiny
Numba32/64 and standalone compiled native implementations; they do not claim
to have run the Linux-only upstream wheel on macOS. Actual scientific eligibility
requires all seven complete same-cohort reports, exact oracles, full input and
environment continuity, clean owned-process exits and the separately collected
outer completion. No root launch or scientific result is implied by these files.
