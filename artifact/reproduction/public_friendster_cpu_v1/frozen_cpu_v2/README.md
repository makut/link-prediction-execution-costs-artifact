# Public Friendster matched CPU control v2

This separate v2 addresses only a source-identity portability failure in v1.
The deployed v1 source and first failed attempt remain immutable. Its target
Python3.10 rejected hashes computed with Python3.12's additional empty
`FunctionDef.type_params=[]` field before output, lock, environment or graph
work began. `V1_TARGET310_PORTABILITY_DIAGNOSIS.json` preserves the actual
source-only target observation: adding that one empty field reproduced all
seven original local hashes exactly.

V2 keeps the seven extracted scientific/execution function files byte for byte,
along with every original scientific kernel. A separate canonicalizer omits
only empty `type_params` on FunctionDef, AsyncFunctionDef and ClassDef. It
rejects nonempty or unsupported fields and preserves all other AST fields.
The Python3.13 serializer is explicitly asked to show empty fields to preserve
the Python3.10 display. Both original raw local hashes and actual target3.10
hashes are retained in `PORTABLE_AST_PINS.json`. Full extracted-file SHA checks
remain mandatory. V2 uses fresh code, result, environment, wheel-extraction,
conversion, build and cache paths; the separate pinned wheel packet is reused.

This is a new, public-only candidate. It does not resume or modify the previous
two-graph queue. The only dataset is the full public Friendster graph:
65,608,366 nodes and 3,612,134,270 stored directed entries. Its existing source is
`BASE/public_friendster_repro_v1/new_data/full_csr`. The prepared training graph
has a different edge count and is rejected.

The coordinator defaults to a plan. Execution is restricted to the reviewed
Linux CPU host, canonical base and source location, requires an explicit source
manifest SHA, and refuses existing output, environment, conversion, build or
cache directories. Root owns staging and deployment. No code here downloads
inputs, requests GPUs, changes a pod, installs into the base environment, or
creates/uploads a YT object.

## Measurements retained

- The exact original 768-row Friendster source record archive is included. The
  independently frozen 128-row selection, seed 20261125, strata, original order,
  ordered node sets, edge oracles and DRNL hashes are unchanged.
- The original eight native files and external helper are copied byte for byte.
  Three fresh sequential processes measure the standalone full-clear control,
  persistent touched-reset control and unmodified native torch-sparse direct
  operator. The original seeded arm order, one warmup, three repeats, target
  removal, output cloning, packaging and DRNL timers are retained.
- Two separate Numba workers use int64 offsets with int32 and int64 neighbors.
  Each measures both original fixed scan and pairwise kernels. The three
  scientific helper functions are AST-identical extracts of the former matched
  worker; `exact.measure`, `exact.production_induced` and the kernel module are
  unmodified. The first selected query compiles the actual mapped-array
  signatures before per-record timing.
- All five workers share the same host, boot, CPU model and affinity, and one
  numerical thread. Every result is checked against its archived edge and DRNL
  oracle. Native ordered raw rows, columns and original edge IDs also match.

The complete original full graph receives a streaming SHA check before and
after execution. Actual path, size, inode, device and mtime continuity are
retained. The single derived int64 conversion must match the historical whole
rowptr and column SHA anchors. The int64 Numba worker reuses that conversion;
it does not create a second graph copy.

## CPU environment

The inherited base versions remain Python 3.10.12, Torch 2.2.2+cu121,
NumPy 1.24.4, SciPy 1.10.1 and Numba 0.58.0. Native binary admission is new for
this CPU stack. The separate pinned official PyG packet contains exactly two
cp310 Linux wheels, torch-sparse 0.6.18+pt22cu121 and
torch-scatter 2.1.2+pt22cu121, and their original download manifest.
The exact 15,064,339-byte packet has SHA256
`5e9edbdc067325cd0b41b5fb292e2cc7343cc535da87ff3bef720cbf85da38cf`.
It is a separately staged root-owned asset at this candidate stage. A public
release must provide that exact asset or a separately reviewed wheel-download
adapter; recreating an arbitrary tar from the official URLs need not reproduce
the packet SHA. No published self-contained release is claimed here.

An isolated venv is created without pip. A plain path-only `.pth` exposes the
existing base site directories after the new venv site. The inherited pip is
invoked with the two exact wheel paths, `--isolated --no-index --no-deps
--no-compile --no-cache-dir --ignore-installed --prefix <fresh-venv>`.
No base package is removed or installed. Before/after base and isolated probes
bind interpreter, versions, module origins, wrapper source and all native binary
bytes. The wrapper git blob and every installed native file must match the
official wheels. CUDA visibility is empty; the probe calls no GPU API.

The standalone C++17 library is built from the unchanged source and flags in a
fresh directory. The original real native and C++ tiny fixtures must pass
before the converter can allocate the full derived graph. A failing native
import, ABI or fixture produces preserved failure evidence and no large
benchmark allocation.

## Resource and lifecycle contract

Only the new int64 representation is large: 29,421,941,096 bytes, or
27.401 GiB. The unchanged converter estimate, including original and derived
working sets, association state, a one-million-entry chunk and reserve, is
46,017,953,864 bytes, or 42.858 GiB. Native storage, peak and association budgets
remain 28, 48 and 0.5 GiB. The controller requires 64 GiB free disk initially.
The persistent association map is 524,866,928 bytes per worker; its measured
initialization stays outside query timers and is retained in raw receipts.

The original reviewed owned-process supervisor and memory guard are included
unchanged. An AST-identical extract of the previously reviewed CPU guard samples
the owned tree every two seconds, enforces the 180 GiB bound/cgroup policy,
and bounds logs. The controller holds an exclusive CPU timing lock, records its
real PID/start/source identity and refuses overlapping project work. It has a
finite total deadline of at most six hours; conversion, native-arm, Numba and
environment subprocesses also have separate deadlines. Only its owned trees
can be stopped. Failure outputs and partial conversions remain available.

Scientific completion is distinct from outer completion. After owned processes
are quiescent, a local archive contains the small raw evidence, complete sources
and a whole-file SHA manifest. It excludes graphs, native input arrays, wheels,
environment and caches. The final outer receipt, archive pointer and actual
process quiescence can be collected separately. No YT service is used.

## Launch after root staging and independent review

Stage this directory as `BASE/public_fr_cpu_v2_code`, and the already pinned
wheel packet as `BASE/cpu_pt22_native_wheels_v1.tgz`. Read `SOURCE_MANIFEST.json`
and its SHA from the reviewed source packet. The explicit execution command is:

```text
/opt/python/bin/python3 -B -u BASE/public_fr_cpu_v2_code/cpu_coordinator_v1.py
  --execute --source-manifest-sha256 <reviewed-manifest-sha256> --max-hours 6
```

The CLI default omits `--execute` and creates no output or runtime environment.
Before execution, root must run both source-only checks on the actual target
Python3.10 interpreter and retain their stdout receipts:

```text
/opt/python/bin/python3 -B BASE/public_fr_cpu_v2_code/portable_ast_probe.py
/opt/python/bin/python3 -B BASE/public_fr_cpu_v2_code/cpu_coordinator_v1.py
```

Every portable function hash must equal the captured actual target3.10 hash,
and the coordinator must return `plan_only`. These checks open no graph data,
import no numerical packages and create no output, environment or lock.

## Interpretation

If all seven physical paths finish, compare only their new measurements on
this single documented CPU/runtime and the same 128 ordered sets. Aggregate
the sum of 128 per-case medians for each declared edge/raw/DRNL boundary.
Conversion, whole-content verification, setup, JIT, fixtures and persistent
initialization are separately reported. Native edge time includes invocation,
native clones, NumPy views and local CSR packaging; Numba edge time includes
the unchanged fixed-mode call, target removal and local CSR. DRNL is separate.
No different-host speed ratio, independent-seed uncertainty, universal winning
method, training result, quality score, memory-free state or completed old
two-graph experiment follows from this control.

The source graph is the Graphalytics dense relabelled Friendster release at
<https://doi.org/10.5281/zenodo.5146230> (CC BY 4.0). Original torch-sparse source
notices and pins are retained under `third_party/native_notices`.
