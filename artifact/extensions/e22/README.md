# E22: public Citation2 matched CPU extension

This separate extension adds the recorded seven-arm Citation2 CPU control to
the public SEAL-style execution study. It is a local publication candidate;
the already published v0.5.0 code/tag/assets are unchanged. E23 pipeline-repeat
results are not included or claimed here.

## Verify the recorded result without scientific dependencies

From this directory, run with ordinary Python 3.10 or newer:

```sh
python3 verify_e22.py --output E22_OFFLINE_VERIFICATION.json
```

The verifier uses only the Python standard library. It checks every selected
payload against `RESULT_CATALOG.json`, streams and checks the original raw
archive, verifies the deployed source/cohort/input/environment/conversion and
exit bindings, and recomputes all seven E22 table rows from raw repetition
vectors. Edge+DRNL times are paired within each repetition before taking the
per-case median; those 128 medians are then summed. The result is compared with
the exact Citation2 columns in `recorded/integration_numbers.json`.

No graph arrays, model weights or native libraries are loaded, no numerical
package is imported, and no network, VK, YT, Kubernetes or author path is used.
The source archive's original paths/timestamps are preserved as observations;
they are never opened by the verifier. Decompressed JSON/source data is bounded
to 16 MiB. The historical Friendster columns are supplied for table integration
from their previously validated E20 report; this verifier recomputes E22 only.

`recorded/E22_RESULT_AUDIT.json` is the prior independent actual-result audit.
Its presence is not used as a substitute for checking the raw archive.
`recorded/post_exit_lifecycle.json` binds the successful outer OS exit, process
quiescence and owned timing-lock release to the exact archived result.

## Data and source

The public data source is the original
[OGB Citation2 archive](https://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip).
Its locally measured official-download SHA256 is
`f8502b935ea8f0a7bea5eb130991ca337cd26cad7f231cc9f5546fe076f4be53`
(2,293,454,318 bytes; this is not claimed as a provider-published checksum).

Use the existing strict-clean preparation recipe in the
[v0.5.0 public artifact](https://github.com/makut/link-prediction-execution-costs-artifact/releases/tag/v0.5.0).
Its [`artifact/PUBLIC_COMMANDS.txt`](https://github.com/makut/link-prediction-execution-costs-artifact/blob/v0.5.0/artifact/PUBLIC_COMMANDS.txt)
documents public acquisition and exact numeric regeneration. No private source
is needed. The relevant existing command is
`python -m artifact_public.reproduce_citation2 --work-parent "$DATA/c2_public" --threads 8`,
under that artifact's supplied preparation environment/instructions.

Fresh E22 execution consumes an already prepared strict-clean `train_csr`
directory with only these two files:

| Payload | Bytes | SHA256 |
|---|---:|---|
| edge_starts.bin | 23423704 | f06070adc368892f8441ea15e691dce400f9d44b6ec3e6e4505134c276c1072d |
| edge_ends.bin | 242814016 | d3d8d1e2016675321c1d3d9ae817d2c6dca4a9dd2d5cdbba3e638ea7201057a4 |

These encode N=2,927,963 vertices and M=60,703,502 stored entries. Features,
timestamps, held-out candidates and quality-model files are not needed.
The similar `prepared_original` graph is not interchangeable and will fail
whole-file admission. Both source arrays are read-only; no source copy occurs.

The scientific/execution source under `fresh/frozen_e22` is the exact actual E22
source: 39 payloads and its manifest. The source manifest SHA256 is
`d30698edff9d9d6b1d959e78bedc7ea226359fb667941e70f43a87945443012b`.
`fresh/ADAPTER_LINEAGE.json` records reuse of the v0.5.0 E20 public wrapper.
Only operational path binding, the Citation2 budget/labels and fixed-cohort
staging differ. No scientific kernel or source record has been rewritten.

## Fresh Linux replay of the frozen cohort

This optional procedure produces new timing measurements on the reader's
machine. It does not promise historical latency values. It requires Linux
x86_64 Python 3.10 with Torch 2.2.2+cu121, NumPy 1.24.4, SciPy 1.10.1,
Numba 0.58.0, pip/venv and a C++17 compiler. It uses CPU only.

Supply the exact two-wheel packet already published with v0.5.0:
[`pt22_native_wheels_v1.tgz`](https://github.com/makut/link-prediction-execution-costs-artifact/releases/download/v0.5.0/pt22_native_wheels_v1.tgz),
15,064,339 bytes, SHA256
`5e9edbdc067325cd0b41b5fb292e2cc7343cc535da87ff3bef720cbf85da38cf`.
The runner does no downloading or package installation into the supplied
environment. It creates a fresh isolated environment and installs only those
two exact official native wheels into that environment, offline and without
dependencies. Base and isolated source/binary/version probes run before/after.

Set ordinary local paths. `CPU_PY` retains the supplied interpreter's lexical
path if it belongs to a venv. `E22_OUT` must be new and disjoint from inputs and
sources. `TIMING_COORDINATION` is a writable directory shared with other
cooperating physical benchmarks on that host.

```sh
"$CPU_PY" fresh/run_public_cpu.py \
  --python "$CPU_PY" --train-csr "$C2_TRAIN_CSR" \
  --wheel-packet "$CPU_NATIVE_WHEELS" --out "$E22_OUT" \
  --coordination-dir "$TIMING_COORDINATION"
```

The default prints a source-verified plan, with no graph-array read, output,
environment, lock or job creation. Review it, obtain the SHA256 of
`fresh/SOURCE_MANIFEST.json`, then add the pin and `--execute`:

```sh
"$CPU_PY" fresh/run_public_cpu.py \
  --python "$CPU_PY" --train-csr "$C2_TRAIN_CSR" \
  --wheel-packet "$CPU_NATIVE_WHEELS" --out "$E22_OUT" \
  --coordination-dir "$TIMING_COORDINATION" \
  --source-manifest-sha256 "$E22_PORTABLE_SOURCE_PIN" --execute
```

The complete original cohort is copied byte for byte from
`fresh/recorded_cohort`. No node set or query is resampled. Its 128 cases use
80 distinct physical pairs and 126 distinct ordered sets across 12 strata;
they are not 128 independent queries. All seven arms use identical S and
edge/DRNL oracles, one excluded warmup and three measured repetitions. Actual
installed upstream and standalone compiled native fixtures must pass before
derived whole-graph allocation. Numba32/64 use the original compiled scan and
pair-probing functions. No model, sampler experiment or quality test is run.

At least 8 GiB of free scratch is admitted initially. The single derived int64
representation has two arrays totalling 509,051,728 bytes. The persistent map
is 23,423,704 private bytes per worker, with initialization outside query time.
Conversion, verification, compilation and warmup are separately reported.
The original 1 GiB native / 8 GiB conservative peak / 0.05 GiB association
budgets remain; the sampled external CPU guard defaults to 8 GiB. The runner
also checks current host/cgroup headroom, retains a 1.5 GiB disk reserve and
limits raw evidence to 96 MiB. Runtime arrays, wheels, venv and caches remain
under `E22_OUT/runtime`, outside that raw-evidence budget.

Native arms and width-specific Numba workers run sequentially in fresh processes
with one numerical thread and the
same observed host/boot/CPU affinity. Shared flock plus an owned active marker
prevents overlapping cooperating runs; stale/unknown markers are retained.
The six-hour total deadline and owned PID/start-time cleanup are inherited
from the reviewed E20 public wrapper. Failures and partial outputs are preserved.
Successful replay writes raw reports and `public_cpu7_raw.tgz` under `E22_OUT`.
Its current controller cannot invent its own future outer OS exit; preserve
the actual launcher exit separately when collecting a fresh experiment.

## Interpretation

The recorded E22 persistent native route has the smallest sum of per-case
medians for both edge-only and edge+DRNL boundaries. Scan has lower aggregate
time than pair probing in this Citation2 cohort. These are graph-specific,
repeated-cache observations with unpinned broad CPU affinity and differing
low-level output materialization. Neither the ratios nor logical array/map
bytes establish language superiority, first-batch or complete SEAL speedup,
time-to-quality, a new PSS result or a population confidence interval.

The offline verifier has been run on the actual recorded E22 archive. The new
portable Linux replay wrapper is supplied for reproduction; a full replay has
not been executed by this extension's preparation. Its actual target fixtures
and resource admission remain mandatory. Original own-code MIT and upstream
notices are retained under `fresh/frozen_e22`.
