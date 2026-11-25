# Public E7/E12 fresh-run adapters

These new adapters replace historical filesystem and timestamp admission with
full public numerical identity checks. Original E7/E12 sources, metadata,
checkpoint declarations, cohort selection and scientific functions are retained
byte-identically. No historical timestamp or completion receipt is forged.

Use the recorded Linux numerical environment: Python 3.10.12, NumPy 1.24.4,
SciPy 1.10.1 and Numba 0.58.0. The packet contains exact scientific helper
sources and the small original plan/freeze/config/cohort files. It contains no
full graph, weights or full model-score files. It requires the public Friendster
prepared data reconstructed with the source artifact's public recipe.

Inspect a plan first; this reads no graph and creates no output:

```sh
python -B run_public.py --experiment e7 \
  --prepared-root /absolute/path/to/friendster/prepared \
  --out /absolute/path/to/new_public_e7
python -B run_public.py --experiment e12 \
  --prepared-root /absolute/path/to/friendster/prepared \
  --out /absolute/path/to/new_public_e12
```

Append `--execute` to perform a fresh reproduction. The input tree must contain
all 23 prepared numerical files, including the full training CSR and original
10,000-row evaluation arrays. Initial admission streams SHA256 over exactly
18,788,800,187 bytes with a 4 MiB buffer. It makes no graph copy. Metadata paths
and generated times may differ from the historical environment.

Admission checks the unchanged original plan/freeze against the exact public
quality manifest; six validation-selected checkpoint identities are bound by
that already verified asset, while structural evaluation never opens models.
It verifies the exact 512 validation and 4,096 test query IDs, positives, warm
masks and all 1,000 provided negatives per row. The missing original test-negative
subset NPY is never created: its full expected NPY SHA is reconstructed in
roughly 1 MiB blocks from the public candidate arrays. Repeated physical source
vertices retain their separate original query rows.

E7 invokes the original `main`, replacing only `load_inputs`. Its 512-query
shared-priority visited-policy audit retains seed 20261125, cap 200, the
2,000,000-neighbor/10,000-node/1,000-control-node guards, 4 GiB scratch budget,
one Numba thread and 3,600-second scientific wall budget. Guarded and incomplete
denominators remain explicit.

E12 invokes the original `run`, replacing only `verify_test_gate` and
`load_inputs`. Its five fixed structural scores, midpoint ties, all 4,096
queries/all 1,000 negatives, eight Numba threads, 20-billion search-step guard,
180 GiB conservative memory budget and 64-query chunks remain unchanged.
The derived in-memory admission plan records actual new file paths and stat
observations; it is never saved over the original plan or freeze.

Both routines retain original numerical implementations and write their new
results below a fresh wrapper output directory. Additional `PUBLIC_ADMISSION.json`
and `COMPLETE.json` identify the new gate and bind the original sources. Failure
produces `FAILED.json`; it cannot produce a wrapper completion. Numba cache stays
inside the new output. Original source/metadata/input paths are read-only.

Use an idle Linux CPU host with sufficient RAM for the read-only 12.1 GB train
CSR, imported numerical libraries and the declared work budgets. The complete
23-file input check includes about 18.8 GB of storage, even though the evaluation
does not read every file again. Initial full hashes are followed by actual
size/mtime/inode/device continuity checks; no old mtime is required or set.

Local tiny tests exercise a real unchanged E12 kernel against an independent
structural oracle, E7 priority/guard semantics, exact streamed NPY hashes,
whole-corpus corruption despite unchanged size/mtime, and restored gate bindings
after failure. A real full-scale CPU rerun is a separate, explicitly admitted
operation; preparation of this packet alone is not a new result or timing claim.
