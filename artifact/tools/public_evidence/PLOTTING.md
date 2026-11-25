# Recreate the three public figures

This is a derived plotting adapter, separate from the original experimental
sources. It recomputes plotted quantities from recorded public data and checks
them against the saved analyses. It performs no new benchmark, training or
inference. It writes a fresh output directory and does not modify manuscript
files, input assets or source packages.

Use Python 3.10 or later, NumPy and Matplotlib. Local verification used Python
3.12, NumPy 1.26.4 and Matplotlib 3.10.3. The original experiment's numerical
environment is preserved separately; plotting does not rerun its kernels.

After extracting the public systems evidence asset, run:

```sh
python -B plot_public_figures_v1.py \
  --asset-root /absolute/path/to/extracted/systems/root \
  --quality-asset /absolute/path/to/lpp_public_quality_evidence_v1.tgz \
  --out /absolute/path/to/a/new/figures/directory
```

The systems root must contain `MANIFEST.json` and the logical `E1`, `E4`, `E8`
and `E17` evidence paths declared in that manifest. The separate quality asset
is pinned to its exact 404,147,012-byte original file and SHA256. The adapter
streams only twelve small reciprocal-rank/query-ID NPY files and the manifest
from it. It never extracts or loads model weights or full candidate-score
NPZs. The only NPZ loaded is the 81 KB degree geometry, with an additional
8 MiB uncompressed safety bound.

The output contains three PNGs and `NUMERIC_PANELS.json` with recomputed values,
input hashes, source identity and generated figure hashes. Matplotlib's font
cache stays inside the new output directory. At least 1.5 GiB free disk reserve
plus a 16 MiB rendering allowance is required. Existing output is refused.

- E8 uses the median of eight query-wise means of five recorded repetitions,
  for each virtual N and SciPy version. Forced-int64 reports verify the same
  logical inputs; the displayed curves use native-int32 temporaries.
- E17 derives every point from the 768 original Friendster scan/pair timing
  records and verifies the public analysis, including correlation and limited
  hindsight/proxy headroom. No new timings are produced.
- E4 aligns all 4,096 original query IDs, retains the 44 cold queries, and
  recomputes six degree-bin differences for all three paired training seeds.
  Error bars are the sample SD of those three differences.

PNG bytes can differ with rendering libraries and fonts. Numerical panels must
match the recorded inputs. These figures do not establish independent-sample
confidence intervals, causal effects, time-to-quality or predictor superiority.
