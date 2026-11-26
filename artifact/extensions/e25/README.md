# E25 v6: source and offline verification

**Results pending.** This addition contains frozen source, source reviews and
synthetic metadata checks. It contains no actual v6 block archive, root collection
sidecar, per-block audit, complete-24 aggregate, scientific figure or new timing
result. Source readiness and synthetic checks establish no experiment outcome.
The v4 tree remains archived source and author-deployment provenance;
this v6 entry point supplies the v6 offline CLI.

This is the v6 source entry point for `artifact/extensions/e25/`. The source,
notices and source reviews are prepared; actual v6 results remain pending.
`COPY_LINEAGE.json` records the exact v6 source-copy origins;
`RELOCATION_LINEAGE.json` records the byte-preserved v4 relocation and this
README revision. `SOURCE_STAGE_MANIFEST.json` catalogs the full current tree.

## Archived ENV v4 source and provenance

`historical/env_v4/` preserves the entire previous E25 extension tree byte for
byte, including its source archives, notices, auditor, source reviews and
original author preflight/qualification/schema records. These are historical
source/provenance materials, not a completed v6 result or an active v6 entry point.
They contribute no application timings to the new three-block cohort.

The archived README and source comments retain their original path examples,
including `$ARTIFACT/extensions/e25/source/packet/run_lifecycle.py` and
`$ARTIFACT/extensions/e25/independent/audit_e25_env_v4.py`. Those examples refer
to the original v4 directory layout before relocation. They must not be followed
against the current v6 root. The archived v4 four-hour single-cohort CLI, source/protocol
hashes, v4 auditor with `independent_disk.py`, original platform snapshots and
future v4 figure/collection instructions apply only to that archived version.
Relative links inside the preserved documents also retain their original layout
meaning; their bytes have not been rewritten to turn v4 into a new replay path.
Use the v6 offline audit and aggregation commands below. The current author-bound
replay limits also apply; archived historical prose establishes no portable v6
deployment or actual-platform qualification.

- `source/packet/` and `source_archives/` preserve the exact frozen producer.
- `source/outer/` preserves the exact author-bound wrapper and its readiness.
- `independent/` supplies the standard-library block auditor with its adjacent
  hash-pinned `independent_env.py`, readiness and source-peer report.
- `aggregate/` supplies the standard-library three-block aggregation tool and
  clearly labeled synthetic metadata test source/report.
- `reference/admit_block_platform_v6.py.txt` preserves the exact root admission
  helper as reference text. It is author-bound provenance, not a public launcher.

The MIT `LICENSE` covers the authors' own additions. Exact upstream notices remain
in `source/packet/third_party/` and `source/packet/native_core/third_party/`.
See `DATA_ATTRIBUTION.md` for the single small public workload exception.
The frozen `THIRD_PARTY.md` retains historical artifact-wide references to
`DATA_ACCESS.md`, `PACKAGING_NOTES.md`, `metadata/PRIOR_WORK_PROVENANCE.json`,
`metadata/SYSTEMS_FROZEN_INPUTS.json` and PENCIL source paths. Resolve these at the
base artifact root; they are not claims that unrelated PENCIL material is bundled
inside this E25 source addition. `reference/` scripts are not replay entry points.

## E25 v6: recorded audit and replay scope

E25 v6 splits the original 24-application protocol into three fresh allocation
blocks, each containing eight applications. Paired timing ratios stay within
their allocation block; hardware/topology and retained OS/filesystem-cache
differences between allocations are disclosed. Source readiness, tiny
qualification, one completed block and a verified complete24 cohort are separate
states. Failed or partial historical attempts supply no timing observations.

### Prepared one-command recorded verifier

`verify_recorded_v6.py` and [OFFLINE_VERIFICATION.txt](OFFLINE_VERIFICATION.txt)
provide a single entry point to the existing standard-library block auditor and
three-block aggregator. This is prepared source with static review only. The real
recorded catalog, all three genuine blocks and a genuine public-layout roundtrip
remain pending; this source preparation creates no catalog or result placeholders
and establishes no operational verification or complete-24 result.

After verifying the complete published code archive with its externally supplied
release SHA256, and once its real recorded catalog and all three input sets are
included, set `ARTIFACT` to the absolute canonical artifact directory and
`E25_FRESH_OUT` to a new directory outside it whose parent already exists:

```sh
python3 -B -S "$ARTIFACT/extensions/e25/verify_recorded_v6.py" \
  --artifact "$ARTIFACT" --out "$E25_FRESH_OUT"
```

The prepared helper checks the fixed catalog/input/source identities and original
PASS8/PASS24 metadata before creating fresh output, then requires regenerated
audits and aggregate to match the cataloged bytes and SHA256 values. The exact
catalog schema and trust boundary are documented in `OFFLINE_VERIFICATION.txt`.
The reviewed source record is `reference/PUBLIC_VERIFY_ENTRYPOINT_SOURCE_REVIEW.json`;
a genuine all-three-block roundtrip must still be performed before claiming this
entry point works on the released evidence.

### Offline audit of a recorded block

Use Python 3.10+ and the standard library. No GPU, graph download, ML packages,
container image, cluster account or live Kubernetes access is required. Keep
audit_block_v6.py and the supplied independent_env.py in the same directory.

Each block needs both its original block_raw.tgz and its separate original root
collection JSON. The latter records observations made after the raw archive,
including actual wrapper exit0 and owned-tree quiescence; it cannot be replaced
by an in-archive COMPLETE. Obtain both whole SHA256 values from the release
catalog/checksums anchored by the verified release, not by trusting an unchecked
archive's self-manifest.

Set E25_V6_AUDITOR to the directory containing the two auditor files. For each
of blocks 1, 2 and 3, set E25_BLOCK_RAW and E25_BLOCK_COLLECTION to the published
files for that block; set E25_BLOCK_RAW_SHA256 and E25_BLOCK_COLLECTION_SHA256
to their cataloged whole hashes. E25_NEW_AUDIT must be a new output filename
in an existing writable directory. Run:

```sh
python3 -B "$E25_V6_AUDITOR/audit_block_v6.py" \
  --archive "$E25_BLOCK_RAW" \
  --archive-sha256 "$E25_BLOCK_RAW_SHA256" \
  --root-collection "$E25_BLOCK_COLLECTION" \
  --root-collection-sha256 "$E25_BLOCK_COLLECTION_SHA256" \
  --output "$E25_NEW_AUDIT"
```

The helper streams and rehashes the archive without extraction, checks frozen
source/platform/exit bindings, and independently recomputes application-clock
metrics, within-block ratios and sampled PSS eligibility/maxima from saved raw
records. CPU/disk uncertainty does not filter PSS; missing observations remain
unobserved. It does not remeasure performance, independently regenerate consumed
tensor bytes, or establish current cluster state. A successful report says
PASS_ACTUAL_E25_V6_EIGHT_APP_BLOCK_AUDIT and global24_complete=false. Run it for
all three blocks; three per-block checks must still be reconciled by the
release's separately documented complete-cohort aggregation. No global24 audit
command is supplied by this per-block tool.

### What a fresh replay requires

The frozen producer/envelope and root admission sources document the measured
deployment. They are not a turnkey portable Kubernetes launcher. In particular,
the envelope checks the authors' original PVC UID/subPath and internal image
digest. That image reference and the saved platform JSON are provenance; they
are not public access instructions or valid evidence for another allocation.
The producer's plan CLI also requires genuine actual-platform JSON.

The measured harness requires Linux x86_64, Python 3.10.12, a C++17 compiler,
NumPy 1.24.4, SciPy 1.10.1, Numba 0.58.0, Torch 2.5.0+cu121 and PyG 2.7.0,
with the declared public environment extras. It admits one idle whole 80GB
H100, an actual pool of 27 logical CPUs, quota 2700000/100000 and a 256GiB
cgroup memory limit, with readable cgroup-v2/proc/sysfs observations. These
controls are not claims of minimum hardware requirements; no CPU, MIG or DDP
fallback has been established.

Use the existing public Friendster full-CSR preparation recipe, not the quality
split. Source, graph, proof, results and temp/cache paths must use the admitted
writable project mount. At least 40GiB must remain free before each block,
above the prepared graph; the runtime reserve is 4GiB, with a separate 4GiB
writable-root check. Each native app writes and fsyncs a fresh
29,421,941,096-byte representation inside its application clock. Outputs and
Numba cache are fresh per block; OS/filesystem caches are retained. The
3900-second producer and 4200-second envelope caps are failure limits, not
completion guarantees.

External replay therefore needs a separately reviewed admission/deployment
binding that records that operator's real platform and preserves the scientific
protocol. Such a portable binding is not supplied or claimed tested here.
Do not substitute fabricated Pod/PVC identities or reuse the authors' proof.
Measurements on a different qualified platform are new observations, not a
promise to reproduce the authors' latency.

### Offline complete-cohort aggregation

After all three genuine archive/collection pairs are independently audited, set
`E25_V6_AGGREGATOR` to the directory containing `aggregate_blocks_v6.py`. Set the
three `E25_AUDIT_*` filenames and whole hashes from the verified release catalog,
`E25_AUDITOR_SOURCE_SHA256` to the cataloged exact block-auditor source hash, and
`E25_NEW_AGGREGATE` to a fresh output filename. Run:

```sh
python3 -B "$E25_V6_AGGREGATOR/aggregate_blocks_v6.py" \
  --block-audit "$E25_AUDIT_1" "$E25_AUDIT_1_SHA256" \
  --block-audit "$E25_AUDIT_2" "$E25_AUDIT_2_SHA256" \
  --block-audit "$E25_AUDIT_3" "$E25_AUDIT_3_SHA256" \
  --auditor-source-sha256 "$E25_AUDITOR_SOURCE_SHA256" \
  --output "$E25_NEW_AGGREGATE"
```

The aggregator checks all three complete block audits, exact scientific/auditor
source pins, distinct original allocations/archives, the complete original job
grid, and cross-block source-bound input/prefix identity. Paired ratios are formed
within blocks. Across-allocation median, min/max and sample SD are descriptive;
retained OS/PVC caches and platform differences do not support a population
uncertainty or causal hardware claim. PSS preserves missing values and coverage.

For source-only synthetic checks, keep `aggregate/source_review_aggregate_v6/`
next to `aggregate/aggregate_blocks_v6.py`, set `E25_SYNTHETIC_CHECK_OUTPUT` to a
fresh filename, then run:

```sh
python3 -B "$E25_V6_AGGREGATOR/source_review_aggregate_v6/synthetic_metadata_checks.py" \
  --output "$E25_SYNTHETIC_CHECK_OUTPUT"
```

These checks create only a labeled source-test outcome. Their in-memory fixtures
are synthetic, and no actual aggregate report or scientific result is serialized.
The supplied synthetic-review JSON is not an actual block/complete-24 audit.

Actual raw archives, post-archive root collection sidecars, independently pinned
audit reports and aggregate evidence will be added only after genuine collection,
audit and public-data/credential review. No placeholders stand in for them here.
