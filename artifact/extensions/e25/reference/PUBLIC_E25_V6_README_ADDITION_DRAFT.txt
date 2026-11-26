## E25 v6: recorded audit and replay scope

E25 v6 splits the original 24-application protocol into three fresh allocation
blocks, each containing eight applications. Paired timing ratios stay within
their allocation block; hardware/topology and retained OS/filesystem-cache
differences between allocations are disclosed. Source readiness, tiny
qualification, one completed block and a verified complete24 cohort are separate
states. Failed or partial historical attempts supply no timing observations.

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
