Native reuse and two-graph application lifetimes, prospective closure v1

This new source extends the published E25 frozen sampler/model/native core.
It neither changes E25 results nor pools them with these new measurements.
Two source-pinned public inputs: full Friendster and strict-clean Citation2
trainCSR. Workloads are the original fixed featureless E23 query sequences.
The numerical stack, sampling law, ordered selected vertices, DRNL labels,
DGCNN, optimizer, prefetch and consumed-input hashing are retained.

Design per dataset: compiled, native_fresh and native_reuse; eight spawn workers;
short320/long6400 measured graphs after256warmup; three prospective allocation
blocks. Each invocation runs6 fresh applications; all3 complete blocks yield18
perdataset,36 acrossbothgraphs. Failed/partial attempts remain excluded, no
blind retry or pooling with historical E25. Native preparation is once per
DATASET, outside ALL fresh application clocks, and retained across3blocks.
Nativefresh still writes/fsyncs its own int64 arrays inside each appclock.
Bothnativearms sample from compactCSR and allocate one private mutable8N
mapper perworker inside theclock; readonly int64 arrays may sharefilepages.

One-time preparation, CPU only (no data download/YT mutation):
 python3 -B prepare_native.py --dataset friendster --graph-root CANONICAL_CSR \
   --out FRESH_REUSE_ROOT --max-seconds2400
The converter verifies the compact source and all derived native bytes against
independently established E20/E22 identities. It records promotion/fsync and
separate full-validation durations and produces readonly rowptr.i8,col.i8 and
READY.json. Host/boot identifies preparation clockscope: never compare its
monotonic timestamps across another host/boot. Retain the exact receipt whole
SHA; admission copies it byte-for-byte to each allocation's raw outputs.

Controller, PLAN ONLY without --execute:
 python3 -B run_lifecycle.py --dataset friendster --graph-root CANONICAL_CSR \
   --native-reuse-root IMMUTABLE_REUSE_ROOT \
   --native-reuse-receipt-sha256 WHOLE_READY_JSON_SHA \
   --out FRESH_BLOCK_OUTPUT --coordination-dir SHARED_TIMING_COORDINATION \
   --repetition-block 1 --platform-admission-json ROOT_PLATFORM_JSON \
   --platform-admission-sha256 WHOLE_ROOT_PLATFORM_SHA
To execute, add
--source-manifest-sha256 EXACT_FROZEN_SOURCE_SHA --execute. Dataset citation2
uses the same interface with its distinct pinned graph and workload. Default
and maximum block duration3900s; outer4200s cap remains root-owned. These are
failure limits, not speed guarantees. No idle-stop bypass or dummy GPU work.

Preparation st_dev is retained as a historical host-local value. Each current
controller/app/worker binds reusable files to the current allocation PVC device;
bytes, mtime, inode and full content identities remain unchanged.

Root supplies CLOSURE_V1_ACTUAL_BLOCK_PLATFORM_ADMISSION for the exact dataset
and block, with actual Pod/STS/image/PVC/device/mount/host/boot/GPU,27CPU pool,
2700000/100000 quota and256GiB limit. Source/input/reuse/output/cache/proof paths
must stay on admitted writable PVC. Initial pinned native-array bytes plus6GiB free and runtime4GiB reserves,
latched hard failures, observed180GiB process-tree memory budget and output
caps remain. Existingcache requires extra storage and nativefresh additionally
requires its complete derived arrays plus6GiB free before launch. Nothing
writes fullgraphs to laptop or nativeprepared storage to YT.

Original E25 fixed27_env_v5 CPU/task containment and genuine Popen-to-actual-wait
clocks remain. Sampled PSS uses complete stable same-phase process-tree samples;
missing observations stay missing, maxima are non-atomic/partial-residency.
Common graph/native full-integrity scans and build/qualification remainoutside
application clocks and can warm pages. Numba cache isfresh perallocation and
shared within it including tiny qualification. OS/PVC caches are retained.
The preparednative graph isfile-backed, not a fully prefaulted/shared-memory
promotion arm; no cold-I/O, universalimplementation or time-to-quality claim.

Ordering: PythonRandom(20261127) seededbasepermutation per dataset/horizon;
shuffleconsumption datasets friendster,citation2 thenshort320,long6400;
blockr rotatesbase byr-1. Horizonordersshort,long inblocks1/3,long,short inblock2.
Root allocationorder F1,C1,F2,C2,F3,C3. Everyarmoccupieseverypositiononce per
horizon; no outcome-dependentordering. Ratiospair withinexactdataset/block.

Each output retains plan.json, byte-exact PLATFORM_ADMISSION.json and
native_reuse_PREPARATION_READY.json, completeinput_identity.json,
native_reuse_admission.json, launch/exit/PSS and worker/app records, freshnative
post-exit content checks and scratch-onlycleanup, reusablefinalfullhashes,
scientific_summary.json, SCIENTIFIC_TERMINAL.json and COMPLETE.json. Every
consumed typed/shaped/ordered CPUinput and queryID ishashed before H2D. Exact
prefix576 equality issemantic only forlongapps; no addedlong576CUDAcheckpoint.
Root must separately retain genuine wrapper/controller exit, quiescence/lock,
collection/rawarchive and resource-release proofs. No fabricated/root-derived
platform identities or independent tensor-regeneration claim is permitted.
