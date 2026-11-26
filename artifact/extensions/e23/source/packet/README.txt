Independent-process repetitions of the public V4 pipeline control

Scientific purpose
Quantify process/window variability of the same three physical induction substitutions. The selected graph queries, extraction RNG, initial model seed and train-step protocol remain fixed. Each scientific job runs in a fresh benchmark process and a fresh child training process. These are repeated-cache, fixed-workload execution repetitions, not independent graph samples or quality training seeds.

Predeclared grid
Friendster full public CSR and strict-clean Citation2; workers1/8; reference_union_stock/reference_union_corrected/reference_union_compiled; three fresh-process repetitions. Twelve randomized dataset-worker blocks use a cyclic Latin order of the three arms, placing each arm once at each position within each configuration. Runtime seed20261124; order seed20261126. Exactly36 timing jobs,2 manifest checks, one impossible-GPU-capacity refusal and one three-arm tiny GPU fixture:40 jobs. The original two8000-row workload NPY files are included,192128bytes each. They are from the already published public systems evidence and retain exact whole-file hashes.

Unchanged measurement boundary
DGCNN hidden32,3layers,sortpool281,featureless; batch32;8 excluded warmup steps=256graphs;100 measured steps=3200graphs; prefetch1;2hops;cap200; one CUDA-synchronized window per job. Same iterator crosses warmup/window boundaries. The32 audited queries are solely inside excluded warmup. There are36 pairwise prefix comparisons across12 blocks. These do not independently verify every measured graph. Shared OS/Numba disk caches are retained and not evicted; JIT/setup/warmup remain excluded.

Software and input requirements
Linux Python3.10; NumPy1.24.4, SciPy1.10.1, Numba0.58.0, Torch2.5.0+cu121, PyG2.7.0. Original model and source hashes are checked. One whole H100, no MIG/DDP/CPU fallback; idle before each GPU phase. Torch allocator cap24GiB; CPU process-tree outer guard180GiB and finite cgroup90% guard. This program does not install or alter any environment.
Citation2 root must contain num_nodes.txt, train_csr/{edge_starts.bin,edge_ends.bin} and train_pairs_csr/{edge_starts.bin,edge_ends.bin}, matching the included strict-clean pins. Friendster root contains num_nodes.txt and the full graph edge_starts.bin/edge_ends.bin, matching the included public dense graph pins. All eight used inputs are whole-hashed once in bounded2MiB reads, then stat/inode/source/pair hashes are checked between jobs. No node features or held-out negatives are needed.

Plan only (no output directory or GPU required)
python run_pipeline_repeats.py --citation2-root /PUBLIC/CITATION2_PREPARED --friendster-root /PUBLIC/FRIENDSTER_FULL_CSR --out /SCRATCH/NEW_PIPELINE_REPEAT_OUTPUT

Execute only after root resource/identity admission
python -u run_pipeline_repeats.py --citation2-root /PUBLIC/CITATION2_PREPARED --friendster-root /PUBLIC/FRIENDSTER_FULL_CSR --out /SCRATCH/NEW_PIPELINE_REPEAT_OUTPUT --coordination-dir /SCRATCH/SHARED_TIMING_LOCKS --execute

Output must be new and disjoint from source/input trees. There is no automatic overwrite or resume. The6hour total deadline,3600second phase deadline,2GiB total output and1GiB scratch reserve are enforced. Only owned descendants are signaled during cleanup; foreign processes are not killed. A shared local timing lock serializes this queue with other cooperating physical benchmarks on that host. Root manages deployment, persistence, collection and GPU release separately; this runner has no cluster/YT/network code and performs zero YT writes. Persist one coarse final archive rather than per-job snapshots.

Guard and timer scope
The exact historical V4 benchmark retains its parent resource sampler approximately every0.25seconds while its isolated training child runs. No new per-query memory sampling is introduced. The new outer coordinator additionally samples its owned process tree approximately every2seconds using the reviewed V3 PSS/cgroup policy. These parent observations are not separately subtracted from the child wall timer and can consume CPU; the experiment is a controlled repeated-window benchmark with stated instrumentation, not zero-overhead serving throughput.

Validation and reporting
Every timing must have clean outer and inner exits, no abort_reason, correct denominator/native-auto width/source/software/model/workload identity, and complete32-query warmup audit. A status=complete string cannot override failure. Retain every failed or guarded arm; any missing required result prevents a successful scientific summary. The summary exposes all three wall times with descriptive mean/sampleSD/median/range and paired ratios per block. No p-values, population confidence intervals, historical-to-new causal speedup, full upstream application throughput or time-to-quality is claimed. Source/runtime portability fixtures are not scientific real-data results.

Lineage
The benchmark, kernels, model, exact graph code and V4 GPU helper functions are byte-identical copies from the reviewed public v0.5.0 artifact. Only orchestration/validation is new. The old35-job suite is never invoked and no VK or production sampler arm is allowed in this new grid. SOURCE_PINS.json records exact originals; PACKET_MANIFEST.json binds the complete new packet. Original licenses and upstream notices are retained.
