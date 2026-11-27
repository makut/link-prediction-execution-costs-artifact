"""Small stdlib-only contracts shared by the E25 controller and app."""
from pathlib import Path
import hashlib,json,os,time
HERE=Path(__file__).resolve().parent
GIB=1<<30
ARMS=('reference_union_compiled', 'reference_union_native_fresh', 'reference_union_native_reuse')
NATIVE_ARMS=ARMS[1:]
WORKERS=(8,)
HORIZONS={'long6400': {'checkpoints': [32, 256, 1280, 3456, 6656],
              'consumed_graphs': 6656,
              'measured_graphs': 6400,
              'steps': 200},
 'short320': {'checkpoints': [32, 256, 576],
              'consumed_graphs': 576,
              'measured_graphs': 320,
              'steps': 10}}
MODEL_SHA='f88254079fe6c152b4c0153c857a63516348353d578f918622e7f6719559618e'
PROTOCOL={'allocation_blocks': {'version': 'six_prospective_dataset_allocations_v1',
                       'global_applications_per_dataset': 18,
                       'global_two_dataset_applications': 36,
                       'blocks': [1, 2, 3],
                       'applications_per_block': 6,
                       'workers': [8],
                       'selection': 'one dataset and repetition_block per allocation; all6 fresh applications required',
                       'order': 'Seeded arm permutation per dataset/horizon, cyclic block rotation; every arm occupies every '
                                'position once. Horizon order short/long in blocks1/3, long/short inblock2. Dataset allocation '
                                'orderF1,C1,F2,C2,F3,C3.',
                       'statistics': 'Pair arm ratios only within the same dataset/horizon/block; three complete block '
                                     'descriptive medians/ranges, no populationCI.',
                       'completion': 'All three independently complete6-app datasetblocks required pergraph; failed/partial '
                                     'attempts excluded and never appended; no blind retry.',
                       'platform': 'Fresh rootplatform proof perallocation; exact sameGPU/CPUenvelope withinblock, '
                                   'placementmayvarybetweenblocks.'},
 'arms': ['reference_union_compiled', 'reference_union_native_fresh', 'reference_union_native_reuse'],
 'batch_size': 32,
 'cache': 'Reusable native representation prepared once per dataset before all applications, full integrity scans may warm '
          'compact/native file pages. Fresh output/Numba cache per allocation block, shared within block including '
          'qualification. OS/PVC caches retained; no cold-I/O claim.',
 'cap': 200,
 'controller_guard_amendment': {'disk_limits': '4GiB live reserve and2GiB non-runtime cap on the actual admitted output backing '
                                               'filesystem; no reservation or node eviction-threshold guarantee',
                                'disk_observer': 'At actual approximately2s controller checks, bounded fsynced JSONL records '
                                                 'statvfs bytes/inodes available and readonly flag on output backing '
                                                 'filesystem;8MiB history cap; observer cost included without subtraction',
                                'hard_negatives': 'Controller deadline/environment/disk/output errors latch once, persist a '
                                                  'bounded failure receipt when storage permits, and abort owned processes '
                                                  'independently of transient PSS sampling errors; no throttle or successful '
                                                  'sample clears them',
                                'output_observer': 'Non-atomic sampled output-byte cap, not instantaneous quota; one lstat per '
                                                   'entry, vanished entries skipped and counted in bounded latest inventory '
                                                   'receipt; observed bytes and known negatives never heal',
                                'scope': 'Fresh complete6-app datasetblocks; all3blocks pergraph, no failedpartial pooling',
                                'storage_admission': {'binding': 'Separate actual root proof JSON and whole SHA supplied by '
                                                                 '--platform-admission-json/--platform-admission-sha256, bound '
                                                                 'in each selected block plan and copied byte-exactly; no actual '
                                                                 'platform hash in this global protocol',
                                                      'preblock_minimum_bytes_available': {'friendster': 35864392040,
                                                                                           'citation2': 6951502672},
                                                      'runtime_minimum_bytes_available': 4294967296,
                                                      'schema_kind': 'CLOSURE_V1_ACTUAL_BLOCK_PLATFORM_ADMISSION',
                                                      'scope': 'Actual Pod/host/boot/GPU/CPU pool/quota/memory/topology and '
                                                               'root-bound PVC/image metadata fixed within each block. '
                                                               'Source/graph/results/native/temp/cache use admitted project '
                                                               'mount. Storage placement can affect timing; no reservation or '
                                                               'continuous/exclusive health/bandwidth claim.',
                                                      'preblock_minimum_bytes_available_rule': 'Pinned dataset native '
                                                                                               'rowptr_bytes + col_bytes + '
                                                                                               'sixGiB; caches already occupy '
                                                                                               'storage and no old-data cleanup '
                                                                                               'is assumed. Runtime fourGiB '
                                                                                               'reserve remains.'},
                                'version': 'latched_disk_guard_v4'},
 'cpu_memory_gib': 180,
 'datasets': ['friendster', 'citation2'],
 'environment_policy': {'boundary_rescan': {'eligibility': 'Only ancillary task membership/observation churn with known '
                                                           'unchanged own PID/start, exact leader mask, known exact cgroup and '
                                                           'every observable hard predicate true; eventual fully stable PASS '
                                                           'required within both caps',
                                            'failure': 'Any observed resource/mask/PID/TID/cgroup negative fails immediately '
                                                       'without another attempt; budget exhaustion fails; no repin or repair; '
                                                       'all attempts retained, not just final PASS',
                                            'max_attempts': 8,
                                            'max_wall_seconds': 0.5,
                                            'pause_seconds': 0.01,
                                            'wall_budget_scope': 'From before own identity read through terminal decision; a '
                                                                 'scan returning after deadline cannot PASS. OS I/O can overrun '
                                                                 'before failure is reported; not a hard real-time cancellation '
                                                                 'guarantee.'},
                        'cohort': 'Three fresh6-app allocationblocks perdataset; six totalblocks. No pooling with historicalE25 '
                                  'orfailedattempts.',
                        'invariants': 'same host/boot/cgroup identity and memory limit; chosen CPUs remain in effective cpuset '
                                      'and online; externally observed process leaders and runtime threads have nonempty subset '
                                      'masks; own controller/app/worker boundary leaders retain exact chosen mask; any observed '
                                      'empty/outside mask or quota change aborts',
                        'leader_envelope_amendment': {'claims': 'Fixed27 CPU envelope plus exact owner boundary gates; not '
                                                                'exact27 for every external descendant leader at every sample or '
                                                                'continuous/exclusive placement',
                                                      'cohort': 'Historical external-leader containment policy retained; current '
                                                                'prospective cohort is three complete six-application blocks per '
                                                                'dataset, no failed/partial or prior E25 pooling.',
                                                      'motivation': 'Separate aborted ENVv4 cohort observed one new single-task '
                                                                    'descendant leader at[141]inside chosen27 during worker '
                                                                    'startup; all other observed masks/resources valid. Runtime '
                                                                    'startup is a plausible explanation, not a proven OpenMP '
                                                                    'mechanism or outside-envelope escape.',
                                                      'no_healing': 'Every observed empty/outside task/leader mask remains an '
                                                                    'immediate hard negative even in unstable scans; no extra '
                                                                    'retry, repin, affinity mutation or '
                                                                    'science/supervision/order change',
                                                      'scope': 'External descendant-tree observations admit nonempty leader '
                                                               'subsets inside chosen27; all own boundary scans still require '
                                                               'the exact27 leader mask and unchanged PID/start/cgroup',
                                                      'version': 'external_leader_containment_v5'},
                        'limits': 'shared27CPU time quota and27logical CPU mask are not exclusive physical cores, absence of '
                                  'contention, bandwidth reservation, or continuous observation',
                        'logical_cpu_count': 27,
                        'mutation': 'one controller affinity pin before admission; children inherit; app/worker guards never '
                                    'repin',
                        'observer': 'read-only named-predicate boundary guards plus complete before/after PID/TID/cgroup '
                                    'brackets in CPU/PSS observations approximately every2s; metadata work included, no '
                                    'subtraction; ambient pool changes outside chosen mask recorded',
                        'quota': 'exact cpu.max2700000 100000',
                        'selection': 'highest27 currently allowed online logical CPU IDs, chosen once before '
                                     'admission/build/qualification; optional root-bound E25_EXPECTED_CPU_AFFINITY pins the same '
                                     'exact list for prelaunch qualification and controller',
                        'version': 'fixed27_env_v5'},
 'gpu_allocator_gib': 24,
 'hops': 2,
 'horizons': {'long6400': {'checkpoints': [32, 256, 1280, 3456, 6656],
                           'consumed_graphs': 6656,
                           'measured_graphs': 6400,
                           'steps': 200},
              'short320': {'checkpoints': [32, 256, 576], 'consumed_graphs': 576, 'measured_graphs': 320, 'steps': 10}},
 'input_equality': 'typed/shaped ordered CPU PyG batch stream SHA and exact query indices for every consumed graph; short576 '
                   'must equal long prefix576; hash time included without subtraction',
 'long_semantic_prefix576': 'full input SHA snapshot only; no added CUDA sync or timing checkpoint',
 'max_block_seconds': 3900,
 'max_job_seconds': 2400,
 'model': 'unchanged featureless DGCNN32/3/k281/maxz1000 Adam1e-4',
 'name': 'native_reuse_and_two_graph_lifetimes_v1',
 'native': 'Exact same frozen NativeCSR persistent_touched_reset/pack_local in native-fresh and native-reuse; private mutable '
           'mapper per worker; no shared mapper',
 'order_seed': 20261127,
 'outer_block_seconds': 4200,
 'prefetch_factor': 1,
 'repetitions': 3,
 'sampler_seconds': 2,
 'sampling': 'unchanged reference_union, original pair_seed and ordered S',
 'scope': 'Two prospective public datasets and three execution/preparation boundaries under the original reference-union '
          'featureless DGCNN consumer; no quality/full-workflow/cold-cache/population inference',
 'seed': 20261124,
 'setup': 'Native-fresh promotes compactCSR once inside each app; native-reuse opens once-per-dataset immutable preconverted '
          'arrays, prepared and fully validated outside all app clocks. Both native arms allocate private8N mapper inside each '
          'spawnworker. Compiled natural JIT/cache loading unchanged.',
 'timing': 'common checkpoint-synchronized E25; each short/long job genuinely terminates; interior synchronized prefixes are not '
           'independently terminated lifetimes or original E23 windows',
 'timing_jobs': 18,
 'warmup_steps': 8,
 'workers': [8],
 'global_two_dataset_applications': 36}
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(path,check=lambda:None):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for b in iter(lambda:stream.read(4<<20),b''):check();h.update(b)
 return h.hexdigest()
def pin(path,check=lambda:None):
 p=Path(path);require(p.is_file()and not p.is_symlink(),'regular immutable payload required: '+str(p))
 return {'bytes':p.stat().st_size,'sha256':sha(p,check)}
def read(path):
 p=Path(path);require(p.is_file()and not p.is_symlink()and p.stat().st_size<=32<<20,'bounded regular JSON required')
 return json.loads(p.read_text(),parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 tmp.replace(p)
def identity(pid):
 try:
  fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'start_ticks':int(fields[19]),'state':fields[0]}
 except FileNotFoundError:return None
def verify_sources():
 m=read(HERE/'SOURCE_MANIFEST.json')
 observed=set()
 for p in HERE.rglob('*'):
  require(not p.is_symlink(),'source symlink refused')
  if p.is_file()and p.name!='SOURCE_MANIFEST.json':observed.add(p.relative_to(HERE).as_posix())
 require(observed==set(m['files']),'complete source inventory changed')
 for name,value in m['files'].items():
  p=HERE/name;require(p.resolve().is_relative_to(HERE)and pin(p)==value,'prepared closure source differs: '+name)
 return m
