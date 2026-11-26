"""Small stdlib-only contracts shared by the E25 controller and app."""
from pathlib import Path
import hashlib,json,os,time
HERE=Path(__file__).resolve().parent
GIB=1<<30
ARMS=('reference_union_compiled','reference_union_native_persistent')
HORIZONS={'short320':{'steps':10,'measured_graphs':320,'consumed_graphs':576,'checkpoints':[32,256,576]},
 'long6400':{'steps':200,'measured_graphs':6400,'consumed_graphs':6656,'checkpoints':[32,256,1280,3456,6656]}}
MODEL_SHA='f88254079fe6c152b4c0153c857a63516348353d578f918622e7f6719559618e'
WORKLOAD_SHA='340e61cdd1eb8d329d52be164f919480e6a41feb4c620852df0393c2e9a192f0'
PROTOCOL={'name':'friendster_native_lifecycle_prospective_allocation_blocks_v6','datasets':['friendster'],'arms':list(ARMS),
 'workers':[1,8],'repetitions':3,'seed':20261124,'order_seed':20261126,'batch_size':32,'warmup_steps':8,
 'horizons':HORIZONS,'timing_jobs':24,'hops':2,'cap':200,'prefetch_factor':1,
 'long_semantic_prefix576':'full input SHA snapshot only; no added CUDA sync or timing checkpoint',
 'sampling':'unchanged reference_union, original pair_seed and ordered S',
 'model':'unchanged featureless DGCNN32/3/k281/maxz1000 Adam1e-4',
 'input_equality':'typed/shaped ordered CPU PyG batch stream SHA and exact query indices for every consumed graph; short576 must equal long prefix576; hash time included without subtraction',
 'setup':'worker_init only sets Torch threads; lazy arm-specific setup, natural first-use compiled JIT/cache load and fresh native promotion/state initialization inside app clock',
 'cache':'Fresh output and Numba cache for each allocation block; shared within that block including qualification. OS/PVC caches retained, not cleared; no prefault, eviction or cold-cache claim',
 'timing':'common checkpoint-synchronized E25; each short/long job genuinely terminates; interior synchronized prefixes are not independently terminated lifetimes or original E23 windows',
 'native':'unchanged standalone persistent NativeCSR and pack_local; no PENCIL NativeGraph/Facade or sampler',
 'cpu_memory_gib':180,'gpu_allocator_gib':24,'sampler_seconds':2,'max_block_seconds':3900,'outer_block_seconds':4200,'max_job_seconds':2400,
 'controller_guard_amendment':{'version':'latched_disk_guard_v4','scope':'Fresh complete8-app blocks only; all three complete global repetitions required for24, no partial-attempt pooling',
  'hard_negatives':'Controller deadline/environment/disk/output errors latch once, persist a bounded failure receipt when storage permits, and abort owned processes independently of transient PSS sampling errors; no throttle or successful sample clears them',
  'disk_observer':'At actual approximately2s controller checks, bounded fsynced JSONL records statvfs bytes/inodes available and readonly flag on output backing filesystem;8MiB history cap; observer cost included without subtraction',
  'output_observer':'Non-atomic sampled output-byte cap, not instantaneous quota; one lstat per entry, vanished entries skipped and counted in bounded latest inventory receipt; observed bytes and known negatives never heal',
  'disk_limits':'4GiB live reserve and2GiB non-runtime cap on the actual admitted output backing filesystem; no reservation or node eviction-threshold guarantee',
  'storage_admission':{'schema_kind':'E25_V6_ACTUAL_BLOCK_PLATFORM_ADMISSION',
   'binding':'Separate actual root proof JSON and whole SHA supplied by --platform-admission-json/--platform-admission-sha256, bound in each selected block plan and copied byte-exactly; no actual platform hash in this global protocol',
   'preblock_minimum_bytes_available':40*GIB,'runtime_minimum_bytes_available':4*GIB,
   'scope':'Actual Pod/host/boot/GPU/CPU pool/quota/memory/topology and root-bound PVC/image metadata fixed within each block. Source/graph/results/native/temp/cache use admitted project mount. Storage placement can affect timing; no reservation or continuous/exclusive health/bandwidth claim.'}},
 'allocation_blocks':{'version':'prospective_three_allocations_v6','global_applications':24,'blocks':[1,2,3],'applications_per_block':8,
  'selection':'--repetition-block selects the original global repetition in the original full24 randomized order; original r1/r2/r3 names and all within-block pairs retained',
  'platform':'Fresh actual allocation and root platform proof per block; Pod/GPU/CPU pool fixed within a block and allowed to differ between blocks',
  'statistics':'Four horizon/worker configuration rows per block, two raw arm values each; paired ratios only within block. No fake three-repetition summaries in a single block.',
  'completion':'All three independently complete8-app blocks required for global24 aggregate. Timed-out/failed blocks remain failed and are never salvaged, appended or pooled with earlier cohorts; no automatic retry.',
  'deadline':'Producer3900s/outer4200s caps are admission/failure limits, not a speed or completion guarantee'},
 'environment_policy':{'version':'fixed27_env_v5','logical_cpu_count':27,'selection':'highest27 currently allowed online logical CPU IDs, chosen once before admission/build/qualification; optional root-bound E25_EXPECTED_CPU_AFFINITY pins the same exact list for prelaunch qualification and controller',
  'quota':'exact cpu.max2700000 100000','mutation':'one controller affinity pin before admission; children inherit; app/worker guards never repin',
  'invariants':'same host/boot/cgroup identity and memory limit; chosen CPUs remain in effective cpuset and online; externally observed process leaders and runtime threads have nonempty subset masks; own controller/app/worker boundary leaders retain exact chosen mask; any observed empty/outside mask or quota change aborts',
  'leader_envelope_amendment':{'version':'external_leader_containment_v5','scope':'External descendant-tree observations admit nonempty leader subsets inside chosen27; all own boundary scans still require the exact27 leader mask and unchanged PID/start/cgroup',
   'motivation':'Separate aborted ENVv4 cohort observed one new single-task descendant leader at[141]inside chosen27 during worker startup; all other observed masks/resources valid. Runtime startup is a plausible explanation, not a proven OpenMP mechanism or outside-envelope escape.',
   'no_healing':'Every observed empty/outside task/leader mask remains an immediate hard negative even in unstable scans; no extra retry, repin, affinity mutation or science/supervision/order change',
   'claims':'Fixed27 CPU envelope plus exact owner boundary gates; not exact27 for every external descendant leader at every sample or continuous/exclusive placement',
   'cohort':'All prior failed cohorts remain diagnostic only; prospectively chosen three fresh8-app blocks, fresh output/Numba cache per block, all24 required for aggregate, no failed-cohort pooling'},
  'observer':'read-only named-predicate boundary guards plus complete before/after PID/TID/cgroup brackets in CPU/PSS observations approximately every2s; metadata work included, no subtraction; ambient pool changes outside chosen mask recorded',
  'boundary_rescan':{'max_attempts':8,'max_wall_seconds':0.5,'pause_seconds':0.01,'eligibility':'Only ancillary task membership/observation churn with known unchanged own PID/start, exact leader mask, known exact cgroup and every observable hard predicate true; eventual fully stable PASS required within both caps',
   'failure':'Any observed resource/mask/PID/TID/cgroup negative fails immediately without another attempt; budget exhaustion fails; no repin or repair; all attempts retained, not just final PASS',
   'wall_budget_scope':'From before own identity read through terminal decision; a scan returning after deadline cannot PASS. OS I/O can overrun before failure is reported; not a hard real-time cancellation guarantee.'},
  'limits':'shared27CPU time quota and27logical CPU mask are not exclusive physical cores, absence of contention, bandwidth reservation, or continuous observation',
  'cohort':'Three prospectively specified fresh8-app allocation blocks cover the original global24 grid; fresh output/Numba cache per block, no failed/partial-cohort pooling'},
 'scope':'two physical induction routes within one reference-sampling DGCNN consumer; no quality, full-upstream or population inference'}
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
  p=HERE/name;require(p.resolve().is_relative_to(HERE)and pin(p)==value,'prepared E25 source differs: '+name)
 return m
