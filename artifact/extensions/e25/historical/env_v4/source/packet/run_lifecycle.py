#!/usr/bin/env python3
"""E25 Friendster lifecycle controller. Plan-only by default; no network/YT."""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib.metadata,importlib.util,json,math,os,platform,random,shutil,signal,stat,subprocess,sys,time,uuid
from pathlib import Path
from common import *
import environment as environment

class ControllerGuardError(RuntimeError):
 """A latched controller negative, never a transient sampling failure."""

DISK_OBSERVATION_CAP=8<<20

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

def configs(args):
 rng=random.Random(PROTOCOL['order_seed']);first=list(ARMS);rng.shuffle(first)
 combinations=[(h,w)for h in HORIZONS for w in(1,8)]
 base={key:first if index%2==0 else list(reversed(first))for index,key in enumerate(combinations)};jobs=[]
 for repetition in range(1,4):
  current=combinations[:];rng.shuffle(current)
  for horizon,w in current:
   definition=HORIZONS[horizon];order=base[horizon,w]if repetition%2 else list(reversed(base[horizon,w]))
   for position,arm in enumerate(order,1):
    name=f'{horizon}_w{w}_r{repetition}_{arm}';out=args.out/'jobs'/name
    config={'horizon':horizon,'fixture':False,'graph_path':str(args.friendster_root),'file_endian':'little','pairs_path':str(args.out/'friendster_pairs.npy'),
     'seed':20261124,'hops':2,'cap':200,'oversample':1.5,'auto_ratio':1.0,'max_neighbors':20000000,'warmup_pairs':256,
     'steps':definition['steps'],'prefetch_factor':1,'audit_pairs':32,'device':'cuda','torch_threads':1,'gpu_memory_budget_gib':24.0,
     'stock_index_dtype':'auto','batch_size':32,'num_workers':w,'arm':arm,'out':str(out),
     'worker_receipts':str(out/'workers'),'native_root':str(out/'runtime/native'),'native_library':str(args.out/'build/native_assoc_v1.so'),
     'checkpoints':definition['checkpoints'],'semantic_input_prefixes':sorted(set(definition['checkpoints']+[576]))}
    jobs.append({'name':name,'horizon':horizon,'workers':w,'repetition':repetition,'position':position,'arm':arm,'config':config})
 require(len(jobs)==24,'exact24-job grid required');return jobs

def plan(args):
 verify_sources()
 return {'kind':'E25_FRIENDSTER_TWO_ARM_LIFECYCLE_ENV_V2','execute':args.execute,'protocol':PROTOCOL,
  'protocol_sha256':hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest(),
  'source_manifest_sha256':sha(HERE/'SOURCE_MANIFEST.json'),'friendster_root':str(args.friendster_root),
  'out':str(args.out),'coordination_dir':str(args.coordination_dir),'max_hours':args.max_hours,
  'jobs':configs(args),'environment_policy':environment.POLICY,'selected_cpu_affinity':environment.selected(),
  'no_periodic_YT_writes':True,'graph_or_model_reads_in_plan':0}

class Controller:
 def __init__(self,args,definition):
  self.args,self.plan=args,definition;self.records=[];self.current=None;self.closed=False
  self.deadline=time.monotonic()+args.max_hours*3600
  self.supervisor=module(HERE/'support/owned_supervision.py','e25_owned_supervision')
  self.memory=module(HERE/'frozen_src/scripts/seal_revision/memory_benchmark_v3.py','e25_memory_guard')
  self.helpers=module(HERE/'frozen_src/experiment_configs/seal_revision_2026-11/run_pipeline_suite_v4.py','e25_gpu_admission')
  self.env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
   PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',NUMBA_CACHE_DIR=str(args.out/'shared_numba_cache'),WANDB_MODE='disabled',HF_HUB_OFFLINE='1')
  self.hardware=None;self.snapshots={};self.last_disk_check=0.;self.environment=None;self.environment_failed=False
  self.controller_guard_failure=None;self.last_disk_observation=None
 def environment_guard(self,label):
  require(self.environment is not None,'ENVv2 admission must precede guards')
  try:observed=environment.guard(self.environment,label,self.args.out/'ENVIRONMENT_FAILURE.json')
  except BaseException:self.environment_failed=True;raise
  environment.append(self.args.out/'environment_events.jsonl',observed);return observed
 def status(self,phase,**extra):save(self.args.out/'status.json',{'phase':phase,'monotonic_ns':time.monotonic_ns(),**extra})
 def cache_inventory(self):
  root=self.args.out/'shared_numba_cache';rows=[]
  if root.exists():
   for p in sorted(root.rglob('*')):
    require(not p.is_symlink(),'shared cache symlink refused')
    if p.is_file():
     st=p.stat();rows.append({'name':str(p.relative_to(root)),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
  return {'files':rows,'count':len(rows),'total_bytes':sum(r['bytes']for r in rows),
   'metadata_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
   'cache_payloads_read_or_prefaulted_by_inventory':False,'observed_monotonic_ns':time.monotonic_ns()}
 def latch_controller_guard(self,predicate,error):
  if self.controller_guard_failure is None:
   self.controller_guard_failure={'status':'FAIL','kind':'LATCHED_CONTROLLER_HARD_GUARD',
    'predicate':predicate,'error_type':type(error).__name__,'message':str(error)[:2000],
    'observed_monotonic_ns':time.monotonic_ns(),'disk_observation':self.last_disk_observation,
    'source_manifest_sha256':self.plan['source_manifest_sha256'],'clearing_or_retry_allowed':False}
   try:save(self.args.out/'CONTROLLER_GUARD_FAILURE.json',self.controller_guard_failure)
   except Exception as receipt_error:
    # Readonly/full/unavailable storage may prevent persistence; the in-memory
    # negative and owned-process abort must still survive that second error.
    self.controller_guard_failure['receipt_error_type']=type(receipt_error).__name__
  raise ControllerGuardError('latched controller hard guard: '+self.controller_guard_failure['predicate'])from error
 def disk_observation(self):
  path=self.args.out;value=os.statvfs(path)
  observation={'observed_monotonic_ns':time.monotonic_ns(),'path':str(path),'device':path.stat().st_dev,
   'bytes_available':value.f_bavail*value.f_frsize,'bytes_total':value.f_blocks*value.f_frsize,
   'inodes_available':value.f_favail,'inodes_total':value.f_files,'inodes_reported':value.f_files>0,
   'mount_readonly':bool(value.f_flag&os.ST_RDONLY),'scope':'statvfs of current output backing filesystem; no capacity reservation or node eviction-threshold claim'}
  self.last_disk_observation=observation
  raw=(json.dumps(observation,sort_keys=True,allow_nan=False)+'\n').encode();target=path/'disk_observations.jsonl'
  require(len(raw)<=4096 and not target.is_symlink(),'bounded regular disk observation required')
  require(not target.exists()or(target.is_file()and target.stat().st_size+len(raw)<=DISK_OBSERVATION_CAP),'disk observation8MiB cap reached')
  with target.open('ab')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
  return observation
 def check(self):
  if self.controller_guard_failure is not None:
   raise ControllerGuardError('latched controller hard guard: '+self.controller_guard_failure['predicate'])
  predicate='controller_deadline'
  try:
   require(time.monotonic()<self.deadline-30,'finite total controller deadline reached')
   if time.monotonic()-self.last_disk_check<2:return False
   self.last_disk_check=time.monotonic();predicate='controller_environment'
   if self.environment is not None:self.environment_guard('periodic_controller')
   predicate='disk_observation';observed=self.disk_observation()
   predicate='disk_readonly';require(not observed['mount_readonly'],'output backing filesystem is readonly')
   predicate='disk_free_bytes';require(observed['bytes_available']>=4*GIB,'fourGiB live disk reserve reached')
   predicate='disk_free_inodes';require(not observed['inodes_reported']or observed['inodes_available']>0,'output backing filesystem has no available inodes')
   predicate='output_inventory';total=0;files=0;vanished=0
   for p in self.args.out.rglob('*'):
    try:entry=p.lstat()
    except FileNotFoundError:vanished+=1;continue
    require(not stat.S_ISLNK(entry.st_mode),'output symlink refused')
    if stat.S_ISREG(entry.st_mode)and 'runtime'not in p.relative_to(self.args.out).parts:
     total+=entry.st_size;files+=1
   inventory={'observed_monotonic_ns':time.monotonic_ns(),'disk_observed_monotonic_ns':observed['observed_monotonic_ns'],
    'observed_non_runtime_bytes':total,'observed_non_runtime_files':files,'vanished_entries':vanished,
    'complete_entry_observations':vanished==0,'scope':'Non-atomic sampled output-byte cap, not an instantaneous quota; vanished entries are skipped and counted, observed bytes are never subtracted.'}
   observed['output_inventory']=inventory
   predicate='output_bytes';require(total<=2*GIB,'twoGiB non-runtime output cap exceeded')
   predicate='output_inventory_receipt';require(len(json.dumps(inventory,sort_keys=True,allow_nan=False).encode())<=4096,'output inventory4KiB cap reached')
   save(self.args.out/'output_inventory_observation.json',inventory)
   return True
  except Exception as error:self.latch_controller_guard(predicate,error)
 def idle(self):
  h=self.helpers;inventory=h.permitted_inventory(h.parse_inventory(h.nvidia_query('--query-gpu=index,uuid,name,memory.total,driver_version')),self.env)
  target=h.select_target(inventory,self.env);hardware={'permitted_inventory':inventory,'target':target}
  require(self.hardware is None or self.hardware==hardware,'GPU/driver inventory changed');self.hardware=hardware
  observations=[]
  for index in range(2):
   self.check();apps=h.parse_compute_apps(h.nvidia_query('--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory'))
   require(not any(x['gpu_uuid']in{v['uuid']for v in inventory}for x in apps),'GPU busy; no foreign signals sent')
   observations.append({'unix':time.time(),'busy_apps':[],'environment':self.environment_guard('gpu_idle_'+str(index))})
   if index==0:time.sleep(15)
  self.env['CUDA_VISIBLE_DEVICES']=target['uuid'];save(self.args.out/'hardware.json',hardware);return observations
 def admission(self):
  require(sys.platform.startswith('linux')and sys.version_info[:2]==(3,10)and platform.machine()=='x86_64','real target Linuxx86_64Python3.10 required')
  self.environment=environment.admit(self.plan['selected_cpu_affinity'],failure_path=self.args.out/'ENVIRONMENT_FAILURE.json');self.env.update(environment.exported(self.environment))
  for job in self.plan['jobs']:job['config']['environment_contract']=self.environment
  save(self.args.out/'plan.json',self.plan)
  save(self.args.out/'ENVIRONMENT_ADMISSION.json',{'status':'PASS','policy':environment.POLICY,
   'chosen_cpu_affinity':self.environment['chosen_cpu_affinity'],'contract':self.environment,
   'after':self.environment_guard('controller_admission'),'scope':'Fixed inherited logical CPU mask; shared quota/pool, no exclusive or continuous-placement guarantee.'})
  pins=read(HERE/'SOURCE_PINS.json');versions={name:importlib.metadata.version(name)for name in pins['runtime_versions']}
  require(versions==pins['runtime_versions'],'original E23 numerical/model stack required')
  save(self.args.out/'host.json',{'hostname':platform.node(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
   'cpu_affinity':sorted(os.sched_getaffinity(0)),'python':sys.version,'versions':versions})
  files=read(HERE/'pins/friendster_full_expected.json')['files']
  for name,expected in files.items():
   p=self.args.friendster_root/name;require(pin(p,self.check)==expected,'whole public Friendster input differs: '+name)
   st=p.stat();self.snapshots[str(p)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev,'sha256':expected['sha256']}
  save(self.args.out/'input_identity.json',{'whole_hashes':True,'files':self.snapshots})
  shutil.copyfile(HERE/'workloads/friendster_pairs.npy',self.args.out/'friendster_pairs.npy')
  require(sha(self.args.out/'friendster_pairs.npy')==WORKLOAD_SHA,'original public workload changed')
 def assert_inputs(self):
  verify_sources();self.check()
  self.environment_guard('assert_inputs')
  for name,value in self.snapshots.items():
   s=Path(name).stat();require((s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)==tuple(value[k]for k in('bytes','mtime_ns','inode','device')),'original graph stat changed')
  require(sha(self.args.out/'friendster_pairs.npy')==WORKLOAD_SHA,'query stream changed')
 def command(self,name,argv,cwd,timeout=2400,env_extra=None,phase_folder=None):
  self.check();self.environment_guard('before_command_'+name)
  log=self.args.out/(name+'.log');samples=self.args.out/(name+'.pss.jsonl');abort=[None];last=[0.];errors=[0];launch={};child=[None]
  def clock():return time.monotonic()+(10**9 if abort[0]else 0)
  def popen(*values,**kwargs):
   self.environment_guard('immediately_before_Popen_'+name)
   kwargs['env']=dict(self.env,**(env_extra or{}));start=time.monotonic_ns()
   kwargs['env']['E25_LAUNCH_MONOTONIC_NS']=str(start)
   child[0]=subprocess.Popen(*values,**kwargs);self.current=child[0]
   observed=identity(child[0].pid);require(observed is not None,'actual child PID/start unavailable')
   launch.update(pid=child[0].pid,start_ticks=observed['start_ticks'],launch_monotonic_ns=start,argv=argv)
   save(self.args.out/(name+'.launch.json'),launch);return child[0]
  def phase():
   if phase_folder is None:return 'qualification_or_build'
   p=phase_folder/'phase.json';return read(p)['phase']if p.exists()else'interpreter_and_imports'
  def groups(owner):
   current=self.supervisor.owned_groups(owner)
   # Hard controller negatives run outside the transient sampler-error handler.
   # A latched negative is checked before the throttle and cannot be cleared.
   try:self.check()
   except ControllerGuardError:
    abort[0]=abort[0]or('controller hard guard: '+self.controller_guard_failure['predicate']);return current
   if child[0]is not None and child[0].poll()is None and time.monotonic()-last[0]>=2:
    last[0]=time.monotonic()
    try:
     before=phase();sample=self.memory.sample_tree(child[0].pid);after=phase()
     decision=self.memory.budget_decision(sample,180*GIB,self.memory.cgroup_memory());abort[0]=abort[0]or decision['abort_reason']
     members,complete=self.memory.tree_members(child[0].pid)
     env_observation=environment.observe(self.environment,[{'pid':p,'start_ticks':s}for p,s in sorted(members.items())])if complete and members else{
      'status':'TRANSIENT_PROCESS_TREE_UNOBSERVED','policy':environment.POLICY,'observed_monotonic_ns':time.monotonic_ns()}
     if env_observation['status']=='FAIL':
      self.environment_failed=True;save(self.args.out/'ENVIRONMENT_FAILURE.json',env_observation)
      abort[0]=abort[0]or('ENVv2 '+','.join(env_observation['failed_predicates']))
     sample.update(phase_before=before,phase_after=after,phase_stable=before==after,
      phase_publication_eligible=sample['publication_eligible']and before==after,monotonic_ns=time.monotonic_ns(),decision=decision,
      environment=env_observation)
     errors[0]=0
    except Exception as exc:
     errors[0]+=1;sample={'sampling_error':type(exc).__name__,'monotonic_ns':time.monotonic_ns()}
     if self.environment_failed:abort[0]=abort[0]or'ENVv2 controller guard failed; see ENVIRONMENT_FAILURE.json'
     if errors[0]>=3:abort[0]=abort[0]or'three consecutive external sampling failures'
    with samples.open('a')as stream:stream.write(json.dumps(sample,allow_nan=False)+'\n')
   if log.exists()and log.stat().st_size>8<<20:abort[0]='stdout8MiB cap'
   return current
  result=self.supervisor.run_owned(argv,log,cwd=cwd,timeout=min(timeout,self.deadline-time.monotonic()-60),
   clock=clock,popen=popen,groups_reader=groups)
  result.update(guard_abort_reason=abort[0],actual_wait_returned=True,wait_returned_monotonic_ns=time.monotonic_ns(),
   start_ticks=launch['start_ticks'],launch_monotonic_ns=launch['launch_monotonic_ns'],guard_policy=self.memory.GUARD_POLICY,
   PSS_scope='external sampled app process tree, complete stable membership only; file residency partial, not full-resident layout or RSS sum')
  result['launch_to_actual_wait_seconds']=(result['wait_returned_monotonic_ns']-result['launch_monotonic_ns'])/1e9
  save(self.args.out/(name+'.exit.json'),result);self.current=None
  self.environment_guard('after_actual_wait_'+name)
  require(type(result['returncode'])is int and result['returncode']==0 and not result['timed_out']and not result['orphan_cleanup']
   and result['owned_tree_quiescent']is True and not result['owned_pgids_signaled']and abort[0]is None,'owned child failed/guarded/cleanup required: '+name)
  return result
 def native_postcheck_and_cleanup(self,job):
  root=Path(job['config']['native_root']);expected=read(HERE/'pins/friendster_native_expected.json');observed={}
  for name,size_key,sha_key in(('rowptr.i8','rowptr_bytes','rowptr_sha256'),('col.i8','col_bytes','col_sha256')):
   p=root/name;require(p.is_file()and not p.is_symlink()and p.stat().st_size==expected[size_key],'fresh derived native bytes differ')
   observed[name]=pin(p,self.check);require(observed[name]['sha256']==expected[sha_key],'full post-app native content hash differs')
  save(Path(job['config']['out'])/'native_post_exit_integrity.json',{'files':observed,'outside_application_lifetime':True,
   'observed_monotonic_ns':time.monotonic_ns(),'expected_proof':expected})
  # Exactly the known fresh owned scratch after actual app wait/quiescence;
  # never search/delete a graph input or another job's runtime.
  require(root.resolve()==root and root.is_relative_to(self.args.out/'jobs')and root.name=='native','owned canonical native scratch only')
  require({p.name for p in root.iterdir()}=={'rowptr.i8','col.i8','READY.json'},'unexpected scratch entries; preserve them')
  for name in('rowptr.i8','col.i8','READY.json'):(root/name).unlink()
  root.rmdir();save(Path(job['config']['out'])/'native_scratch_cleanup.json',{'owned_files_removed':3,'after_actual_exit_and_hash':True,'monotonic_ns':time.monotonic_ns()})
 def validate(self,job,result):
  folder=Path(job['config']['out']);r=read(folder/'report.json');config=job['config']
  horizon=HORIZONS[job['horizon']];consumed=horizon['consumed_graphs']
  require(r['status']=='complete'and r['config']==config and r['consumed_graphs']==consumed and r['measured_graphs']==horizon['measured_graphs'],'full E25 consumer completion required')
  require(r['launch_monotonic_ns']==result['launch_monotonic_ns']and r['entry_monotonic_ns']>=result['launch_monotonic_ns'],'same-kernel actual launch clock differs')
  require([x['total_trained_graphs']for x in r['checkpoints']]==horizon['checkpoints']
   and len({x['label']for x in r['checkpoints']})==len(horizon['checkpoints']),'exact predeclared synchronized checkpoints required')
  times=[p['monotonic_ns']for p in r['checkpoints']]
  require(r['first_batch_ready_ns']<=times[0]and all(b>a for a,b in zip(times,times[1:]))
   and times[1]<=r['steady_start_ns']<times[2]and r['training_end_ns']==times[-1]
   and r['shutdown_begin_ns']>=times[-1]and r['shutdown_end_ns']>=r['shutdown_begin_ns']
   and result['wait_returned_monotonic_ns']>=r['shutdown_end_ns'],'ordered true prefix/shutdown/wait milestones required')
  require(all(abs(p['launch_to_complete_seconds']-(p['monotonic_ns']-r['launch_monotonic_ns'])/1e9)<1e-10 for p in r['checkpoints'])
   and r['input_stream_sha256']==r['checkpoints'][-1]['input_stream_sha256'],'direct cumulative clocks and final stream digest required')
  require(r['model']['initial_state_sha256']==MODEL_SHA and len(r['losses'])==horizon['steps']and all(math.isfinite(v)and v>=0 for v in r['losses']),'initial model and all finite losses required')
  require(r['device']['type']=='cuda'and'H100'in r['device']['name']and r['gpu_memory_guard']['allocator_budget_bytes']==24*GIB
   and r['gpu_reserved_peak_gib']<=24,'one H100 and original24GiB allocator contract required')
  require(r['completed_extractions_including_unconsumed_prefetch']>=consumed
   and r['completed_extractions_including_unconsumed_prefetch']<=consumed+config['num_workers']*32,'bounded consumed plus unconsumed tail accounting required')
  workers=r['worker_receipts'];require(len(workers)==config['num_workers']and len({w['pid']for w in workers})==config['num_workers']
   and all(w['status']=='closed'and w['native_closed']is True for w in workers),'all spawn workers normal-close proof required')
  require(all(r[k]['status']=='PASS'for k in('environment_start','environment_end'))
   and all(w[k]['status']=='PASS'for w in workers for k in('environment_start','environment_end')),'ENVv2 app/worker boundary guards required')
  if job['arm']==ARMS[1]:
   require(r['native_promotion']['rowptr_bytes']+r['native_promotion']['col_bytes']==29421941096,'one fresh full native promotion required')
   require(all(w['initialization_count']==1 and w['native_initialization']['persistent_assoc_bytes_per_worker']==524866928 for w in workers),'exact one private8N map per native worker required')
   require(all(w['native_initialization']['owner_pid']==w['pid']and w['native_initialization']['library_sha256']==sha(self.args.out/'build/native_assoc_v1.so')
    and w['native_arrays']['readonly']is True and w['native_arrays']['rowptr_bytes']+w['native_arrays']['col_bytes']==29421941096 for w in workers),'actual per-worker library/readonly representation identity differs')
  else:require(r['native_promotion']is None and all(w['initialization_count']==0 for w in workers),'compiled must not allocate native state')
  return r
 def summarize(self,reports):
  import statistics
  metrics={};pss=[]
  for record in self.records:
   r=reports[record['name']];values={'launch_to_first_batch_ready_seconds':r['launch_to_first_batch_ready_seconds'],
    'launch_to_actual_wait_seconds':record['exit']['launch_to_actual_wait_seconds'],
    'checkpointed_measured_seconds':r['checkpointed_measured_seconds'],'worker_shutdown_seconds':r['worker_shutdown_seconds'],
    'input_hash_seconds_included':r['input_hash_seconds_included']}
   values.update({p['label']+'_launch_to_complete_seconds':p['launch_to_complete_seconds']for p in r['checkpoints']})
   metrics[record['name']]=values;phases={}
   for line in(self.args.out/(record['name']+'.pss.jsonl')).read_text().splitlines():
    sample=json.loads(line)
    if sample.get('phase_publication_eligible')is True:
     phase=sample['phase_before'];entry=phases.setdefault(phase,{'eligible_samples':0,'max_sampled_tree_pss_bytes':0})
     entry['eligible_samples']+=1;entry['max_sampled_tree_pss_bytes']=max(entry['max_sampled_tree_pss_bytes'],sample['tree_pss_bytes'])
   pss.append({'name':record['name'],'phases':phases,'scope':'Stable complete process-tree samples only, sampled non-atomic maxima; no total RSS sum, cache subtraction or full-resident claim.'})
  def stats(values):return {'values':values,'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),
   'median':statistics.median(values),'minimum':min(values),'maximum':max(values)}
  rows=[];ratios=[]
  for horizon in HORIZONS:
   for w in(1,8):
    byarm={}
    for arm in ARMS:
     chosen=sorted((r for r in self.records if r['horizon']==horizon and r['workers']==w and r['arm']==arm),key=lambda r:r['repetition'])
     require([r['repetition']for r in chosen]==[1,2,3],'all3 fresh lifecycle process repetitions required')
     byarm[arm]={key:[metrics[r['name']][key]for r in chosen]for key in metrics[chosen[0]['name']]}
     rows.append({'horizon':horizon,'workers':w,'arm':arm,'metrics':{key:stats(value)for key,value in byarm[arm].items()}})
    ratios.append({'horizon':horizon,'workers':w,'numerator':ARMS[0],'denominator':ARMS[1],
     'metrics':{key:stats([a/b for a,b in zip(byarm[ARMS[0]][key],byarm[ARMS[1]][key])])for key in byarm[ARMS[0]]},
     'scope':'paired same-horizon/worker/repetition descriptive ratios;>1 means shorter observed native lifetime for this metric, not a population speedup.'})
  return {'status':'complete','jobs':24,'rows':rows,'paired_compiled_over_native':ratios,'sampled_PSS':pss,
   'scope':'Actual short and long jobs genuinely terminate; interior synchronized prefixes are not independent lifetimes. Intervals are observed directly, never assembled by adding overlapping worker service timers; no historical E23 timing equivalence.'}
 def run(self):
  a=self.args;a.out.mkdir();self.supervisor.enable_subreaper();lock=None;active=None;token=None;success=False;failure=None
  def interrupted(sig,frame):raise KeyboardInterrupt('E25 controller signal '+str(sig))
  previous={sig:signal.signal(sig,interrupted)for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
  signal.setitimer(signal.ITIMER_REAL,a.max_hours*3600-15)
  try:
   a.coordination_dir.mkdir(parents=True,exist_ok=True);lockpath=a.coordination_dir/'PUBLIC_SYSTEMS_RUNNING.lock'
   require(not lockpath.is_symlink(),'timing lock must not be symlink');lock=lockpath.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   active=a.coordination_dir/'PUBLIC_SYSTEMS_ACTIVE.json';require(not os.path.lexists(active),'unknown predecessor active marker preserved')
   token={'pid':os.getpid(),'start_ticks':identity(os.getpid())['start_ticks'],'nonce':uuid.uuid4().hex,'source_manifest_sha256':self.plan['source_manifest_sha256']}
   with active.open('x')as stream:json.dump(token,stream)
   save(a.out/'plan.json',self.plan);self.status('admission');self.admission();self.status('build_native')
   self.command('build_native',[sys.executable,'-B',str(HERE/'native_core/build_native_v1.py'),'--output-dir',str(a.out/'build')],HERE,180)
   save(a.out/'cache_before_qualification.json',self.cache_inventory())
   self.status('actual_cuda_qualification');self.idle();self.command('actual_cuda_qualification',[sys.executable,'-B',str(HERE/'qualify.py'),'--library',str(a.out/'build/native_assoc_v1.so'),
    '--out',str(a.out/'qualification'),'--device','cuda','--workers','1','8'],HERE,900)
   qualification=read(a.out/'qualification/QUALIFIED.json');require(qualification['success']is True and qualification['device']=='cuda'
    and qualification['workers']==[1,8]and qualification['actual_consumers']==4,'actual GPU tiny1/8 qualification required')
   save(a.out/'cache_after_qualification.json',self.cache_inventory())
   reports={}
   for job in self.plan['jobs']:
    self.status('running',job=job['name'],completed_jobs=len(self.records))
    self.assert_inputs();idle=self.idle();folder=Path(job['config']['out']);folder.mkdir(parents=True)
    if job['arm']==ARMS[1]:require(shutil.disk_usage(folder).free>=29421941096+6*GIB,'fresh native copy plus6GiB reserve required')
    config=folder/'config.json';save(config,job['config'])
    save(folder/'cache_before.json',self.cache_inventory())
    outcome=self.command(job['name'],[sys.executable,'-B','-u',str(HERE/'lifecycle_child.py'),'--config',str(config)],HERE,2400,phase_folder=folder)
    r=self.validate(job,outcome)
    if reports:
     reference=next(iter(reports.values()))
     require(r['semantic_input_prefixes']['576']==reference['semantic_input_prefixes']['576']
      and r['excluded_warmup_prefix_audits']==reference['excluded_warmup_prefix_audits'],'immediate hard gate: same576 input prefix across every app')
     same_horizon=[value for value in reports.values()if value['config']['horizon']==job['horizon']]
     if same_horizon:
      reference=same_horizon[0]
      require(r['input_stream_sha256']==reference['input_stream_sha256']and r['semantic_input_prefixes']==reference['semantic_input_prefixes'],
       'immediate hard gate: same full consumer stream within each horizon')
    save(folder/'cache_after.json',self.cache_inventory())
    if job['arm']==ARMS[1]:self.native_postcheck_and_cleanup(job)
    reports[job['name']]=r;self.records.append({'name':job['name'],'horizon':job['horizon'],'arm':job['arm'],'workers':job['workers'],'repetition':job['repetition'],
     'report_sha256':sha(folder/'report.json'),'exit':outcome,'GPU_idle_observations':idle})
    save(a.out/'job_records.json',self.records)
   comparisons=[]
   for horizon in HORIZONS:
    for w in(1,8):
     for rep in(1,2,3):
      pair=[j for j in self.plan['jobs']if j['horizon']==horizon and j['workers']==w and j['repetition']==rep]
      left,right=[reports[j['name']]for j in sorted(pair,key=lambda j:ARMS.index(j['arm']))]
      require(left['input_stream_sha256']==right['input_stream_sha256']and left['excluded_warmup_prefix_audits']==right['excluded_warmup_prefix_audits']
       and left['semantic_input_prefixes']==right['semantic_input_prefixes'],'all consumed inputs/checkpoint prefixes must match across physical routes')
      comparisons.append({'horizon':horizon,'workers':w,'repetition':rep,'consumed_graphs':HORIZONS[horizon]['consumed_graphs'],
       'all_consumed_inputs_exact':True,'stream_sha256':left['input_stream_sha256']})
    require(len({r['input_stream_sha256']for r in reports.values()if r['config']['horizon']==horizon})==1,'same full input stream required within every horizon')
   short=next(r for r in reports.values()if r['config']['horizon']=='short320')
   long=next(r for r in reports.values()if r['config']['horizon']=='long6400')
   require(len({r['semantic_input_prefixes']['576']for r in reports.values()})==1
    and short['input_stream_sha256']==long['semantic_input_prefixes']['576'],'short final must match long prefix576 across all apps')
   cross={'consumed_input_graphs':576,'short_final_stream_sha256':short['input_stream_sha256'],
    'long_prefix_stream_sha256':long['semantic_input_prefixes']['576'],'all_arms_workers_repetitions_exact':True}
   self.assert_inputs();save(a.out/'scientific_summary.json',self.summarize(reports))
   save(a.out/'SCIENTIFIC_TERMINAL.json',{'success':True,'jobs':24,'stream_comparisons':comparisons,'cross_horizon_input_prefix576':cross,'records':self.records,
    'scope':PROTOCOL['scope'],'timing_protocol':PROTOCOL['timing'],'no_new_quality_claim':True,'YT_writes':0})
   success=True
  except BaseException as error:
   failure={'type':type(error).__name__,'message':str(error)[:2000]}
  finally:
   signal.setitimer(signal.ITIMER_REAL,0)
   for sig in(signal.SIGTERM,signal.SIGKILL):
    self.supervisor.signal_confirmed(self.supervisor.owned_groups(os.getpid()),sig)
    until=time.monotonic()+5
    while time.monotonic()<until and self.supervisor.owned_groups(os.getpid()):time.sleep(.1)
   quiet=not self.supervisor.owned_groups(os.getpid());owned=active is not None and active.is_file()and read(active)==token
   save(a.out/'COMPLETE.json',{'success':success and quiet and owned,'controller_finished':True,'owned_tree_quiescent':quiet,
    'environment_policy':environment.POLICY,'environment_guard_success':self.environment is not None and not self.environment_failed,
    'environment_admission_sha256':sha(a.out/'ENVIRONMENT_ADMISSION.json')if(a.out/'ENVIRONMENT_ADMISSION.json').is_file()else None,
    'failure':failure,'controller_guard_failure':self.controller_guard_failure,'records':self.records,'controller_identity':token,'source_manifest_sha256':self.plan['source_manifest_sha256'],
    'elapsed_seconds':a.max_hours*3600-(self.deadline-time.monotonic()),'YT_writes':0,'statefulset_actions':0,
    'scope':'This terminal precedes own future OS exit and lock.close; root must separately observe actual exit/quiescence/lock availability.'})
   if quiet and owned:active.unlink()
   if lock is not None:lock.close()
   self.status('complete'if success and quiet and owned else'failed',completed_jobs=len(self.records),failure=failure)
   for sig,old in previous.items():signal.signal(sig,old)
  return 0 if success and quiet and owned else 1

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--friendster-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
 parser.add_argument('--coordination-dir',type=Path,required=True);parser.add_argument('--max-hours',type=float,default=4);parser.add_argument('--source-manifest-sha256');parser.add_argument('--execute',action='store_true')
 a=parser.parse_args();a.friendster_root=a.friendster_root.resolve();a.out=a.out.resolve();a.coordination_dir=a.coordination_dir.resolve()
 require(math.isfinite(a.max_hours)and 0<a.max_hours<=4,'positive maximum4hours required');require(not os.path.lexists(a.out),'fresh output required')
 for p in(HERE,a.friendster_root):require(not a.out.is_relative_to(p)and not p.is_relative_to(a.out),'source/input/output trees must be disjoint')
 require(not any(a.coordination_dir.is_relative_to(p)for p in(HERE,a.friendster_root,a.out)),'coordination outside source/input/output trees required')
 definition=plan(a);print(json.dumps(definition,indent=2),flush=True)
 if not a.execute:return 0
 require(a.source_manifest_sha256==definition['source_manifest_sha256'],'root-reviewed manifest SHA required')
 return Controller(a,definition).run()
if __name__=='__main__':raise SystemExit(main())
