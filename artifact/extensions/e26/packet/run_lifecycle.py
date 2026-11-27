#!/usr/bin/env python3
"""Prospective native reuse/two-graph lifetime controller; plan-only by default."""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib.metadata,importlib.util,json,math,os,platform,random,shutil,signal,stat,subprocess,sys,time,uuid
from pathlib import Path
from common import *
import environment as environment
from bindings import binding
from native_storage import reuse_descriptor, verify_native_files

class ControllerGuardError(RuntimeError):
 """A latched controller negative, never a transient sampling failure."""

DISK_OBSERVATION_CAP=8<<20

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

def configs(args):
 rng=random.Random(PROTOCOL['order_seed']);bases={}
 for dataset_id in PROTOCOL['datasets']:
  bases[dataset_id]={}
  for horizon in ('short320','long6400'):
   order=list(ARMS);rng.shuffle(order);bases[dataset_id][horizon]=order
 jobs=[];b=binding(args.dataset)
 for repetition in range(1,4):
  horizons=['short320','long6400'] if repetition%2 else ['long6400','short320']
  for horizon in horizons:
   definition=HORIZONS[horizon];base=bases[args.dataset][horizon];offset=repetition-1
   order=base[offset:]+base[:offset]
   for position,arm in enumerate(order,1):
    name=f'{args.dataset}_{horizon}_w8_r{repetition}_{arm}';out=args.out/'jobs'/name
    native_root=args.native_reuse_root if arm==ARMS[2] else out/'runtime/native'
    config={'dataset_id':args.dataset,'horizon':horizon,'fixture':False,'graph_path':str(args.graph_root),'file_endian':'little','pairs_path':str(args.out/'pairs.npy'),
     'seed':20261124,'hops':2,'cap':200,'oversample':1.5,'auto_ratio':1.0,'max_neighbors':20000000,'warmup_pairs':256,
     'steps':definition['steps'],'prefetch_factor':1,'audit_pairs':32,'device':'cuda','torch_threads':1,'gpu_memory_budget_gib':24.0,
     'stock_index_dtype':'auto','batch_size':32,'num_workers':8,'arm':arm,'out':str(out),
     'worker_receipts':str(out/'workers'),'native_root':str(native_root),'native_library':str(args.out/'build/native_assoc_v1.so'),
     'native_expected':b['native'],'native_reuse_receipt_sha256':args.native_reuse_receipt_sha256,
     'native_storage_policy':('fresh_inside_application' if arm==ARMS[1] else 'immutable_preconverted_reuse' if arm==ARMS[2] else 'compact_compiled'),
     'checkpoints':definition['checkpoints'],'semantic_input_prefixes':sorted(set(definition['checkpoints']+[576]))}
    jobs.append({'name':name,'dataset_id':args.dataset,'horizon':horizon,'workers':8,'repetition':repetition,'position':position,'arm':arm,'config':config})
 require(len(jobs)==18,'exact18-job prospective per-dataset grid required');return jobs

def plan(args):
 verify_sources()
 path=args.platform_admission_json.resolve();require(path.is_file()and not path.is_symlink()and path.stat().st_size<=32<<20,'bounded regular actual platform proof required')
 require(isinstance(args.platform_admission_sha256,str)and len(args.platform_admission_sha256)==64
  and set(args.platform_admission_sha256)<=set('0123456789abcdef')and sha(path)==args.platform_admission_sha256,'root-bound whole platform proof SHA required')
 proof=read(path);require(proof['kind']=='CLOSURE_V1_ACTUAL_BLOCK_PLATFORM_ADMISSION' and proof['dataset_id']==args.dataset and type(proof['block'])is int and proof['block']==args.repetition_block
  and proof['actual_scientific_jobs_started']==0,'actual platform proof must precede this exact selected block')
 chosen=proof['cpu_affinity'];require(isinstance(chosen,list)and len(chosen)==27 and all(type(v)is int and v>=0 for v in chosen)
  and chosen==sorted(set(chosen)),'actual declared27 CPU pool required')
 global_jobs=configs(args);jobs=[job for job in global_jobs if job['repetition']==args.repetition_block]
 require(len(jobs)==6,'exact six prospective jobs for selected dataset block required')
 for job in jobs:job['config']['native_reuse_expected_device']=proof['storage']['device']
 b=binding(args.dataset);reuse=read(args.native_reuse_root/'READY.json')
 require(sha(args.native_reuse_root/'READY.json')==args.native_reuse_receipt_sha256 and reuse['dataset_id']==args.dataset and reuse['expected_native']==b['native'],'source-bound one-time native reuse proof required')
 return {'kind':'CLOSURE_V1_PROSPECTIVE_REPETITION_BLOCK','execute':args.execute,'protocol':PROTOCOL,
  'protocol_sha256':hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest(),
  'source_manifest_sha256':sha(HERE/'SOURCE_MANIFEST.json'),'dataset_id':args.dataset,'graph_root':str(args.graph_root),'dataset_binding':b,
  'preblock_minimum_bytes_available':b['native']['rowptr_bytes']+b['native']['col_bytes']+6*GIB,
  'native_reuse_root':str(args.native_reuse_root),'native_reuse_receipt_sha256':args.native_reuse_receipt_sha256,'native_reuse_preparation':reuse,
  'out':str(args.out),'coordination_dir':str(args.coordination_dir),'max_hours':args.max_hours,
  'repetition_block':args.repetition_block,'global_protocol_jobs':18,'global_two_dataset_jobs':36,'global_job_order':[job['name']for job in global_jobs],
  'jobs':jobs,'environment_policy':environment.POLICY,'selected_cpu_affinity':chosen,
  'platform_admission':{'source_path':str(path),'bytes':path.stat().st_size,'sha256':args.platform_admission_sha256,'proof':proof},
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
  self.platform=self.plan['platform_admission']['proof']
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
  expected=self.platform['gpu'];require(target['uuid']==expected['uuid']and target['name']==expected['name']
   and target['memory_total_mib']==expected['memory_total_mib']and target['driver_version']==expected['driver'],'actual block GPU differs from predeclared platform')
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
  self.platform_guard()
  source=Path(self.plan['platform_admission']['source_path']);target=self.args.out/'PLATFORM_ADMISSION.json'
  require(sha(source)==self.plan['platform_admission']['sha256'],'actual platform proof changed before admission copy')
  with source.open('rb')as incoming,target.open('xb')as outgoing:
   shutil.copyfileobj(incoming,outgoing,1<<20);outgoing.flush();os.fsync(outgoing.fileno())
  require(sha(target)==self.plan['platform_admission']['sha256'],'exact original platform proof copy differs')
  for job in self.plan['jobs']:job['config']['environment_contract']=self.environment
  save(self.args.out/'plan.json',self.plan)
  save(self.args.out/'ENVIRONMENT_ADMISSION.json',{'status':'PASS','policy':environment.POLICY,
   'chosen_cpu_affinity':self.environment['chosen_cpu_affinity'],'contract':self.environment,
   'after':self.environment_guard('controller_admission'),'scope':'Fixed inherited logical CPU mask; shared quota/pool, no exclusive or continuous-placement guarantee.'})
  pins=read(HERE/'SOURCE_PINS.json');versions={name:importlib.metadata.version(name)for name in pins['runtime_versions']}
  require(versions==pins['runtime_versions'],'original E23 numerical/model stack required')
  save(self.args.out/'host.json',{'hostname':platform.node(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
   'cpu_affinity':sorted(os.sched_getaffinity(0)),'python':sys.version,'versions':versions})
  b=self.plan['dataset_binding'];files=b['graph']['files']
  for name,expected in files.items():
   p=self.args.graph_root/name;require(pin(p,self.check)==expected,'whole source-pinned public graph differs: '+name)
   st=p.stat();self.snapshots[str(p)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev,'sha256':expected['sha256']}
  save(self.args.out/'input_identity.json',{'whole_hashes':True,'files':self.snapshots})
  shutil.copyfile(Path(b['workload_path']),self.args.out/'pairs.npy')
  require(sha(self.args.out/'pairs.npy')==b['workload_sha256'],'original frozen E23 dataset workload changed')
  descriptor=reuse_descriptor(self.args.native_reuse_root,b['native'],self.args.native_reuse_receipt_sha256,full=True,check=self.check,expected_device=self.platform['storage']['device'])
  save(self.args.out/'native_reuse_admission.json',{'descriptor':descriptor,'preparation':self.plan['native_reuse_preparation'],'outside_application_lifetimes':True})
  reuse_copy=self.args.out/'native_reuse_PREPARATION_READY.json'
  with (self.args.native_reuse_root/'READY.json').open('rb') as incoming,reuse_copy.open('xb') as outgoing:
   shutil.copyfileobj(incoming,outgoing,1<<20);outgoing.flush();os.fsync(outgoing.fileno())
  require(sha(reuse_copy)==self.args.native_reuse_receipt_sha256,'byte-exact reusable preparation receipt copy differs')
 def platform_guard(self):
  p=self.platform;base=Path(p['base']);storage=p['storage'];resources=self.environment['resources']
  require(p['kind']=='CLOSURE_V1_ACTUAL_BLOCK_PLATFORM_ADMISSION' and p['dataset_id']==self.args.dataset and p['block']==self.plan['repetition_block']
   and p['actual_scientific_jobs_started']==p['YT_writes']==p['statefulset_actions']==0,'fresh exact-block platform proof required')
  require(os.environ.get('POD_UID')==p['pod_uid']and platform.node()==p['hostname']
   and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==p['boot_id'],'actual block Pod/host/boot differs')
  require(p['cpu_max']==resources['cpu_max']=='2700000 100000'and p['memory_max']==resources['memory_max']=='274877906944'
   and p['cpu_affinity']==self.environment['chosen_cpu_affinity']==sorted(os.sched_getaffinity(0))
   and p['topology']==self.environment['topology'],'actual block CPU/quota/memory/topology differs')
  require(all(isinstance(p[k],str)and p[k]for k in('pod_uid','sts_uid','hostname','boot_id','node','image_digest'))
   and p['image_digest'].find('sha256:')>=0,'root-bound block identity/image metadata required')
  binding=p['kubernetes_binding'];require(all(binding[k]==p[k]for k in('pod_uid','sts_uid','hostname','node','image_digest'))
   and all(binding[k]==storage[k]for k in('PVC_name','PVC_uid','volume_name','subPath')),'root Kubernetes/PVC identity binding differs')
  require(base.resolve()==base and base.is_dir()and not base.is_symlink(),'canonical actual block BASE required')
  for value in(HERE,self.args.graph_root,self.args.native_reuse_root,self.args.out,self.args.coordination_dir,Path(self.plan['platform_admission']['source_path'])):
   require(value.is_relative_to(base)and value.resolve()==value,'source/input/output/proof/coordination must share actual BASE')
   while not value.exists():value=value.parent
   require(value.stat().st_dev==storage['device'],'block path is on another storage device')
  for key in('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','TORCH_HOME'):
   require(key in self.env,'root must declare persistent temp/cache paths');value=Path(self.env[key])
   require(value.is_relative_to(base)and value.resolve()==value,'temporary/cache path must remain under admitted BASE')
   while not value.exists():value=value.parent
   require(value.stat().st_dev==storage['device'],'temporary/cache path is on another device')
  mount=[]
  for line in environment.text('/proc/self/mountinfo').splitlines():
   left,right=line.split(' - ',1);a,z=left.split(),right.split()
   if environment.unescape(a[4])==str(base):mount.append({'root':environment.unescape(a[3]),'mount_point':environment.unescape(a[4]),
    'major_minor':a[2],'filesystem':z[0],'source':environment.unescape(z[1]),'options':a[5].split(','),'super_options':z[2].split(',')})
  require(len(mount)==1 and mount[0]==storage['mount']and mount[0]['root']=='/'+storage['subPath']
   and'rw'in mount[0]['options']and'ro'not in mount[0]['options']and'ro'not in mount[0]['super_options'],'actual block project mount differs')
  minimum=self.plan['preblock_minimum_bytes_available']
  require(minimum==self.plan['dataset_binding']['native']['rowptr_bytes']+self.plan['dataset_binding']['native']['col_bytes']+6*GIB,'source-bound dataset disk reserve formula differs')
  capacity=os.statvfs(base);require(base.stat().st_dev==storage['device']and not bool(capacity.f_flag&os.ST_RDONLY)
   and capacity.f_bavail*capacity.f_frsize>=minimum and(capacity.f_files==0 or capacity.f_favail>0),'actual preblock native-bytes-plus-sixGiB/readwrite/inodes required')
  require(storage['mount_readonly']is False and storage['bytes_available']>=minimum
   and(storage['inodes_total']==0 or storage['inodes_available']>0)and storage['write_fsync_readback_unlink']is True,'root actual PVC probe/capacity proof required')
  root_layer=p['root_layer'];require(root_layer['mount_readonly']is False and root_layer['bytes_available']>=4*GIB
   and root_layer['write_fsync_readback_unlink']is True,'root actual writable-layer probe/reserve proof required')
  actual_root=os.statvfs('/home/jovyan');require(not bool(actual_root.f_flag&os.ST_RDONLY)
   and actual_root.f_bavail*actual_root.f_frsize>=4*GIB,'actual preblock root-layer readwrite/reserve differs')
 def assert_inputs(self):
  verify_sources();self.check()
  self.environment_guard('assert_inputs')
  for name,value in self.snapshots.items():
   s=Path(name).stat();require((s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)==tuple(value[k]for k in('bytes','mtime_ns','inode','device')),'original graph stat changed')
  require(sha(self.args.out/'pairs.npy')==self.plan['dataset_binding']['workload_sha256'],'query stream changed')
  reuse_descriptor(self.args.native_reuse_root,self.plan['dataset_binding']['native'],self.args.native_reuse_receipt_sha256,expected_device=self.platform['storage']['device'])
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
  root=Path(job['config']['native_root']);expected=self.plan['dataset_binding']['native'];observed={}
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
  if job['arm'] in NATIVE_ARMS:
   expected=self.plan['dataset_binding']['native'];native_bytes=expected['rowptr_bytes']+expected['col_bytes']
   if job['arm']==ARMS[1]:
    require(r['native_promotion']['rowptr_bytes']+r['native_promotion']['col_bytes']==native_bytes and r['native_promotion']['full_native_copy_once_per_app'] is True and r['native_reuse'] is None,'one fresh source-bound promotion required')
   else:
    require(r['native_promotion'] is None and r['native_reuse']['immutable_open_only'] is True and r['native_reuse']['receipt_sha256']==self.args.native_reuse_receipt_sha256,'reuse must open immutable prior arrays without promotion')
   require(all(w['initialization_count']==1 and w['native_initialization']['persistent_assoc_bytes_per_worker']==8*expected['nodes'] for w in workers),'exact one private8N map per native worker required')
   require(all(w['native_initialization']['owner_pid']==w['pid']and w['native_initialization']['library_sha256']==sha(self.args.out/'build/native_assoc_v1.so')
    and w['native_arrays']['readonly']is True and w['native_arrays']['rowptr_bytes']+w['native_arrays']['col_bytes']==native_bytes for w in workers),'actual per-worker library/readonly representation identity differs')
  else:require(r['native_promotion']is None and r['native_reuse'] is None and all(w['initialization_count']==0 for w in workers),'compiled must not allocate native state')
  return r
 def summarize(self,reports):
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
  rows=[];ratios=[];rep=self.plan['repetition_block']
  require(len(self.records)==len(reports)==6,'all six prospective dataset-block apps required')
  for horizon in HORIZONS:
   for w in WORKERS:
    arms={}
    for arm in ARMS:
     chosen=[r for r in self.records if r['horizon']==horizon and r['workers']==w and r['arm']==arm and r['repetition']==rep]
     require(len(chosen)==1,'one actual observation per block configuration/arm required')
     record=chosen[0];arms[arm]={'name':record['name'],'metrics':metrics[record['name']]}
    rows.append({'horizon':horizon,'workers':w,'repetition':rep,'arms':arms})
    for numerator,denominator in ((ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[2],ARMS[1])):
     ratios.append({'horizon':horizon,'workers':w,'repetition':rep,'numerator':numerator,'denominator':denominator,
      'metrics':{key:arms[numerator]['metrics'][key]/arms[denominator]['metrics'][key]for key in arms[numerator]['metrics']},
      'scope':'One actual within-dataset/horizon/allocation paired ratio; no cross-block causal speedup or populationCI.'})
  return {'kind':'CLOSURE_V1_SINGLE_REPETITION_BLOCK_RAW_SUMMARY','status':'complete','dataset_id':self.args.dataset,'jobs':6,'global_protocol_jobs':18,'global_two_dataset_jobs':36,
   'repetition_block':rep,'platform_admission_sha256':self.plan['platform_admission']['sha256'],
   'native_reuse_receipt_sha256':self.args.native_reuse_receipt_sha256,
   'rows':rows,'paired_arm_ratios':ratios,'sampled_PSS':pss,
   'scope':'Two horizon rows at W8, three fresh application observations each. All three independently complete blocks required per dataset; no pooling with E25/failedattempts. Reuse preparation cost is separately recorded outside all app clocks.'}

 def run(self):
  a=self.args;a.out.mkdir();self.supervisor.enable_subreaper();lock=None;active=None;token=None;success=False;failure=None
  def interrupted(sig,frame):raise KeyboardInterrupt('E25 controller signal '+str(sig))
  previous={sig:signal.signal(sig,interrupted)for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
  signal.setitimer(signal.ITIMER_REAL,a.max_hours*3600-15)
  try:
   a.coordination_dir.mkdir(parents=True,exist_ok=True);lockpath=a.coordination_dir/'PUBLIC_SYSTEMS_RUNNING.lock'
   require(not lockpath.is_symlink(),'timing lock must not be symlink');lock=lockpath.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   active=a.coordination_dir/'PUBLIC_SYSTEMS_ACTIVE.json';require(not os.path.lexists(active),'unknown predecessor active marker preserved')
   token={'pid':os.getpid(),'start_ticks':identity(os.getpid())['start_ticks'],'nonce':uuid.uuid4().hex,'source_manifest_sha256':self.plan['source_manifest_sha256'],'repetition_block':self.plan['repetition_block']}
   with active.open('x')as stream:json.dump(token,stream)
   save(a.out/'plan.json',self.plan);self.status('admission');self.admission();self.status('build_native')
   self.command('build_native',[sys.executable,'-B',str(HERE/'native_core/build_native_v1.py'),'--output-dir',str(a.out/'build')],HERE,180)
   save(a.out/'cache_before_qualification.json',self.cache_inventory())
   self.status('actual_cuda_qualification');self.idle();self.command('actual_cuda_qualification',[sys.executable,'-B',str(HERE/'qualify.py'),'--library',str(a.out/'build/native_assoc_v1.so'),
    '--out',str(a.out/'qualification'),'--device','cuda','--workers','8'],HERE,900)
   qualification=read(a.out/'qualification/QUALIFIED.json');require(qualification['success']is True and qualification['device']=='cuda'
    and qualification['workers']==[8]and qualification['actual_consumers']==3,'actual GPU tiny8 qualification for all3arms required')
   save(a.out/'cache_after_qualification.json',self.cache_inventory())
   reports={}
   for job in self.plan['jobs']:
    self.status('running',job=job['name'],completed_jobs=len(self.records))
    self.assert_inputs();idle=self.idle();folder=Path(job['config']['out']);folder.mkdir(parents=True)
    if job['arm']==ARMS[1]:require(shutil.disk_usage(folder).free>=sum(self.plan['dataset_binding']['native'][k]for k in('rowptr_bytes','col_bytes'))+6*GIB,'fresh native copy plus6GiB reserve required')
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
   comparisons=[];rep=self.plan['repetition_block']
   for horizon in HORIZONS:
    for w in WORKERS:
     pair=[j for j in self.plan['jobs']if j['horizon']==horizon and j['workers']==w and j['repetition']==rep]
     require(len(pair)==3,'exact three-arm comparison in selected block required')
     values=[reports[j['name']]for j in sorted(pair,key=lambda j:ARMS.index(j['arm']))];left=values[0]
     require(all(left['input_stream_sha256']==right['input_stream_sha256']and left['excluded_warmup_prefix_audits']==right['excluded_warmup_prefix_audits']
      and left['semantic_input_prefixes']==right['semantic_input_prefixes']for right in values[1:]),'all consumed inputs/checkpoint prefixes must match across physical routes')
     comparisons.append({'horizon':horizon,'workers':w,'repetition':rep,'consumed_graphs':HORIZONS[horizon]['consumed_graphs'],
      'all_consumed_inputs_exact':True,'stream_sha256':left['input_stream_sha256']})
    require(len({r['input_stream_sha256']for r in reports.values()if r['config']['horizon']==horizon})==1,'same full input stream required within every horizon')
   cross_workers=[]
   for w in WORKERS:
    short=[r for r in reports.values()if r['config']['horizon']=='short320'and r['config']['num_workers']==w]
    long=[r for r in reports.values()if r['config']['horizon']=='long6400'and r['config']['num_workers']==w]
    require(len(short)==len(long)==3 and len({r['semantic_input_prefixes']['576']for r in short+long})==1
     and short[0]['input_stream_sha256']==long[0]['semantic_input_prefixes']['576'],'worker-specific exact cross-horizon576 prefix required')
    cross_workers.append({'workers':w,'repetition':rep,'consumed_input_graphs':576,'short_final_stream_sha256':short[0]['input_stream_sha256'],
     'long_prefix_stream_sha256':long[0]['semantic_input_prefixes']['576'],'all_block_arms_horizons_exact':True})
   require(len({r['semantic_input_prefixes']['576']for r in reports.values()})==1,'same576 prefix across all six prospective block apps required')
   cross={'consumed_input_graphs':576,'short_final_stream_sha256':cross_workers[0]['short_final_stream_sha256'],
    'long_prefix_stream_sha256':cross_workers[0]['long_prefix_stream_sha256'],'all_arms_workers_in_block_exact':True,'repetition_block':rep}
   self.assert_inputs()
   final_reuse_descriptor=reuse_descriptor(a.native_reuse_root,self.plan['dataset_binding']['native'],a.native_reuse_receipt_sha256,full=True,check=self.check,expected_device=self.platform['storage']['device'])
   save(a.out/'native_reuse_final_integrity.json',{'descriptor':final_reuse_descriptor,'readonly_stat_checks_passed':True,'outside_application_lifetimes':True,'receipt_sha256':a.native_reuse_receipt_sha256,'files':final_reuse_descriptor['verified_files'],'observed_monotonic_ns':time.monotonic_ns()})
   save(a.out/'scientific_summary.json',self.summarize(reports))
   save(a.out/'SCIENTIFIC_TERMINAL.json',{'success':True,'jobs':6,'dataset_id':a.dataset,'global_protocol_jobs':18,'global_two_dataset_jobs':36,'repetition_block':rep,
    'platform_admission_sha256':self.plan['platform_admission']['sha256'],'platform_admission_file':'PLATFORM_ADMISSION.json',
    'stream_comparisons':comparisons,'cross_horizon_input_prefix576':cross,'cross_horizon_worker_comparisons':cross_workers,'records':self.records,
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
    'dataset_id':a.dataset,'repetition_block':self.plan['repetition_block'],'global_protocol_jobs':18,'global_two_dataset_jobs':36,
    'platform_admission_sha256':self.plan['platform_admission']['sha256'],'platform_admission_file':'PLATFORM_ADMISSION.json',
    'environment_policy':environment.POLICY,'environment_guard_success':self.environment is not None and not self.environment_failed,
    'environment_admission_sha256':sha(a.out/'ENVIRONMENT_ADMISSION.json')if(a.out/'ENVIRONMENT_ADMISSION.json').is_file()else None,
    'failure':failure,'controller_guard_failure':self.controller_guard_failure,'records':self.records,'controller_identity':token,'source_manifest_sha256':self.plan['source_manifest_sha256'],
    'elapsed_seconds':a.max_hours*3600-(self.deadline-time.monotonic()),'YT_writes':0,'statefulset_actions':0,
    'scope':'One prospective6-app datasetblock only; all3blocks required for18 pergraph,36two-graph. This terminal precedes own future OS exit/lock.close; actual exit/quiescence/lock proof remains separate.'})
   if quiet and owned:active.unlink()
   if lock is not None:lock.close()
   self.status('complete'if success and quiet and owned else'failed',completed_jobs=len(self.records),failure=failure)
   for sig,old in previous.items():signal.signal(sig,old)
  return 0 if success and quiet and owned else 1

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',choices=('friendster','citation2'),required=True);parser.add_argument('--graph-root',type=Path,required=True);parser.add_argument('--native-reuse-root',type=Path,required=True);parser.add_argument('--native-reuse-receipt-sha256',required=True);parser.add_argument('--out',type=Path,required=True)
 parser.add_argument('--coordination-dir',type=Path,required=True);parser.add_argument('--max-hours',type=float,default=3900/3600);parser.add_argument('--repetition-block',type=int,choices=(1,2,3),required=True);parser.add_argument('--platform-admission-json',type=Path,required=True);parser.add_argument('--platform-admission-sha256',required=True);parser.add_argument('--source-manifest-sha256');parser.add_argument('--execute',action='store_true')
 a=parser.parse_args();a.graph_root=a.graph_root.resolve();a.native_reuse_root=a.native_reuse_root.resolve();a.out=a.out.resolve();a.coordination_dir=a.coordination_dir.resolve()
 require(math.isfinite(a.max_hours)and 0<a.max_hours<=3900/3600,'positive maximum65minutes per block required');require(not os.path.lexists(a.out),'fresh output required')
 for p in(HERE,a.graph_root,a.native_reuse_root):require(not a.out.is_relative_to(p)and not p.is_relative_to(a.out),'source/input/output trees must be disjoint')
 require(not any(a.coordination_dir.is_relative_to(p)for p in(HERE,a.graph_root,a.native_reuse_root,a.out)),'coordination outside source/input/output trees required')
 definition=plan(a);print(json.dumps(definition,indent=2),flush=True)
 if not a.execute:return 0
 require(a.source_manifest_sha256==definition['source_manifest_sha256'],'root-reviewed manifest SHA required')
 return Controller(a,definition).run()
if __name__=='__main__':raise SystemExit(main())
