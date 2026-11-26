"""ENVv3: fixed27 mask, full task brackets and bounded boundary-only rescans.

Linux scientific gates fail closed. Portable Mac CPU1 qualification reports an
explicit unsupported status and supplies no evidence about target placement.
"""
from pathlib import Path
import json,os,platform,time
from common import require,save,identity

COUNT=27
QUOTA='2700000 100000'
POLICY='fixed27_env_v3'
BOUNDARY_ATTEMPTS=8
BOUNDARY_WALL_SECONDS=0.5
BOUNDARY_PAUSE_SECONDS=0.01
KEY='E25_EXPECTED_CPU_AFFINITY'
CONTRACT_KEY='E25_ENV_CONTRACT'
LIMIT=16384

def text(path):
 with Path(path).open()as stream:value=stream.read(65537)
 require(len(value)<=65536,'ENVv3 bounded resource metadata');return value.strip()
def cpus(value):
 out=set()
 for item in value.split(','):
  if not item:continue
  parts=item.split('-');require(len(parts)in(1,2),'ENVv3 invalid CPU list')
  low,high=int(parts[0]),int(parts[-1]);require(0<=low<=high<65536,'ENVv3 CPU list range')
  out.update(range(low,high+1));require(len(out)<=4096,'ENVv3 CPU list cap')
 return sorted(out)
def selected():
 if not hasattr(os,'sched_getaffinity'):return None
 raw=os.environ.get(KEY)
 if raw is None:return sorted(os.sched_getaffinity(0))[-COUNT:]
 require(len(raw)<=65536,'ENVv3 selected mask env cap');value=json.loads(raw)
 require(isinstance(value,list)and len(value)==COUNT and all(type(x)is int and x>=0 for x in value)
  and value==sorted(set(value)),'ENVv3 expected exact sorted27 mask');return value
def unescape(value):
 for a,b in(('\\040',' '),('\\011','\t'),('\\012','\n'),('\\134','\\')):value=value.replace(a,b)
 return value
def cgroup_location():
 rows=[line.split(':',2)for line in text('/proc/self/cgroup').splitlines()]
 current=[r[2]for r in rows if r[:2]==['0','']];require(len(current)==1,'ENVv3 cgroupv2 membership required')
 membership=current[0];matches=[]
 for line in text('/proc/self/mountinfo').splitlines():
  left,right=line.split(' - ',1)
  if right.split()[0]!='cgroup2':continue
  fields=left.split();root,mount=unescape(fields[3]),unescape(fields[4])
  if membership==root or membership.startswith(root.rstrip('/')+'/'):
   relative=membership[len(root):].lstrip('/');path=Path(mount)/relative
   matches.append((len(root),path,root,mount))
 require(matches,'ENVv3 visible actual cgroupv2 mount required')
 _,path,root,mount=max(matches,key=lambda x:x[0]);st=path.stat()
 return {'membership':membership,'path':str(path),'mount_root':root,'mount_point':mount,'inode':st.st_ino,'device':st.st_dev}
def resources():
 location=cgroup_location();root=Path(location['path'])
 ancestors=[];current=root
 while True:
  st=current.stat();ancestors.append({'path':str(current),'inode':st.st_ino,'device':st.st_dev,
   'cpu_max':text(current/'cpu.max'),'memory_max':text(current/'memory.max')})
  if str(current)==location['mount_point']:break
  require(len(ancestors)<32 and current.parent!=current,'ENVv3 bounded visible cgroup ancestry');current=current.parent
 return {'cgroup':location,'cpu_max':text(root/'cpu.max'),'cpu_weight':text(root/'cpu.weight'),
  'visible_quota_hierarchy':ancestors,
  'cpu_stat':text(root/'cpu.stat'),'memory_max':text(root/'memory.max'),
  'ambient_cpu_pool':cpus(text(root/'cpuset.cpus.effective')),'ambient_mems':text(root/'cpuset.mems.effective'),
  'online_cpus':cpus(text('/sys/devices/system/cpu/online')),
  'hostname':platform.node(),'boot_id':text('/proc/sys/kernel/random/boot_id')}
def topology(mask):
 rows=[]
 for cpu in mask:
  root=Path('/sys/devices/system/cpu')/('cpu'+str(cpu));top=root/'topology'
  rows.append({'cpu':cpu,'physical_package_id':text(top/'physical_package_id'),'core_id':text(top/'core_id'),
   'thread_siblings_list':text(top/'thread_siblings_list'),'numa_nodes':sorted(p.name for p in root.glob('node[0-9]*'))})
 return rows
def process(pid,expected_start=None):
 began=time.monotonic_ns();root=Path('/proc')/str(pid);errors=[]
 def ident(tid,stage):
  try:return identity(tid)
  except ProcessLookupError:errors.append({'stage':stage,'tid':tid,'type':'ProcessLookupError'});return None
 def cgroup(stage):
  try:return text(root/'cgroup')
  except(FileNotFoundError,ProcessLookupError):errors.append({'stage':stage,'type':'process_disappeared'});return None
 def task_ids(stage):
  try:rows=sorted(int(p.name)for p in(root/'task').iterdir())
  except(FileNotFoundError,ProcessLookupError):errors.append({'stage':stage,'type':'process_disappeared'});return []
  require(len(rows)<=256,'ENVv3 task count cap');return rows
 before=ident(pid,'process_before');cg_before=cgroup('cgroup_before');ids_before=task_ids('task_list_before')
 tasks_before=[{'tid':tid,'identity':ident(tid,'task_before')}for tid in ids_before]
 observations=[];masks=[];observed=[];vanished=[]
 for tid in ids_before:
  a=ident(tid,'mask_before');affinity=None
  if a is not None:
   try:affinity=sorted(os.sched_getaffinity(tid))
   except ProcessLookupError:errors.append({'stage':'task_affinity','tid':tid,'type':'ProcessLookupError'})
  b=ident(tid,'mask_after');observations.append({'tid':tid,'before':a,'cpu_affinity':affinity,'after':b})
  if a is None or b is None or affinity is None or a['start_ticks']!=b['start_ticks']:vanished.append(tid);continue
  if affinity not in masks:masks.append(affinity)
  observed.append({'tid':tid,'start_ticks':a['start_ticks'],'mask_index':masks.index(affinity)})
 ids_after=task_ids('task_list_after');tasks_after=[{'tid':tid,'identity':ident(tid,'task_after')}for tid in ids_after]
 cg_after=cgroup('cgroup_after');after=ident(pid,'process_after')
 reasons=[];pid_starts=[r['start_ticks']for r in(before,after)if r is not None]
 pid_exact=None if len(pid_starts)<2 else len(set(pid_starts+[expected_start]if expected_start is not None else pid_starts))==1
 if before is None or after is None:reasons.append('process_identity_unobserved')
 if expected_start is not None and any(s!=expected_start for s in pid_starts):pid_exact=False
 if pid_exact is False:reasons.append('process_start_changed')
 if cg_before is None or cg_after is None:reasons.append('process_cgroup_unobserved')
 elif cg_before!=cg_after:reasons.append('process_cgroup_changed')
 if ids_before!=ids_after:reasons.append('task_membership_changed')
 starts={};unobserved=set()
 for row in tasks_before+tasks_after:
  if row['identity']is None:unobserved.add(row['tid'])
  else:starts.setdefault(row['tid'],set()).add(row['identity']['start_ticks'])
 for row in observations:
  for entry in(row['before'],row['after']):
   if entry is None:unobserved.add(row['tid'])
   else:starts.setdefault(row['tid'],set()).add(entry['start_ticks'])
  if row['cpu_affinity']is None:unobserved.add(row['tid'])
 tid_exact=all(len(values)==1 for values in starts.values())
 if not tid_exact:reasons.append('task_start_changed')
 if unobserved:reasons.append('leader_task_unobserved'if pid in unobserved else'ancillary_task_unobserved')
 leader=next((r['cpu_affinity']for r in observations if r['tid']==pid),None)
 if leader is None:reasons.append('leader_mask_unobserved')
 if not ids_before:reasons.append('task_list_unobserved')
 stable=not reasons and pid_exact is True and len(observed)==len(ids_before)
 return {'pid':pid,'expected_start_ticks':expected_start,'start_ticks':None if before is None else before['start_ticks'],
  'stable':stable,'instability_reasons':reasons,'process_identity_exact':pid_exact,'task_identities_exact':tid_exact,
  'task_masks':masks,'tasks':observed,'cgroup_membership':cg_before,'vanished_tids':vanished,'leader_mask':leader,
  'brackets':{'begin_monotonic_ns':began,'end_monotonic_ns':time.monotonic_ns(),'process_before':before,'process_after':after,
   'cgroup_before':cg_before,'cgroup_after':cg_after,'task_ids_before':ids_before,'task_ids_after':ids_after,
   'tasks_before':tasks_before,'tasks_after':tasks_after,'task_affinity_observations':observations,'read_transients':errors}}
def observe(contract,owners=None):
 result={'policy':POLICY,'observed_monotonic_ns':time.monotonic_ns()}
 if contract.get('portable_unsupported'):
  return {**result,'status':'NOT_APPLICABLE_PORTABLE_CPU1','predicates':{},'target_environment_proof':False}
 try:
  resource=resources();result['resources']=resource;chosen=contract['chosen_cpu_affinity'];base=contract['resources'];owned=owners or[identity(os.getpid())]
  require(len(owned)<=32 and all(x is not None for x in owned),'ENVv3 bounded identified owner tree required')
  processes=[];result['processes']=processes
  for p in owned:processes.append(process(p['pid'],p['start_ticks']))
  stable=[p for p in processes if p['stable']]
  predicates={'resource_observation_readable':True,'same_hostname':resource['hostname']==base['hostname'],
   'same_boot_id':resource['boot_id']==base['boot_id'],'same_cgroup_identity':resource['cgroup']==base['cgroup'],
   'same_visible_quota_hierarchy':resource['visible_quota_hierarchy']==base['visible_quota_hierarchy'],
   'same_exact_cpu_quota':resource['cpu_max']==base['cpu_max']==QUOTA,'same_memory_limit':resource['memory_max']==base['memory_max'],
   'same_cpu_weight':resource['cpu_weight']==base['cpu_weight'],'same_effective_mems':resource['ambient_mems']==base['ambient_mems'],
   'chosen_cpus_still_allowed':set(chosen)<=set(resource['ambient_cpu_pool']),
   'chosen_cpus_still_online':set(chosen)<=set(resource['online_cpus']),
   'stable_process_cgroups_exact':all(p['cgroup_membership']=='0::'+base['cgroup']['membership']for p in stable),
   'stable_process_leader_masks_exact':all(p['leader_mask']==chosen for p in stable),
   'observed_thread_masks_within_chosen':all(row['cpu_affinity']and set(row['cpu_affinity'])<=set(chosen)for p in processes for row in p['brackets']['task_affinity_observations']if row['cpu_affinity']is not None),
   'observed_process_PID_starts_exact':all(p['process_identity_exact']is not False for p in processes),
   'observed_TID_starts_exact':all(p['task_identities_exact']is True for p in processes),
   'observed_process_cgroups_exact':all(value=='0::'+base['cgroup']['membership']for p in processes for value in(p['brackets']['cgroup_before'],p['brackets']['cgroup_after'])if value is not None),
   'observed_process_leader_masks_exact':all(p['leader_mask']==chosen for p in processes if p['leader_mask']is not None)}
  result.update(resources=resource,processes=processes,predicates=predicates,
   process_scan_complete_stable=len(stable)==len(processes),
   ambient_pool_changed=resource['ambient_cpu_pool']!=base['ambient_cpu_pool'])
 except Exception as error:
  result.update(predicates={'resource_observation_readable':False},error={'type':type(error).__name__,'message':str(error)[:2000]})
 failed=[name for name,ok in result['predicates'].items()if not ok]
 result.update(status='FAIL'if failed else'PASS',failed_predicates=failed)
 return result
def _boundary_scan(contract,owners=None,*,observer=None,clock=None,sleeper=None):
 """Retry ONLY ancillary task churn; preserve every attempt, never repair masks."""
 observer=observe if observer is None else observer;clock=time.monotonic if clock is None else clock;sleeper=time.sleep if sleeper is None else sleeper
 started=clock();deadline=started+BOUNDARY_WALL_SECONDS;chosen=contract.get('chosen_cpu_affinity')
 try:expected=owners or[identity(os.getpid())]
 except ProcessLookupError:expected=[None]
 attempts=[];last=None
 def terminal(sample,status,failed,kind):
  elapsed=clock()-started
  if status=='PASS'and elapsed>BOUNDARY_WALL_SECONDS:status='FAIL';failed=['boundary_rescan_within_wall_budget'];kind='boundary_budget_exhaustion'
  result=dict(sample);result['predicates']=dict(sample.get('predicates',{}));result['status']=status;result['failed_predicates']=list(failed)
  for name in failed:result['predicates'][name]=False
  result['boundary_rescan']={'max_attempts':BOUNDARY_ATTEMPTS,'max_wall_seconds':BOUNDARY_WALL_SECONDS,
   'pause_seconds':BOUNDARY_PAUSE_SECONDS,'expected_owners':expected,'attempt_count':len(attempts),
   'elapsed_seconds':elapsed,'terminal_kind':kind,'attempts':attempts}
  return result
 for index in range(BOUNDARY_ATTEMPTS):
  if clock()>=deadline:break
  sample=observer(contract,expected);last=sample
  if sample['status']=='NOT_APPLICABLE_PORTABLE_CPU1':
   attempts.append({'attempt':index+1,'classification':'NOT_APPLICABLE','observation':sample});return terminal(sample,sample['status'],[],'portable_unsupported')
  if sample['status']=='FAIL':
   attempts.append({'attempt':index+1,'classification':'HARD_FAILURE','observation':sample});return terminal(sample,'FAIL',sample['failed_predicates'],'observed_hard_failure')
  processes=sample['processes'];base_cgroup='0::'+contract['resources']['cgroup']['membership']
  own_identity=len(processes)==len(expected)==1 and expected[0]is not None
  if own_identity:
   p=processes[0];br=p['brackets'];target=expected[0]
   own_identity=p['pid']==target['pid']and all(r is not None and r['pid']==target['pid']and r['start_ticks']==target['start_ticks']for r in(br['process_before'],br['process_after']))
  boundary={'boundary_own_PID_start_observed_consistent':own_identity,
   'boundary_own_leader_mask_exact':len(processes)==1 and processes[0]['leader_mask']==chosen,
   'boundary_own_cgroup_observed_exact':len(processes)==1 and all(v==base_cgroup for v in(processes[0]['brackets']['cgroup_before'],processes[0]['brackets']['cgroup_after'])),
   'boundary_instability_is_ancillary_task_only':all(set(p['instability_reasons'])<= {'task_membership_changed','ancillary_task_unobserved'}for p in processes)}
  sample=dict(sample);sample['predicates']={**sample['predicates'],**boundary};last=sample
  negative=[name for name,value in sample['predicates'].items()if not value]
  if negative:
   attempts.append({'attempt':index+1,'classification':'HARD_FAILURE','observation':sample});return terminal(sample,'FAIL',negative,'observed_hard_failure')
  if clock()>deadline:
   attempts.append({'attempt':index+1,'classification':'WALL_BUDGET_EXCEEDED','observation':sample});return terminal(sample,'FAIL',['boundary_rescan_within_wall_budget'],'boundary_budget_exhaustion')
  if sample['process_scan_complete_stable']is True:
   attempts.append({'attempt':index+1,'classification':'FULLY_STABLE_PASS','observation':sample});return terminal(sample,'PASS',[],'fully_stable_pass')
  attempts.append({'attempt':index+1,'classification':'TRANSIENT_ANCILLARY_TASK_SCAN','observation':sample})
  if index+1<BOUNDARY_ATTEMPTS:
   remaining=deadline-clock()
   if remaining<=0:break
   sleeper(min(BOUNDARY_PAUSE_SECONDS,remaining))
 if last is None:last={'policy':POLICY,'observed_monotonic_ns':time.monotonic_ns(),'predicates':{}}
 failed='boundary_rescan_within_wall_budget'if clock()>=deadline else'own_boundary_process_tasks_stable'
 return terminal(last,'FAIL',[failed],'boundary_budget_exhaustion')
def guard(contract,label,failure_path=None,owners=None,require_stable=True):
 require(require_stable,'ENVv3 boundaries require an eventual fully stable scan')
 result=_boundary_scan(contract,owners);result['label']=label
 if result['status']=='FAIL':
  if failure_path is not None:save(failure_path,result)
  raise RuntimeError('ENVv3 '+label+' failed predicates: '+','.join(result['failed_predicates']))
 return result
def admit(expected=None,portable=False,failure_path=None):
 if not hasattr(os,'sched_getaffinity'):
  require(portable,'ENVv3 Linux affinity required')
  return {'portable_unsupported':True,'policy':POLICY,'chosen_cpu_affinity':None}
 inherited=os.environ.get(CONTRACT_KEY)
 if inherited:
  require(len(inherited)<=65536,'ENVv3 inherited contract cap');contract=json.loads(inherited)
  require(contract['policy']==POLICY and contract['chosen_cpu_affinity']==(expected or selected()),'ENVv3 inherited chosen mask differs')
  guard(contract,'inherited_admission',failure_path);return contract
 chosen=expected or selected();require(chosen is not None and len(chosen)==COUNT,'ENVv3 exactly27 allowed logical CPUs required')
 before=resources();allowed=sorted(os.sched_getaffinity(0))
 tasks=list((Path('/proc/self/task')).iterdir())
 predicates={'chosen_cpus_available_at_admission':set(chosen)<=set(allowed)&set(before['ambient_cpu_pool'])&set(before['online_cpus']),
  'admission_exact_cpu_quota':before['cpu_max']==QUOTA,'single_thread_before_once_pin':len(tasks)==1}
 failed=[k for k,v in predicates.items()if not v]
 if failed:
  if failure_path is not None:save(failure_path,{'policy':POLICY,'status':'FAIL','label':'admission_before_pin',
   'failed_predicates':failed,'predicates':predicates,'resources':before,'chosen_cpu_affinity':chosen,'actual_affinity':allowed,'task_count':len(tasks)})
  raise RuntimeError('ENVv3 admission failed predicates: '+','.join(failed))
 os.sched_setaffinity(0,chosen)  # The only affinity mutation, before admission/build/qualification.
 contract={'policy':POLICY,'chosen_cpu_affinity':chosen,'resources':before,'topology':topology(chosen),
  'admission_before_affinity':allowed,'pin_calls':1,'no_exclusive_or_continuous_placement_guarantee':True}
 guard(contract,'post_once_pin',failure_path);return contract
def exported(contract):
 if contract.get('portable_unsupported'):return {}
 return {KEY:json.dumps(contract['chosen_cpu_affinity'],separators=(',',':')),CONTRACT_KEY:json.dumps(contract,separators=(',',':'))}
def append(path,result):
 p=Path(path);line=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
 require(len(line)<=1<<20 and(not p.exists()or p.stat().st_size+len(line)<=32<<20),'ENVv3 bounded event log')
 with p.open('a')as stream:stream.write(line)
