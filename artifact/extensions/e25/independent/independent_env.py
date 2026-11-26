"""Independent bounded interpretation of E25 ENVv3 recorded observations.

No producer code or recorded filesystem path is imported/opened. Resource/mask
predicates and PID/TID/cgroup stability are recomputed from full raw brackets.
Boundary absolute start and actual sleep durations remain source-pinned receipts.
"""
import json

POLICY='fixed27_env_v3'
QUOTA='2700000 100000'
MEMORY='274877906944'
PREDICATES={'resource_observation_readable','same_hostname','same_boot_id','same_cgroup_identity',
 'same_visible_quota_hierarchy','same_exact_cpu_quota','same_memory_limit','same_cpu_weight',
 'same_effective_mems','chosen_cpus_still_allowed','chosen_cpus_still_online',
 'stable_process_cgroups_exact','stable_process_leader_masks_exact','observed_thread_masks_within_chosen',
 'observed_process_PID_starts_exact','observed_TID_starts_exact','observed_process_cgroups_exact','observed_process_leader_masks_exact'}
PROJECTION=('cgroup','cpu_max','cpu_weight','visible_quota_hierarchy','memory_max','ambient_mems','hostname','boot_id')
RESOURCE_KEYS=set(PROJECTION)|{'cpu_stat','ambient_cpu_pool','online_cpus'}

def need(ok,message):
 if not ok:raise ValueError('Independent ENVv3: '+message)
def positive(value):return type(value)is int and value>0
def bounded_string(value):return isinstance(value,str)and 0<len(value)<=65536
def mask(value):
 need(isinstance(value,list)and 0<len(value)<=4096 and all(type(v)is int and 0<=v<65536 for v in value)
  and value==sorted(set(value)),'nonempty canonical numeric CPU mask required');return set(value)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def projection(resource):
 need(isinstance(resource,dict)and set(resource)==RESOURCE_KEYS,'complete recorded resource schema required')
 need(all(bounded_string(resource[k])for k in('cpu_max','cpu_weight','memory_max','ambient_mems','hostname','boot_id','cpu_stat')),'bounded resource strings required')
 location=resource['cgroup'];need(set(location)=={'membership','path','mount_root','mount_point','inode','device'},'recorded cgroup identity schema')
 need(all(bounded_string(location[k])and location[k].startswith('/')for k in('membership','path','mount_root','mount_point'))
  and positive(location['inode'])and type(location['device'])is int and location['device']>=0,'recorded cgroup identity values')
 hierarchy=resource['visible_quota_hierarchy'];need(isinstance(hierarchy,list)and 0<len(hierarchy)<=32,'bounded visible quota ancestry')
 need(len({r['path']for r in hierarchy})==len(hierarchy),'duplicate visible quota ancestor')
 for row in hierarchy:
  need(set(row)=={'path','inode','device','cpu_max','memory_max'}and bounded_string(row['path'])and row['path'].startswith('/')
   and positive(row['inode'])and type(row['device'])is int and row['device']>=0 and bounded_string(row['cpu_max'])and bounded_string(row['memory_max']),'visible quota ancestor schema')
 need(hierarchy[0]=={'path':location['path'],'inode':location['inode'],'device':location['device'],
  'cpu_max':resource['cpu_max'],'memory_max':resource['memory_max']}and hierarchy[-1]['path']==location['mount_point'],'visible quota leaf/mount bindings')
 for child,parent in zip(hierarchy,hierarchy[1:]):
  expected_parent=child['path'].rstrip('/').rsplit('/',1)[0]or'/'
  need(expected_parent==parent['path'],'visible quota ancestor order')
 mask(resource['ambient_cpu_pool']);mask(resource['online_cpus'])
 return {k:resource[k]for k in PROJECTION}


BOUNDARY_PREDICATES={'boundary_own_PID_start_observed_consistent','boundary_own_leader_mask_exact',
 'boundary_own_cgroup_observed_exact','boundary_instability_is_ancillary_task_only'}
ANCILLARY_REASONS={'task_membership_changed','ancillary_task_unobserved'}

def ident(value,pid):
 if value is None:return None
 need(isinstance(value,dict)and set(value)=={'pid','start_ticks','state'}and value['pid']==pid and positive(value['pid'])
  and positive(value['start_ticks'])and isinstance(value['state'],str)and len(value['state'])==1,'raw PID/TID identity schema')
 return value['start_ticks']
def ids(value):
 need(isinstance(value,list)and len(value)<=256 and all(positive(v)for v in value)and value==sorted(set(value)),'bounded canonical task list')
 return value

def process_record(p,observation_ns):
 need(set(p)=={'pid','expected_start_ticks','start_ticks','stable','instability_reasons','process_identity_exact','task_identities_exact',
  'task_masks','tasks','cgroup_membership','vanished_tids','leader_mask','brackets'}and positive(p['pid'])and positive(p['expected_start_ticks']),'complete ENVv3 process schema')
 pid,expected=p['pid'],p['expected_start_ticks'];b=p['brackets']
 need(set(b)=={'begin_monotonic_ns','end_monotonic_ns','process_before','process_after','cgroup_before','cgroup_after','task_ids_before','task_ids_after',
  'tasks_before','tasks_after','task_affinity_observations','read_transients'}and positive(b['begin_monotonic_ns'])and positive(b['end_monotonic_ns'])
  and observation_ns<=b['begin_monotonic_ns']<=b['end_monotonic_ns'],'full raw process bracket schema/clocks')
 before,after=ident(b['process_before'],pid),ident(b['process_after'],pid);known=[v for v in(before,after)if v is not None]
 exact=None if len(known)<2 else len(set(known+[expected]))==1
 if any(v!=expected for v in known):exact=False
 cg=[b['cgroup_before'],b['cgroup_after']]
 need(all(v is None or bounded_string(v)for v in cg),'bounded raw cgroup brackets')
 left,right=ids(b['task_ids_before']),ids(b['task_ids_after']);starts={};missing=set();masks=[];tasks=[];vanished=[];all_masks=[]
 for rows,declared in((b['tasks_before'],left),(b['tasks_after'],right)):
  need(isinstance(rows,list)and len(rows)==len(declared)and[r['tid']for r in rows]==declared,'raw task identity table/list binding')
  for row in rows:
   need(set(row)=={'tid','identity'},'raw task table schema');tick=ident(row['identity'],row['tid'])
   if tick is None:missing.add(row['tid'])
   else:starts.setdefault(row['tid'],set()).add(tick)
 rows=b['task_affinity_observations'];need(isinstance(rows,list)and len(rows)==len(left)and[r['tid']for r in rows]==left,'per-TID affinity scan/list order')
 leader=None
 for row in rows:
  need(set(row)=={'tid','before','cpu_affinity','after'},'raw TID affinity bracket schema')
  a,z=ident(row['before'],row['tid']),ident(row['after'],row['tid']);actual=row['cpu_affinity']
  need(a is not None or actual is None,'affinity cannot be observed after missing before identity')
  if actual is not None:mask(actual);all_masks.append(actual)
  for tick in(a,z):
   if tick is None:missing.add(row['tid'])
   else:starts.setdefault(row['tid'],set()).add(tick)
  if actual is None:missing.add(row['tid'])
  if row['tid']==pid:leader=actual
  if a is None or z is None or actual is None or a!=z:vanished.append(row['tid']);continue
  if actual not in masks:masks.append(actual)
  tasks.append({'tid':row['tid'],'start_ticks':a,'mask_index':masks.index(actual)})
 reasons=[]
 if before is None or after is None:reasons.append('process_identity_unobserved')
 if exact is False:reasons.append('process_start_changed')
 if None in cg:reasons.append('process_cgroup_unobserved')
 elif cg[0]!=cg[1]:reasons.append('process_cgroup_changed')
 if left!=right:reasons.append('task_membership_changed')
 tid_exact=all(len(v)==1 for v in starts.values())
 if not tid_exact:reasons.append('task_start_changed')
 if missing:reasons.append('leader_task_unobserved'if pid in missing else'ancillary_task_unobserved')
 if leader is None:reasons.append('leader_mask_unobserved')
 if not left:reasons.append('task_list_unobserved')
 stable=not reasons and exact is True and len(tasks)==len(left)
 need(p['start_ticks']==before and p['cgroup_membership']==cg[0]and p['task_masks']==masks and p['tasks']==tasks
  and p['vanished_tids']==vanished and p['leader_mask']==leader,'legacy compact fields disagree with raw brackets')
 need(type(p['stable'])is bool and p['stable']is stable and p['instability_reasons']==reasons
  and p['process_identity_exact']is exact and p['task_identities_exact']is tid_exact,'process stability/reasons/identity flags contradict full raw brackets')
 errors=b['read_transients'];need(isinstance(errors,list)and len(errors)<=1024,'bounded raw transient diagnostic list')
 for error in errors:need(set(error)<={'stage','tid','type'}and bounded_string(error.get('stage'))and bounded_string(error.get('type'))
  and('tid'not in error or positive(error['tid'])),'raw transient diagnostic schema')
 return {'pid':pid,'expected':expected,'known_starts':known,'stable':stable,'reasons':reasons,'leader':leader,'masks':all_masks,
  'cgroups':[v for v in cg if v is not None],'pid_exact':exact,'tid_exact':tid_exact,'before':before,'after':after,
  'begin':b['begin_monotonic_ns'],'end':b['end_monotonic_ns'],'tasks':len(tasks)}

class Facts:
 """Bounded stream aggregates; all resource/mask/task assertions rederived."""
 def __init__(self):
  self.count=self.transient=self.complete=self.incomplete=self.boundaries=self.boundary_attempts=0
  self.projections={};self.pools={};self.pool_flags={};self.online=None;self.allowed=None
  self.thread_union=set();self.leaders=set();self.cgroups=set();self.pid_intersection=None;self.starts={};self.labels={};self.first=None;self.last=None
  self.stable_processes=self.unstable_processes=self.recorded_tasks=0;self.max_attempts=0;self.max_elapsed=0.0
 def raw(self,value,boundary_owners=None):
  need(self.count+self.transient<262144,'raw observation count cap')
  need(isinstance(value,dict)and value.get('policy')==POLICY and positive(value.get('observed_monotonic_ns')),'recorded policy/time required')
  timestamp=value['observed_monotonic_ns'];self.first=timestamp if self.first is None else min(self.first,timestamp);self.last=timestamp if self.last is None else max(self.last,timestamp)
  if value['status']=='TRANSIENT_PROCESS_TREE_UNOBSERVED':
   need(boundary_owners is None and set(value)=={'status','policy','observed_monotonic_ns'},'transient scan cannot invent CPU evidence');self.transient+=1;return None
  allowed_predicates=PREDICATES|(BOUNDARY_PREDICATES if boundary_owners is not None else set())
  need(value['status']=='PASS'and value['failed_predicates']==[]and set(value['predicates'])==allowed_predicates
   and all(v is True for v in value['predicates'].values()),'successful artifact may contain no negative attempt or omitted predicate')
  resource=value['resources'];project=projection(resource);key=canonical(project)
  need(key in self.projections or not self.projections,'raw invariant values changed');self.projections[key]=project
  pool=tuple(resource['ambient_cpu_pool']);need(pool in self.pools or len(self.pools)<128,'ambient pool variants cap');self.pools[pool]=self.pools.get(pool,0)+1
  need(type(value['ambient_pool_changed'])is bool,'raw ambient change flag');self.pool_flags.setdefault(pool,set()).add(value['ambient_pool_changed'])
  actual_allowed,actual_online=mask(resource['ambient_cpu_pool']),mask(resource['online_cpus'])
  self.allowed=actual_allowed if self.allowed is None else self.allowed&actual_allowed;self.online=actual_online if self.online is None else self.online&actual_online
  need(isinstance(value['processes'],list)and 0<len(value['processes'])<=32,'bounded nonempty CPU tree')
  rows=[process_record(p,timestamp)for p in value['processes']];need(len({r['pid']for r in rows})==len(rows),'duplicate raw CPU PID')
  need(all(a['end']<=b['begin']for a,b in zip(rows,rows[1:])),'source process scan ordering')
  self.last=max(self.last,max(r['end']for r in rows));stable=sum(r['stable']for r in rows)
  need(type(value['process_scan_complete_stable'])is bool and value['process_scan_complete_stable']is(stable==len(rows)),'CPU tree completeness differs from independently derived brackets')
  need(all(r['pid_exact']is not False and r['tid_exact']is True for r in rows),'observed PID/TID identity violation cannot heal')
  for r in rows:
   self.stable_processes+=r['stable'];self.unstable_processes+=not r['stable'];self.recorded_tasks+=r['tasks']
   if r['leader']is not None:self.leaders.add(tuple(r['leader']))
   self.cgroups.update(r['cgroups'])
   for actual in r['masks']:self.thread_union|=mask(actual)
   if r['known_starts']:
    self.starts.setdefault(r['pid'],set()).update(r['known_starts']);need(len(self.starts[r['pid']])<=32,'PID reuse cap')
  pids={r['pid']for r in rows};self.pid_intersection=pids if self.pid_intersection is None else self.pid_intersection&pids
  self.complete+=stable==len(rows);self.incomplete+=stable!=len(rows);self.count+=1
  need(len(self.starts)<=4096 and len(self.leaders)<=128 and len(self.cgroups)<=128,'CPU identity evidence cap')
  if boundary_owners is not None:
   need(len(rows)==len(boundary_owners)==1,'own boundary needs one fixed owner');r=rows[0];owner=boundary_owners[0]
   need(r['pid']==owner['pid']and r['expected']==owner['start_ticks']and r['before']==r['after']==owner['start_ticks'],
    'boundary required PID/start not completely observed or changed')
   need(all(set(r['reasons'])<=ANCILLARY_REASONS for r in rows),'only ancillary task churn may precede final boundary PASS')
  return rows
 def ingest(self,value,boundary=False):
  if not boundary:self.raw(value);return
  meta=value['boundary_rescan'];need(set(meta)=={'max_attempts','max_wall_seconds','pause_seconds','expected_owners','attempt_count','elapsed_seconds','terminal_kind','attempts'},'full bounded rescan receipt schema')
  owners=meta['expected_owners'];need(isinstance(owners,list)and len(owners)==1 and ident(owners[0],owners[0]['pid'])is not None,'fixed boundary owner identity')
  attempts=meta['attempts'];need(meta['max_attempts']==8 and type(meta['max_attempts'])is int and meta['max_wall_seconds']==0.5 and meta['pause_seconds']==0.01
   and isinstance(attempts,list)and type(meta['attempt_count'])is int and meta['attempt_count']==len(attempts)and 1<=len(attempts)<=8,'8/.5/.01 configuration and actual attempt count')
  elapsed=meta['elapsed_seconds'];need(type(elapsed)in(int,float)and 0<=elapsed<=0.5 and meta['terminal_kind']=='fully_stable_pass','declared actual boundary budget/final kind')
  times=[];last_end=None
  for index,attempt in enumerate(attempts,1):
   need(set(attempt)=={'attempt','classification','observation'}and type(attempt['attempt'])is int and attempt['attempt']==index,'complete sequential attempts')
   observation=attempt['observation'];rows=self.raw(observation,owners);complete=all(r['stable']for r in rows)
   need(complete is(index==len(attempts)),'no stable earlier attempt may be bypassed; final scan must be fully stable')
   need(attempt['classification']==('FULLY_STABLE_PASS'if complete else'TRANSIENT_ANCILLARY_TASK_SCAN'),'attempt class differs from raw brackets')
   start=observation['observed_monotonic_ns'];end=max(r['end']for r in rows)
   need(last_end is None or last_end<=start,'non-overlapping sequential raw attempt scans');last_end=end;times.append((start,end))
  need((times[-1][1]-times[0][0])/1e9<=elapsed+1e-6,'observed raw attempt span exceeds declared total elapsed')
  need({k:v for k,v in value.items()if k not in('boundary_rescan','label')}==attempts[-1]['observation'],'top boundary is not an exact copy of final observation')
  self.boundaries+=1;self.boundary_attempts+=len(attempts);self.max_attempts=max(self.max_attempts,len(attempts));self.max_elapsed=max(self.max_elapsed,float(elapsed))
  need(bounded_string(value.get('label'))and(len(self.labels)<1024 or value['label']in self.labels),'bounded boundary role labels')
  timestamp=value['observed_monotonic_ns'];entry=self.labels.setdefault(value['label'],{'count':0,'first_ns':timestamp,'last_ns':timestamp,'first_scan_ns':times[0][0],'last_bracket_ns':times[-1][1],'identities':set()})
  entry['count']+=1;entry['first_ns']=min(entry['first_ns'],timestamp);entry['last_ns']=max(entry['last_ns'],timestamp)
  entry['first_scan_ns']=min(entry['first_scan_ns'],times[0][0]);entry['last_bracket_ns']=max(entry['last_bracket_ns'],times[-1][1])
  entry['identities'].add((owners[0]['pid'],owners[0]['start_ticks']))
 def bind(self,contract,identity=None,interval=None,require_observed=True):
  chosen=mask(contract['chosen_cpu_affinity']);need(len(chosen)==27 and contract['policy']==POLICY,'exact chosen27 ENVv3 contract')
  expected=projection(contract['resources']);need(expected['cpu_max']==QUOTA and expected['memory_max']==MEMORY,'own CPU quota and memory contract')
  if require_observed:need(self.count>0,'required raw CPU observations absent')
  need(all(v==expected for v in self.projections.values()),'raw hostname/boot/cgroup/quota/hierarchy/weight/mems drift')
  if self.count:need(chosen<=self.allowed and chosen<=self.online,'chosen CPU disappeared from observed allowed/online set')
  need(all(flags=={list(pool)!=contract['resources']['ambient_cpu_pool']}for pool,flags in self.pool_flags.items()),'ambient pool change flag differs from raw pool')
  need(self.thread_union<=chosen and all(set(m)==chosen for m in self.leaders),'observed thread widening or leader exact-mask violation')
  need(all(c=='0::'+expected['cgroup']['membership']for c in self.cgroups),'any observed cgroup mismatch cannot heal')
  if identity is not None and self.count:
   pid,tick=identity;need(pid in self.pid_intersection,'required root PID missing in a CPU observation')
   if pid in self.starts:need(self.starts[pid]=={tick},'required root PID/start mismatch')
  if interval is not None and self.first is not None:need(interval[0]<=self.first<=self.last<=interval[1],'complete CPU observation brackets outside actual lifetime')
  return {'status':'RECORDED_CPU_OBSERVATIONS_CHECKED'if self.count else'CPU_UNOBSERVED','raw_observations':self.count,
   'independently_complete_process_scans':self.complete,'independently_incomplete_process_scans':self.incomplete,'transient_tree_unobserved':self.transient,
   'recorded_stable_processes':self.stable_processes,'recorded_unstable_processes':self.unstable_processes,'recorded_tasks':self.recorded_tasks,
   'resource_invariant_variants':len(self.projections),'ambient_pool_variants':[{'cpu_affinity':list(v),'observations':n}for v,n in sorted(self.pools.items())],
   'first_observed_ns':self.first,'last_bracket_ns':self.last,'boundary_receipts_checked':self.boundaries,'all_boundary_attempts_checked':self.boundary_attempts,
   'maximum_boundary_attempts':self.max_attempts,'maximum_declared_boundary_elapsed_seconds':self.max_elapsed,
   'masks_resources_PID_TID_cgroup_stability_independently_recomputed':bool(self.count),'exact_boundary_wall_start_and_sleep_durations_independently_recomputed':False,
   'scope':'Full recorded PID/TID/cgroup brackets and every boundary attempt recomputed; observed hard violation cannot heal. Declared elapsed/limits/source-bound10ms pause checked with raw duration lower-bound consistency; absolute boundary start and actual sleep durations were not serialized. Sampled, non-atomic, no continuous or exclusive placement guarantee.'}

def boundary(value,contract,owner=None,interval=None,label=None):
 need(bounded_string(value.get('label'))and(label is None or value['label']==label),'original own-boundary role label')
 facts=Facts();facts.ingest(value,boundary=True);return facts.bind(contract,owner,interval)
