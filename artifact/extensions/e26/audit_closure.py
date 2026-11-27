#!/usr/bin/env python3
"""Independent bounded standard-library audit of the new two-dataset closure controls.

Never imports producer code or opens recorded remote paths. Complete raw archives
and root-collected process exits are required; prior E25 measurements are excluded.
The embedded ENV decoder is the previously independently reviewed v6 decoder.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,os,random,statistics,tarfile
from decimal import Decimal,localcontext
from pathlib import Path,PurePosixPath
from types import SimpleNamespace

MIB,GIB=1<<20,1<<30

# BEGIN embedded independent ENV bracket decoder (prior v6).
"""Independent bounded interpretation of E25 ENVv5 recorded observations.

No producer code or recorded filesystem path is imported/opened. Resource/mask
predicates and PID/TID/cgroup stability are recomputed from full raw brackets.
Boundary absolute start and actual sleep durations remain source-pinned receipts.
"""
import json

POLICY='fixed27_env_v5'
QUOTA='2700000 100000'
MEMORY='274877906944'
PREDICATES={'resource_observation_readable','same_hostname','same_boot_id','same_cgroup_identity',
 'same_visible_quota_hierarchy','same_exact_cpu_quota','same_memory_limit','same_cpu_weight',
 'same_effective_mems','chosen_cpus_still_allowed','chosen_cpus_still_online',
 'stable_process_cgroups_exact','stable_process_leader_masks_within_chosen','observed_thread_masks_within_chosen',
 'observed_process_PID_starts_exact','observed_TID_starts_exact','observed_process_cgroups_exact','observed_process_leader_masks_within_chosen'}
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
  self.thread_union=set();self.leaders=set();self.boundary_leaders=set();self.cgroups=set();self.pid_intersection=None;self.starts={};self.labels={};self.first=None;self.last=None
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
   if r['leader']is not None:self.boundary_leaders.add(tuple(r['leader']))
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
  need(self.thread_union<=chosen and all(bool(m)and set(m)<=chosen for m in self.leaders),'external observed task/leader mask must remain a nonempty chosen-pool subset')
  need(all(set(m)==chosen for m in self.boundary_leaders),'own boundary leader must retain the exact declared27 mask')
  need(all(c=='0::'+expected['cgroup']['membership']for c in self.cgroups),'any observed cgroup mismatch cannot heal')
  if identity is not None and self.count:
   pid,tick=identity;need(pid in self.pid_intersection,'required root PID missing in a CPU observation')
   if pid in self.starts:need(self.starts[pid]=={tick},'required root PID/start mismatch')
  if interval is not None and self.first is not None:need(interval[0]<=self.first<=self.last<=interval[1],'complete CPU observation brackets outside actual lifetime')
  return {'status':'RECORDED_CPU_OBSERVATIONS_CHECKED'if self.count else'CPU_UNOBSERVED','raw_observations':self.count,
   'independently_complete_process_scans':self.complete,'independently_incomplete_process_scans':self.incomplete,'transient_tree_unobserved':self.transient,
   'observed_process_leader_mask_variants':[list(v)for v in sorted(self.leaders)],
   'own_boundary_leader_mask_variants':[list(v)for v in sorted(self.boundary_leaders)],
   'recorded_stable_processes':self.stable_processes,'recorded_unstable_processes':self.unstable_processes,'recorded_tasks':self.recorded_tasks,
   'resource_invariant_variants':len(self.projections),'ambient_pool_variants':[{'cpu_affinity':list(v),'observations':n}for v,n in sorted(self.pools.items())],
   'first_observed_ns':self.first,'last_bracket_ns':self.last,'boundary_receipts_checked':self.boundaries,'all_boundary_attempts_checked':self.boundary_attempts,
   'maximum_boundary_attempts':self.max_attempts,'maximum_declared_boundary_elapsed_seconds':self.max_elapsed,
   'masks_resources_PID_TID_cgroup_stability_independently_recomputed':bool(self.count),'exact_boundary_wall_start_and_sleep_durations_independently_recomputed':False,
   'scope':'Full recorded PID/TID/cgroup brackets and every boundary attempt recomputed; external nonempty subset masks and own exact27 boundary leaders are checked separately without repin or PSS filtering; observed hard violation cannot heal. Declared elapsed/limits/source-bound10ms pause checked with raw duration lower-bound consistency; absolute boundary start and actual sleep durations were not serialized. Sampled, non-atomic, no continuous or exclusive placement guarantee.'}

def boundary(value,contract,owner=None,interval=None,label=None):
 need(bounded_string(value.get('label'))and(label is None or value['label']==label),'original own-boundary role label')
 facts=Facts();facts.ingest(value,boundary=True);return facts.bind(contract,owner,interval)

# END embedded independent ENV bracket decoder.
env=SimpleNamespace(POLICY=POLICY,QUOTA=QUOTA,MEMORY=MEMORY,mask=mask,projection=projection,boundary=boundary,Facts=Facts)
def need(ok,message):
 if not ok:raise ValueError(message)


def decode(raw):
 def unique(pairs):
  out={}
  for key,value in pairs:need(key not in out,'duplicate JSON key');out[key]=value
  return out
 def finite_float(token):
  value=float(token);need(math.isfinite(value),'nonfinite JSON exponent');return value
 return json.loads(raw,object_pairs_hook=unique,parse_float=finite_float,
  parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))


def sha_hex(value):return isinstance(value,str)and len(value)==64 and all(c in'0123456789abcdef'for c in value)


def recorded_path(value):
 need(isinstance(value,str)and 0<len(value)<=2048,'bounded recorded path required')
 path=PurePosixPath(value)
 need(path.is_absolute()and'..'not in path.parts and'\\'not in value and str(path)==value,'canonical absolute recorded POSIX path required')
 return path


def pin(path,maximum):
 p=Path(path);need(p.is_file()and not p.is_symlink()and p.stat().st_size<=maximum,'bounded regular input required')
 digest=hashlib.sha256();size=0
 with p.open('rb')as stream:
  for block in iter(lambda:stream.read(MIB),b''):size+=len(block);need(size<=maximum,'growing input');digest.update(block)
 need(size==p.stat().st_size,'input changed while hashing');return {'bytes':size,'sha256':digest.hexdigest()}


def same(actual,expected,label):
 if isinstance(expected,dict):
  need(isinstance(actual,dict)and set(actual)==set(expected),label+': keys')
  for key,value in expected.items():same(actual[key],value,label+'.'+key)
 elif isinstance(expected,list):
  need(isinstance(actual,list)and len(actual)==len(expected),label+': length')
  for i,value in enumerate(expected):same(actual[i],value,label+'['+str(i)+']')
 elif type(expected)is float:need(type(actual)in(int,float)and math.isfinite(actual)and math.isclose(actual,expected,rel_tol=2e-12,abs_tol=1e-10),label+': number')
 else:need(type(actual)is type(expected)and actual==expected,label+': identity')


def clean(value,guard=True):
 need(type(value['returncode'])is int and value['returncode']==0 and value['timed_out']is False
  and value['orphan_cleanup']is False and value['owned_tree_quiescent']is True and not value['owned_pgids_signaled']
  and value['actual_wait_returned']is True,'actual clean wait0/quiescence required')
 need(type(value['pid'])is int and value['pid']>0 and type(value['start_ticks'])is int and value['start_ticks']>0,'actual positive PID/start required')
 if guard:need(value['guard_abort_reason']is None,'resource guard abort cannot be successful')


def pss_row(sample):
 """Derive eligibility from original memberships/process records, not flags."""
 if 'sampling_error'in sample:return None,'sampling_error'
 before,after=sample['membership_before'],sample['membership_after'];processes=sample['processes']
 def members(rows):
  need(len({r['pid']for r in rows})==len(rows),'duplicate PSS PID')
  need(all(type(r['pid'])is int and r['pid']>0 and type(r['start_ticks'])is int and r['start_ticks']>0 for r in rows),'invalid PSS membership identity')
  return {r['pid']:r['start_ticks']for r in rows}
 a,b=members(before),members(after)
 stable=sample['membership_before_complete']is True and sample['membership_after_complete']is True and a==b
 complete=bool(processes)and len(processes)==len(a)and all(type(r['pss_bytes'])is int and r['pss_bytes']>=0 for r in processes)
 need(len({r['pid']for r in processes})==len(processes),'duplicate PSS process record')
 if complete:need({r['pid']for r in processes}==set(a),'complete PSS records/membership differ')
 need(sample['membership_stable']is stable and sample['pss_complete']is complete
  and sample['publication_eligible']is(stable and complete),'raw PSS membership/completeness flags contradict records')
 need(sample['decision']['abort_reason']is None,'observed resource guard abort')
 need(all(isinstance(sample[k],str)and 0<len(sample[k])<=128 for k in('phase_before','phase_after')),
  'bounded named PSS phases required')
 same_phase=sample['phase_before']==sample['phase_after'];eligible=stable and complete and same_phase
 need(sample['phase_stable']is same_phase and sample['phase_publication_eligible']is eligible,'phase eligibility flag contradicts recorded phases')
 if complete:need(sample['tree_pss_bytes']==sum(r['pss_bytes']for r in processes),'PSS tree sum differs from original process PSS')
 if eligible:
  need(sample['backend']=='linux-proc-smaps_rollup','actual Linux PSS backend required')
  return (sample['phase_before'],sum(r['pss_bytes']for r in processes)),'eligible'
 return None,'ineligible'


def counter(value):return type(value)is int and 0<=value<1<<128


def linux_major_minor(device):
 need(type(device)is int and 0<=device<1<<64,'recorded Linux dev_t range')
 major=((device>>8)&0xfff)|((device>>32)&0xfffff000)
 minor=(device&0xff)|((device>>12)&0xffffff00)
 return str(major)+':'+str(minor)


class DiskFacts:
 def __init__(self):
  self.count=0;self.first=self.last=None;self.devices=set();self.paths=set()
  self.minimum_bytes=None;self.minimum_inodes=None;self.inodes_exposed=0;self.inodes_unreported=0
 def ingest(self,value):
  need(set(value)==DISK_KEYS,'complete raw observation schema')
  tick=value['observed_monotonic_ns'];need(type(tick)is int and tick>0 and(self.last is None or tick>self.last),'strictly ordered raw disk clocks')
  need(isinstance(value['path'],str)and 0<len(value['path'])<=2048 and counter(value['device']),'bounded original path/device')
  need(all(counter(value[k])for k in('bytes_available','bytes_total','inodes_available','inodes_total')),'nonnegative original integer capacities')
  need(4*GIB<=value['bytes_available']<=value['bytes_total']and value['mount_readonly']is False,'observed low bytes or readonly mount cannot heal')
  reported=value['inodes_total']>0
  need(value['inodes_reported']is reported and(not reported or 0<value['inodes_available']<=value['inodes_total']),
   'positive available inodes required only where a total is exposed')
  need(isinstance(value['scope'],str)and 0<len(value['scope'])<=1024,'bounded raw scope')
  self.first=tick if self.first is None else self.first;self.last=tick;self.count+=1
  self.devices.add(value['device']);self.paths.add(value['path']);need(len(self.devices)==len(self.paths)==1,'output device/path changed between observations')
  self.minimum_bytes=value['bytes_available']if self.minimum_bytes is None else min(self.minimum_bytes,value['bytes_available'])
  if reported:
   self.inodes_exposed+=1;self.minimum_inodes=value['inodes_available']if self.minimum_inodes is None else min(self.minimum_inodes,value['inodes_available'])
  else:self.inodes_unreported+=1
 def bind(self,device,path,inventory):
  need(self.count>0 and self.devices=={device}and self.paths=={path},'all runtime rows must match root-admitted output device/path')
  need(set(inventory)=={'observed_monotonic_ns','disk_observed_monotonic_ns','observed_non_runtime_bytes',
   'observed_non_runtime_files','vanished_entries','complete_entry_observations','scope'},'bounded latest inventory schema')
  need(all(counter(inventory[k])for k in('observed_monotonic_ns','disk_observed_monotonic_ns','observed_non_runtime_bytes',
   'observed_non_runtime_files','vanished_entries'))and inventory['disk_observed_monotonic_ns']==self.last<=inventory['observed_monotonic_ns'],
   'latest inventory must bind final raw disk observation')
  need(inventory['observed_non_runtime_bytes']<=2*GIB and inventory['complete_entry_observations']is(inventory['vanished_entries']==0),
   'sampled output cap/vanished-entry completeness contradicts raw values')
  need(isinstance(inventory['scope'],str)and 0<len(inventory['scope'])<=1024,'bounded latest inventory scope')
  return {'raw_disk_observations_checked':self.count,'first_observed_ns':self.first,'last_observed_ns':self.last,
   'device':device,'path':path,'minimum_reported_bytes_available':self.minimum_bytes,
   'inode_reporting_observations':self.inodes_exposed,'inode_unreported_observations':self.inodes_unreported,
   'minimum_reported_inodes_available':self.minimum_inodes,'latest_inventory':inventory,
   'runtime_thresholds_independently_checked':True,'raw_statvfs_block_arithmetic_independently_recomputed':False,
   'scope':'Each recorded available-byte/inode/readonly/device threshold and latest sampled inventory checked. Byte availability was derived by pinned statvfs code; raw block components were not serialized. Initial dataset-native-bytes-plus6GiB/positive exposed inodes admission is checked separately from runtime4GiB/positive exposed inodes. No instantaneous quota, capacity reservation, node eviction or continuous-health guarantee.'}


def load_archive(path,expected_sha):
 whole=pin(path,128*MIB);need(whole['sha256']==expected_sha,'independently collected whole archive SHA differs')
 payloads,infos,pss={}, {}, {};total=kept=0;events=env.Facts();disk_facts=DiskFacts()
 class BoundedTarBytes:
  def __init__(self,stream):self.stream,self.size=stream,0
  def read(self,size=-1):
   block=self.stream.read(min(MIB,size)if size>=0 else MIB);self.size+=len(block)
   need(self.size<=400*MIB,'whole decompressed tar/header cap');return block
 with gzip.open(path,'rb')as compressed,tarfile.open(fileobj=BoundedTarBytes(compressed),mode='r|')as tar:
  for member in tar:
   name=member.name;rel=PurePosixPath(name)
   need(name=='SHA256_MANIFEST.json'or name.startswith('closure/'),'exact closure raw archive namespace required')
   need(member.isfile()and not rel.is_absolute()and'..'not in rel.parts and'\\'not in name and str(rel)==name
    and name not in infos and len(infos)<4096 and 0<=member.size<=32*MIB and set(member.pax_headers)<={'path'}
    and member.pax_headers.get('path',name)==name,'bounded regular canonical member required')
   total+=member.size;need(total<=384*MIB,'whole raw payload cap')
   digest=hashlib.sha256();size=0;data=bytearray();is_pss=name.startswith('closure/results/')and name.endswith('.pss.jsonl')
   is_events=name=='closure/results/environment_events.jsonl'
   is_disk=name=='closure/results/disk_observations.jsonl'
   if is_disk:need(0<member.size<=8*MIB,'bounded nonempty8MiB disk history required')
   phases={};counts={'eligible':0,'ineligible':0,'sampling_error':0};peak=None;env_facts=env.Facts();first_sample=last_sample=None
   with tar.extractfile(member)as stream:
    if is_pss or is_events or is_disk:
     while True:
      line_cap=4096 if is_disk else MIB;line=stream.readline(line_cap+1)
      if not line:break
      need(len(line)<=line_cap and line.endswith(b'\n'),'complete bounded JSONL line required');size+=len(line);digest.update(line)
      sample=decode(line)
      if is_disk:disk_facts.ingest(sample);continue
      if is_events:events.ingest(sample,boundary=True);continue
      tick=sample['monotonic_ns'];need(type(tick)is int and tick>0 and(last_sample is None or tick>last_sample),'strictly ordered raw PSS/error sample clocks required')
      first_sample=tick if first_sample is None else first_sample;last_sample=tick
      row,status=pss_row(sample);counts[status]+=1
      if status!='sampling_error':env_facts.ingest(sample['environment'])
      if row:
       phase,value=row;entry=phases.setdefault(phase,{'eligible_samples':0,'max_sampled_tree_pss_bytes':0})
       entry['eligible_samples']+=1;entry['max_sampled_tree_pss_bytes']=max(entry['max_sampled_tree_pss_bytes'],value)
       need(type(sample['monotonic_ns'])is int and sample['monotonic_ns']>0
        and type(sample['process_count'])is int and sample['process_count']==len(sample['processes']),'eligible PSS timestamp/process count differs')
       if peak is None or value>peak['tree_pss_bytes']:
        peak={'tree_pss_bytes':value,'phase':phase,'scan_end_monotonic_ns':sample['monotonic_ns'],
         'process_count':sample['process_count'],'sample_index_1based':sum(counts.values())}
    else:
     for block in iter(lambda:stream.read(MIB),b''):
      size+=len(block);digest.update(block)
      if name.endswith('.json'):
       kept+=len(block);need(kept<=64*MIB,'retained JSON metadata cap');data.extend(block)
   need(size==member.size,'truncated member');infos[name]={'bytes':size,'sha256':digest.hexdigest()}
   if is_pss:pss[name]={'phases':phases,'raw_observation_counts':counts,'all_phases_maximum':peak,'environment_facts':env_facts,
    'first_raw_sample_ns':first_sample,'last_raw_sample_ns':last_sample}
   elif name.endswith('.json'):payloads[name]=decode(data)
 need('SHA256_MANIFEST.json'in payloads,'whole-member manifest required')
 need(payloads['SHA256_MANIFEST.json']['files']=={k:v for k,v in infos.items()if k!='SHA256_MANIFEST.json'},'every member must match internal manifest')
 need(pin(path,128*MIB)==whole,'archive changed while reading');return payloads,infos,pss,whole,events,disk_facts


def environment_job_labels(events,name,launch_ns,wait_ns,controller):
 labels=[]
 for prefix in('before_command_','immediately_before_Popen_','after_actual_wait_'):
  key=prefix+name;need(key in events.labels,'missing original ENVv3 named command guard: '+key);value=events.labels[key]
  need(value['count']==1 and value['identities']=={controller},'unique actual controller boundary identity required')
  labels.append(value)
 need(labels[0]['last_bracket_ns']<=labels[1]['first_scan_ns']<=labels[1]['last_bracket_ns']<=launch_ns<wait_ns<=labels[2]['first_scan_ns'],
  'ENVv3 complete before/Popen/actual-wait guard spans differ')
 return {prefix:{'first_scan_ns':value['first_scan_ns'],'last_bracket_ns':value['last_bracket_ns']}for prefix,value in zip(('before_command','immediately_before_Popen','after_actual_wait'),labels)}



DISK_KEYS={'observed_monotonic_ns','path','device','bytes_available','bytes_total','inodes_available','inodes_total','inodes_reported','mount_readonly','scope'}

ARMS=('reference_union_compiled','reference_union_native_fresh','reference_union_native_reuse')

def grid(dataset):
 need(dataset in ('friendster','citation2'),'declared dataset required')
 rng=random.Random(20261127);bases={}
 for graph in ('friendster','citation2'):
  bases[graph]={}
  for horizon in ('short320','long6400'):
   order=list(ARMS);rng.shuffle(order);bases[graph][horizon]=order
 jobs=[]
 for repetition in (1,2,3):
  horizons=('short320','long6400')if repetition%2 else('long6400','short320')
  for horizon in horizons:
   base=bases[dataset][horizon];offset=repetition-1
   for position,arm in enumerate(base[offset:]+base[:offset],1):
    jobs.append(dict(name=f'{dataset}_{horizon}_w8_r{repetition}_{arm}',dataset_id=dataset,
     horizon=horizon,workers=8,repetition=repetition,position=position,arm=arm))
 return jobs

def native_files(expected):
 return {name:{'bytes':expected[size],'sha256':expected[digest]}for name,size,digest in
  (('rowptr.i8','rowptr_bytes','rowptr_sha256'),('col.i8','col_bytes','col_sha256'))}

def preblock_minimum(dataset):
 expected=BINDINGS[dataset]['native']
 return expected['rowptr_bytes']+expected['col_bytes']+6*GIB

def input_prefixes(reports):
 """Cross-arm/full-horizon and cross-horizon semantic identity, not tensor regeneration."""
 need(len(reports)==6,'all six applications required for input identity')
 streams={};prefixes={}
 for horizon in H:
  selected=[r for r in reports.values()if r['config']['horizon']==horizon]
  need(len(selected)==3 and {r['config']['arm']for r in selected}==set(ARMS),
   'exact three physical routes per horizon')
  first=selected[0]
  need(all(r['input_stream_sha256']==first['input_stream_sha256']and
   r['semantic_input_prefixes']==first['semantic_input_prefixes']and
   r['excluded_warmup_prefix_audits']==first['excluded_warmup_prefix_audits']for r in selected),
   'cross-arm ordered consumed-input/prefix/warmup audit equality differs')
  streams[horizon]=first['input_stream_sha256'];prefixes[horizon]=first['semantic_input_prefixes']
 short=streams['short320']
 need(sha_hex(short)and short==prefixes['short320']['576']==prefixes['long6400']['576']
  and streams['long6400']==prefixes['long6400']['6656'],
  'actual short terminal stream must equal long semantic576 prefix')
 need(len({json.dumps(r['excluded_warmup_prefix_audits'],sort_keys=True)for r in reports.values()})==1,
  'all six applications must retain the same32 warmup query audits')
 return streams,prefixes

def paired_rows(reports,metrics,block):
 rows=[];ratios=[]
 for horizon in H:
  arms={arm:{'name':next(name for name,r in reports.items()if r['config']['horizon']==horizon and
   r['config']['arm']==arm),'metrics':None}for arm in ARMS}
  for arm,value in arms.items():value['metrics']=metrics[value['name']]
  rows.append(dict(horizon=horizon,workers=8,repetition=block,arms=arms))
  for numerator,denominator in ((ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[2],ARMS[1])):
   left,right=arms[numerator]['metrics'],arms[denominator]['metrics'];values={}
   need(set(left)==set(right),'paired raw metric axes differ')
   for key in left:
    need(type(left[key])in(int,float)and type(right[key])in(int,float)and math.isfinite(left[key])and
     math.isfinite(right[key])and left[key]>=0 and right[key]>0,'positive finite paired denominator required')
    with localcontext()as context:
     context.prec=50;values[key]=float(Decimal(str(left[key]))/Decimal(str(right[key])))
   ratios.append(dict(horizon=horizon,workers=8,repetition=block,numerator=numerator,denominator=denominator,metrics=values))
 return rows,ratios

def environment_admission(get,infos,plan,done,events,platform,source_sha):
 root='closure/results/';chosen=platform['cpu_affinity']
 need(len(env.mask(chosen))==27 and plan['environment_policy']==done['environment_policy']==env.POLICY
  and plan['selected_cpu_affinity']==chosen and done['environment_guard_success']is True
  and done['source_manifest_sha256']==source_sha,'actual source/CPU envelope differs')
 receipt=get(root+'ENVIRONMENT_ADMISSION.json');contract=receipt['contract']
 need(infos[root+'ENVIRONMENT_ADMISSION.json']['sha256']==done['environment_admission_sha256']
  and receipt['status']=='PASS'and receipt['policy']==env.POLICY and
  receipt['chosen_cpu_affinity']==chosen==contract['chosen_cpu_affinity']and contract['policy']==env.POLICY
  and contract['pin_calls']==1 and not contract.get('portable_unsupported',False),'actual once-pin Linux admission required')
 base=contract['resources'];env.projection(base)
 need(base['hostname']==platform['hostname']and base['boot_id']==platform['boot_id']and
  base['cpu_max']==platform['cpu_max']==env.QUOTA and base['memory_max']==platform['memory_max']==env.MEMORY
  and contract['topology']==platform['topology']and env.mask(chosen)<=env.mask(contract['admission_before_affinity']),
  'actual host/boot/CPU/quota/memory/topology binding differs')
 controller=(done['controller_identity']['pid'],done['controller_identity']['start_ticks'])
 observed=env.boundary(receipt['after'],contract,controller,label='controller_admission')
 need(events.labels.get('controller_admission',{}).get('count')==1 and 'assert_inputs'in events.labels,
  'unique controller admission and source continuity events required')
 return contract,dict(admission=observed,controller_events=events.bind(contract,controller))

def storage_admission(get,infos,plan,done,disk,platform,platform_sha):
 root='closure/results/';storage=platform['storage'];base=recorded_path(platform['base'])
 minimum=preblock_minimum(plan['dataset_id'])
 need(plan['preblock_minimum_bytes_available']==minimum and
  plan['protocol']['controller_guard_amendment']['storage_admission']['preblock_minimum_bytes_available']==
  {dataset:preblock_minimum(dataset)for dataset in BINDINGS}and
  plan['protocol']['controller_guard_amendment']['storage_admission']['runtime_minimum_bytes_available']==4*GIB,
  'independently derived per-dataset native-bytes-plus6GiB preblock/runtime guard thresholds differ')
 need(done['controller_guard_failure']is None and not any(n.startswith(root)and
  (n.endswith('ENVIRONMENT_FAILURE.json')or n.endswith('CONTROLLER_GUARD_FAILURE.json')or n.endswith('FAILED.json'))for n in infos),
  'any actual environment/controller/application failure excludes block')
 need(platform['kind']=='CLOSURE_V1_ACTUAL_BLOCK_PLATFORM_ADMISSION'and platform['block']==plan['repetition_block']
  and platform['dataset_id']==plan['dataset_id']and platform['actual_scientific_jobs_started']==0,
  'actual pre-science selected-dataset platform proof required')
 binding=plan['platform_admission']
 need(binding['proof']==platform and binding['sha256']==platform_sha and
  binding['bytes']==infos[root+'PLATFORM_ADMISSION.json']['bytes']and done['platform_admission_sha256']==platform_sha,
  'dynamic original root platform file binding differs')
 need(storage['mount']['mount_point']==str(base)and storage['mount']['root']=='/'+storage['subPath']
  and storage['mount']['major_minor']==linux_major_minor(storage['device'])and
  'rw'in storage['mount']['options']and'ro'not in storage['mount']['options']and
  'ro'not in storage['mount']['super_options']and storage['mount_readonly']is False and
  storage['bytes_available']>=minimum and(storage['inodes_total']==0 or storage['inodes_available']>0)
  and storage['write_fsync_readback_unlink']is True,'actual admitted writablePVC native-bytes-plus6GiB/probe differs')
 layer=platform['root_layer'];need(layer['mount_readonly']is False and layer['bytes_available']>=4*GIB
  and layer['write_fsync_readback_unlink']is True,'actual writable-root reserve/probe differs')
 kube=platform['kubernetes_binding'];need(all(kube[k]==platform[k]for k in('pod_uid','sts_uid','hostname','node','image_digest'))
  and all(kube[k]==storage[k]for k in('PVC_name','PVC_uid','volume_name','subPath')),'actual API/image/PVC identity projection differs')
 need('H100'in platform['gpu']['name']and platform['gpu']['memory_total_mib']>=79000,'whole H10080 class required')
 for value in(plan['out'],plan['graph_root'],plan['native_reuse_root'],plan['coordination_dir'],binding['source_path']):
  need(recorded_path(value).is_relative_to(base),'recorded path outside admitted project mount')
 return disk.bind(storage['device'],plan['out'],get(root+'output_inventory_observation.json'))

def source_and_inputs(get,infos,plan,platform,pins):
 source='closure/sources/packet/';root='closure/results/';dataset=plan['dataset_id'];known=BINDINGS[dataset]
 need(infos[source+'SOURCE_MANIFEST.json']['sha256']==pins['source_manifest_sha256'],'whole reviewed packet manifest differs')
 manifest=get(source+'SOURCE_MANIFEST.json')['files']
 need({n[len(source):]for n in infos if n.startswith(source)}==set(manifest)|{'SOURCE_MANIFEST.json'},
  'complete source inventory required')
 for name,value in manifest.items():need(infos[source+name]==value,'pinned source member differs: '+name)
 protocol=get(source+'PROTOCOL.json');same(protocol,plan['protocol'],'source/plan exact protocol')
 need(hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()==pins['protocol_sha256']==plan['protocol_sha256'],
  'canonical reviewed protocol SHA differs')
 need(protocol['datasets']==['friendster','citation2']and protocol['arms']==list(ARMS)and protocol['workers']==[8]
  and protocol['repetitions']==3 and protocol['timing_jobs']==18 and protocol['order_seed']==20261127
  and protocol['seed']==20261124 and protocol['batch_size']==32 and protocol['warmup_steps']==8,
  'declared two-dataset W8 three-arm36-app protocol differs')
 need(protocol['allocation_blocks']['applications_per_block']==6 and protocol['allocation_blocks']['blocks']==[1,2,3]
  and protocol['allocation_blocks']['global_applications_per_dataset']==18 and
  protocol['allocation_blocks']['global_two_dataset_applications']==36,'three complete datasetblocks required')
 for h,(steps,measured,consumed,points)in H.items():
  same(protocol['horizons'][h],dict(steps=steps,measured_graphs=measured,consumed_graphs=consumed,checkpoints=points),'fixed horizon')
 graph=get(source+'pins/'+dataset+'_graph_expected.json');native=get(source+'pins/'+dataset+'_native_expected.json')
 need(graph['num_nodes']==known['nodes']and graph['stored_entries']==known['stored_entries']and graph['files']==known['graph_files'],
  'independently known compact graph identities differ')
 need(all(native[k]==v for k,v in known['native'].items()),'independently known full native identities differ')
 binding=plan['dataset_binding'];need(binding['dataset_id']==dataset and binding['graph']==graph and binding['native']==native
  and binding['workload_sha256']==known['workload_sha256'],'exact dataset binding differs')
 identity=get(root+'input_identity.json');graphpath=recorded_path(plan['graph_root'])
 need(identity['whole_hashes']is True and set(identity['files'])=={str(graphpath/name)for name in known['graph_files']},
  'all three actual compact source-file identities required')
 for name,pinvalue in known['graph_files'].items():
  value=identity['files'][str(graphpath/name)];need(all(value[k]==v for k,v in pinvalue.items())and
   all(counter(value[k])for k in('mtime_ns','inode','device'))and value['device']==platform['storage']['device'],
   'source-bound whole compact hash/stat/current device differs')
 workload=infos[source+'workloads/'+dataset+'_pairs.npy']
 need(workload==infos[root+'pairs.npy']and workload=={'bytes':192128,'sha256':known['workload_sha256']},'existing E23 query workload differs')
 host=get(root+'host.json');hardware=get(root+'hardware.json');target=hardware['target'];gpu=platform['gpu']
 need(host['hostname']==platform['hostname']and host['boot_id']==platform['boot_id']and host['cpu_affinity']==platform['cpu_affinity']
  and host['versions']==get(source+'SOURCE_PINS.json')['runtime_versions']and host['python'].startswith('3.10.'),
  'actual hostname/boot/CPU/Python/library versions differ')
 need(all(target[k]==gpu[v]for k,v in(('uuid','uuid'),('name','name'),('memory_total_mib','memory_total_mib'),('driver_version','driver')))
  and hardware['permitted_inventory']==[target],'actual GPU/driver inventory differs')
 return native

def qualification(get,pss,contract,events,controller,source_sha):
 root='closure/results/';folder=root+'qualification/';q=get(folder+'QUALIFIED.json')
 need(q['success']is True and q['device']=='cuda'and q['workers']==[8]and q['actual_consumers']==3
  and q['source_manifest_sha256']==source_sha and q['environment_policy']==env.POLICY and
  q['chosen_cpu_affinity']==contract['chosen_cpu_affinity']and q['environment_target_proof']is True,
  'actual same-contract three-arm W8 CUDA qualification required')
 need(get(folder+'ENVIRONMENT_ADMISSION.json')=={'status':'PASS','contract':contract},'qualification contract differs')
 support=[]
 for name in ('build_native','actual_cuda_qualification'):
  launch=get(root+name+'.launch.json');exit=get(root+name+'.exit.json');clean(exit)
  need(exit['pid']==launch['pid']and exit['start_ticks']==launch['start_ticks']and
   exit['launch_monotonic_ns']==launch['launch_monotonic_ns'],'actual support-command identity differs')
  interval=(launch['launch_monotonic_ns'],exit['wait_returned_monotonic_ns'])
  labels=environment_job_labels(events,name,*interval,controller);path=root+name+'.pss.jsonl'
  value=pss.get(path)
  if value and value['first_raw_sample_ns']is not None:
   need(interval[0]<=value['first_raw_sample_ns']<=value['last_raw_sample_ns']<=interval[1],
    'support PSS/error observations outside actual lifetime')
  proof=value['environment_facts'].bind(contract,(launch['pid'],launch['start_ticks']),interval,
   require_observed=False)if value else{'status':'CPU_UNOBSERVED'}
  support.append(dict(name=name,controller_labels=labels,sampled_CPU=proof))
 outer=(get(root+'actual_cuda_qualification.launch.json')['launch_monotonic_ns'],
  get(root+'actual_cuda_qualification.exit.json')['wait_returned_monotonic_ns'])
 records=[];reports=[]
 for arm in ARMS:
  current=folder+'w8_'+arm+'/';r=get(current+'report.json');exit=get(current+'exit.json');cfg=get(current+'config.json')
  need(type(exit['returncode'])is int and exit['returncode']==0 and exit['actual_wait_returned']is True and
   r['status']=='complete'and r['config']==cfg and cfg['environment_contract']==contract and
   cfg['fixture']is True and cfg['num_workers']==8 and cfg['arm']==arm and r['device']['type']=='cuda'
   and 'H100'in r['device']['name']and r['model']['initial_state_sha256']==MODEL_SHA and
   r['consumed_graphs']==20 and r['measured_graphs']==4,'actual tiny3-arm consumer completion differs')
  interval=(exit['launch_monotonic_ns'],exit['wait_returned_monotonic_ns'])
  need(outer[0]<=interval[0]<interval[1]<=outer[1]and r['launch_monotonic_ns']==interval[0],
   'tiny launch/wait outside original qualifier')
  owner=(exit['pid'],r['environment_start']['processes'][0]['start_ticks'])
  for key,label in (('environment_start','application_start'),('environment_end','application_end')):
   env.boundary(r[key],contract,owner,interval,label)
  workers=r['worker_receipts'];need(len(workers)==8 and{w['worker_id']for w in workers}==set(range(8))
   and len({w['pid']for w in workers})==8,'eight distinct tiny spawn workers required')
  for w in workers:
   need(w['status']=='closed'and w['native_closed']is True and
    w==get(current+'workers/worker'+str(w['worker_id'])+'.json'),'tiny normal worker close receipt differs')
   owner=(w['pid'],w['environment_start']['processes'][0]['start_ticks'])
   for key,label in (('environment_start','worker_start'),('environment_end','worker_end')):
    env.boundary(w[key],contract,owner,interval,label)
  records.append(exit);reports.append(r)
 need(q['actual_child_exits']==records and q['initial_model_sha256']==MODEL_SHA and
  len({r['input_stream_sha256']for r in reports})==1,'tiny actual exits/model/input equality differs')
 same(q['comparisons'],[dict(workers=8,consumed_graphs=20,all_actual_consumed_inputs_exact=True,
  stream_sha256=reports[0]['input_stream_sha256'])],'qualification all-three input streams')
 return dict(support_commands=support,actual_H100_W8_three_arm_consumers=3,all_normal_workers=24)

def reuse_preparation(get,infos,plan,native,source_sha):
 root='closure/results/';pre=get(root+'native_reuse_PREPARATION_READY.json')
 receipt_sha=plan['native_reuse_receipt_sha256'];need(sha_hex(receipt_sha)and
  infos[root+'native_reuse_PREPARATION_READY.json']['sha256']==receipt_sha and
  pre==plan['native_reuse_preparation'],'byte-exact once-per-dataset prior preparation receipt required')
 need(pre['kind']=='CLOSURE_V1_IMMUTABLE_NATIVE_REUSE'and pre['dataset_id']==plan['dataset_id']and
  pre['source_manifest_sha256']==source_sha and pre['source_graph_root']==plan['graph_root']and
  pre['source_graph_files']==BINDINGS[plan['dataset_id']]['graph_files']and pre['expected_native']==native,
  'source-bound dataset/source/native preparation identity differs')
 need(pre['nodes']==native['nodes']and pre['stored_entries']==native['stored_entries']and
  pre['files']==native_files(native)and pre['one_time_per_dataset']is True and
  pre['outside_all_application_clocks']is True and pre['reusable_arrays_readonly']is True and
  pre['no_private_worker_mapper_prepared']is True,'whole native preparation/read-only/private-map boundary differs')
 for name,value in pre['file_stats'].items():
  need(name in pre['files']and value['bytes']==pre['files'][name]['bytes']and
   all(counter(value[k])for k in('mtime_ns','inode','device')),'native preparation full file/stat binding differs')
 need(set(pre['file_stats'])==set(pre['files']),'both immutable array stat records required')
 promotion=pre['native_promotion'];a=pre['source_admission_begin_ns'];b=pre['native_validation_begin_ns'];c=pre['native_validation_end_ns']
 need(pre['preparation_entry_monotonic_ns']<=a<=promotion['begin_ns']<=promotion['end_ns']<=b<=c
  <=pre['preparation_complete_monotonic_ns'],'one-process preparation milestone order differs')
 need(promotion['rowptr_bytes']==native['rowptr_bytes']and promotion['col_bytes']==native['col_bytes']and
  promotion['files_fsynced']is True and promotion['sentinel_added']is True and
  promotion['promotion_scope']=='once_per_dataset'and promotion['full_native_copy_once_per_app']is False,
  'genuine once-per-dataset compact-to-int64 fsynced preparation differs')
 need(bounded_string(pre['hostname'])and bounded_string(pre['boot_id'])and type(pre['observed_unix'])in(int,float)
  and math.isfinite(pre['observed_unix'])and pre['observed_unix']>0,'preparation host/boot/wallclock provenance required')
 same(pre['conversion_seconds'],(promotion['end_ns']-promotion['begin_ns'])/1e9,'independent one-time conversion duration')
 same(pre['validation_seconds'],(c-b)/1e9,'independent one-time full-readback duration')
 admission=get(root+'native_reuse_admission.json');descriptor=admission['descriptor']
 device=plan['platform_admission']['proof']['storage']['device']
 expected=dict(root=plan['native_reuse_root'],receipt_sha256=receipt_sha,expected_native=native,
  immutable_open_only=True,preparation_outside_all_application_clocks=True,
  full_integrity_readback_inside_application=False,
  preparation_file_devices={name:pre['file_stats'][name]['device']for name in pre['files']},
  actual_file_devices={name:device for name in pre['files']},actual_reuse_root_device=device,
  bound_expected_device=device,verified_files=None,full_integrity_readback_outside_application=False,
  device_scope='Preparation device values are historical host-local observations; current devices are bound to this allocation PVC proof.')
 full_expected={**expected,'verified_files':native_files(native),'full_integrity_readback_outside_application':True}
 same(descriptor,full_expected,'immutable native reuse admission full-hash/current-device descriptor')
 need(admission['outside_application_lifetimes']is True and admission['preparation']==pre,
  'preexisting reusable native receipt must be admitted before applications')
 return pre,expected

def config_expected(plan,job,contract,native):
 out=recorded_path(plan['out']);folder=out/'jobs'/job['name'];steps,_,_,points=H[job['horizon']]
 return dict(dataset_id=plan['dataset_id'],horizon=job['horizon'],fixture=False,graph_path=plan['graph_root'],file_endian='little',
  pairs_path=str(out/'pairs.npy'),seed=20261124,hops=2,cap=200,oversample=1.5,auto_ratio=1.0,
  max_neighbors=20000000,warmup_pairs=256,steps=steps,prefetch_factor=1,audit_pairs=32,device='cuda',
  torch_threads=1,gpu_memory_budget_gib=24.0,stock_index_dtype='auto',batch_size=32,num_workers=8,arm=job['arm'],
  out=str(folder),worker_receipts=str(folder/'workers'),
  native_root=plan['native_reuse_root']if job['arm']==ARMS[2]else str(folder/'runtime/native'),
  native_library=str(out/'build/native_assoc_v1.so'),native_expected=native,
  native_reuse_receipt_sha256=plan['native_reuse_receipt_sha256'],
  native_reuse_expected_device=plan['platform_admission']['proof']['storage']['device'],
  native_storage_policy='fresh_inside_application'if job['arm']==ARMS[1]else
   'immutable_preconverted_reuse'if job['arm']==ARMS[2]else'compact_compiled',
  checkpoints=points,semantic_input_prefixes=sorted(set(points+[576])),environment_contract=contract)
H={'short320':(10,320,576,[32,256,576]),'long6400':(200,6400,6656,[32,256,1280,3456,6656])}
LABELS={32:'first_trained_batch',256:'warmup_complete',576:'measured_prefix320',1280:'measured_prefix1024',3456:'measured_prefix3200',6656:'measured_prefix6400'}
MODEL_SHA='f88254079fe6c152b4c0153c857a63516348353d578f918622e7f6719559618e'
MODEL={'name':'DGCNN','sortpool_k':281,'hidden_channels':32,'num_layers':3,'max_z':1000,'learning_rate':0.0001,'optimizer':'Adam','use_feature':False,'node_embedding':False,'parameters':596674}
BINDINGS={'friendster': {'nodes': 65608366, 'stored_entries': 3612134270, 'graph_files': {'edge_ends.bin': {'bytes': 14448537088, 'sha256': '0f74897efa3172252a8371bb0d2c8e0129602475ebb84a863ec1210f9c5e5973'}, 'edge_starts.bin': {'bytes': 524866928, 'sha256': 'b6ebf026fa54498f241af834b96b5496b2a1bcbf8be3edea2f4b3a1cf0a674e9'}, 'num_nodes.txt': {'bytes': 9, 'sha256': '8551541f6f2ef955481af83ef5e6f92c5f206c968d5d9db6a4437efecf767753'}}, 'native': {'nodes': 65608366, 'stored_entries': 3612134270, 'rowptr_bytes': 524866936, 'col_bytes': 28897074160, 'rowptr_sha256': '28060771441d71ba71877e55c0a48230c54d1b2715dd4edd08042019cf4a7a81', 'col_sha256': '012e1d0a907d2eb6b46d72f31be8b3ca722d88432b2c0509162c4ff487f25a5d'}, 'workload_sha256': '340e61cdd1eb8d329d52be164f919480e6a41feb4c620852df0393c2e9a192f0'}, 'citation2': {'nodes': 2927963, 'stored_entries': 60703502, 'graph_files': {'edge_ends.bin': {'bytes': 242814016, 'sha256': 'd3d8d1e2016675321c1d3d9ae817d2c6dca4a9dd2d5cdbba3e638ea7201057a4'}, 'edge_starts.bin': {'bytes': 23423704, 'sha256': 'f06070adc368892f8441ea15e691dce400f9d44b6ec3e6e4505134c276c1072d'}, 'num_nodes.txt': {'bytes': 8, 'sha256': 'ff1799f8bf9bd540c71427feb1bc07bcab1331eec0d2fcb847a8e68395b8e81f'}}, 'native': {'nodes': 2927963, 'stored_entries': 60703502, 'rowptr_bytes': 23423712, 'col_bytes': 485628016, 'rowptr_sha256': 'ab36a416ba782c06fbb04d5e536852dfc6e6478215137de4c0a4191b3b0fb82a', 'col_sha256': 'e02f86c875c0a257d79024f266c7ef43d0d0fac0c26bcfedab8506cceb439d57'}, 'workload_sha256': '648332c22cb6a25825e8487e2e82e39e2cb5dc53c8890cb3acb71b6324d7b750'}}

def audit_applications(get,infos,pss,events,plan,records,block,expected,environment_contract,done,launch_frame,native_expected,build,reuse_descriptor):
 root='closure/results/'
 reports={};metrics={};pss_output=[];raw_observations=[];pss_headline_runs=[];environment_jobs=[]
 previous_completion=get(root+'actual_cuda_qualification.exit.json')['wait_returned_monotonic_ns']
 for job,declared,record in zip(expected,plan['jobs'],records):
  need(all(declared[k]==v for k,v in job.items())and all(record[k]==job[k]for k in('name','horizon','workers','repetition','arm')),'original randomized grid/order differs')
  name=job['name'];folder=root+'jobs/'+name+'/';steps,measured,consumed,points=H[job['horizon']];cfg=declared['config']
  same(cfg,config_expected(plan,job,environment_contract,native_expected),'exact closure application defaults/paths/contract')
  need(cfg['horizon']==job['horizon']and cfg['steps']==steps and cfg['checkpoints']==points and cfg['semantic_input_prefixes']==sorted(set(points+[576]))
   and cfg['arm']==job['arm']and cfg['num_workers']==job['workers']and cfg['batch_size']==32 and cfg['warmup_pairs']==256
   and cfg['seed']==20261124 and cfg['prefetch_factor']==1 and cfg['hops']==2 and cfg['cap']==200 and cfg['device']=='cuda'
   and cfg['max_neighbors']==20000000 and cfg['auto_ratio']==1.0,'fixed consumer config differs')
  report=get(folder+'report.json');need(record['report_sha256']==infos[folder+'report.json']['sha256']and report['config']==cfg==get(folder+'config.json'),'report/config/hash binding differs')
  exit=get(root+name+'.exit.json');launch=get(root+name+'.launch.json');need(exit==record['exit'],'raw app exit differs');clean(exit)
  need(exit['pid']==launch['pid']and exit['start_ticks']==launch['start_ticks']and exit['launch_monotonic_ns']==launch['launch_monotonic_ns']==report['launch_monotonic_ns'],'actual app PID/start/clock binding differs')
  t=launch['launch_monotonic_ns'];end=exit['wait_returned_monotonic_ns'];need(t<=report['entry_monotonic_ns']<=report['imports_ready_ns']<=report['model_setup_begin_ns']
   <=report['model_setup_end_ns']<=report['iterator_begin_ns']<=report['iterator_created_ns']<=report['first_batch_ready_ns'],'ordered startup milestones differ')
  need(launch_frame['producer_launch_ns']<=previous_completion<=t<end<=launch_frame['producer_wait_ns']and launch['argv'][1:]==['-B','-u',str(PurePosixPath(launch_frame['packet'])/'lifecycle_child.py'),'--config',str(PurePosixPath(cfg['out'])/'config.json')],
   'actual app argv/config path or lifetime outside original producer')
  previous_completion=end
  need(cfg['environment_contract']==environment_contract,'exact same inherited ENVv5 contract in every app')
  app_environment={key:env.boundary(report[key],environment_contract,(launch['pid'],launch['start_ticks']),(t,end),label)for key,label in(('environment_start','application_start'),('environment_end','application_end'))}
  need(report['imports_ready_ns']<=app_environment['environment_start']['first_observed_ns']<=app_environment['environment_start']['last_bracket_ns']<=report['model_setup_begin_ns']
   and report['shutdown_end_ns']<=app_environment['environment_end']['first_observed_ns']<=app_environment['environment_end']['last_bracket_ns']<=end,'full application guard phase spans differ')
  app_environment['controller_labels']=environment_job_labels(events,name,t,end,(done['controller_identity']['pid'],done['controller_identity']['start_ticks']))
  need(report['status']=='complete'and report['consumed_graphs']==consumed and report['measured_graphs']==measured
   and all(type(report['model'][k])is type(v)and report['model'][k]==v for k,v in MODEL.items())and report['model']['initial_state_sha256']==MODEL_SHA
   and len(report['losses'])==steps and all(type(v)in(int,float)and math.isfinite(v)and v>=0 for v in report['losses']),'actual model/full consumer completion differs')
  need(report['device']['type']=='cuda'and'H100'in report['device']['name']and report['gpu_memory_guard']['allocator_budget_bytes']==24*GIB and report['gpu_reserved_peak_gib']<=24,'actual H100/24GiB consumer required')
  cps=report['checkpoints'];need([p['total_trained_graphs']for p in cps]==points and [p['label']for p in cps]==[LABELS[n]for n in points],'exact sync checkpoints differ')
  times=[p['monotonic_ns']for p in cps];need(report['first_batch_ready_ns']<=times[0]and all(b>a for a,b in zip(times,times[1:]))
   and times[1]<=report['steady_start_ns']<times[2]and report['training_end_ns']==times[-1]<=report['shutdown_begin_ns']<=report['shutdown_end_ns']<=end,'actual completion/shutdown/wait order differs')
  for p in cps:need(sha_hex(p['input_stream_sha256'])and p['measured_graphs']==max(0,p['total_trained_graphs']-256),'checkpoint input identity/count differs');same(p['launch_to_complete_seconds'],(p['monotonic_ns']-t)/1e9,'direct checkpoint clock')
  prefixes=report['semantic_input_prefixes'];need(set(prefixes)=={str(n)for n in sorted(set(points+[576]))}and all(sha_hex(s)for s in prefixes.values())
   and all(prefixes[str(p['total_trained_graphs'])]==p['input_stream_sha256']for p in cps)
   and report['input_stream_sha256']==prefixes[str(consumed)],'full consumed/prefix stream bindings differ')
  workers=report['worker_receipts'];need(len(workers)==job['workers']and{w['worker_id']for w in workers}==set(range(job['workers']))and len({w['pid']for w in workers})==job['workers'],'exact spawn worker identities required')
  for worker in workers:
   need(get(folder+'workers/worker'+str(worker['worker_id'])+'.json')==worker and worker['status']=='closed'and worker['native_closed']is True and worker['arm']==job['arm']
    and worker['torch_cpu_threads']==1 and all(v=='1'for v in worker['thread_environment'].values())and t<=worker['setup_begin_ns']<=worker['setup_end_ns']<=worker['close_begin_ns']<=worker['close_end_ns']<=end,'normal owned worker close/thread/clock evidence differs')
   worker_owner=(worker['pid'],worker['environment_start']['processes'][0]['start_ticks'])
   worker_proof={key:env.boundary(worker[key],environment_contract,worker_owner,(t,end),label)for key,label in(('environment_start','worker_start'),('environment_end','worker_end'))}
   need(worker['setup_begin_ns']<=worker_proof['environment_start']['first_observed_ns']<=worker_proof['environment_start']['last_bracket_ns']<=worker['original_graph_mmap_end_ns']<=worker['setup_end_ns']
    and worker['close_end_ns']<=worker_proof['environment_end']['first_observed_ns']<=worker_proof['environment_end']['last_bracket_ns']<=end,'full worker guard phase spans differ')
   signatures=worker['compiled_induction_signatures_at_close']
   if job['arm']in ARMS[1:]:need(worker['initialization_count']==1 and worker['native_initialization']['owner_pid']==worker['pid']
    and worker['native_initialization']['mode']=='persistent_touched_reset'and worker['native_initialization']['nodes']==native_expected['nodes']
    and worker['native_initialization']['stored_entries']==native_expected['stored_entries']
    and worker['native_initialization']['persistent_assoc_bytes_per_worker']==8*native_expected['nodes'] and worker['native_initialization']['library_sha256']==build['library_sha256']
    and worker['native_arrays']['readonly']is True and worker['native_arrays']['rowptr_bytes']+worker['native_arrays']['col_bytes']==native_expected['rowptr_bytes']+native_expected['col_bytes']
    and not any(signatures.values()),'one native-only private8N state per spawn worker required')
   else:need(worker['initialization_count']==0 and any(signatures.values()),'actual compiled kernel signature required; no native state')
  extracted=sum(w['completed_graphs']for w in workers);need(extracted==report['completed_extractions_including_unconsumed_prefetch']and consumed<=extracted<=consumed+32*job['workers'],'consumed/prefetch extraction bound differs')
  if job['arm']==ARMS[1]:
   promotion=report['native_promotion'];need(promotion['rowptr_bytes']+promotion['col_bytes']==native_expected['rowptr_bytes']+native_expected['col_bytes'] and promotion['files_fsynced']is True and promotion['full_native_copy_once_per_app']is True and report['native_reuse']is None and report['imports_ready_ns']<=promotion['begin_ns']<=promotion['end_ns']<=report['model_setup_begin_ns'],'fresh native preparation timing/bytes differ')
   post,cleanup=get(folder+'native_post_exit_integrity.json'),get(folder+'native_scratch_cleanup.json')
   need(post['outside_application_lifetime']is True and end<=post['observed_monotonic_ns']<=cleanup['monotonic_ns']<=launch_frame['producer_wait_ns']
    and cleanup['after_actual_exit_and_hash']is True and cleanup['owned_files_removed']==3 and post['expected_proof']==native_expected,'post-exit native verification/cleanup order differs')
   previous_completion=cleanup['monotonic_ns']
   for f,size_key,sha_key in(('rowptr.i8','rowptr_bytes','rowptr_sha256'),('col.i8','col_bytes','col_sha256')):need(post['files'][f]=={'bytes':native_expected[size_key],'sha256':native_expected[sha_key]},'whole derived native hash differs')
  elif job['arm']==ARMS[2]:
   need(report['native_promotion']is None,'reuse must not perform native conversion')
   same(report['native_reuse'],reuse_descriptor,'actual reused native open-only descriptor')
   need(folder+'native_post_exit_integrity.json'not in infos and folder+'native_scratch_cleanup.json'not in infos,'reused immutable native cache must not be treated as fresh owned scratch')
  else:need(report['native_promotion']is None and report['native_reuse']is None,'compiled must not prepare/open native format')
  value={'launch_to_first_batch_ready_seconds':(report['first_batch_ready_ns']-t)/1e9,'launch_to_actual_wait_seconds':(end-t)/1e9,
   'checkpointed_measured_seconds':(report['training_end_ns']-report['steady_start_ns'])/1e9,
   'worker_shutdown_seconds':(report['shutdown_end_ns']-report['shutdown_begin_ns'])/1e9,'input_hash_seconds_included':report['input_hash_seconds_included']}
  for key in('launch_to_first_batch_ready_seconds','checkpointed_measured_seconds','worker_shutdown_seconds'):same(report[key],value[key],key)
  same(exit['launch_to_actual_wait_seconds'],value['launch_to_actual_wait_seconds'],'actual wait elapsed')
  need(type(value['input_hash_seconds_included'])in(int,float)and 0<=value['input_hash_seconds_included']<=value['launch_to_actual_wait_seconds'],'bounded included hash observer time required')
  value.update({p['label']+'_launch_to_complete_seconds':(p['monotonic_ns']-t)/1e9 for p in cps})
  metrics[name]=value;reports[name]=report;raw_observations.append({'job':job,'metrics':value})
  pss_path=root+name+'.pss.jsonl';need(pss_path in pss,'original app PSS log required')
  need(pss[pss_path]['first_raw_sample_ns']is not None and t<=pss[pss_path]['first_raw_sample_ns']<=pss[pss_path]['last_raw_sample_ns']<=end,
   'all original app PSS/error clocks must lie within actual lifetime')
  app_environment['sampled_CPU']=pss[pss_path]['environment_facts'].bind(environment_contract,(launch['pid'],launch['start_ticks']),(t,end),require_observed=False)
  app_environment['name']=name;app_environment['worker_boundary_pairs_checked']=len(workers);environment_jobs.append(app_environment)
  pss_output.append({'name':name,'phases':pss[pss_path]['phases']})
  observation=pss[pss_path];peak=observation['all_phases_maximum']
  if peak is not None:need(t<=peak['scan_end_monotonic_ns']<=end,'headline PSS scan timestamp outside actual application lifetime')
  pss_headline_runs.append({'job':job,'status':'observed'if peak is not None else'unobserved',
   'eligible_samples':observation['raw_observation_counts']['eligible'],
   'total_samples':sum(observation['raw_observation_counts'].values()),'maximum':peak})
 return reports,metrics,raw_observations,pss_output,pss_headline_runs,environment_jobs,previous_completion

def collection_binding(get,infos,collection,whole,platform,plan,pins):
 need(collection['kind']=='CLOSURE_V1_ROOT_COLLECTION'and collection['dataset_id']==plan['dataset_id']
  and collection['repetition_block']==plan['repetition_block']and collection['pod_uid']==platform['pod_uid']
  and collection['sts_uid']==platform['sts_uid'],'actual root collection allocation/dataset binding differs')
 expected={'packet_manifest_sha256':pins['source_manifest_sha256'],'protocol_sha256':pins['protocol_sha256'],
  'wrapper_sha256':pins['wrapper_sha256'],'platform_admission_sha256':infos['closure/results/PLATFORM_ADMISSION.json']['sha256']}
 need(all(collection['source_hashes'][k]==v for k,v in expected.items()),'root-collected source/protocol/platform hashes differ')
 need(all(collection['archive'][k]==v for k,v in whole.items()),'root-collected whole archive digest/size differs')
 need(collection['local_fsync']is True and collection['all_archive_members_independently_verified']is True
  and collection['owned_postexit_scan']['live_owned_rows']==[]and collection['timing_lock_available']is True
  and collection['active_marker_absent']is True,'actual post-exit root transfer/fsync/quiescence/lock proof required')
 outer=collection['outer_process'];clean(outer,guard=False)
 need(outer['kind']=='CLOSURE_V1_ENVELOPE_ACTUAL_OS_WAIT'and outer['failure']is None and
  outer['owned_tree_quiescent_after_cleanup']is True and outer['wrapper_sha256']==pins['wrapper_sha256']and
  outer['driver_source_sha256']==infos['closure/sources/envelope/wait_driver.py']['sha256']and
  outer['pod_uid']==platform['pod_uid'],'genuine detached-parent wrapper wait/source proof differs')
 need(infos['closure/sources/envelope/run_block.py']['sha256']==pins['wrapper_sha256'],
  'reviewed actual wrapper source differs')
 child=get('closure/envelope/child_launch.json');exit=get('closure/envelope/child_exit.json');clean(exit)
 need(exit==collection['producer_exit']and exit['pid']==child['pid']and exit['start_ticks']==child['start_ticks'],
  'original producer launch and actual parent wait0 differ')
 packet=recorded_path(child['entrypoint']).parent;out=recorded_path(plan['out']).parent
 need(child['entrypoint']==str(packet/'run_lifecycle.py')and plan['out']==str(out/'producer')
  and outer['launch_monotonic_ns']<=child['launch_monotonic_ns']<exit['wait_returned_monotonic_ns']
  <=outer['wait_returned_monotonic_ns'],'actual nested wrapper/producer lifetime order differs')
 child_tail=['-B','-u',str(packet/'run_lifecycle.py'),'--dataset',plan['dataset_id'],
  '--graph-root',plan['graph_root'],'--native-reuse-root',plan['native_reuse_root'],
  '--native-reuse-receipt-sha256',plan['native_reuse_receipt_sha256'],'--out',plan['out'],
  '--coordination-dir',plan['coordination_dir'],'--max-hours',str(3900/3600),
  '--repetition-block',str(plan['repetition_block']),'--platform-admission-json',plan['platform_admission']['source_path'],
  '--platform-admission-sha256',expected['platform_admission_sha256'],
  '--source-manifest-sha256',pins['source_manifest_sha256'],'--execute']
 need(child['argv'][1:]==child_tail and recorded_path(child['argv'][0]).name.startswith('python'),
  'actual producer argv/configuration differs')
 need(get('closure/envelope/platform_admission.json')==platform,'original outer/inner actual platform proof differs')
 preflight=get('closure/envelope/preflight.json')
 need(preflight['dataset_id']==plan['dataset_id']and preflight['repetition_block']==plan['repetition_block']
  and preflight['source_hashes']==expected and preflight['native_reuse_receipt_sha256']==plan['native_reuse_receipt_sha256']
  and preflight['max_seconds']==4200 and preflight['producer_max_seconds']==3900 and preflight['YT_writes']==0,
  'actual wrapper preflight source/protocol/caps differs')
 terminal=get('closure/envelope/SCIENTIFIC_TERMINAL.json')
 need(terminal['kind']=='CLOSURE_V1_SIX_APP_ENVELOPE_SUCCESS'and terminal['success']is True and terminal['jobs']==6
  and terminal['dataset_id']==plan['dataset_id']and terminal['repetition_block']==plan['repetition_block']
  and terminal['source_manifest_sha256']==pins['source_manifest_sha256']and terminal['producer_exit']==exit
  and terminal['independent_raw_audit_required']is True and terminal['YT_writes']==0,
  'actual successful six-app wrapper terminal required')
 base=recorded_path(platform['base']);need(packet.is_relative_to(base)and out.is_relative_to(base),
  'recorded source/output must use admitted project backing')
 locations=[packet,out,recorded_path(plan['graph_root']),recorded_path(plan['native_reuse_root'])]
 need(all(not(a.is_relative_to(b)or b.is_relative_to(a))for i,a in enumerate(locations)for b in locations[i+1:]),
  'recorded source/compact/native/output trees must be disjoint')
 need(not any(recorded_path(plan['coordination_dir']).is_relative_to(p)for p in locations),
  'timing coordination must be outside source/input/output trees')
 return dict(packet=str(packet),producer_launch_ns=child['launch_monotonic_ns'],
  producer_wait_ns=exit['wait_returned_monotonic_ns'],wrapper_actual_wait0_checked=True)

def audit_block(data,infos,pss,events,disk,collection,whole,pins):
 def get(path):need(path in data,'required original JSON missing: '+path);return data[path]
 root='closure/results/';plan=get(root+'plan.json');records=get(root+'job_records.json')
 dataset=plan['dataset_id'];block=plan['repetition_block']
 need(dataset in BINDINGS and type(block)is int and block in(1,2,3),'declared dataset/repetition required')
 expected=[j for j in grid(dataset)if j['repetition']==block]
 need(plan['kind']=='CLOSURE_V1_PROSPECTIVE_REPETITION_BLOCK'and plan['execute']is True
  and plan['source_manifest_sha256']==pins['source_manifest_sha256']and plan['global_protocol_jobs']==18
  and plan['global_two_dataset_jobs']==36 and plan['global_job_order']==[j['name']for j in grid(dataset)]
  and 0<plan['max_hours']<=3900/3600 and len(plan['jobs'])==len(records)==6,
  'complete prospective selected-dataset six-app plan/order required')
 need(len({r['name']for r in records})==6,'duplicate actual job record')
 allowed={j['name']for j in expected}
 need(all(n[len(root+'jobs/'):].split('/',1)[0]in allowed for n in infos if n.startswith(root+'jobs/')),
  'another dataset/repetition/failed attempt cannot enter six-app block')
 need(not any(n.startswith(root)and any(part in ('runtime','shared_numba_cache','__pycache__')for part in
  PurePosixPath(n[len(root):]).parts)for n in infos),'raw graph/native/cache payloads must not enter evidence archive')
 internal=get('SHA256_MANIFEST.json');need(internal['graphs_included']is False and
  internal['native_arrays_included']is False,'bounded graph-free evidence archive required')
 platform=get(root+'PLATFORM_ADMISSION.json');platform_sha=infos[root+'PLATFORM_ADMISSION.json']['sha256']
 done=get(root+'COMPLETE.json');terminal=get(root+'SCIENTIFIC_TERMINAL.json')
 need(all(done[k]is True for k in('controller_finished','success','owned_tree_quiescent'))and done['failure']is None
  and done['records']==records==terminal['records']and terminal['success']is True and terminal['jobs']==6
  and done['dataset_id']==terminal['dataset_id']==dataset and
  done['repetition_block']==terminal['repetition_block']==block and
  done['global_protocol_jobs']==terminal['global_protocol_jobs']==18 and
  done['global_two_dataset_jobs']==terminal['global_two_dataset_jobs']==36 and
  done['platform_admission_sha256']==terminal['platform_admission_sha256']==platform_sha and
  done['platform_admission_file']==terminal['platform_admission_file']=='PLATFORM_ADMISSION.json'and
  done['YT_writes']==done['statefulset_actions']==terminal['YT_writes']==0 and terminal['no_new_quality_claim']is True,
  'actual complete six-app terminal/source/platform semantics required')
 token=done['controller_identity'];need(token['source_manifest_sha256']==pins['source_manifest_sha256']and
  token['repetition_block']==block and isinstance(token['nonce'],str)and len(token['nonce'])==32,
  'actual owned-controller marker identity required')
 native=source_and_inputs(get,infos,plan,platform,pins)
 frame=collection_binding(get,infos,collection,whole,platform,plan,pins)
 need((token['pid'],token['start_ticks'])==(get('closure/envelope/child_exit.json')['pid'],
  get('closure/envelope/child_exit.json')['start_ticks']),'controller marker and original OS wait identity differ')
 contract,environment=environment_admission(get,infos,plan,done,events,platform,pins['source_manifest_sha256'])
 storage=storage_admission(get,infos,plan,done,disk,platform,platform_sha)
 environment.update(qualification(get,pss,contract,events,(token['pid'],token['start_ticks']),pins['source_manifest_sha256']))
 status=get(root+'status.json');need(status['phase']=='complete'and status['completed_jobs']==6 and status['failure']is None,
  'final complete-six status required')
 need(frame['producer_launch_ns']<=events.first<=events.last<=status['monotonic_ns']<=frame['producer_wait_ns']
  and frame['producer_launch_ns']<=disk.first<=disk.last<=storage['latest_inventory']['observed_monotonic_ns']
  <=status['monotonic_ns'],'raw resource/disk/status bounds outside actual producer lifetime')
 pre,descriptor=reuse_preparation(get,infos,plan,native,pins['source_manifest_sha256'])
 build=get(root+'build/build_receipt_v1.json')
 need(build['source_sha256']==infos['closure/sources/packet/native_core/native_assoc_v1.cpp']['sha256']
  and sha_hex(build['library_sha256'])and build['source']==str(recorded_path(frame['packet'])/'native_core/native_assoc_v1.cpp')
  and build['library']==str(recorded_path(plan['out'])/'build/native_assoc_v1.so'),'actual native source/library build binding differs')
 reports,metrics,raw,phase_pss,headline,app_env,completion=audit_applications(get,infos,pss,events,plan,records,block,
  expected,contract,done,frame,native,build,descriptor)
 streams,prefixes=input_prefixes(reports);rows,ratios=paired_rows(reports,metrics,block)
 final=get(root+'native_reuse_final_integrity.json')
 need(final['outside_application_lifetimes']is True and final['readonly_stat_checks_passed']is True and
  final['receipt_sha256']==plan['native_reuse_receipt_sha256']and final['files']==native_files(native)and
  completion<=final['observed_monotonic_ns']<=status['monotonic_ns'],
  'genuine post-block whole native/read-only proof required after all actual apps')
 same(final['descriptor'],{**descriptor,'verified_files':native_files(native),
  'full_integrity_readback_outside_application':True},'unchanged final immutable native full-hash/current-device descriptor')
 comparisons=[dict(horizon=h,workers=8,repetition=block,consumed_graphs=H[h][2],
  all_consumed_inputs_exact=True,stream_sha256=streams[h])for h in H]
 cross=dict(consumed_input_graphs=576,short_final_stream_sha256=streams['short320'],
  long_prefix_stream_sha256=prefixes['long6400']['576'],all_arms_workers_in_block_exact=True,repetition_block=block)
 cross_workers=[dict(workers=8,repetition=block,consumed_input_graphs=576,short_final_stream_sha256=streams['short320'],
  long_prefix_stream_sha256=prefixes['long6400']['576'],all_block_arms_horizons_exact=True)]
 same(sorted(terminal['stream_comparisons'],key=lambda r:r['horizon']),
  sorted(comparisons,key=lambda r:r['horizon']),'independent full-stream comparisons')
 same(terminal['cross_horizon_input_prefix576'],cross,'independent short/long semantic576 comparison')
 same(terminal['cross_horizon_worker_comparisons'],cross_workers,'independent W8 short/long prefix proof')
 need(terminal['scope']==plan['protocol']['scope']and terminal['timing_protocol']==plan['protocol']['timing'],
  'original declared scientific/timing scope differs')
 summary=get(root+'scientific_summary.json')
 need(summary['kind']=='CLOSURE_V1_SINGLE_REPETITION_BLOCK_RAW_SUMMARY'and summary['status']=='complete'and
  summary['dataset_id']==dataset and summary['jobs']==6 and summary['global_protocol_jobs']==18 and
  summary['global_two_dataset_jobs']==36 and summary['repetition_block']==block and
  summary['platform_admission_sha256']==platform_sha and summary['native_reuse_receipt_sha256']==plan['native_reuse_receipt_sha256'],
  'original six-app summary bindings differ')
 same(sorted(summary['rows'],key=lambda r:r['horizon']),sorted(rows,key=lambda r:r['horizon']),
  'independently rederived raw configuration metrics')
 ratio_key=lambda r:(r['horizon'],r['numerator'],r['denominator'])
 same(sorted([{k:v for k,v in row.items()if k!='scope'}for row in summary['paired_arm_ratios']],key=ratio_key),
  sorted(ratios,key=ratio_key),
  'independently recomputed three-arm within-block ratios')
 same([{k:v for k,v in row.items()if k!='scope'}for row in summary['sampled_PSS']],phase_pss,
  'independently rederived complete/stable/same-phase PSS maxima')
 return dict(dataset_id=dataset,repetition_block=block,block_jobs=6,global_protocol_jobs=18,global_two_dataset_jobs=36,
  all_three_complete_blocks_required=True,global_dataset_complete=False,platform=platform,
  raw6_metrics=raw,rows=rows,paired_arm_ratios=ratios,source_bound_consumed_input_streams=streams,
  source_bound_semantic_prefixes_by_horizon=prefixes,cross_horizon_input_prefix576=cross,
  headline_application_PSS_all_phases=dict(runs=headline,missing_rule='unobserved, never zero',
   scope='Sampled all-phase application-tree maximum; non-atomic and partial residency, not required RAM'),
  PSS_raw_observation_counts={name:v['raw_observation_counts']for name,v in pss.items()},sampled_PSS=phase_pss,
  environment_block={**environment,'timing_jobs':app_env},storage_block=storage,actual_launch_frame=frame,
  native_reuse_preparation=dict(receipt_sha256=plan['native_reuse_receipt_sha256'],conversion_seconds=pre['conversion_seconds'],
   validation_seconds=pre['validation_seconds'],native_total_bytes=native['rowptr_bytes']+native['col_bytes'],
   preparation_clock_scope='Durations internal to preparation process only; no monotonic comparison across host/boot'),
  all_app_waits_zero_normal_worker_close=True,original_summary_independently_recomputed=True,
  source_bound_consumed_digest_equality_checked=True,consumed_tensor_bytes_independently_rehashed=False,
  graph_and_native_payload_bytes_independently_rehashed=False,failed_or_historical_applications_pooled=False)

def describe(values):
 need(set(values)=={'1','2','3'}and all(type(v)in(int,float)and math.isfinite(v)for v in values.values()),
  'three finite actual allocation observations required')
 numbers=list(values.values())
 return dict(n=3,by_block=values,minimum=min(numbers),median=statistics.median(numbers),maximum=max(numbers),
  sample_sd=statistics.stdev(numbers),scope='Descriptive across-allocation variation; retained caches, no independence/population CI claim')

def aggregate(reports):
 need(len(reports)==3 and {r['repetition_block']for r in reports}=={1,2,3},
  'exactly three distinct complete dataset blocks required; partial pooling forbidden')
 reports={r['repetition_block']:r for r in reports};first=reports[1];dataset=first['dataset_id']
 need(dataset in BINDINGS,'one known dataset required')
 pins=first['source_hashes'];script=first['script']['sha256'];streams=first['source_bound_consumed_input_streams']
 prefixes=first['source_bound_semantic_prefixes_by_horizon'];preparation=first['native_reuse_preparation']
 need(len({r['platform']['pod_uid']for r in reports.values()})==3 and
  len({r['archive']['sha256']for r in reports.values()})==3,'three genuinely distinct original allocations/raw archives required')
 cells={};memory={};ratios={};names=set()
 for block,r in reports.items():
  need(r['kind']=='INDEPENDENT_CLOSURE_V1_SIX_APP_BLOCK_AUDIT'and r['status']=='PASS_SIX_APP_DATASET_BLOCK'
   and r['dataset_id']==dataset and r['block_jobs']==6 and r['global_protocol_jobs']==18 and r['global_two_dataset_jobs']==36
   and r['global_dataset_complete']is False and r['all_three_complete_blocks_required']is True and
   r['failed_or_historical_applications_pooled']is False and r['all_app_waits_zero_normal_worker_close']is True and
   r['original_summary_independently_recomputed']is True and r['source_bound_consumed_digest_equality_checked']is True and
   r['consumed_tensor_bytes_independently_rehashed']is False,'only genuinely audited complete six-app new blocks can aggregate')
  need(r['source_hashes']==pins and r['script']['sha256']==script and r['platform']['block']==block and
   r['source_bound_consumed_input_streams']==streams and r['source_bound_semantic_prefixes_by_horizon']==prefixes,
   'frozen source/protocol or consumed-input maps differ across allocations')
  same(r['native_reuse_preparation'],preparation,'one immutable preparation receipt and cost per dataset')
  need(len(r['raw6_metrics'])==6,'six genuine raw app metric rows required')
  local={};fake_reports={}
  expected={j['name']:j for j in grid(dataset)if j['repetition']==block}
  for row in r['raw6_metrics']:
   job=row['job'];name=job['name'];need(name in expected and job==expected[name]and name not in names,
    'original exact unique job identity differs')
   names.add(name);key=(job['horizon'],job['arm']);value=row['metrics']
   need(key not in local and all(type(v)in(int,float)and math.isfinite(v)and v>=0 for v in value.values())and
    value['launch_to_actual_wait_seconds']>0,'finite actual raw clock metrics required')
   local[key]=value;cells.setdefault(key,{})[str(block)]=value
   fake_reports[name]={'config':{'horizon':job['horizon'],'arm':job['arm']}}
  need(set(local)=={(h,a)for h in H for a in ARMS},'complete three-arm/two-horizon raw matrix required')
  metric_map={name:local[(job['horizon'],job['arm'])]for name,job in expected.items()}
  derived_rows,derived_ratios=paired_rows(fake_reports,metric_map,block)
  same(r['rows'],derived_rows,'aggregate rederived raw block rows')
  same(r['paired_arm_ratios'],derived_ratios,'aggregate rederived within-block ratios')
  for row in derived_ratios:
   key=(row['horizon'],row['numerator'],row['denominator']);ratios.setdefault(key,{})[str(block)]=row['metrics']
  seen=set();runs=r['headline_application_PSS_all_phases']['runs'];need(len(runs)==6,'all six PSS outcomes required')
  for row in runs:
   job=row['job'];need(job['name']in expected and job==expected[job['name']],'exact PSS original job identity required')
   key=(job['horizon'],job['arm']);need(key not in seen,'duplicate PSS outcome');seen.add(key);peak=row['maximum']
   if peak is None:
    need(row['status']=='unobserved'and row['eligible_samples']==0,'missing PSS must remain unobserved');value=None
   else:
    need(row['status']=='observed'and row['eligible_samples']>0 and type(peak['tree_pss_bytes'])is int
     and peak['tree_pss_bytes']>=0,'observed sampled PSS maximum required');value=peak['tree_pss_bytes']
   memory.setdefault(key,{})[str(block)]=value
 need(len(names)==18 and streams['short320']==prefixes['long6400']['576'],
  'all18 new complete apps and cross-allocation semantic prefix equality required')
 rows=[]
 for (h,arm),blocks in sorted(cells.items()):
  keys=set(blocks['1']);need(all(set(v)==keys for v in blocks.values()),'same raw metric axes required')
  pss=memory[h,arm];observed=[v for v in pss.values()if v is not None]
  rows.append(dict(horizon=h,workers=8,arm=arm,
   metrics={k:describe({b:v[k]for b,v in blocks.items()})for k in sorted(keys)},
   sampled_application_PSS_bytes=dict(by_block=pss,observed_blocks=len(observed),
    unobserved_blocks=[b for b,v in pss.items()if v is None],complete_three_block_range=len(observed)==3,
    minimum_of_observed=min(observed)if observed else None,maximum_of_observed=max(observed)if observed else None)))
 paired=[]
 for (h,numerator,denominator),blocks in sorted(ratios.items()):
  keys=set(blocks['1']);need(all(set(v)==keys for v in blocks.values()),'same paired metric axes required')
  paired.append(dict(horizon=h,workers=8,numerator=numerator,denominator=denominator,
   metrics={k:describe({b:v[k]for b,v in blocks.items()})for k in sorted(keys)},
   scope='Descriptive statistics of three within-block ratios; never ratio of cross-block medians'))
 return dict(kind='CLOSURE_V1_THREE_BLOCK_DATASET_AGGREGATE',status='PASS_THREE_BLOCKS_18_APPS',dataset_id=dataset,
  blocks=3,jobs=18,global_dataset_complete=True,global_two_dataset_complete=False,source_hashes=pins,
  auditor_source_sha256=script,raw_archive_pins={str(b):r['archive']for b,r in reports.items()},
  root_collection_pins={str(b):r['root_collection']for b,r in reports.items()},
  platforms={str(b):r['platform']for b,r in reports.items()},rows=rows,paired_arm_ratios=paired,
  source_bound_consumed_input_streams=streams,source_bound_semantic_prefixes_by_horizon=prefixes,
  native_reuse_preparation=preparation,failed_or_historical_applications_pooled=False,
  scope='One dataset only: all18 genuinely completed fresh processes in three fresh allocations. Reused native preparation cost reported once separately; private worker maps remain per process. Retained caches and sampled/non-atomic PSS; no population CI, cold I/O, time-to-quality, or cross-dataset pooling.',
  input_identity_scope='Ordered/typed/shaped consumed-input digests are source-bound equality evidence, not independent tensor regeneration')

def read_pinned_json(path,expected,maximum=32*MIB):
 actual=pin(path,maximum);need(sha_hex(expected)and actual['sha256']==expected,'explicit independent metadata SHA differs')
 raw=Path(path).read_bytes();need(hashlib.sha256(raw).hexdigest()==expected and len(raw)==actual['bytes'],
  'metadata changed during bounded read')
 return decode(raw),actual

def write_fresh(path,value):
 need(not path.exists()and not path.is_symlink(),'fresh audit output required')
 with path.open('x')as stream:
  json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def main():
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
 block=sub.add_parser('block',help='Audit exactly one new complete six-app dataset allocation')
 block.add_argument('--archive',type=Path,required=True);block.add_argument('--archive-sha256',required=True)
 block.add_argument('--root-collection',type=Path,required=True);block.add_argument('--root-collection-sha256',required=True)
 for name in ('source-manifest-sha256','protocol-sha256','wrapper-sha256'):block.add_argument('--'+name,required=True)
 block.add_argument('--output',type=Path,required=True)
 combine=sub.add_parser('aggregate',help='Combine exactly three pinned independent block audits for one dataset')
 combine.add_argument('--block-audit',nargs=2,action='append',metavar=('JSON','SHA256'),required=True)
 combine.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 if args.command=='block':
  pins={k:getattr(args,k)for k in('source_manifest_sha256','protocol_sha256','wrapper_sha256')}
  need(all(sha_hex(v)for v in pins.values()),'explicit frozen reviewed source/protocol/wrapper SHA required')
  collection,external=read_pinned_json(args.root_collection,args.root_collection_sha256)
  data,infos,pss,whole,events,disk=load_archive(args.archive,args.archive_sha256)
  result=audit_block(data,infos,pss,events,disk,collection,whole,pins)
  report=dict(kind='INDEPENDENT_CLOSURE_V1_SIX_APP_BLOCK_AUDIT',status='PASS_SIX_APP_DATASET_BLOCK',
   source_hashes=pins,archive=whole,root_collection=external,script=pin(Path(__file__),MIB),
   whole_members_rehashed=len(infos),actual_new_experiments_GPU_cluster_network_YT_actions=0,
   scope='Independent raw clocks/PSS/CPU brackets/source/provenance checks. No producer import or recorded path access. Source-bound graph/native/input identities, not independent payload regeneration.',**result)
 else:
  need(len(args.block_audit)==3,'exactly three pinned independent block audits required')
  values=[];input_pins=[]
  for path,digest in args.block_audit:
   value,identity=read_pinned_json(path,digest);values.append(value);input_pins.append(identity)
  report=aggregate(values);report['block_audit_pins']=input_pins
  need(report['auditor_source_sha256']==pin(Path(__file__),MIB)['sha256'],'same frozen auditor source must audit and aggregate')
 write_fresh(args.output,report)
 print(json.dumps(dict(status=report['status'],dataset_id=report['dataset_id'],output=str(args.output)),sort_keys=True))

if __name__=='__main__':main()
