#!/usr/bin/env python3
"""Bounded independent stdlib peer for a genuine completed E25 v6 eight-app block raw archive.

No producer, summary helper, ML library, recorded path, GPU or cluster is used.
Raw app clocks and per-process PSS observations are independently reaggregated.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,random,tarfile
from decimal import Decimal,localcontext
from pathlib import Path,PurePosixPath
import independent_env as env

MIB,GIB=1<<20,1<<30
ARMS=('reference_union_compiled','reference_union_native_persistent')
H={'short320':(10,320,576,[32,256,576]),'long6400':(200,6400,6656,[32,256,1280,3456,6656])}
LABELS={32:'first_trained_batch',256:'warmup_complete',576:'measured_prefix320',1280:'measured_prefix1024',3456:'measured_prefix3200',6656:'measured_prefix6400'}
PACKET_SHA='3fd8783173e4a850018d860459881aafd1dd49e8d4cda101eb273641e47af086'
PROTOCOL_SHA='e89b371089bd01f71579fdc00ec06004373bb549542658cc223914759f392e18'
WRAPPER_SHA='072575b15555a83f223e15643b869874f9a5e4f83172c06e0f4f5d032bdae4c9'
WRAPPER_READY_SHA='88b06c57d7de42c062fbd2ea76cd268999d3bbabe8bbd31130a9ec79dfa5a785'
ENV_SOURCE_SHA='96959b28102ea112538783843b1fe50634a733fac7d19ee77eeff54732ed7c27'
PSS_RULE_SHA='47b8a0f24c4ec3f386a8b8ba29f6d52a2c3af54ea46b3f2389f6483914bb0a90'
MODEL_SHA='f88254079fe6c152b4c0153c857a63516348353d578f918622e7f6719559618e'
MODEL={'name':'DGCNN','sortpool_k':281,'hidden_channels':32,'num_layers':3,'max_z':1000,
 'learning_rate':0.0001,'optimizer':'Adam','use_feature':False,'node_embedding':False,'parameters':596674}
PRIOR_FAILED_REFERENCE={'archive':'E25_ENV_V4_FAILED_DIAGNOSTIC_v1.tgz','bytes':2697876,
 'sha256':'5a470ea09dcfef2c754273dcb9d63288739b7c779799952e5f24a853b25c2365','completed_jobs':9,
 'status':'FAILED_EXCLUDED_FROM_SCIENCE',
 'scope':'Reference only; preserve this capsule once in the final aggregate, with older diagnostics nested. No old app enters new measurements.'}

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
def grid():
 rng=random.Random(20261126);first=list(ARMS);rng.shuffle(first)
 combinations=[(h,w)for h in H for w in(1,8)]
 base={key:first if i%2==0 else list(reversed(first))for i,key in enumerate(combinations)};out=[]
 for repetition in(1,2,3):
  current=combinations[:];rng.shuffle(current)
  for horizon,w in current:
   order=base[horizon,w]if repetition%2 else list(reversed(base[horizon,w]))
   for position,arm in enumerate(order,1):out.append({'name':f'{horizon}_w{w}_r{repetition}_{arm}','horizon':horizon,'workers':w,'repetition':repetition,'position':position,'arm':arm})
 return out

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
 same_phase=sample['phase_before']==sample['phase_after'];eligible=stable and complete and same_phase
 need(sample['phase_stable']is same_phase and sample['phase_publication_eligible']is eligible,'phase eligibility flag contradicts recorded phases')
 if complete:need(sample['tree_pss_bytes']==sum(r['pss_bytes']for r in processes),'PSS tree sum differs from original process PSS')
 if eligible:
  need(sample['backend']=='linux-proc-smaps_rollup','actual Linux PSS backend required')
  return (sample['phase_before'],sum(r['pss_bytes']for r in processes)),'eligible'
 return None,'ineligible'

DISK_KEYS={'observed_monotonic_ns','path','device','bytes_available','bytes_total',
 'inodes_available','inodes_total','inodes_reported','mount_readonly','scope'}

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
   'scope':'Each recorded available-byte/inode/readonly/device threshold and latest sampled inventory checked. Byte availability was derived by pinned statvfs code; raw block components were not serialized. Initial40GiB/positive exposed inodes admission is separate from runtime4GiB/positive exposed inodes. No instantaneous quota, capacity reservation, node eviction or continuous-health guarantee.'}

def load_archive(path,expected_sha):
 whole=pin(path,128*MIB);need(whole['sha256']==expected_sha,'independently collected whole archive SHA differs')
 payloads,infos,pss={}, {}, {};total=kept=0;known={j['name']for j in grid()}|{'build_native','actual_cuda_qualification'};events=env.Facts();disk_facts=DiskFacts()
 class BoundedTarBytes:
  def __init__(self,stream):self.stream,self.size=stream,0
  def read(self,size=-1):
   block=self.stream.read(min(MIB,size)if size>=0 else MIB);self.size+=len(block)
   need(self.size<=400*MIB,'whole decompressed tar/header cap');return block
 with gzip.open(path,'rb')as compressed,tarfile.open(fileobj=BoundedTarBytes(compressed),mode='r|')as tar:
  for member in tar:
   name=member.name;rel=PurePosixPath(name)
   need(name=='SHA256_MANIFEST.json'or name.startswith('e25/'),'exact E25 raw archive namespace required')
   need(member.isfile()and not rel.is_absolute()and'..'not in rel.parts and'\\'not in name and str(rel)==name
    and name not in infos and len(infos)<4096 and 0<=member.size<=32*MIB and set(member.pax_headers)<={'path'}
    and member.pax_headers.get('path',name)==name,'bounded regular canonical member required')
   total+=member.size;need(total<=384*MIB,'whole raw payload cap')
   digest=hashlib.sha256();size=0;data=bytearray();is_pss=name.startswith('e25/results/')and name.endswith('.pss.jsonl')and name[len('e25/results/'):-len('.pss.jsonl')]in known
   is_events=name=='e25/results/environment_events.jsonl'
   is_disk=name=='e25/results/disk_observations.jsonl'
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

def environment_admission(get,infos,plan,done,events,platform):
 root='e25/results/';chosen=platform['cpu_affinity']
 need(len(env.mask(chosen))==27 and plan['environment_policy']==done['environment_policy']==env.POLICY
  and plan['selected_cpu_affinity']==chosen and done['environment_guard_success']is True
  and done['source_manifest_sha256']==PACKET_SHA,'exact block source/CPU policy and chosen pool required')
 need(not any(name.startswith(root)and name.endswith('ENVIRONMENT_FAILURE.json')for name in infos),'any recorded CPU failure excludes this block')
 receipt=get(root+'ENVIRONMENT_ADMISSION.json');contract=receipt['contract']
 need(infos[root+'ENVIRONMENT_ADMISSION.json']['sha256']==done['environment_admission_sha256']
  and receipt['status']=='PASS'and receipt['policy']==env.POLICY and receipt['chosen_cpu_affinity']==chosen==contract['chosen_cpu_affinity']
  and contract['policy']==env.POLICY and contract['pin_calls']==1 and contract['no_exclusive_or_continuous_placement_guarantee']is True
  and not contract.get('portable_unsupported',False),'original once-pin block CPU admission required')
 base=contract['resources'];env.projection(base)
 need(base['hostname']==platform['hostname']and base['boot_id']==platform['boot_id']
  and base['cpu_max']==platform['cpu_max']==env.QUOTA and base['memory_max']==platform['memory_max']==env.MEMORY
  and contract['topology']==platform['topology']and env.mask(chosen)<=env.mask(contract['admission_before_affinity'])
  and env.mask(chosen)<=env.mask(base['ambient_cpu_pool'])and env.mask(chosen)<=env.mask(base['online_cpus']),
  'original within-block host/boot/CPU/quota/memory/topology differs from actual platform')
 controller=(done['controller_identity']['pid'],done['controller_identity']['start_ticks'])
 admission=env.boundary(receipt['after'],contract,controller,label='controller_admission');event_proof=events.bind(contract,controller)
 need(events.labels.get('controller_admission',{}).get('count')==1
  and events.labels['controller_admission']['first_ns']==receipt['after']['observed_monotonic_ns']and'assert_inputs'in events.labels,
  'original own controller admission and events must bind')
 return contract,{'policy':env.POLICY,'chosen_cpu_affinity':chosen,'admission_sha256':infos[root+'ENVIRONMENT_ADMISSION.json']['sha256'],
  'admission':admission,'controller_events':event_proof}

def environment_qualification(get,pss,contract,events,controller):
 root='e25/results/';folder=root+'qualification/';qualified=get(folder+'QUALIFIED.json');records=[];initials=[];streams=[];consumer_proofs=[]
 need(qualified['success']is True and qualified['device']=='cuda'and qualified['workers']==[1,8]and qualified['actual_consumers']==4
  and qualified['source_manifest_sha256']==PACKET_SHA and qualified['environment_policy']==env.POLICY
  and qualified['chosen_cpu_affinity']==contract['chosen_cpu_affinity']and qualified['environment_target_proof']is True
  and get(folder+'ENVIRONMENT_ADMISSION.json')=={'status':'PASS','contract':contract},'actual same-contract H1001/8 qualification required')
 support=[]
 for name in('build_native','actual_cuda_qualification'):
  launch=get(root+name+'.launch.json');exit=get(root+name+'.exit.json');clean(exit)
  need(exit['pid']==launch['pid']and exit['start_ticks']==launch['start_ticks']and exit['launch_monotonic_ns']==launch['launch_monotonic_ns'],'actual support command identity binding')
  interval=(launch['launch_monotonic_ns'],exit['wait_returned_monotonic_ns'])
  labels=environment_job_labels(events,name,*interval,controller)
  pss_path=root+name+'.pss.jsonl'
  if pss_path in pss and pss[pss_path]['first_raw_sample_ns']is not None:need(interval[0]<=pss[pss_path]['first_raw_sample_ns']<=pss[pss_path]['last_raw_sample_ns']<=interval[1],'support original PSS/error clocks outside actual lifetime')
  proof=pss[pss_path]['environment_facts'].bind(contract,(launch['pid'],launch['start_ticks']),interval,require_observed=False)if pss_path in pss else{'status':'CPU_UNOBSERVED','reason':'No support command external log archived; original boundary/child gates remain checked.'}
  support.append({'name':name,'controller_labels':labels,'sampled_CPU':proof})
 qualification_interval=(get(root+'actual_cuda_qualification.launch.json')['launch_monotonic_ns'],get(root+'actual_cuda_qualification.exit.json')['wait_returned_monotonic_ns'])
 for w in(1,8):
  pair=[]
  for arm in ARMS:
   current=folder+f'w{w}_{arm}/';r=get(current+'report.json');exit=get(current+'exit.json');cfg=get(current+'config.json')
   need(exit['actual_wait_returned']is True and type(exit['returncode'])is int and exit['returncode']==0 and r['status']=='complete'
    and r['config']==cfg and cfg['environment_contract']==contract and cfg['fixture']is True and cfg['num_workers']==w and cfg['arm']==arm
    and r['device']['type']=='cuda'and'H100'in r['device']['name']and r['model']['initial_state_sha256']==MODEL_SHA
    and r['consumed_graphs']==2*w+4 and r['measured_graphs']==4,'actual tiny consumer completion/model/contract required')
   interval=(exit['launch_monotonic_ns'],exit['wait_returned_monotonic_ns']);need(qualification_interval[0]<=interval[0]<interval[1]<=qualification_interval[1],'tiny actual lifetime outside original qualifier')
   need(r['launch_monotonic_ns']==interval[0]<=r['entry_monotonic_ns']<=r['imports_ready_ns'],'tiny original launch/import clock')
   owner=(exit['pid'],r['environment_start']['processes'][0]['start_ticks'])
   proof={key:env.boundary(r[key],contract,owner,interval,label)for key,label in(('environment_start','application_start'),('environment_end','application_end'))}
   need(r['imports_ready_ns']<=proof['environment_start']['first_observed_ns']<=proof['environment_start']['last_bracket_ns']<=r['model_setup_begin_ns']
    and r['shutdown_end_ns']<=proof['environment_end']['first_observed_ns']<=proof['environment_end']['last_bracket_ns']<=interval[1],'tiny full environment guard phase spans')
   workers=r['worker_receipts'];need(len(workers)==w and{worker['worker_id']for worker in workers}==set(range(w))and len({worker['pid']for worker in workers})==w,'actual all tiny workers required')
   for worker in workers:
    need(worker['status']=='closed'and worker['native_closed']is True and worker==get(current+'workers/worker'+str(worker['worker_id'])+'.json'),'original tiny normal worker close receipt')
    owner=(worker['pid'],worker['environment_start']['processes'][0]['start_ticks'])
    for key,label in(('environment_start','worker_start'),('environment_end','worker_end')):env.boundary(worker[key],contract,owner,interval,label)
   records.append(exit);initials.append(r['model']['initial_state_sha256']);pair.append(r)
   consumer_proofs.append({'workers':w,'arm':arm,'actual_wait_zero':True,'worker_boundary_pairs_checked':w,'all_boundary_resource_mask_values_checked':True})
  need(pair[0]['input_stream_sha256']==pair[1]['input_stream_sha256'],'tiny paired actual consumed input streams differ')
  streams.append({'workers':w,'consumed_graphs':2*w+4,'all_actual_consumed_inputs_exact':True,'stream_sha256':pair[0]['input_stream_sha256']})
 need(qualified['actual_child_exits']==records and qualified['comparisons']==streams and set(initials)=={qualified['initial_model_sha256']}=={MODEL_SHA},'original tiny qualified bindings differ')
 return {'support_commands':support,'actual_H100_qualification_consumers':consumer_proofs,'actual_qualification_same_source_contract_and_inputs':True}

def storage_and_controller_guard(get,infos,plan,done,disk_facts,platform,platform_sha):
 root='e25/results/';storage=platform['storage'];base=PurePosixPath(platform['base'])
 need(done['controller_guard_failure']is None and not any(n.endswith('CONTROLLER_GUARD_FAILURE.json')for n in infos),'any hard guard failure excludes block')
 reference=plan['protocol']['controller_guard_amendment'];binding=plan['platform_admission']
 need(reference['version']=='latched_disk_guard_v4'and reference['storage_admission']['schema_kind']=='E25_V6_ACTUAL_BLOCK_PLATFORM_ADMISSION'
  and 'sha256'not in reference['storage_admission']and binding['sha256']==platform_sha
  and binding['bytes']==infos[root+'PLATFORM_ADMISSION.json']['bytes']and binding['proof']==platform
  and infos[root+'PLATFORM_ADMISSION.json']['sha256']==done['platform_admission_sha256']==platform_sha,
  'dynamic original block platform proof binding differs')
 need(platform['kind']=='E25_V6_ACTUAL_BLOCK_PLATFORM_ADMISSION'and platform['block']==plan['repetition_block']
  and type(platform['block'])is int and platform['actual_scientific_jobs_started']==platform['YT_writes']==platform['statefulset_actions']==0,
  'genuine pre-block admission required')
 need(base.is_absolute()and'..'not in base.parts and storage['mount']['mount_point']==str(base)
  and storage['mount']['root']=='/'+storage['subPath']and storage['mount']['major_minor']==linux_major_minor(storage['device'])
  and'rw'in storage['mount']['options']and'ro'not in storage['mount']['options']and'ro'not in storage['mount']['super_options']
  and storage['mount_readonly']is False and storage['bytes_available']>=40*GIB
  and(storage['inodes_total']==0 or storage['inodes_available']>0)and storage['write_fsync_readback_unlink']is True,
  'actual block PVC/subPath/device/readwrite40GiB/probe admission differs')
 layer=platform['root_layer'];need(layer['mount_readonly']is False and layer['bytes_available']>=4*GIB
  and layer['write_fsync_readback_unlink']is True,'actual root-layer admission differs')
 kube=platform['kubernetes_binding'];need(all(kube[k]==platform[k]for k in('pod_uid','sts_uid','hostname','node','image_digest'))
  and all(kube[k]==storage[k]for k in('PVC_name','PVC_uid','volume_name','subPath')),'root API/image/PVC identity projection differs')
 need(platform['image_digest'].endswith('2c0d776d47eb7d3b763f5a42c70d51bff2ba6b1f9cb1f01b4adbad76b658cbb1')
  and'H100'in platform['gpu']['name']and platform['gpu']['memory_total_mib']>=79000,'same software image/whole H100 class required')
 for path in(plan['out'],plan['friendster_root'],plan['coordination_dir'],binding['source_path']):
  parsed=PurePosixPath(path);need(parsed.is_absolute()and'..'not in parsed.parts and parsed.is_relative_to(base),'recorded run paths outside admitted BASE')
 need(infos[root+'disk_observations.jsonl']['bytes']<=8*MIB and infos[root+'output_inventory_observation.json']['bytes']<=4096,'bounded original disk/inventory files required')
 computed=disk_facts.bind(storage['device'],plan['out'],get(root+'output_inventory_observation.json'))
 return {'actual_platform_sha256':platform_sha,'controller_guard_failure':None,'raw_disk_and_latest_inventory':computed,
  'scope':'Original dynamic root proof and recorded disk rows; no recorded path opened. Root API identity is root-attested. Raw statvfs block arithmetic was not serialized; no capacity reservation or continuous/global health claim.'}

def input_host_bindings(get,infos,plan,platform):
 root='e25/results/';source='e25/sources/packet/';graph=recorded_path(plan['friendster_root'])
 expected=get(source+'pins/friendster_full_expected.json')['files'];identity=get(root+'input_identity.json')
 need(identity['whole_hashes']is True and set(identity['files'])=={str(graph/name)for name in expected},'exact source-bound whole FR3 input identity required')
 for path,row in identity['files'].items():
  pin=expected[PurePosixPath(path).name]
  need(row['bytes']==pin['bytes']and row['sha256']==pin['sha256']
   and all(counter(row[k])for k in('mtime_ns','inode','device')),'whole FR3 hash/bytes or original stat metadata differs')
 workload=infos[source+'workloads/friendster_pairs.npy']
 need(infos[root+'friendster_pairs.npy']==workload and workload['sha256']==get(source+'SOURCE_PINS.json')['original_workload_sha256'],
  'actual copied pair-workload whole bytes/SHA differs from frozen source')
 host=get(root+'host.json');hardware=get(root+'hardware.json');target=hardware['target'];gpu=platform['gpu']
 need(host['hostname']==platform['hostname']and host['boot_id']==platform['boot_id']and host['cpu_affinity']==platform['cpu_affinity']
  and host['versions']==get(source+'SOURCE_PINS.json')['runtime_versions']and host['python'].startswith('3.10.'),
  'original actual host/boot/exact27/Python/numerical versions differ')
 need(all(target[k]==gpu[v]for k,v in(('uuid','uuid'),('name','name'),('memory_total_mib','memory_total_mib'),('driver_version','driver')))
  and len(hardware['permitted_inventory'])==1 and hardware['permitted_inventory'][0]==target,'actual original GPU/driver inventory differs from root platform')
 status=get(root+'status.json')
 need(status['phase']=='complete'and status['completed_jobs']==8 and status['failure']is None
  and type(status['monotonic_ns'])is int and status['monotonic_ns']>0,'final actual complete8 controller status required')
 return {'original_whole_FR3_identity_source_bound':True,'original_pair_workload_whole_member_hash_checked':True,
  'original_host_boot_versions_and_GPU_bound':True,'graph_payload_independently_rehashed':False}

def job_configuration(plan,job,contract):
 out=recorded_path(plan['out']);folder=out/'jobs'/job['name'];steps,_,_,points=H[job['horizon']]
 return {'horizon':job['horizon'],'fixture':False,'graph_path':plan['friendster_root'],'file_endian':'little',
  'pairs_path':str(out/'friendster_pairs.npy'),'seed':20261124,'hops':2,'cap':200,'oversample':1.5,'auto_ratio':1.0,
  'max_neighbors':20000000,'warmup_pairs':256,'steps':steps,'prefetch_factor':1,'audit_pairs':32,'device':'cuda',
  'torch_threads':1,'gpu_memory_budget_gib':24.0,'stock_index_dtype':'auto','batch_size':32,'num_workers':job['workers'],
  'arm':job['arm'],'out':str(folder),'worker_receipts':str(folder/'workers'),'native_root':str(folder/'runtime/native'),
  'native_library':str(out/'build/native_assoc_v1.so'),'checkpoints':points,'semantic_input_prefixes':sorted(set(points+[576])),
  'environment_contract':contract}

def wrapper_terminal(get,infos,plan,platform_sha,reports,pss,disk_output,cross):
 root='e25/results/';terminal=get('e25/envelope/SCIENTIFIC_TERMINAL.json');block=plan['repetition_block']
 streams={h:{r['input_stream_sha256']for r in reports.values()if r['config']['horizon']==h}for h in H}
 pss_expected={}
 for name in reports:
  path=root+name+'.pss.jsonl';value=pss[path];peak=value['all_phases_maximum']
  pss_expected[name]={'raw':infos[path],'observation_counts':value['raw_observation_counts'],
   'max_all_eligible_phases_bytes':None if peak is None else peak['tree_pss_bytes']}
 need(terminal['kind']=='E25_V6_COMPLETE_EIGHT_APP_BLOCK'and terminal['success']is True
  and terminal['repetition_block']==block and terminal['block_jobs']==8 and terminal['global_cohort_jobs']==24
  and terminal['all_three_blocks_required']is True and terminal['packet_manifest_sha256']==PACKET_SHA
  and terminal['protocol_sha256']==PROTOCOL_SHA and terminal['platform_admission_sha256']==platform_sha
  and terminal['YT_writes']==0,'original wrapper terminal source/block/success differs')
 same(terminal['report_sha256'],{name:infos[root+'jobs/'+name+'/report.json']['sha256']for name in reports},'wrapper report digest map')
 same(terminal['consumed_input_streams'],{h:next(iter(v))for h,v in streams.items()},'wrapper horizon input streams')
 need(terminal['cross_horizon_prefix576']==cross['short_final_stream_sha256'],'wrapper cross576 differs')
 same({name:{k:v for k,v in item.items()if k!='scope'}for name,item in terminal['PSS'].items()},pss_expected,'wrapper all eligible-phase PSS independently rederived')
 same({k:v for k,v in terminal['disk_guard'].items()if k!='scope'},
  {'disk_history':infos[root+'disk_observations.jsonl'],'observations':disk_output['raw_disk_and_latest_inventory']['raw_disk_observations_checked'],
   'latest_inventory':infos[root+'output_inventory_observation.json']},'wrapper disk journal/count/inventory')
 lock=terminal['timing_lock'];need(lock['path']==str(recorded_path(plan['coordination_dir'])/'PUBLIC_SYSTEMS_RUNNING.lock')
  and lock['active_marker_absent']is True and lock['exclusive_flock_acquired_and_released']is True
  and type(lock['checked_unix'])in(int,float)and math.isfinite(lock['checked_unix'])and lock['checked_unix']>0,'wrapper original post-producer lock proof differs')
 same(terminal['prior_failed_reference'],PRIOR_FAILED_REFERENCE,'wrapper failed/excluded reference only')
 return {'original_wrapper_terminal_independently_bound':True,'wrapper_PSS_and_disk_reaggregated':True,
  'prior_failed_reference_only':PRIOR_FAILED_REFERENCE}

def audit(data,infos,pss,events,disk_facts,collection,whole):
 def get(path):need(path in data,'required original JSON missing: '+path);return data[path]
 root='e25/results/';source='e25/sources/packet/'
 need(infos[source+'SOURCE_MANIFEST.json']['sha256']==PACKET_SHA,'frozen producer source manifest differs')
 for name,value in get(source+'SOURCE_MANIFEST.json')['files'].items():need(infos[source+name]==value,'frozen source payload differs')
 plan,records=get(root+'plan.json'),get(root+'job_records.json');block=collection['block'];expected=[j for j in grid()if j['repetition']==block]
 platform=get(root+'PLATFORM_ADMISSION.json');platform_sha=infos[root+'PLATFORM_ADMISSION.json']['sha256']
 launch_frame=root_collection(get,infos,collection,whole,platform,platform_sha)
 need(plan['execute']is True and plan['source_manifest_sha256']==PACKET_SHA and plan['protocol_sha256']==PROTOCOL_SHA
  and hashlib.sha256(json.dumps(plan['protocol'],sort_keys=True).encode()).hexdigest()==PROTOCOL_SHA
  and type(block)is int and block in(1,2,3)and plan['repetition_block']==block and plan['global_protocol_jobs']==24
  and plan['global_job_order']==[j['name']for j in grid()]and 0<plan['max_hours']<=3900/3600
  and len(plan['jobs'])==len(records)==8,'genuine selected original-order8 plan/records required')
 done,terminal=get(root+'COMPLETE.json'),get(root+'SCIENTIFIC_TERMINAL.json')
 need(all(done[k]is True for k in('controller_finished','success','owned_tree_quiescent'))and done['failure']is None
  and done['records']==records==terminal['records']and terminal['success']is True and terminal['jobs']==8
  and done['repetition_block']==terminal['repetition_block']==block and done['global_protocol_jobs']==terminal['global_protocol_jobs']==24
  and done['platform_admission_file']==terminal['platform_admission_file']=='PLATFORM_ADMISSION.json'
  and done['platform_admission_sha256']==terminal['platform_admission_sha256']==platform_sha
  and done['YT_writes']==done['statefulset_actions']==terminal['YT_writes']==0 and terminal['no_new_quality_claim']is True
  and terminal['timing_protocol']==plan['protocol']['timing']and terminal['scope']==plan['protocol']['scope'],
  'complete eight-app block terminal/platform/source semantics required')
 need(done['controller_identity']['source_manifest_sha256']==PACKET_SHA and done['controller_identity']['repetition_block']==block
  and isinstance(done['controller_identity']['nonce'],str)and len(done['controller_identity']['nonce'])==32,'original owned active-marker token source/block/nonce required')
 environment_contract,environment_output=environment_admission(get,infos,plan,done,events,platform)
 storage_output=storage_and_controller_guard(get,infos,plan,done,disk_facts,platform,platform_sha)
 environment_output.update(environment_qualification(get,pss,environment_contract,events,
  (done['controller_identity']['pid'],done['controller_identity']['start_ticks'])))
 inputs_output=input_host_bindings(get,infos,plan,platform)
 need(launch_frame['producer_launch_ns']<=events.first<=events.last<=get(root+'status.json')['monotonic_ns']<=launch_frame['producer_wait_ns'],
  'complete controller ENV/status spans outside actual producer lifetime')
 need(launch_frame['producer_launch_ns']<=disk_facts.first<=disk_facts.last
  <=storage_output['raw_disk_and_latest_inventory']['latest_inventory']['observed_monotonic_ns']
  <=get(root+'status.json')['monotonic_ns']<=launch_frame['producer_wait_ns'],
  'complete raw disk/latest-inventory span outside actual producer lifetime')
 native_expected=get(source+'pins/friendster_native_expected.json');build=get(root+'build/build_receipt_v1.json')
 need(build['source_sha256']==infos[source+'native_core/native_assoc_v1.cpp']['sha256']and sha_hex(build['library_sha256'])
  and build['source']==str(PurePosixPath(launch_frame['packet'])/'native_core/native_assoc_v1.cpp')
  and build['library']==str(PurePosixPath(plan['out'])/'build/native_assoc_v1.so'),'actual pinned native source/library build identity/path required')
 reports={};metrics={};pss_output=[];raw_observations=[];pss_headline_runs=[];environment_jobs=[]
 previous_completion=get(root+'actual_cuda_qualification.exit.json')['wait_returned_monotonic_ns']
 for job,declared,record in zip(expected,plan['jobs'],records):
  need(all(declared[k]==v for k,v in job.items())and all(record[k]==job[k]for k in('name','horizon','workers','repetition','arm')),'original randomized grid/order differs')
  name=job['name'];folder=root+'jobs/'+name+'/';steps,measured,consumed,points=H[job['horizon']];cfg=declared['config']
  same(cfg,job_configuration(plan,job,environment_contract),'exact actual application defaults/paths/contract')
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
  need(cfg['environment_contract']==environment_contract,'exact same inherited ENVv3 contract in every app')
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
   if job['arm']==ARMS[1]:need(worker['initialization_count']==1 and worker['native_initialization']['owner_pid']==worker['pid']
    and worker['native_initialization']['persistent_assoc_bytes_per_worker']==524866928 and worker['native_initialization']['library_sha256']==build['library_sha256']
    and worker['native_arrays']['readonly']is True and worker['native_arrays']['rowptr_bytes']+worker['native_arrays']['col_bytes']==29421941096
    and not any(signatures.values()),'one native-only private8N state per spawn worker required')
   else:need(worker['initialization_count']==0 and any(signatures.values()),'actual compiled kernel signature required; no native state')
  extracted=sum(w['completed_graphs']for w in workers);need(extracted==report['completed_extractions_including_unconsumed_prefetch']and consumed<=extracted<=consumed+32*job['workers'],'consumed/prefetch extraction bound differs')
  if job['arm']==ARMS[1]:
   promotion=report['native_promotion'];need(promotion['rowptr_bytes']+promotion['col_bytes']==29421941096 and report['imports_ready_ns']<=promotion['begin_ns']<=promotion['end_ns']<=report['model_setup_begin_ns'],'fresh native preparation timing/bytes differ')
   post,cleanup=get(folder+'native_post_exit_integrity.json'),get(folder+'native_scratch_cleanup.json')
   need(post['outside_application_lifetime']is True and end<=post['observed_monotonic_ns']<=cleanup['monotonic_ns']<=launch_frame['producer_wait_ns']
    and cleanup['after_actual_exit_and_hash']is True and cleanup['owned_files_removed']==3 and post['expected_proof']==native_expected,'post-exit native verification/cleanup order differs')
   previous_completion=cleanup['monotonic_ns']
   for f,size_key,sha_key in(('rowptr.i8','rowptr_bytes','rowptr_sha256'),('col.i8','col_bytes','col_sha256')):need(post['files'][f]=={'bytes':native_expected[size_key],'sha256':native_expected[sha_key]},'whole derived native hash differs')
  else:need(report['native_promotion']is None,'compiled must not prepare native format')
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
 rows=[];ratios=[];comparisons=[];cross_workers=[];horizon_streams={};horizon_prefixes={}
 for horizon in H:
  selected=[r for r in reports.values()if r['config']['horizon']==horizon]
  need(len({r['input_stream_sha256']for r in selected})==1
   and all(r['semantic_input_prefixes']==selected[0]['semantic_input_prefixes']for r in selected),'source-bound streams/prefix maps differ within horizon')
  horizon_streams[horizon]=selected[0]['input_stream_sha256'];horizon_prefixes[horizon]=selected[0]['semantic_input_prefixes']
  for w in(1,8):
   arms={}
   for arm in ARMS:
    name=f'{horizon}_w{w}_r{block}_{arm}';arms[arm]={'name':name,'metrics':metrics[name]}
   rows.append({'horizon':horizon,'workers':w,'repetition':block,'arms':arms})
   ratio_metrics={}
   for key in arms[ARMS[0]]['metrics']:
    with localcontext()as context:
     context.prec=50;ratio_metrics[key]=float(Decimal(str(arms[ARMS[0]]['metrics'][key]))/Decimal(str(arms[ARMS[1]]['metrics'][key])))
   ratios.append({'horizon':horizon,'workers':w,'repetition':block,'numerator':ARMS[0],'denominator':ARMS[1],'metrics':ratio_metrics})
   a,b=[reports[f'{horizon}_w{w}_r{block}_{arm}']for arm in ARMS]
   need(a['semantic_input_prefixes']==b['semantic_input_prefixes']and a['excluded_warmup_prefix_audits']==b['excluded_warmup_prefix_audits'],'paired source-reported consumed input/prefix digests differ')
   comparisons.append({'horizon':horizon,'workers':w,'repetition':block,'consumed_graphs':H[horizon][2],'all_consumed_inputs_exact':True,'stream_sha256':a['input_stream_sha256']})
 for w in(1,8):
  short=[reports[f'short320_w{w}_r{block}_{arm}']for arm in ARMS];long=[reports[f'long6400_w{w}_r{block}_{arm}']for arm in ARMS]
  need(len({r['semantic_input_prefixes']['576']for r in short+long})==1 and short[0]['input_stream_sha256']==long[0]['semantic_input_prefixes']['576'],'worker-specific cross-horizon576 digest equality differs')
  cross_workers.append({'workers':w,'repetition':block,'consumed_input_graphs':576,'short_final_stream_sha256':short[0]['input_stream_sha256'],
   'long_prefix_stream_sha256':long[0]['semantic_input_prefixes']['576'],'all_block_arms_horizons_exact':True})
 prefix={r['semantic_input_prefixes']['576']for r in reports.values()};need(len(prefix)==1,'short/long576 source-bound digests differ across block')
 cross={'consumed_input_graphs':576,'short_final_stream_sha256':next(iter(prefix)),'long_prefix_stream_sha256':next(iter(prefix)),
  'all_arms_workers_in_block_exact':True,'repetition_block':block}
 same(terminal['stream_comparisons'],comparisons,'four within-block stream comparisons');same(terminal['cross_horizon_input_prefix576'],cross,'cross-horizon576')
 same(terminal['cross_horizon_worker_comparisons'],cross_workers,'two worker-specific cross-horizon comparisons')
 recorded=get(root+'scientific_summary.json');need(recorded['status']=='complete'and recorded['jobs']==8 and recorded['repetition_block']==block and recorded['global_protocol_jobs']==24
  and recorded['platform_admission_sha256']==platform_sha,'complete original single-block summary required')
 same(recorded['rows'],rows,'four independently regenerated paired raw configuration rows')
 same([{k:v for k,v in item.items()if k!='scope'}for item in recorded['paired_compiled_over_native']],ratios,'four independently recomputed within-block paired ratios')
 same([{k:v for k,v in item.items()if k!='scope'}for item in recorded['sampled_PSS']],pss_output,'eight independently eligible PSS phase maxima')
 wrapper_output=wrapper_terminal(get,infos,plan,platform_sha,reports,pss,storage_output,cross)
 producer_exit=get('e25/envelope/child_exit.json');clean(producer_exit,guard=False)
 need(done['controller_identity']['pid']==producer_exit['pid']and done['controller_identity']['start_ticks']==producer_exit['start_ticks'],'actual successful producer exit binding required')
 return {'block':block,'block_jobs':8,'global_protocol_jobs':24,'global24_complete':False,'all_three_complete_blocks_required':True,
  'raw8_metrics':raw_observations,'rows':rows,'paired_compiled_over_native':ratios,'sampled_PSS':pss_output,
  'cross_horizon_input_prefix576':cross,'cross_horizon_worker_comparisons':cross_workers,
  'headline_application_PSS_all_phases':{'scope':'Per-run sampled application-tree maximum over all independently eligible complete/stable/same-phase observations; not full-resident memory requirement or atomic peak.',
   'missing_rule':'No eligible sample means unobserved, never zero. No three-run range exists in a single block.',
   'tie_rule':'First eligible raw observation among tied maxima; scan_end is a non-atomic scan endpoint.','runs':pss_headline_runs},
  'all_app_waits_zero_normal_worker_close':True,'all_initial_models_exact':True,'original_summary_independently_recomputed':True,
  'environment_block':{**environment_output,'timing_jobs':environment_jobs},'storage_block':storage_output,
  'original_input_host_binding':inputs_output,'original_wrapper_binding':wrapper_output,'actual_launch_frame':launch_frame,
  'source_bound_consumed_input_streams':horizon_streams,'source_bound_semantic_prefixes_by_horizon':horizon_prefixes,
  'source_bound_consumed_digest_equality_checked':True,'consumed_tensor_bytes_independently_rehashed':False,
  'PSS_raw_observation_counts':{k:v['raw_observation_counts']for k,v in pss.items()}}


def root_collection(get,infos,c,whole,p,platform_sha):
 need(c['kind']=='ROOT_E25_V6_ACTUAL_BLOCK_COLLECTION'and type(c['block'])is int and c['block']in(1,2,3)
  and c['block']==p['block']and c['pod_uid']==p['pod_uid']and c['sts_uid']==p['sts_uid'],'root collection block/actual UID binding differs')
 expected={'packet_manifest_sha256':PACKET_SHA,'protocol_sha256':PROTOCOL_SHA,'wrapper_sha256':WRAPPER_SHA,
  'ready_sha256':WRAPPER_READY_SHA,'platform_admission_sha256':platform_sha}
 need(all(c['source_hashes'][k]==v for k,v in expected.items()),'root collection frozen source/platform hashes differ')
 need(c['archive']['bytes']==whole['bytes']and c['archive']['sha256']==whole['sha256'],'externally collected archive bytes/SHA differ')
 complete=c['outer_complete']
 need(complete['kind']=='E25_V6_LOCAL_BLOCK_WRAPPER_TERMINAL'and complete['success']is True
  and complete['owned_tree_quiescent']is True and complete['failure']is None
  and complete['repetition_block']==c['block']and complete['block_jobs']==8 and complete['global_cohort_jobs']==24
  and complete['all_three_blocks_required']is True and complete['YT_reads']==complete['YT_writes']==complete['statefulset_actions']==0
  and complete['automatic_retry']is False,'known original outer failure/partial state excludes successful block')
 archived=complete['archive']
 need(archived['bytes']==whole['bytes']and archived['sha256']==whole['sha256']and archived['archive']==c['archive']['file']
  and archived['files']==len(infos)-1 and archived['manifest_sha256']==infos['SHA256_MANIFEST.json']['sha256']
  and archived['all_whole_member_hashes_verified']is True and archived['raw_payload_bytes']==sum(v['bytes']for v in infos.values())
  and archived['compressed_cap_bytes']==128*MIB and archived['raw_cap_bytes']==384*MIB and archived['no_graph_payloads']is True,
  'original outer terminal archive/member/cap provenance differs from independently collected archive')
 actual=c['actual_launch'];outer=c['outer_process'];launch=actual['outer']
 need(actual['kind']=='ROOT_E25_V6_ACTUAL_BLOCK_LAUNCH'and actual['block']==c['block']
  and actual['pod_uid']==c['pod_uid']and actual['sts_uid']==c['sts_uid']and actual['source_hashes']==c['source_hashes']
  and actual['automatic_retry']is False and actual['YT_writes']==0,'genuine new root actual-launch frame required')
 need(type(outer['pid'])is int and outer['pid']>0 and type(outer['start_ticks'])is int and outer['start_ticks']>0
  and outer['state']=='Z'and type(outer['actual_kernel_wait_status'])is int and outer['actual_kernel_wait_status']==0
  and type(outer['actual_kernel_exit_code'])is int and outer['actual_kernel_exit_code']==0
  and launch['pid']==outer['pid']and launch['start_ticks']==outer['start_ticks'],'actual original wrapper zombie kernel exit0 required; absence is not wait0')
 need(c['owned_postexit_scan']['live_owned_rows']==[]and c['timing_lock_available']is True and c['active_marker_absent']is True,
  'root actual post-exit owned quiescence/lock/marker proof required')
 root='e25/results/';producer_exit=get('e25/envelope/child_exit.json');clean(producer_exit,guard=False)
 need(c['producer_exit']==producer_exit,'root producer-exit object differs from original wrapper receipt')
 child=get('e25/envelope/child_launch.json')
 need(child['pid']==producer_exit['pid']and child['start_ticks']==producer_exit['start_ticks'],'original producer child launch/wait identity differs')
 plan=get(root+'plan.json');base=recorded_path(p['base']);entry=recorded_path(launch['entrypoint']);out=recorded_path(launch['out'])
 packet=recorded_path(child['entrypoint']).parent;proof_path=recorded_path(actual['platform_path']);coordination=recorded_path(plan['coordination_dir'])
 need(entry.is_relative_to(base)and entry.name=='run_block.py'and out.is_relative_to(base)and packet.is_relative_to(base)
  and child['entrypoint']==str(packet/'run_lifecycle.py')and plan['out']==str(out/'producer')
  and actual['platform_path']==plan['platform_admission']['source_path']and c['archive']['file']==str(out/'block_raw.tgz')
  and recorded_path(launch['log']).is_relative_to(base),'recorded actual wrapper/producer/proof/archive path frame differs')
 outer_tail=['-B','-u',str(entry),'--repetition-block',str(c['block']),'--packet',str(packet),
  '--friendster-root',plan['friendster_root'],'--platform-admission-json',str(proof_path),'--platform-admission-sha256',platform_sha,
  '--out',str(out),'--coordination',str(coordination),'--execute']
 child_tail=['-B','-u',str(packet/'run_lifecycle.py'),'--friendster-root',plan['friendster_root'],'--out',plan['out'],
  '--coordination-dir',str(coordination),'--max-hours',str(3900/3600),'--repetition-block',str(c['block']),
  '--platform-admission-json',str(proof_path),'--platform-admission-sha256',platform_sha,'--source-manifest-sha256',PACKET_SHA,'--execute']
 need(isinstance(launch['argv'],list)and launch['argv'][1:]==outer_tail
  and isinstance(child['argv'],list)and child['argv'][1:]==child_tail
  and recorded_path(launch['argv'][0]).name.startswith('python')and recorded_path(child['argv'][0]).name.startswith('python')
  and all(type(v)is int and v>0 for v in(launch['launch_monotonic_ns'],child['launch_monotonic_ns'],producer_exit['wait_returned_monotonic_ns']))
  and launch['launch_monotonic_ns']<=child['launch_monotonic_ns']<producer_exit['wait_returned_monotonic_ns'],
  'root original argv/options/launch/actual-wait clocks differ')
 need(type(actual['observed_unix'])in(int,float)and math.isfinite(actual['observed_unix'])
  and actual['observed_unix']>=p['observed_unix']>0,'root actual launch must follow original platform observation')
 same(get('e25/envelope/platform_admission.json'),p,'wrapper derivative of exact original platform proof')
 preflight=get('e25/envelope/preflight.json')
 need(preflight['block']==c['block']and preflight['platform']==infos[root+'PLATFORM_ADMISSION.json']
  and preflight['wrapper']==infos['e25/sources/envelope/run_block.py']and preflight['max_seconds']==4200
  and preflight['producer_max_seconds']==3900 and preflight['YT_writes']==0,'original wrapper preflight source/proof/caps differs')
 need(recorded_path(plan['friendster_root'])==base/'data/friendster/full_csr'
  and not(out.is_relative_to(packet)or packet.is_relative_to(out)or out.is_relative_to(recorded_path(plan['friendster_root']))
  or recorded_path(plan['friendster_root']).is_relative_to(out))
  and not any(coordination.is_relative_to(path)for path in(out,packet,recorded_path(plan['friendster_root']))),
  'original source/input/output/coordination separation differs')
 source='e25/sources/packet/';manifest=get(source+'SOURCE_MANIFEST.json')['files']
 need({n[len(source):]for n in infos if n.startswith(source)}==set(manifest)|{'SOURCE_MANIFEST.json'},'complete exact frozen source inventory required')
 envelope='e25/sources/envelope/'
 need({n[len(envelope):]for n in infos if n.startswith(envelope)}=={'run_block.py','SOURCE_READY.json'}
  and infos[envelope+'run_block.py']['sha256']==WRAPPER_SHA and infos[envelope+'SOURCE_READY.json']['sha256']==WRAPPER_READY_SHA,
  'exact complete frozen wrapper source/readiness required')
 ready=get(envelope+'SOURCE_READY.json')
 need(ready['status']=='SOURCE_READY_FOR_ROOT_PLATFORM_ADMISSION'and ready['ready_for_allocation']is True
  and ready['producer_manifest_sha256']==PACKET_SHA and ready['protocol_sha256']==PROTOCOL_SHA
  and ready['files']['run_block.py']==infos[envelope+'run_block.py'],'original wrapper source readiness differs')
 expected_jobs={j['name']for j in grid()if j['repetition']==c['block']}
 need(all(n[len(root+'jobs/'):].split('/',1)[0]in expected_jobs for n in infos if n.startswith(root+'jobs/')),
  'another repetition/prior-job subtree cannot enter this block')
 need(not any(n.startswith(root+j['name']+'.')for j in grid()if j['repetition']!=c['block']for n in infos),
  'another repetition raw launch/exit/PSS cannot enter this block')
 need(get(root+'PROTOCOL.json')==plan['protocol']if root+'PROTOCOL.json'in infos else True,'stored protocol differs')
 need(get(source+'PROTOCOL.json')==plan['protocol'],'frozen protocol bytes/object differ from original block plan')
 need(plan['protocol']['timing_jobs']==24 and plan['protocol']['max_block_seconds']==3900 and plan['protocol']['outer_block_seconds']==4200
  and plan['protocol']['allocation_blocks']['blocks']==[1,2,3]and plan['protocol']['allocation_blocks']['applications_per_block']==8,
  'prospective global24/three8/caps protocol differs')
 need(get('SHA256_MANIFEST.json')['graphs_included']is False and get('SHA256_MANIFEST.json')['numba_cache_included']is False,
  'raw archive must preserve bounded evidence rather than graph/cache payloads')
 for name in('build_native','actual_cuda_qualification'):
  support_launch=get(root+name+'.launch.json');support_exit=get(root+name+'.exit.json')
  need(child['launch_monotonic_ns']<=support_launch['launch_monotonic_ns']<support_exit['wait_returned_monotonic_ns']<=producer_exit['wait_returned_monotonic_ns'],
   'original support command outside actual producer lifetime')
 return {'packet':str(packet),'producer_launch_ns':child['launch_monotonic_ns'],
  'producer_wait_ns':producer_exit['wait_returned_monotonic_ns'],'outer_pid':outer['pid'],'outer_start_ticks':outer['start_ticks'],
  'recorded_remote_archive':c['archive']['file'],'source_proof_argv_and_actual_exit_checked':True}


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--archive-sha256',required=True)
 parser.add_argument('--root-collection',type=Path,required=True);parser.add_argument('--root-collection-sha256',required=True)
 parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 need(all(sha_hex(v)for v in(PACKET_SHA,PROTOCOL_SHA,WRAPPER_SHA,WRAPPER_READY_SHA,ENV_SOURCE_SHA,a.archive_sha256,a.root_collection_sha256)),
  'exact frozen source plus independently known actual archive/collection SHA required')
 need(not a.output.exists()and not a.output.is_symlink(),'fresh actual single-block audit output required')
 need(pin(Path(__file__).with_name('independent_env.py'),MIB)['sha256']==ENV_SOURCE_SHA,'exact role-aware independent CPU decoder required')
 external=pin(a.root_collection,32*MIB);need(external['sha256']==a.root_collection_sha256,'independently known root collection SHA differs')
 with a.root_collection.open('rb')as stream:collection_bytes=stream.read(32*MIB+1)
 need(len(collection_bytes)<=32*MIB and pin(a.root_collection,32*MIB)==external
  and hashlib.sha256(collection_bytes).hexdigest()==external['sha256'],'bounded stable root collection read required')
 collection=decode(collection_bytes);del collection_bytes
 need(collection['kind']=='ROOT_E25_V6_ACTUAL_BLOCK_COLLECTION','actual root collection schema required')
 data,infos,pss,whole,events,disk_facts=load_archive(a.archive,a.archive_sha256)
 need(external['bytes']+sum(v['bytes']for n,v in infos.items()if n.endswith('.json'))<=64*MIB,'root collection plus retained JSON metadata64MiB cap')
 result=audit(data,infos,pss,events,disk_facts,collection,whole)
 report={'kind':'INDEPENDENT_ACTUAL_E25_V6_EIGHT_APP_BLOCK_AUDIT','status':'PASS_ACTUAL_E25_V6_EIGHT_APP_BLOCK_AUDIT',
  'archive':whole,'root_collection':external,'script':pin(Path(__file__),MIB),
  'independent_environment_module':pin(Path(__file__).with_name('independent_env.py'),MIB),
  'source_manifest_sha256':PACKET_SHA,'protocol_sha256':PROTOCOL_SHA,'wrapper_sha256':WRAPPER_SHA,
  'wrapper_ready_sha256':WRAPPER_READY_SHA,'whole_members_rehashed':len(infos),'pre_outcome_headline_PSS_rule_sha256':PSS_RULE_SHA,
  'platform':get_platform(data),'scope':'Only one complete prospective8-app block; no global24 success before three independently complete blocks. Raw parent/checkpoint/shutdown clocks and within-block ratios are rederived; serialized consumed-input digests are source-bound equality evidence, not independently rehashed tensor bytes. Source-reported hashing cost is bounded, not re-timed. Every raw CPU bracket/mask/boundary and recorded disk threshold is checked; raw statvfs block multiplication was not serialized. PSS eligibility/maxima retain every eligible phase and explicit unknowns, with no CPU/disk filtering. No recorded path/graph/model access, producer import, new experiment, live cluster/YT check, population/cross-allocation causal claim.',
  'actual_new_experiments_GPU_cluster_network_YT_actions':0,**result}
 with a.output.open('x')as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush()
 print(json.dumps({'status':report['status'],'block':report['block'],'jobs':8,'global24_complete':False,'output':str(a.output),'archive':whole},indent=2))


def get_platform(data):
 return data['e25/results/PLATFORM_ADMISSION.json']


if __name__=='__main__':main()
