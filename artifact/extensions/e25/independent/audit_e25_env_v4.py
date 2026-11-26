#!/usr/bin/env python3
"""Bounded independent stdlib peer for a genuine completed E25 v4 raw archive.

No producer, summary helper, ML library, recorded path, GPU or cluster is used.
Raw app clocks and per-process PSS observations are independently reaggregated.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,random,tarfile
from decimal import Decimal,localcontext
from pathlib import Path,PurePosixPath
import independent_env as env
import independent_disk as disk

MIB,GIB=1<<20,1<<30
ARMS=('reference_union_compiled','reference_union_native_persistent')
H={'short320':(10,320,576,[32,256,576]),'long6400':(200,6400,6656,[32,256,1280,3456,6656])}
LABELS={32:'first_trained_batch',256:'warmup_complete',576:'measured_prefix320',1280:'measured_prefix1024',3456:'measured_prefix3200',6656:'measured_prefix6400'}
PACKET_SHA='82f93337ba4b4333cc319b610f6bad1ad24c5f9edf1c79731ae6127eed94c005'
PROTOCOL_SHA='ff42b44bf17ff54d9994731543fbb3d73caaabb56c7150f576e208ad6c3f1fc2'
PSS_RULE_SHA='47b8a0f24c4ec3f386a8b8ba29f6d52a2c3af54ea46b3f2389f6483914bb0a90'
MODEL_SHA='f88254079fe6c152b4c0153c857a63516348353d578f918622e7f6719559618e'
MODEL={'name':'DGCNN','sortpool_k':281,'hidden_channels':32,'num_layers':3,'max_z':1000,
 'learning_rate':0.0001,'optimizer':'Adam','use_feature':False,'node_embedding':False,'parameters':596674}

def need(ok,message):
 if not ok:raise ValueError(message)
def decode(raw):
 def unique(pairs):
  out={}
  for key,value in pairs:need(key not in out,'duplicate JSON key');out[key]=value
  return out
 return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def sha_hex(value):return isinstance(value,str)and len(value)==64 and all(c in'0123456789abcdef'for c in value)
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

def load_archive(path,expected_sha):
 whole=pin(path,128*MIB);need(whole['sha256']==expected_sha,'independently collected whole archive SHA differs')
 payloads,infos,pss={}, {}, {};total=kept=0;known={j['name']for j in grid()}|{'build_native','actual_cuda_qualification'};events=env.Facts();disk_facts=disk.Facts()
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
   phases={};counts={'eligible':0,'ineligible':0,'sampling_error':0};peak=None;env_facts=env.Facts()
   with tar.extractfile(member)as stream:
    if is_pss or is_events or is_disk:
     while True:
      line_cap=4096 if is_disk else MIB;line=stream.readline(line_cap+1)
      if not line:break
      need(len(line)<=line_cap and line.endswith(b'\n'),'complete bounded JSONL line required');size+=len(line);digest.update(line)
      sample=decode(line)
      if is_disk:disk_facts.ingest(sample);continue
      if is_events:events.ingest(sample,boundary=True);continue
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
   if is_pss:pss[name]={'phases':phases,'raw_observation_counts':counts,'all_phases_maximum':peak,'environment_facts':env_facts}
   elif name.endswith('.json'):payloads[name]=decode(data)
 need('SHA256_MANIFEST.json'in payloads,'whole-member manifest required')
 need(payloads['SHA256_MANIFEST.json']['files']=={k:v for k,v in infos.items()if k!='SHA256_MANIFEST.json'},'every member must match internal manifest')
 need(pin(path,128*MIB)==whole,'archive changed while reading');return payloads,infos,pss,whole,events,disk_facts

def stats(values):
 need(len(values)==3 and all(type(v)in(int,float)and math.isfinite(v)and v>=0 for v in values),'three finite nonnegative observations required')
 with localcontext()as context:
  context.prec=50;n=[Decimal(str(v))for v in values];mean=sum(n)/3;sd=(sum((x-mean)**2 for x in n)/2).sqrt()
  return {'values':values,'mean':float(mean),'sample_sd':float(sd),'median':float(sorted(n)[1]),'minimum':float(min(n)),'maximum':float(max(n))}

def environment_job_labels(events,name,launch_ns,wait_ns,controller):
 labels=[]
 for prefix in('before_command_','immediately_before_Popen_','after_actual_wait_'):
  key=prefix+name;need(key in events.labels,'missing original ENVv3 named command guard: '+key);value=events.labels[key]
  need(value['count']==1 and value['identities']=={controller},'unique actual controller boundary identity required')
  labels.append(value)
 need(labels[0]['last_bracket_ns']<=labels[1]['first_scan_ns']<=labels[1]['last_bracket_ns']<=launch_ns<wait_ns<=labels[2]['first_scan_ns'],
  'ENVv3 complete before/Popen/actual-wait guard spans differ')
 return {prefix:{'first_scan_ns':value['first_scan_ns'],'last_bracket_ns':value['last_bracket_ns']}for prefix,value in zip(('before_command','immediately_before_Popen','after_actual_wait'),labels)}

def environment_admission(get,infos,plan,done,events):
 root='e25/results/';source='e25/sources/outer/';outer=get(source+'CONTRACT.json')
 chosen=outer['expected_cpu_affinity'];need(len(env.mask(chosen))==27,'root-bound exact27 mask required')
 need(outer['environment_policy']==env.POLICY and outer['packet_manifest_sha256']==PACKET_SHA and outer['protocol_sha256']==PROTOCOL_SHA
  and outer['expected_cpu_max']==env.QUOTA and outer['expected_memory_max']==env.MEMORY,'new ENVv3 outer/source/quota binding required')
 need(plan['environment_policy']==done['environment_policy']==env.POLICY and plan['selected_cpu_affinity']==chosen
  and done['environment_guard_success']is True and done['source_manifest_sha256']==PACKET_SHA,'new complete ENVv3 plan/controller gate required')
 need(not any(name.startswith(root)and name.endswith('ENVIRONMENT_FAILURE.json')for name in infos),'any original ENVv3 failure prevents success')
 receipt=get(root+'ENVIRONMENT_ADMISSION.json');contract=receipt['contract']
 need(infos[root+'ENVIRONMENT_ADMISSION.json']['sha256']==done['environment_admission_sha256']
  and receipt['status']=='PASS'and receipt['policy']==env.POLICY and receipt['chosen_cpu_affinity']==chosen==contract['chosen_cpu_affinity']
  and contract['policy']==env.POLICY and contract['pin_calls']==1 and contract['no_exclusive_or_continuous_placement_guarantee']is True
  and not contract.get('portable_unsupported',False),'original once-pin target admission identity required')
 base=contract['resources'];env.projection(base)
 need(base['hostname']==outer['hostname']and base['boot_id']==outer['expected_boot_id']and base['cpu_max']==env.QUOTA
  and base['memory_max']==env.MEMORY and env.mask(chosen)<=env.mask(contract['admission_before_affinity'])
  and env.mask(chosen)<=env.mask(base['ambient_cpu_pool'])and env.mask(chosen)<=env.mask(base['online_cpus']),'raw target admission resources differ')
 declaration=get(source+'ROOT_CPU_PREDECLARATION.json')
 need(infos[source+'ROOT_CPU_PREDECLARATION.json']['sha256']==outer['root_cpu_predeclaration_sha256']
  and declaration['selected_cpu_affinity']==declaration['cpu_affinity']==chosen and declaration['boot_id']==base['boot_id']
  and declaration['pod_uid']==outer['pod_uid']and declaration['cgroup']['cpu.max']==base['cpu_max']
  and declaration['cgroup']['memory.max']==base['memory_max']and declaration['cgroup']['cpu.weight']==base['cpu_weight']
  and declaration['cgroup']['cpuset.mems.effective']==base['ambient_mems']and declaration['statefulset_uid']==outer['statefulset_uid']
  and declaration['storage_amendment_sha256']==outer['root_storage_amendment_sha256']
  and declaration['source_actual_storage_admission_sha256']==outer['root_actual_storage_admission_sha256'],
  'original pre-science v4 CPU/storage admission differs')
 topology=[]
 for row in declaration['cpu_topology']:
  need(set(row)=={'logical_cpu','package','core','numa_nodes'}and type(row['package'])is int and type(row['core'])is int
   and isinstance(row['numa_nodes'],list)and all(type(v)is int and v>=0 for v in row['numa_nodes']),'actual v4 CPU topology schema')
  topology.append({'cpu':row['logical_cpu'],'physical_package_id':str(row['package']),'core_id':str(row['core']),
   'numa_nodes':['node'+str(v)for v in row['numa_nodes']]})
 need([{k:r[k]for k in('cpu','physical_package_id','core_id','numa_nodes')}for r in contract['topology']]==topology
  and[r['cpu']for r in topology]==chosen and len({(r['physical_package_id'],r['core_id'])for r in topology})==declaration['physical_cores'],
  'root/admission observed package/core/NUMA topology differs; root predeclaration did not serialize complete sibling lists')
 controller=(done['controller_identity']['pid'],done['controller_identity']['start_ticks'])
 admission_proof=env.boundary(receipt['after'],contract,controller,label='controller_admission')
 event_proof=events.bind(contract,controller)
 need(events.labels.get('controller_admission',{}).get('count')==1
  and events.labels['controller_admission']['first_ns']==receipt['after']['observed_monotonic_ns']and'assert_inputs'in events.labels,
  'original admission/event label binding required')
 prior='e25/prior_aborted/E25_ENV_V3_EXTERNAL_STOP_DIAGNOSTIC_v1.tgz';prior_pin={'bytes':1168934,'sha256':'34d1dbc0aba48f3ce72b766a655f48ee42b857edf70a7d19415d12aa90490d39'}
 need(infos.get(prior)==prior_pin and outer['prior_aborted_archive_bytes']==prior_pin['bytes']
  and outer['prior_aborted_archive_sha256']==prior_pin['sha256'],'exact separately preserved abortedENVv3 capsule identity required')
 prior_receipt=get('e25/prior_aborted/PRESERVATION_RECEIPT.json')
 need(infos['e25/prior_aborted/PRESERVATION_RECEIPT.json']['sha256']==outer['prior_aborted_receipt_sha256']
  and infos['e25/prior_aborted/PRESERVATION_RECEIPT.json']==infos[source+'PRIOR_ABORTED_PRESERVATION_RECEIPT.json']
  and all(prior_receipt[k]==v for k,v in prior_pin.items())and prior_receipt['kind']=='ENV_V3_AVAILABLE_LOCAL_EVIDENCE_PRESERVATION_RECEIPT'
  and prior_receipt['full24_success']is False and prior_receipt['raw_partial_job_metrics_preserved']is False
  and prior_receipt['actual_OS_exit0_proven']is False and prior_receipt['prior_failed_env_v2_capsule_nested_once']is True
  and prior_receipt['prior_failed_v1_nested_inside_v2']is True and prior_receipt['observed_completed_jobs_only']==2
  and prior_receipt['members']==81 and prior_receipt['raw_payload_bytes']==1762608
  and prior_receipt['all_available_selected_member_hashes_verified']is True and prior_receipt['local_fsync']is True,
  'available local observations/source capsule only; no recovered partial metric or OS-exit proof may be invented')
 return contract,{'policy':env.POLICY,'chosen_cpu_affinity':chosen,'admission_sha256':infos[root+'ENVIRONMENT_ADMISSION.json']['sha256'],
  'root_predeclaration_sha256':outer['root_cpu_predeclaration_sha256'],'admission':admission_proof,'controller_events':event_proof,
  'prior_aborted':{'archive':prior_pin,'scope':'Nested bytes independently rehashed; no expansion, timing reads, pooling or metric use. Internal-member verification is the separately pinned preservation receipt.'}}

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

def storage_and_controller_guard(get,infos,plan,done,disk_facts):
 root='e25/results/';source='e25/sources/outer/';outer=get(source+'CONTRACT.json')
 need(done['controller_guard_failure']is None and not any(name.endswith('CONTROLLER_GUARD_FAILURE.json')for name in infos),
  'any latched controller failure prevents successful full-grid audit')
 reference=plan['protocol']['controller_guard_amendment']
 need(reference['version']=='latched_disk_guard_v4'and reference['storage_admission']['artifact']=='ROOT_STORAGE_AMENDMENT.json'
  and reference['storage_admission']['sha256']==outer['root_storage_amendment_sha256']
  and reference['storage_admission']['actual_admission_sha256']==outer['root_actual_storage_admission_sha256'],'frozen controller/storage protocol reference differs')
 amendment=get(source+'ROOT_STORAGE_AMENDMENT.json');actual=get(source+'ROOT_STORAGE_ACTUAL_ADMISSION.json')
 need(infos[source+'ROOT_STORAGE_AMENDMENT.json']['sha256']==outer['root_storage_amendment_sha256']
  and infos[source+'ROOT_STORAGE_ACTUAL_ADMISSION.json']['sha256']==outer['root_actual_storage_admission_sha256']==amendment['actual_admission_sha256']
  and actual==amendment['actual_admission'],'immutable original/nested storage admission binding differs')
 need(amendment['kind']=='E25_ENV_V4_PREDECLARED_STORAGE_PLATFORM_AMENDMENT'
  and amendment['status']=='FROZEN_ACTUAL_PLATFORM_BEFORE_QUALIFICATION_AND_FULL_SCIENCE'
  and amendment['base']==outer['base']and amendment['pod_uid']==outer['pod_uid']and amendment['statefulset_uid']==outer['statefulset_uid']
  and amendment['pod_name']==outer['hostname']and amendment['boot_id']==outer['expected_boot_id']and amendment['node']==outer['expected_node']
  and amendment['cpu_affinity']==outer['expected_cpu_affinity']and amendment['gpu_uuid']==outer['gpu_uuid']
  and amendment['expected_storage_device']==outer['expected_storage_device']and amendment['scientific_jobs_started']==0
  and amendment['YT_writes']==0,'predeclared actual v4 platform differs')
 need(plan['out']==outer['science_out']and plan['friendster_root']==outer['friendster_root']and plan['coordination_dir']==outer['coordination'],
  'actual protocol output/input/coordination paths differ from bound outer')
 need(actual['kind']=='ROOT_ACTUAL_PVC_STORAGE_ADMISSION_V4'and actual['status']=='PASS_ACTUAL_MOUNT_AND_OWN_DIRECTORY_ONLY'
  and actual['base']==outer['base']and actual['expected_storage_device']==outer['expected_storage_device']
  and actual['boot_id']==outer['expected_boot_id']and actual['cpu_affinity']==outer['expected_cpu_affinity']and actual['large_payloads_on_PVC']is True
  and actual['actual_scientific_jobs_started']==0 and actual['YT_writes']==0,
  'actual owned-mount admission differs')
 kube=actual['kubernetes'];need(kube['namespace']==outer['namespace']and kube['pod_uid']==outer['pod_uid']
  and kube['statefulset_uid']==outer['statefulset_uid']and kube['pod']==outer['hostname']and kube['node']==outer['expected_node']and kube['qos']=='Guaranteed'
  and kube['PVC_claim']==amendment['PVC_claim_name']==outer['intended_existing_claim']
  and kube['PVC_uid']==amendment['PVC_claim_uid']==outer['intended_existing_claim_uid']
  and kube['PV_volume_name']==amendment['PV_volume_name']and kube['storage_class']==amendment['storage_class']
  and kube['PV_claimRef_directly_visible']is False and isinstance(kube['PV_binding_scope'],str)and kube['PV_binding_scope'],
  'actual PVC/Pod/volumeName identity differs; no direct PV claimRef proof was available')
 need(amendment['project_subdirectory']==outer['storage_project_subdirectory']and actual['mount']==amendment['mount']
  and amendment['mount']['mount_point']==outer['base']and amendment['mount']['root']=='/'+outer['storage_project_subdirectory']
  and amendment['mount']['major_minor']==disk.linux_major_minor(outer['expected_storage_device'])
  and amendment['all_source_graph_results_native_and_temp_cache_paths_under_BASE']is True
  and actual['project_owner']['claim']==outer['intended_existing_claim']and actual['project_owner']['claim_uid']==outer['intended_existing_claim_uid'],
  'actual project subPath/mount scope differs')
 for probe in('PVC_write_probe','root_layer_write_probe'):
  need(actual[probe]['write_fsync_readback_and_unlink']is True and actual[probe]['directory_fsync']is True,'actual bounded owned-write/fsync/readback/unlink proof')
 need(actual['root_layer_filesystem']['mount_readonly']is False,'root layer was observed readonly at initial admission')
 need(amendment['initial_free_threshold_bytes']==80*GIB and amendment['initial_inode_threshold']==65536
  and amendment['live_free_threshold_bytes']==4*GIB and amendment['max_nonruntime_science_outputs_bytes']==2*GIB,'initial/runtime thresholds conflated')
 initial=actual['after'];need(type(initial['device'])is int and initial['device']==outer['expected_storage_device']
  and type(initial['bytes_available'])is int and amendment['initial_free_bytes']==initial['bytes_available']>=80*GIB
  and initial['mount_readonly']is False and type(initial['inodes_total'])is int and initial['inodes_total']>=0
  and type(initial['inodes_available'])is int and amendment['initial_inodes_available']==initial['inodes_available']>=0
  and(initial['inodes_total']==0 or initial['inodes_available']>=65536),'actual initial filesystem admission differs')
 binding=get('e25/envelope/admission.json')['storage'];mount=binding['mount'];expected_mount=amendment['mount']
 need(binding['policy']==outer['storage_policy']=='persistent_project_mount_v4'
  and binding['device']==outer['expected_storage_device']and binding['amendment']==infos[source+'ROOT_STORAGE_AMENDMENT.json']
  and binding['original_actual_admission']==infos[source+'ROOT_STORAGE_ACTUAL_ADMISSION.json']
  and binding['root_kubernetes_binding']==kube and mount['mount_point']==outer['base']and mount['mount_root']==expected_mount['root']
  and mount['filesystem_type']==expected_mount['filesystem']and mount['major_minor']==expected_mount['major_minor']and mount['source']==expected_mount['source']
  and'rw'in mount['mount_options']and'ro'not in mount['mount_options']and'ro'not in mount['super_options']
  and type(binding['current_bytes_available'])is int and binding['current_bytes_available']>=4*GIB
  and type(binding['current_inodes_reported'])is bool and(not binding['current_inodes_reported']or binding['current_inodes_available']>0),
  'recorded outer current mount/device/readwrite/runtime admission differs')
 need(infos[root+'disk_observations.jsonl']['bytes']<=8*MIB and infos[root+'output_inventory_observation.json']['bytes']<=4096,'bounded disk/inventory original files required')
 inventory=get(root+'output_inventory_observation.json');computed=disk_facts.bind(outer['expected_storage_device'],outer['science_out'],inventory)
 recorded=get('e25/envelope/SCIENTIFIC_TERMINAL.json')['controller_guard_proof']
 expected={'controller_guard_failure':None,'disk_history':infos[root+'disk_observations.jsonl'],'disk_observations':computed['raw_disk_observations_checked'],
  'first_observed_monotonic_ns':computed['first_observed_ns'],'last_observed_monotonic_ns':computed['last_observed_ns'],
  'minimum_recorded_bytes_available':computed['minimum_reported_bytes_available'],'inode_reporting_observations':computed['inode_reporting_observations'],
  'minimum_recorded_inodes_available':computed['minimum_reported_inodes_available'],'latest_output_inventory':infos[root+'output_inventory_observation.json'],
  'expected_storage_device':outer['expected_storage_device']}
 need({k:v for k,v in recorded.items()if k!='scope'}==expected,'outer disk guard proof differs from independently streamed raw values')
 return {'storage_amendment_sha256':outer['root_storage_amendment_sha256'],'actual_admission_sha256':outer['root_actual_storage_admission_sha256'],
  'controller_guard_failure':None,'raw_disk_and_latest_inventory':computed,'outer_guard_proof_independently_recomputed':True,
  'scope':'Actual Bound PVC UID/volumeName plus Pod mount/subPath/probes and recorded runtime device thresholds; direct PV claimRef was not visible. Historical node807 readonly episode is not treated as a current global-health verdict. No ephemeral partial v3 metric/OS-exit claim.'}

def audit(data,infos,pss,events,disk_facts):
 def get(path):need(path in data,'required original JSON missing: '+path);return data[path]
 root='e25/results/';source='e25/sources/packet/'
 need(infos[source+'SOURCE_MANIFEST.json']['sha256']==PACKET_SHA,'frozen producer source manifest differs')
 for name,value in get(source+'SOURCE_MANIFEST.json')['files'].items():need(infos[source+name]==value,'frozen source payload differs')
 plan,records=get(root+'plan.json'),get(root+'job_records.json');expected=grid()
 need(plan['execute']is True and plan['source_manifest_sha256']==PACKET_SHA and plan['protocol_sha256']==PROTOCOL_SHA
  and hashlib.sha256(json.dumps(plan['protocol'],sort_keys=True).encode()).hexdigest()==PROTOCOL_SHA
  and 0<plan['max_hours']<=4 and len(plan['jobs'])==len(records)==24,'genuine full24 plan/records required')
 done,terminal=get(root+'COMPLETE.json'),get(root+'SCIENTIFIC_TERMINAL.json')
 need(all(done[k]is True for k in('controller_finished','success','owned_tree_quiescent'))and done['failure']is None
  and done['records']==records==terminal['records']and terminal['success']is True and terminal['jobs']==24,'complete producer terminal required')
 environment_contract,environment_output=environment_admission(get,infos,plan,done,events)
 storage_output=storage_and_controller_guard(get,infos,plan,done,disk_facts)
 environment_output.update(environment_qualification(get,pss,environment_contract,events,
  (done['controller_identity']['pid'],done['controller_identity']['start_ticks'])))
 native_expected=get(source+'pins/friendster_native_expected.json');build=get(root+'build/build_receipt_v1.json')
 need(build['source_sha256']==infos[source+'native_core/native_assoc_v1.cpp']['sha256']and sha_hex(build['library_sha256']),'actual pinned native source/library build identity required')
 reports={};metrics={};pss_output=[];raw_observations=[];pss_headline_runs=[];environment_jobs=[]
 for job,declared,record in zip(expected,plan['jobs'],records):
  need(all(declared[k]==v for k,v in job.items())and all(record[k]==job[k]for k in('name','horizon','workers','repetition','arm')),'original randomized grid/order differs')
  name=job['name'];folder=root+'jobs/'+name+'/';steps,measured,consumed,points=H[job['horizon']];cfg=declared['config']
  need(cfg['horizon']==job['horizon']and cfg['steps']==steps and cfg['checkpoints']==points and cfg['semantic_input_prefixes']==sorted(set(points+[576]))
   and cfg['arm']==job['arm']and cfg['num_workers']==job['workers']and cfg['batch_size']==32 and cfg['warmup_pairs']==256
   and cfg['seed']==20261124 and cfg['prefetch_factor']==1 and cfg['hops']==2 and cfg['cap']==200 and cfg['device']=='cuda'
   and cfg['max_neighbors']==20000000 and cfg['auto_ratio']==1.0,'fixed consumer config differs')
  report=get(folder+'report.json');need(record['report_sha256']==infos[folder+'report.json']['sha256']and report['config']==cfg==get(folder+'config.json'),'report/config/hash binding differs')
  exit=get(root+name+'.exit.json');launch=get(root+name+'.launch.json');need(exit==record['exit'],'raw app exit differs');clean(exit)
  need(exit['pid']==launch['pid']and exit['start_ticks']==launch['start_ticks']and exit['launch_monotonic_ns']==launch['launch_monotonic_ns']==report['launch_monotonic_ns'],'actual app PID/start/clock binding differs')
  t=launch['launch_monotonic_ns'];end=exit['wait_returned_monotonic_ns'];need(t<=report['entry_monotonic_ns']<=report['imports_ready_ns']<=report['model_setup_begin_ns']
   <=report['model_setup_end_ns']<=report['iterator_begin_ns']<=report['iterator_created_ns']<=report['first_batch_ready_ns'],'ordered startup milestones differ')
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
   need(get(folder+'workers/worker'+str(worker['worker_id'])+'.json')==worker and worker['status']=='closed'and worker['native_closed']is True
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
   need(post['outside_application_lifetime']is True and post['observed_monotonic_ns']>=end and cleanup['after_actual_exit_and_hash']is True and cleanup['owned_files_removed']==3 and cleanup['monotonic_ns']>=post['observed_monotonic_ns'],'post-exit native verification/cleanup order differs')
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
  app_environment['sampled_CPU']=pss[pss_path]['environment_facts'].bind(environment_contract,(launch['pid'],launch['start_ticks']),(t,end),require_observed=False)
  app_environment['name']=name;app_environment['worker_boundary_pairs_checked']=len(workers);environment_jobs.append(app_environment)
  pss_output.append({'name':name,'phases':pss[pss_path]['phases']})
  observation=pss[pss_path];peak=observation['all_phases_maximum']
  if peak is not None:need(t<=peak['scan_end_monotonic_ns']<=end,'headline PSS scan timestamp outside actual application lifetime')
  pss_headline_runs.append({'job':job,'status':'observed'if peak is not None else'unobserved',
   'eligible_samples':observation['raw_observation_counts']['eligible'],
   'total_samples':sum(observation['raw_observation_counts'].values()),'maximum':peak})
 rows=[];ratios=[];comparisons=[];pss_headline_groups=[]
 for horizon in H:
  need(len({r['input_stream_sha256']for r in reports.values()if r['config']['horizon']==horizon})==1,'full streams differ within horizon')
  for w in(1,8):
   groups={}
   for arm in ARMS:
    chosen=sorted((j for j in expected if j['horizon']==horizon and j['workers']==w and j['arm']==arm),key=lambda j:j['repetition'])
    groups[arm]={k:[metrics[j['name']][k]for j in chosen]for k in metrics[chosen[0]['name']]}
    rows.append({'horizon':horizon,'workers':w,'arm':arm,'metrics':{k:stats(v)for k,v in groups[arm].items()}})
    pss_runs=sorted((r for r in pss_headline_runs if r['job']['horizon']==horizon and r['job']['workers']==w and r['job']['arm']==arm),key=lambda r:r['job']['repetition'])
    maxima=[None if r['maximum']is None else r['maximum']['tree_pss_bytes']for r in pss_runs];observed=sum(v is not None for v in maxima)
    pss_headline_groups.append({'horizon':horizon,'workers':w,'arm':arm,
     'status':'observed_all_three'if observed==3 else('unobserved'if observed==0 else'incomplete'),
     'repetitions':[r['job']['repetition']for r in pss_runs],'per_run_maxima_bytes':maxima,
     'observed_runs':observed,'three_run_range_bytes':{'minimum':min(maxima),'maximum':max(maxima)}if observed==3 else None,
     'runs':pss_runs})
   ratio_metrics={}
   for key in groups[ARMS[0]]:
    with localcontext()as context:
     context.prec=50;values=[float(Decimal(str(a))/Decimal(str(b)))for a,b in zip(groups[ARMS[0]][key],groups[ARMS[1]][key])]
    ratio_metrics[key]=stats(values)
   ratios.append({'horizon':horizon,'workers':w,'numerator':ARMS[0],'denominator':ARMS[1],'metrics':ratio_metrics})
   for rep in(1,2,3):
    a,b=[reports[f'{horizon}_w{w}_r{rep}_{arm}']for arm in ARMS]
    need(a['semantic_input_prefixes']==b['semantic_input_prefixes']and a['excluded_warmup_prefix_audits']==b['excluded_warmup_prefix_audits'],'paired actual input/prefix audits differ')
    comparisons.append({'horizon':horizon,'workers':w,'repetition':rep,'consumed_graphs':H[horizon][2],'all_consumed_inputs_exact':True,'stream_sha256':a['input_stream_sha256']})
 prefix={r['semantic_input_prefixes']['576']for r in reports.values()};need(len(prefix)==1,'short/long576 inputs differ across any app')
 cross={'consumed_input_graphs':576,'short_final_stream_sha256':next(iter(prefix)),'long_prefix_stream_sha256':next(iter(prefix)),'all_arms_workers_repetitions_exact':True}
 same(terminal['stream_comparisons'],comparisons,'12 raw stream comparisons');same(terminal['cross_horizon_input_prefix576'],cross,'cross-horizon576')
 recorded=get(root+'scientific_summary.json');need(recorded['status']=='complete'and recorded['jobs']==24,'complete original summary required')
 same(recorded['rows'],rows,'8 independently regenerated clock/stat rows')
 same([{k:v for k,v in item.items()if k!='scope'}for item in recorded['paired_compiled_over_native']],ratios,'4 independently paired ratio groups')
 same([{k:v for k,v in item.items()if k!='scope'}for item in recorded['sampled_PSS']],pss_output,'24 independently eligible PSS phase maxima')
 producer_exit=get('e25/envelope/child_exit.json');clean(producer_exit,guard=False)
 need(done['controller_identity']['pid']==producer_exit['pid']and done['controller_identity']['start_ticks']==producer_exit['start_ticks'],'actual successful producer exit binding required')
 return {'raw24_metrics':raw_observations,'rows':rows,'paired_compiled_over_native':ratios,'sampled_PSS':pss_output,'cross_horizon_input_prefix576':cross,
  'headline_application_PSS_all_phases':{'scope':'Sampled application-process-tree PSS maxima across ALL independently eligible complete, stable-membership, same-phase samples; setup and promotion included when eligible. Not full-resident footprint, memory requirement, atomic peak, cgroup charge or population uncertainty.',
   'tie_rule':'First eligible raw-log observation among tied maximum values; scan_end_monotonic_ns timestamps the end of a non-atomic scan.',
   'missing_rule':'No eligible sample is unobserved, never zero; configuration range requires all three run maxima.',
   'runs':pss_headline_runs,'groups':pss_headline_groups},
  'all_app_waits_zero_normal_worker_close':True,'all_initial_models_exact':True,'original_summary_independently_recomputed':True,
  'environment_v4':{**environment_output,'timing_jobs':environment_jobs},'storage_v4':storage_output,
  'PSS_raw_observation_counts':{k:v['raw_observation_counts']for k,v in pss.items()}}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--archive-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 need(sha_hex(a.archive_sha256),'independent whole archive SHA required');need(not a.output.exists()and not a.output.is_symlink(),'fresh actual audit receipt required')
 data,infos,pss,whole,events,disk_facts=load_archive(a.archive,a.archive_sha256);result=audit(data,infos,pss,events,disk_facts)
 report={'kind':'INDEPENDENT_ACTUAL_E25_ENV_V4_RAW_NUMERICAL_PSS_BRACKETS_AND_DISK_PEER','status':'PASS_ACTUAL_RAW_E25_ENV_V4_AUDIT','archive':whole,'whole_members_rehashed':len(infos),
  'script':pin(Path(__file__),MIB),'independent_environment_module':pin(Path(__file__).with_name('independent_env.py'),MIB),
  'independent_disk_module':pin(Path(__file__).with_name('independent_disk.py'),MIB),'source_manifest_sha256':PACKET_SHA,'protocol_sha256':PROTOCOL_SHA,
  'pre_outcome_headline_PSS_rule_sha256':PSS_RULE_SHA,
  'scope':'Existing actual ENVv4 observations only, retaining fixed27_env_v3 CPU policy. Decimal50 descriptive3-process statistics, per-repetition paired ratios, raw monotonic clocks, independently reclassified stable complete phase-PSS, and raw ENVv3 resource/mask/identity bindings. ENV PID/TID/cgroup stability and all boundary attempts independently recomputed from full raw brackets. Exact boundary wall start and actual sleep durations remain source-pinned receipts; declared limits/elapsed and raw-span consistency are checked. Original/nested PVC admission hashes and recorded disk/device/inode/readonly thresholds and latest non-atomic output inventory independently checked; raw statvfs block counters were not serialized. No producer imports, full graph/PyG-byte oracle recomputation, live PID/lock/YT check, final outer/YT publication check, population inference, or new experiment.',
  'new_experiments_graph_model_GPU_cluster_network_YT_actions':0,**result}
 with a.output.open('x')as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps({'status':report['status'],'output':str(a.output),'archive':whole,'rows':len(result['rows']),'paired_groups':len(result['paired_compiled_over_native'])},indent=2))
if __name__=='__main__':main()
