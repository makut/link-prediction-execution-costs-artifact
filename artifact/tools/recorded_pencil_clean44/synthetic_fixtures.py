"""SYNTHETIC fixtures only; no real platform, graph, model or measurement claims."""
from pathlib import Path
import copy,hashlib,io,json,tarfile
import numpy as np
import audit_clean_v1 as a
from archive_contract import *
from plan_contract import ARMS,GOLDEN,make_jobs
from evidence_checks import EXPECTED
from trace_audit import NAMES,array_hash,numpy_fp,tensor_fp
def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return path

def empty_fp(dtype=np.int64):return numpy_fp(np.asarray([],dtype=dtype))

def fp(count,tag='same'):
    return {'kind':'tensor','dtype':'torch.float32','shape':[count],'stride':[1],'sha256':hashlib.sha256(tag.encode()).hexdigest()}

def model(tiny=False):
    return {'checkpoint':None,'initialization_seed':42,'tiny_fixture_model':tiny,'features':128,'feature_fusion':'early','encoding':'adjacency_row',
        'settings':{'name':REMOTE_BRIDGE+'/bert_config','hidden_size':32 if tiny else 512,
        'num_hidden_layers':1 if tiny else 3,'num_attention_heads':4 if tiny else 8,'intermediate_size':64 if tiny else 2048,
        'max_position_embeddings':2048,'from_pretrained':False},'bert_config_sha256':'0'*64}

def trace(folder,tiny=True):
    """Literal toy adjacency reference; real-shaped fixture has intentionally empty edges."""
    folder=Path(folder);folder.mkdir();q=7 if tiny else 1024
    adjacency={0:[1,2],1:[0,3],2:[0,4],3:[1,4],4:[2,3],5:[],6:[7],7:[6],8:[]}
    pairs=np.asarray([[3,0],[0,1],[8,5],[8,0],[0,0],[7,6],[8,8]],dtype=np.int64)if tiny else np.asarray(
        [[i,i+1000]for i in range(512)]+[[i,i+2000]for i in range(512)],dtype=np.int64)
    labels=np.asarray([0,1,0,0,0,1,0],dtype=np.int64)if tiny else np.r_[np.ones(512,dtype=np.int64),np.zeros(512,dtype=np.int64)]
    arcs=[(u,v)for u in range(9)for v in adjacency[u]];slots={e:i for i,e in enumerate(arcs)}
    sets=[];roots=[];records=[]
    words_py=np.tile(np.arange(625,dtype=np.uint32),(q,1));words_np=np.tile(np.arange(624,dtype=np.uint32),(q,1))
    torch=np.tile(np.arange(32,dtype=np.uint8),(q,1));scalars=np.zeros((q,5),dtype=np.float64);scalars[:,2]=624
    features=(np.arange(9*128,dtype=np.float32).reshape(9,128)%97)/97
    for i,(u,v)in enumerate(pairs):
        S=np.asarray(sorted({int(u),int(v),*(adjacency[int(u)]+adjacency[int(v)] if tiny else[])}),dtype=np.int64)
        mapping={node:j for j,node in enumerate(S)};root=np.asarray([mapping[u],mapping[v]],dtype=np.int64)
        picked=[(mapping[x],mapping[y],slot,slots[(y,x)])for slot,(x,y)in enumerate(arcs)if tiny and x in mapping and y in mapping]
        rows=np.asarray([x[0]for x in picked],dtype=np.int64);cols=np.asarray([x[1]for x in picked],dtype=np.int64)
        raw_ids=np.asarray([x[2]for x in picked],dtype=np.int64);stored=np.asarray([x[3]for x in picked],dtype=np.int64)
        post=np.asarray(sorted((int(c),int(r))for r,c in zip(rows,cols)if(r,c)not in((root[0],root[1]),(root[1],root[0]))),dtype=np.int64).reshape(-1,2).T.copy()
        rng={'py_version':3,'py_words':numpy_fp(words_py[i]),'py_gauss':None,'np_algorithm':'MT19937','np_words':numpy_fp(words_np[i]),
            'np_position':624,'np_has_gauss':0,'np_cached_gauss':0.0,'torch_cpu':numpy_fp(torch[i])}
        records.append({'query_index':i,'nodes_hash':array_hash(S),'sum_selected_degrees':sum(len(adjacency[int(n)])for n in S)if tiny else 0,
            'raw':dict(zip(('rows','cols','csr_edge_slots'),map(numpy_fp,(rows,cols,raw_ids)))),'stored_values':numpy_fp(stored),
            'data':{'num_nodes':len(S),'root_n_id':tensor_fp(root,'torch.int64'),'seed_node':tensor_fp(pairs[i],'torch.int64'),
                'y':tensor_fp(labels[i:i+1],'torch.int64'),'edge_index':tensor_fp(post,'torch.int64'),
                'x':tensor_fp(features[S]if tiny else np.zeros((len(S),128),dtype=np.float32),'torch.float32'),
                'id':tensor_fp(S,'torch.int64'),'idx':i},'post_raw_rng':rng,
            'native_sampler_seconds':0.001,'native_public_induction_seconds':0.002,'native_raw_data_seconds':0.003})
        sets.append(S);roots.append(root)
    arrays={'pairs.npy':pairs,'labels.npy':labels,'node_ptr.npy':np.r_[0,np.cumsum([len(v)for v in sets],dtype=np.int64)],
        'nodes.npy':np.concatenate(sets).astype(np.int32),'root_positions.npy':np.asarray(roots,dtype=np.int64),
        'rng_python_words.npy':words_py,'rng_numpy_words.npy':words_np,'rng_torch_cpu.npy':torch,'rng_scalars.npy':scalars}
    for name,array in arrays.items():np.save(folder/name,array,allow_pickle=False)
    put(folder/'records.json',{'records':records})
    meta={'fixture':tiny,'eval_seed':20261126,'query_seed_policy':'SeedSequence([base,0,query_index])',
        'sampler':{'actual_native_operator':'ego_k_hop_sample_adj','depth':1,'neighbors_per_endpoint':40,'replacement':False,'torch_threads':1},
        'post_raw_rng_schema':{'py_version':3,'np_algorithm':'MT19937','torch_cpu_bytes':32},'binding':{'synthetic':True},
        'cohort':{'sources':512,'original_negatives_per_source':1000,'queries':q,'selection_depends_on_timing_or_scores':False,
            'pairs_hash':array_hash(pairs),'labels_hash':array_hash(labels)}}
    put(folder/'manifest.json',{'schema_version':1,'status':'complete','metadata':meta,
        'files':[{'name':name,**identity(folder/name)}for name in sorted(NAMES)]})
    return folder

def refresh_trace(folder):
    manifest=a.read(folder/'manifest.json');manifest['files']=[{'name':name,**identity(folder/name)}for name in sorted(NAMES)];put(folder/'manifest.json',manifest)

def raw(job,contract,arms):
    result={'schema_version':1,'command':job['kind'],'status':'preflight'if job['kind']=='preflight'else'complete',
        'configuration':a.expected_configuration(job,arms),'source_files':contract['driver_sources'],'quality_claim':'none'}
    if job['kind']=='preflight':result['preflight_passed']=True
    if job['kind']=='audit'or(job['kind']=='benchmark'and job['arm'].startswith('cpp_')):
        result['cpp_artifacts']={k:contract['cpp'][k]for k in('module_sha256','source_sha256','library_sha256','build_receipt_sha256')}
    return result

def semantic(job,tracepath,contract,arms):
    result=raw(job,contract,arms);tiny=job['name']=='tiny_audit';count=7 if tiny else 1024;batch=job.get('batch',128)
    result['trace_manifest_sha256']=sha(tracepath/'manifest.json')
    checks=[]
    for training in(False,True):
        for start in range(0,count,batch):
            size=min(batch,count-start);checks.append({'training_wrapper':training,'through_query_index':start+size-1,'batch_size':size,
                'all_input_tensors_exact':True,'logits':{arm:{'hash':fp(size),'max_abs':0,'bitwise_equal':True}for arm in arms}})
    gates=[{'arm':arm,'training_wrapper':training,'queries':count,'reference':'actual online native public sampler and wrapper',
        **{key:True for key in('raw_csr_slots_order_exact','postmask_data_exact','wrapped_and_collated_inputs_exact','native_postraw_cpu_rng_restored_before_candidate_wrapper')}}
        for arm in arms for training in(False,True)]
    result['result']={'model_forward_checked':True,'max_logit_abs_tolerance':0,'model':model(tiny),'semantic_gates':gates,'batch_checks':checks}
    return result

def good_exit(gpu=False):
    return {'returncode':0,'timed_out':False,'owned_tree_quiescent':True,'orphan_cleanup':False,'guard_abort_reason':None,'timeout_seconds':1800,'exactly_one_visible_gpu':gpu,'memory_budget_gib':96,'guard_policy':'unchanged stable_pss_individual_rss_cgroup90_v3 sampled2s'}

def good_guard(path):Path(path).write_text(json.dumps({'decision':{'abort_reason':None,'memory_budget_bytes':96*2**30,'own_pss_estimate_bytes':10}})+'\n')

def benchmark(job,payload,contract,arms):
    result=raw(job,contract,arms);name=f"audit_bf16_b{job['batch']}_w{job['workers']}";audit={'audit_environment':{'synthetic':True},
        'input_binding':{'synthetic':True},'trace_manifest_sha256':'1'*64,'result':{'model':model()}}
    put(payload/(name+'.json'),audit);result.update({k:audit[k]for k in('audit_environment','input_binding','trace_manifest_sha256')})
    sweeps=[]
    for rep,seconds in enumerate((100.,3.,4.,5.)):
        stages={key:a.exact_summary([0.01]*1024)for key in('sampler_seconds','induction_seconds','raw_data_seconds','wrapper_seconds','S_verification_seconds')}
        if job['mode']=='replay':stages['sampler_seconds']=a.exact_summary([0.]*1024)
        cpp=[]
        if job['arm'].startswith('cpp_'):
            n=contract['input_contract']['n'];m=contract['input_contract']['m'];mode='reference_full_clear'if job['arm']=='cpp_full_clear'else'persistent_touched_reset'
            cpp=[[{'pid':123,'mode':mode,'assoc_budget_bytes':2**30,'metadata':{'owner_pid':123,'mode':mode,'nodes':n,'stored_entries':m,
                'version':'persistent_assoc_touched_reset_v1','assoc_bytes_during_each_call':8*n,'persistent_assoc_bytes_per_worker':8*n if mode=='persistent_touched_reset'else 0,
                'library_sha256':contract['cpp']['library_sha256'],'initialization_seconds':0.1,
                'initialization_policy':'once per worker'if mode=='persistent_touched_reset'else'full-N allocation/init inside every native call'}}]]
        sweeps.append({'repetition':rep,'warmup':rep==0,'queries':1024,'elapsed_seconds':seconds,'queries_per_second':1024/seconds,'stages':stages,
            'collator_seconds_total':.1,'h2d_forward_score_copy_seconds_total':.2,'scores':fp(1024),'worker_memory':[{'pid':123,'maxrss_bytes':None,'max_pss_bytes':None,'memory_measurement':'disabled_unobserved; not zero or a worker peak'}],
            'memory_measurement_policy':a.MEMORY_POLICY,'parent_memory_before_sweep':{'maxrss_bytes':3,'rss_bytes':2,'pss_bytes':1},'parent_memory_after_sweep':{'maxrss_bytes':3,'rss_bytes':2,'pss_bytes':1},'parent_memory':{'maxrss_bytes':3,'rss_bytes':2,'pss_bytes':1},
            'cpp_initializations':cpp,'gpu':{'max_allocated_bytes':1,'max_reserved_bytes':2}})
    result['result']={'arm':job['arm'],'pipeline_mode':job['mode'],'batch_size':job['batch'],'num_workers':job['workers'],
        'prefetch_factor':1 if job['workers']else None,'model':model(),'audit_receipt':{'path':str(Path(job['output']).parent/(name+'.json')),'sha256':sha(payload/(name+'.json'))},
        'memory_measurement_policy':a.MEMORY_POLICY,'sweeps':sweeps,'elapsed_summary':a.exact_summary([3.,4.,5.])}
    return result,{name:audit}

def policy(contract,isolated):
    return {'version':'graph_only_transformers_optional_vision_v2','transformers_version':'4.46.2',
        'import_utils_origin':str(Path(isolated['extra_modules']['transformers']['origin']).parent/'utils/import_utils.py'),
        'import_utils_sha256':contract['optional_vision_import_utils_sha256'],'policy_source_sha256':contract['driver_sources']['graph_only_transformers_v2.py'],
        'installed_torchvision_version':isolated['packages'].get('torchvision'),'torchvision_available_before':isolated['packages'].get('torchvision')is not None,
        'torchvision_available_for_graph_worker':False,'torchvision_imported':False,'upstream_source_modified':False,'base_package_modified':False,
        'fake_operator_registered':False,'scope':'Only unused optional vision discovery in pinned Transformers; graph sampler, sparse operator, BERT parameters/forward, precision and input tensors unchanged.'}

def launch_admission(contract):
    admission={'kind':'PENCIL_CLEAN_CONSUMER_V1_ROOT_ADMISSION','authorized_to_execute':True,'fresh_owned_statefulset':True,'namespace':'m-utushkin','pod_name':'seal-clean-consumer-20261125-0','statefulset_name':'seal-clean-consumer-20261125','hostname':'seal-clean-consumer-20261125-0','pod_uid':'00000000-0000-0000-0000-000000000001','statefulset_uid':'00000000-0000-0000-0000-000000000002','observed_unix':100.,'base':BASE,'image':'SYNTHETIC_IMAGE_NOT_A_REAL_LAUNCH','source_manifest_sha256':contract['source_manifest_sha256'],'source_packet_sha256':contract['source_packet_sha256'],'gpu_count':1,'gpu_name_contains':'H100','cpu_request':27,'memory_request_gib':205,'pod_memory_limit_gib':256,'owned_tree_guard_gib':96}
    data=(json.dumps(admission,indent=2)+'\n').encode();admission_sha=hashlib.sha256(data).hexdigest();config={'execute':True,'base':BASE,'out':REMOTE_OUT,'prepared_root':BASE+'/data/citation2/prepared','validation_root':BASE+'/clean_validation512','python':BASE+'/supplied_env/bin/python','max_hours':4.,'memory_guard_gib':96.,'pod_memory_limit_gib':256.,'target':YT,'admission':BASE+'/ROOT_ADMISSION.json','admission_sha256':admission_sha,'source_manifest_sha256':contract['source_manifest_sha256'],'source_packet':BASE+'/controller.tgz','source_packet_sha256':contract['source_packet_sha256']}
    argv=[REMOTE_CODE+'/controller.py','--execute']
    for k,v in config.items():
        if k!='execute':argv.extend(['--'+k.replace('_','-'),str(v)])
    launch={'kind':'actual_clean_consumer_controller_launch','identity':{'pid':123,'start_ticks':456,'pgid':123,'ppid':1,'state':'R'},'entrypoint':REMOTE_CODE+'/controller.py','source_sha256':contract['controller_source_sha256'],'source_manifest_sha256':contract['source_manifest_sha256'],'source_packet_sha256':contract['source_packet_sha256'],'root_admission_sha256':admission_sha,'argv':argv,'started_unix':101.,'hostname':admission['hostname']}
    return config,launch,admission,data

def plan(config):
    protocol=read(GOLDEN/'candidate/PROTOCOL.json');inherited=protocol['GPU_target'];protocol['GPU_target']='one visible H10080Single per job; active controller27CPU205/256GiB contract'
    return {'status':'ROOT_REVIEWABLE_CONTROLLER_PLAN','execution_authorized':False,'jobs':make_jobs(config),'scientific_jobs':44,'protocol':protocol,'source_job_builder_sha256':sha(GOLDEN/'candidate/metadata/ORIGINAL_SCIENTIFIC_JOB_BUILDER.py'),'output_created':False,'inherited_candidate_GPU_target_reference':inherited,'controller_protocol':read(GOLDEN/'PROTOCOL.json'),
    'setup_jobs':[{'name':'platform_before','gpu':True,'timeout_seconds':120,'argv':[config['python'],'-B',REMOTE_CODE+'/platform_probe.py','--output',REMOTE_OUT+'/platform_before.json']},{'name':'build_cpp','gpu':False,'timeout_seconds':180,'argv':[config['python'],'-B',REMOTE_BRIDGE+'/corrected_native/build_native_v1.py','--output-dir',REMOTE_OUT+'/cpp_build']}],
    'after_jobs':[{'name':'platform_after','gpu':True,'timeout_seconds':120,'argv':[config['python'],'-B',REMOTE_CODE+'/platform_probe.py','--output',REMOTE_OUT+'/platform_after.json']}]}

def environment(payload,contract,declared,config):
    versions={k:EXPECTED[k]for k in('torch','numpy','scipy','numba','torch-geometric')};versions['torch-sparse']='0.6.18'
    observed={'schema_version':1,'mode':'base','python':'3.10.99 SYNTHETIC','platform':'Linux-SYNTHETIC','executable':config['python'],'prefix':str(Path(config['python']).parent.parent),'base_prefix':'/usr','torch':'2.5.0+cu121','torch_cuda':'12.1','numpy':'1.24.4','pyg':'2.7.0','torch_sparse_version':'0.6.18','packages':{**EXPECTED,'torch-sparse':'0.6.18'},'critical_versions':versions,
    'critical_modules':{k:{'origin':'/opt/python/lib/python3.10/site-packages/'+k+'/__init__.py','source_sha256':'a'*64}for k in('torch','numpy','scipy','numba','torch_sparse','torch_geometric')},'extra_modules':{k:{'origin':BASE+'/supplied_env/lib/python3.10/site-packages/'+k+'/__init__.py','version':EXPECTED[k],'sha256':'b'*64}for k in('transformers','wandb','rich','tqdm')},'saint_wrapper':{'sha256':contract['native']['saint_wrapper_sha256'],'git_blob':contract['native']['saint_wrapper_git_blob']},'torch_sparse_binaries':{'_saint_cpu.so':{'bytes':1,'sha256':contract['native']['saint_cpu_sha256']}},'visible_gpu':{'count':1,'name':'SYNTHETIC H100','uuid':'GPU-SYNTHETIC','total_memory_bytes':80*(1<<30)},'cpu_observation':{'models_from_bounded_cpuinfo':['SYNTHETIC CPU'],'cpuinfo_prefix_truncated':False,'cpu_affinity':[0,1],'cgroup_cpu_max':'max 100000'}}
    observed['runtime_compatibility']=policy(contract,observed)
    put(payload/'platform_before.json',observed);put(payload/'platform_after.json',observed);put(payload/'initial_cgroup.json',{'max_bytes':256*(1<<30)})
    for job in [*declared['setup_jobs'],*declared['after_jobs']]:
        ex=good_exit(job['gpu']);ex['timeout_seconds']=job['timeout_seconds'];put(payload/(job['name']+'.exit.json'),ex);good_guard(payload/(job['name']+'.guard.jsonl'));(payload/(job['name']+'.log')).write_text('SYNTHETIC\n')
    return observed

def complete_payload(folder,contract):
    payload=Path(folder);payload.mkdir();config,launch,admission,admission_bytes=launch_admission(contract);declared=plan(config);jobs=declared['jobs'];put(payload/'LAUNCH.json',launch);(payload/'ROOT_ADMISSION.json').write_bytes(admission_bytes);(payload/'SOURCES.json').write_bytes((GOLDEN/'SOURCE_MANIFEST.json').read_bytes());put(payload/'plan.json',declared)
    lib=payload/'cpp_build/native_assoc_v1.so';lib.parent.mkdir();lib.write_bytes(b'SYNTHETIC NOT EXECUTABLE')
    source=REMOTE_BRIDGE+'/corrected_native/native_assoc_v1.cpp';wanted=REMOTE_OUT+'/cpp_build/native_assoc_v1.so';build={'schema_version':1,'kind':'corrected_native_build_v1','source':source,'source_sha256':contract['cpp']['source_sha256'],'compiler':'/usr/bin/c++','compiler_version':'SYNTHETIC compiler','argv':['/usr/bin/c++','-std=c++17','-O3','-DNDEBUG','-fPIC','-pthread','-shared',source,'-o',wanted+'.building'],'platform':'Linux-SYNTHETIC','library':wanted,'library_sha256':sha(lib)};put(payload/'cpp_build/build_receipt_v1.json',build)
    contract=copy.deepcopy(contract);contract['cpp'].update(library_sha256=sha(lib),build_receipt_sha256=sha(payload/'cpp_build/build_receipt_v1.json'))
    observed=environment(payload,contract,declared,config);paths={True:trace(payload/'tiny_trace',True),False:trace(payload/'real_trace',False)};upstream={**contract['upstream'],'all_source_files_verified':53}
    strict={'variant':'strict_clean','quality_dataset':True,'files':copy.deepcopy(contract['public_inputs']),'clean_heldout_audits':{'train_csr':{s:{'forward':0,'reverse':0}for s in('valid','test')}}};strict_bytes=(json.dumps(strict)+'\n').encode();strict_sha=hashlib.sha256(strict_bytes).hexdigest()
    actual={config['prepared_root']+'/'+r['path']:{k:r[k]for k in('bytes','sha256')}for r in contract['public_inputs']}
    for name,key in(('valid_edge.npy','valid_positive'),('valid_edge_neg.npy','valid_negative')):actual[config['validation_root']+'/'+name]={'bytes':100,'sha256':contract['input_contract']['payloads'][key]['sha256']}
    for row in actual.values():row.update(mtime_ns=1,inode=2,device=3)
    put(payload/'INPUT_IDENTITY.json',{'status':'pass','numeric_payloads':19,'validation_sources':512,'negatives_per_source':1000,'actual_files':actual,'strict_manifest_path':config['prepared_root']+'/manifest.json','strict_manifest':{'bytes':len(strict_bytes),'sha256':strict_sha}})
    native={};bindings={};metadata={}
    for tiny,path in paths.items():
        n,m=(9,12)if tiny else(contract['input_contract']['n'],contract['input_contract']['m']);setup={'torch':observed['torch'],'torch_sparse':observed['torch_sparse_version'],'pyg':observed['pyg'],'native_setup_seconds':.1,'native_setup_memory':{},'int64_csr_bytes':8*(n+1+m),'public_coo_row_bytes':8*m,'stored_value_bytes':8*m,'symmetric_full_CSR_verified':True,'extension_binaries':[{'name':k,'sha256':v['sha256']}for k,v in observed['torch_sparse_binaries'].items()],'wrapper_sha256':observed['saint_wrapper']['sha256'],'wrapper_git_blob':observed['saint_wrapper']['git_blob'],'software_versions':{k:observed['packages'][k]for k in('numpy','scipy','numba','transformers','wandb')}}
        native[tiny]=(setup,{'native':{k:v for k,v in setup.items()if k not in('native_setup_seconds','native_setup_memory')},'runtime_compatibility':observed['runtime_compatibility'],'python':observed['python'],'platform':observed['platform'],'torch_threads':1,'torch_interop_threads':1,'thread_environment':{k:'1'for k in a.THREADS}})
        manifest=read(path/'manifest.json');binding=manifest['metadata']['binding'];meta=manifest['metadata']['cohort']
        if not tiny:
            payloads={name:{k:v[k]for k in('bytes','sha256')if k in v}for name,v in contract['input_contract']['payloads'].items()}
            for row in payloads.values():row.setdefault('bytes',100)
            binding={'n':n,'m':m,'input_contract_sha256':contract['input_contract_sha256'],'payloads':payloads};meta.update(strict_manifest={'content':strict,'sha256':strict_sha},payload_binding_verification={'status':'pass','quality_plan_checked':False,'payloads':payloads},positive_file=payloads['valid_positive'],negative_file=payloads['valid_negative'])
        bindings[tiny]=binding;metadata[tiny]=meta;manifest['metadata'].update(binding=binding,cohort=meta,upstream=upstream,native=setup);put(path/'manifest.json',manifest)
    audits={};raws={}
    for job in jobs:
        tiny=job['name'].startswith('tiny');path=paths[tiny];count=7 if tiny else 1024
        if job['kind']=='audit':value=semantic(job,path,contract,ARMS);value['result']['model']['bert_config_sha256']=contract['driver_sources']['bert_config/config.json']
        elif job['kind']=='benchmark':
            value,_=benchmark(job,payload,contract,ARMS);name=f"audit_bf16_b{job['batch']}_w{job['workers']}";put(payload/(name+'.json'),audits[name]);value['result']['audit_receipt']['sha256']=sha(payload/(name+'.json'));value['result']['model']=copy.deepcopy(audits[name]['result']['model'])
        else:value=raw(job,contract,ARMS)
        value.update(input_binding=bindings[tiny],input_metadata=metadata[tiny],upstream=upstream)
        if job['kind']!='preflight':setup,stable=native[tiny];value.update(native_setup=setup,audit_environment=stable,runtime_compatibility=observed['runtime_compatibility'],native_admission={'status':'pass','actual_native_cases':7,'before_full_graph_allocation':True})
        if job['kind']=='capture':value['result']={'queries':count,'trace':value['configuration']['trace_dir'],'trace_manifest_sha256':sha(path/'manifest.json'),'trace_files':read(path/'manifest.json')['files'],'native_sampler_summary':a.exact_summary([.001]*count),'native_induction_summary':a.exact_summary([.002]*count)}
        elif job['kind']in('audit','benchmark'):value['trace_manifest_sha256']=sha(path/'manifest.json')
        put(payload/(job['name']+'.json'),value);raws[job['name']]=value
        if job['kind']=='audit':audits[job['name']]=copy.deepcopy(value)
    for job in jobs:put(payload/(job['name']+'.exit.json'),good_exit(job['gpu']));good_guard(payload/(job['name']+'.guard.jsonl'));(payload/(job['name']+'.log')).write_text('SYNTHETIC\n')
    refresh_completed(payload,jobs)
    groups=[]
    for b in(128,512):
        for w in(0,1):groups.append({'precision':'bf16','batch':b,'workers':w,'mode':'online','arms':list(ARMS),'warmup_and_3repeats_all_model_scores_exact':True,'score_fingerprint':fp(1024)})
    put(payload/'crossjob_scores.json',{'status':'pass','groups':groups})
    terminal={'kind':'PENCIL_CLEAN_CONSUMER_SCIENTIFIC_TERMINAL_V1','scientific_success':True,'completed_jobs':44,'planned_jobs':44,'failure':None,'stop_requested':False,'owned_children_quiescent_before_publication':True,'all_planned_science_job_exits_clean':True,'source_manifest_sha256':contract['source_manifest_sha256'],'source_packet_sha256':contract['source_packet_sha256'],'launch_sha256':sha(payload/'LAUNCH.json'),'root_admission_sha256':sha(payload/'ROOT_ADMISSION.json'),'outer_final_pid_exit':'UNKNOWN','finished_unix':110.};put(payload/'SCIENTIFIC_TERMINAL.json',terminal)
    return payload

def refresh_completed(payload,jobs):
    put(payload/'completed_jobs.json',{'count':len(jobs),'jobs':[{'name':j['name'],'kind':j['kind'],'output':identity(payload/(j['name']+'.json')),'exit':read(payload/(j['name']+'.exit.json'))}for j in jobs]})

def package(payload,folder,contract):
    payload=Path(payload);folder=Path(folder);folder.mkdir();values={str(p.relative_to(payload)):p.read_bytes()for p in payload.rglob('*')if p.is_file()and p.name!='SHA256_MANIFEST.json'};rows=[{'path':k,'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}for k,v in sorted(values.items())];mf={'kind':'whole_clean_consumer_evidence_v1','files':rows,'file_count':len(rows),'payload_bytes':sum(r['bytes']for r in rows)};values['SHA256_MANIFEST.json']=(json.dumps(mf)+'\n').encode()
    with tarfile.open(folder/'final_snapshot.tgz','x:gz')as target:
        for name,data in values.items():member=tarfile.TarInfo(name);member.size=len(data);target.addfile(member,io.BytesIO(data))
    for name,data in values.items():p=folder/'extracted'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    put(folder/'extraction_manifest.json',{'files':[{'path':k,'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}for k,v in values.items()]});archive=identity(folder/'final_snapshot.tgz');science=read(payload/'SCIENTIFIC_TERMINAL.json');admission=read(payload/'ROOT_ADMISSION.json')
    context={'target':YT,'archive':archive,'scientific_terminal_sha256':sha(payload/'SCIENTIFIC_TERMINAL.json'),'scientific_success':science['scientific_success'],'all_planned_science_job_exits_clean':science['all_planned_science_job_exits_clean'],'completed_jobs':science['completed_jobs'],'planned_jobs':44,'source_manifest_sha256':contract['source_manifest_sha256'],'source_packet_sha256':contract['source_packet_sha256'],'controller_source_sha256':contract['controller_source_sha256'],'launch_sha256':sha(payload/'LAUNCH.json'),'root_admission_sha256':sha(payload/'ROOT_ADMISSION.json'),'pod_uid':admission['pod_uid'],'statefulset_uid':admission['statefulset_uid'],'stop_requested_at_context_creation':False,'outer_final_pid_exit':'UNKNOWN','readback':{**archive,'verified_unix':111.}}
    text=canonical(context);attribute={'schema_version':1,'kind':ATTRIBUTE,'archive_sha256':archive['sha256'],'archive_bytes':archive['bytes'],'scientific_terminal_sha256':context['scientific_terminal_sha256'],'canonical_receipt_json':text,'canonical_receipt_sha256':hashlib.sha256(text.encode()).hexdigest(),'outer_final_pid_exit':'UNKNOWN'};put(folder/'persistence_attribute.json',attribute)
    put(folder/'export.json',{'status':'verified_existing_persisted_archive','role':'pencil_clean_consumer','all_internal_manifest_files_verified':True,'remote_archive_created':False,'backup_calls':0,'YT_writes':0,'pod_lifecycle_mutations':0,'yt_path':YT,'attribute_name':ATTRIBUTE,'archive':archive,'attribute_canonical_json_sha256':hashlib.sha256(canonical(attribute).encode()).hexdigest(),'scientific_success':science['scientific_success']})
    rootlaunch=folder.with_name(folder.name+'_ROOT_LAUNCH.json');rootadmission=folder.with_name(folder.name+'_ROOT_ADMISSION.json');rootlaunch.write_bytes((payload/'LAUNCH.json').read_bytes());rootadmission.write_bytes((payload/'ROOT_ADMISSION.json').read_bytes())
    return folder,rootlaunch,rootadmission
