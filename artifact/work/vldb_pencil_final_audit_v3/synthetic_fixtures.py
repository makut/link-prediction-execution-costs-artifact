"""Small synthetic regression inputs only; never scientific measurements."""
from pathlib import Path
import copy,hashlib,io,json,tarfile
import numpy as np
import audit_pencil_v3_v1 as a
from archive_contract import BASE,PREFIX,REMOTE_OUT,YT,identity,sha,read
from trace_audit import NAMES,array_hash,numpy_fp,tensor_fp

def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return path
def empty_fp(dtype=np.int64):return numpy_fp(np.asarray([],dtype=dtype))
def fp(count,tag='same'):
    return {'kind':'tensor','dtype':'torch.float32','shape':[count],'stride':[1],'sha256':hashlib.sha256(tag.encode()).hexdigest()}
def model(tiny=False):
    return {'checkpoint':None,'initialization_seed':42,'tiny_fixture_model':tiny,'features':128,'feature_fusion':'early','encoding':'adjacency_row',
        'settings':{'name':BASE+'/pencil_queue_v3_code/bridge/vldb_pencil_bridge_v2/bert_config','hidden_size':32 if tiny else 512,
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

def plan(contract,q,jobs):
    local=a.GOLDEN_QUEUE;remote=Path(BASE)/'pencil_queue_v3_code'
    names=set(q.PINS)|{'pencil_queue_v3.py','runtime_probe_v3.py','README.txt','protocol_spec.json','bridge/vldb_pencil_bridge_v2/SOURCE_MANIFEST.json'}
    names.update('bridge/vldb_pencil_bridge_v2/'+v['path']for v in a.read(local/'bridge/vldb_pencil_bridge_v2/SOURCE_MANIFEST.json')['files'])
    bindings={str(remote/name):identity(local/name)for name in names};L=q.layout(Path(BASE));cpp=contract['cpp']
    bindings[str(L['cpp_module'])]={'bytes':8499,'sha256':cpp['module_sha256']};bindings[str(L['cpp_library'])]={'bytes':1,'sha256':cpp['library_sha256']}
    bindings[str(L['cpp_receipt'])]={'bytes':cpp['build_receipt_bytes'],'sha256':cpp['build_receipt_sha256']}
    return {'recovery_admission_sha256':contract['recovery_admission']['sha256'],'kind':'pencil_queue_v3','jobs':copy.deepcopy(jobs),'protocol_spec':a.read(local/'protocol_spec.json'),'source_bindings':bindings,
        'bridge_source_package_sha256':q.SOURCE_PACKAGE_SHA,'external_sampled_memory_guard_gib':180,'per_job_timeout_seconds':1800,
        'max_admission_hours':1,'max_total_hours_including_wait':4}
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
    return {'returncode':0,'timed_out':False,'owned_tree_quiescent':True,'orphan_cleanup':False,'guard_abort_reason':None,'timeout_seconds':1800,'exactly_one_visible_gpu':gpu}
def good_guard(path):Path(path).write_text(json.dumps({'decision':{'abort_reason':None,'memory_budget_bytes':180*2**30,'own_pss_estimate_bytes':10}})+'\n')

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
            'collator_seconds_total':.1,'h2d_forward_score_copy_seconds_total':.2,'scores':fp(1024),'worker_memory':[{'pid':123,'maxrss_bytes':1,'max_pss_bytes':1}],
            'cpp_initializations':cpp,'gpu':{'max_allocated_bytes':1,'max_reserved_bytes':2}})
    result['result']={'arm':job['arm'],'pipeline_mode':job['mode'],'batch_size':job['batch'],'num_workers':job['workers'],
        'prefetch_factor':1 if job['workers']else None,'model':model(),'audit_receipt':{'path':str(Path(job['output']).parent/(name+'.json')),'sha256':sha(payload/(name+'.json'))},
        'sweeps':sweeps,'elapsed_summary':a.exact_summary([3.,4.,5.])}
    return result,{name:audit}

def export(folder,queue_sha,*,science_success=False,plan_value=None,extra=None,bad_internal=False,science_overrides=None):
    """A complete tiny collector export, with no genuine measurements."""
    folder=Path(folder);folder.mkdir();controller={'pid':123,'start_ticks':456,'source_sha256':queue_sha,'entrypoint':BASE+'/pencil_queue_v3_code/pencil_queue_v3.py'}
    plan_value=copy.deepcopy(plan_value or{});plan_value['controller_identity']=controller
    science={'controller_jobs_finished':True,'success':science_success,'completed_jobs':0,'failure':None if science_success else{'type':'SyntheticFailure','message':'test only'},
        'owned_tree_quiescent':True,'timing_lock_retained':True}
    science.update(science_overrides or{})
    values={PREFIX+'plan.json':(json.dumps(plan_value,indent=2)+'\n').encode(),PREFIX+'SCIENTIFIC_TERMINAL.json':(json.dumps(science,indent=2)+'\n').encode()}
    values.update(extra or{});entries={name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}for name,data in values.items()}
    internal=copy.deepcopy(entries)
    if bad_internal:internal[PREFIX+'plan.json']['sha256']='f'*64
    manifest=json.dumps({'files':internal}).encode();values['SHA256_MANIFEST.json']=manifest
    with tarfile.open(folder/'final_snapshot.tgz','x:gz')as tf:
        for name,data in values.items():entry=tarfile.TarInfo(name);entry.size=len(data);tf.addfile(entry,io.BytesIO(data))
    for name,data in values.items():p=folder/'extracted'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    put(folder/'extraction_manifest.json',{'files':[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}for name,data in values.items()]})
    packed={**identity(folder/'final_snapshot.tgz'),'files':len(entries),'manifest_sha256':hashlib.sha256(manifest).hexdigest()}
    pointer={'yt_path':YT,**identity(folder/'final_snapshot.tgz'),'readback_verified':True,'new_YT_file_nodes':1}
    done={'controller_finished':True,'owned_tree_quiescent':True,'timing_lock_released':True,'persistence_complete':True,'scientific_complete':science_success,
        'success':science_success,'failure':None if science_success else science['failure'],'controller_identity':controller,'archive_pointer':pointer}
    remote=folder/'remote_receipts'/REMOTE_OUT
    for name,value in(('COMPLETE.json',done),('SCIENTIFIC_TERMINAL.json',science),('plan.json',plan_value),('archive_pointer.json',pointer),('archive_local_receipt.json',packed)):put(remote/name,value)
    launch={'pid':123,'start_ticks':456,'source_sha256':queue_sha,'argv':['python',controller['entrypoint'],'--execute']};put(folder/'remote_receipts/pencil_queue_v3_launch.json',launch)
    put(folder/'export.json',{'status':'verified_existing_persisted_archive','role':'pencil','all_internal_manifest_files_verified':True,'remote_archive_created':False,
        'backup_calls':0,'YT_writes':0,'pod_lifecycle_mutations':0,'scientific_success':science_success,'controller_success':science_success,
        'pointer':pointer,'archive_metadata':packed,'download':{**identity(folder/'final_snapshot.tgz'),'file_count':len(values)},'yt_path':YT,
        'receipt':{'path':PREFIX+'SCIENTIFIC_TERMINAL.json','sha256':entries[PREFIX+'SCIENTIFIC_TERMINAL.json']['sha256']}})
    return folder

def recovery(contract):
    pin=contract['recovery_admission'];return {'admitted':True,'kind':'root_pinned_recovery_admission','receipt_sha256':pin['sha256'],
        'replacement_hostname':pin['content']['replacement_hostname'],'prior_archives':copy.deepcopy(pin['content']['prior_archives']),
        'bindings':{BASE+'/RECOVERY_ADMISSION.json':{k:pin[k]for k in('bytes','sha256')}}}
def policy(contract,isolated):
    return {'version':'graph_only_transformers_optional_vision_v2','transformers_version':'4.46.2',
        'import_utils_origin':str(Path(isolated['extra_modules']['transformers']['origin']).parent/'utils/import_utils.py'),
        'import_utils_sha256':contract['optional_vision_import_utils_sha256'],'policy_source_sha256':contract['driver_sources']['graph_only_transformers_v2.py'],
        'installed_torchvision_version':isolated['packages'].get('torchvision'),'torchvision_available_before':isolated['packages'].get('torchvision')is not None,
        'torchvision_available_for_graph_worker':False,'torchvision_imported':False,'upstream_source_modified':False,'base_package_modified':False,
        'fake_operator_registered':False,'scope':'Only unused optional vision discovery in pinned Transformers; graph sampler, sparse operator, BERT parameters/forward, precision and input tensors unchanged.'}
def environment(payload,contract,q,plan):
    native=contract['native'];base_site='/opt/python/lib/python3.10/site-packages';isolated_site=BASE+'/pencil_env_v3/lib/python3.10/site-packages'
    modules={name:{'origin':base_site+'/'+name+'/__init__.py','source_sha256':'a'*64}for name in('torch','numpy','scipy','numba','torch_sparse','torch_geometric')}
    versions={'torch':'2.5.0+cu121','numpy':'1.24.4','scipy':'1.15.3','numba':'0.63.1','torch-sparse':'0.6.18','torch-geometric':'2.7.0'}
    before={'prefix':'/opt/python','base_prefix':'/usr','critical_versions':versions,'critical_modules':modules,
        'torch_sparse_binaries':{'_saint_cpu.so':{'bytes':1,'sha256':native['saint_cpu_sha256']}},
        'saint_wrapper':{'sha256':native['saint_wrapper_sha256'],'git_blob':native['saint_wrapper_git_blob']},
        'torch':'2.5.0+cu121','numpy':'1.24.4','pyg':'2.7.0','torch_sparse_version':'0.6.18','site_packages':[base_site],
        'python':'synthetic Python3.10','platform':'synthetic Linux','packages':{**versions,'torchvision':'0.17.2+cu121'}}
    isolated=copy.deepcopy(before);isolated.update(prefix=BASE+'/pencil_env_v3',visible_gpu={'count':1,'name':'synthetic GPU'})
    extra={name:{'origin':isolated_site+'/'+name+'/__init__.py','sha256':'b'*64,'version':version}for name,version in
        (('transformers','4.46.2'),('wandb','0.18.7'),('rich','14.2.0'),('tqdm','4.67.1'))}
    isolated['extra_modules']=extra;isolated['packages'].update({k:v['version']for k,v in extra.items()})
    content=base_site+'\n';route_info={'bytes':len(content),'sha256':hashlib.sha256(content.encode()).hexdigest()}
    route_path=isolated_site+'/pencil_base_site_v3.pth';route={'path':route_path,**route_info,'content':content,'base_paths':[base_site],'isolated_site_precedes_all_base_paths':True}
    isolated['base_site_route']=route;isolated['runtime_compatibility']=policy(contract,isolated)
    admission=recovery(contract);bindings={**plan['source_bindings'],**admission['bindings'],route_path:route_info,
        BASE+'/pencil_env_v3/pyvenv.cfg':{'bytes':100,'sha256':'c'*64}}
    bindings[str(q.layout(Path(BASE))['wheel_archive'])]={k:contract['wheel'][k]for k in('bytes','sha256')}
    for value in modules.values():bindings[value['origin']]={'bytes':1,'sha256':value['source_sha256']}
    for value in extra.values():bindings[value['origin']]={'bytes':1,'sha256':value['sha256']}
    native_root=base_site+'/torch_sparse';bindings[native_root+'/_saint_cpu.so']=before['torch_sparse_binaries']['_saint_cpu.so']
    bindings[native_root+'/saint.py']={'bytes':1,'sha256':native['saint_wrapper_sha256']}
    bindings[isolated['runtime_compatibility']['import_utils_origin']]={'bytes':1,'sha256':contract['optional_vision_import_utils_sha256']}
    bindings.setdefault(BASE+'/pencil_queue_v3_code/bridge/vldb_pencil_bridge_v2/graph_only_transformers_v2.py',{'bytes':1,'sha256':contract['driver_sources']['graph_only_transformers_v2.py']})
    for name,value in(('base_environment_before',before),('base_environment_after',before),('isolated_environment',isolated),('isolated_environment_after',isolated),
        ('base_site_binding',{'path':route_path,**route_info,'base_paths':[base_site]}),('environment_runtime_bindings',bindings),('admission',admission),
        ('wheel_extraction',{'files':35,'manifest_sha256':contract['wheel']['manifest_sha256'],'uncompressed_bytes':100}),
        ('environment_verification',{'status':'pass','base_unchanged':True,'numerical_and_native_origins_exact':True,'isolated_site_precedes_base_fallback':True})):
        put(payload/(name+'.json'),value)
    for phase in a.ENV_PHASES:put(payload/(phase+'.exit.json'),good_exit(phase.startswith('isolated_environment')));good_guard(payload/(phase+'.guard.jsonl'))
    return isolated

def complete_scientific_export(folder,contract,q,jobs):
    """Integration fixture: synthetic48-job receipts, not real model/graph scores."""
    payload=Path(folder).with_name(Path(folder).name+'_fixture_payload');payload.mkdir()
    declared=plan(contract,q,jobs);isolated=environment(payload,contract,q,declared)
    paths={True:trace(payload/'tiny_trace',True),False:trace(payload/'real_trace',False)}
    upstream={**contract['upstream'],'all_source_files_verified':53}
    # Keep the upstream receipt's exact public field names; extra pin metadata is
    # harmless and is bound identically into raw capture and trace metadata.
    native={};bindings={};metadata={}
    for tiny,path in paths.items():
        n,m=(9,12)if tiny else(contract['input_contract']['n'],contract['input_contract']['m'])
        setup={'torch':isolated['torch'],'torch_sparse':isolated['torch_sparse_version'],'pyg':isolated['pyg'],
            'native_setup_seconds':.1,'native_setup_memory':{},'int64_csr_bytes':8*(n+1+m),'public_coo_row_bytes':8*m,'stored_value_bytes':8*m,
            'symmetric_full_CSR_verified':True,'extension_binaries':[{'name':k,'sha256':v['sha256']}for k,v in isolated['torch_sparse_binaries'].items()],
            'wrapper_sha256':isolated['saint_wrapper']['sha256'],'wrapper_git_blob':isolated['saint_wrapper']['git_blob'],
            'software_versions':{name:isolated['packages'][name]for name in('numpy','scipy','numba','transformers','wandb')}}
        stable={k:v for k,v in setup.items()if k not in('native_setup_seconds','native_setup_memory')}
        native[tiny]=(setup,{'native':stable,'runtime_compatibility':isolated['runtime_compatibility'],'python':isolated['python'],'platform':isolated['platform'],
            'torch_threads':1,'torch_interop_threads':1,'thread_environment':{k:'1'for k in a.THREADS}})
        manifest=read(path/'manifest.json');binding=manifest['metadata']['binding'];meta=manifest['metadata']['cohort']
        if not tiny:
            wanted=contract['input_contract'];payloads={name:{k:v[k]for k in('bytes','sha256')if k in v}for name,v in wanted['payloads'].items()}
            for value in payloads.values():value.setdefault('bytes',100)
            binding={'n':n,'m':m,'input_contract_sha256':contract['input_contract_sha256'],'payloads':payloads}
            strict={'variant':'strict_clean','quality_dataset':True,'files':[{k:v[k]for k in('path','bytes','sha256')}for name,v in wanted['payloads'].items()if name in('graph_starts','graph_columns','features')],
                'clean_heldout_audits':{'train_csr':{split:{'forward':0,'reverse':0}for split in('valid','test')}}}
            meta.update(strict_manifest={'content':strict},payload_binding_verification={'status':'pass','quality_plan_checked':True,'payloads':payloads},
                positive_file=payloads['valid_positive'],negative_file=payloads['valid_negative'])
        bindings[tiny]=binding;metadata[tiny]=meta
        manifest['metadata'].update(binding=binding,cohort=meta,upstream=upstream,native=setup);put(path/'manifest.json',manifest)
    raws={};audits={}
    for job in jobs:
        tiny=job['name'].startswith('tiny');path=paths[tiny];count=7 if tiny else 1024
        if job['kind']=='audit':
            value=semantic(job,path,contract,q.ARMS);value['result']['model']['bert_config_sha256']=contract['driver_sources']['bert_config/config.json']
        elif job['kind']=='benchmark':
            # Fixture helper writes a small audit placeholder; restore the full
            # actually generated audit immediately before binding its SHA.
            value,_=benchmark(job,payload,contract,q.ARMS);name=f"audit_bf16_b{job['batch']}_w{job['workers']}"
            put(payload/(name+'.json'),audits[name]);value['result']['audit_receipt']['sha256']=sha(payload/(name+'.json'))
            value['result']['model']=copy.deepcopy(audits[name]['result']['model'])
        else:value=raw(job,contract,q.ARMS)
        value.update(input_binding=bindings[tiny],input_metadata=metadata[tiny],upstream=upstream)
        if job['kind']!='preflight':
            setup,stable=native[tiny];value.update(native_setup=setup,audit_environment=stable,runtime_compatibility=isolated['runtime_compatibility'],
                native_admission={'status':'pass','actual_native_cases':7,'before_full_graph_allocation':True})
        if job['kind']=='capture':
            value['result']={'queries':count,'trace':value['configuration']['trace_dir'],'trace_manifest_sha256':sha(path/'manifest.json'),
                'trace_files':read(path/'manifest.json')['files'],'native_sampler_summary':a.exact_summary([.001]*count),
                'native_induction_summary':a.exact_summary([.002]*count)}
        elif job['kind']in('audit','benchmark'):value['trace_manifest_sha256']=sha(path/'manifest.json')
        put(payload/(job['name']+'.json'),value);raws[job['name']]=value
        if job['kind']=='audit':audits[job['name']]=copy.deepcopy(value)
    completed=[]
    for job in jobs:
        ex=good_exit(job['gpu']);put(payload/(job['name']+'.exit.json'),ex);good_guard(payload/(job['name']+'.guard.jsonl'))
        (payload/(job['name']+'.log')).write_text('SYNTHETIC integration fixture; not an actual run\n')
        completed.append({'name':job['name'],'kind':job['kind'],'output':identity(payload/(job['name']+'.json')),'exit':ex})
    put(payload/'completed_jobs.json',{'count':48,'jobs':completed})
    # Use frozen producer in tests only as a counterparty; the production auditor
    # never calls its validate_job/crossjob_scores scientific validators.
    scores=q.crossjob_scores([{**job,'output':str(payload/(job['name']+'.json'))}for job in jobs])
    put(payload/'crossjob_model_scores.json',scores)
    extra={PREFIX+str(p.relative_to(payload)):p.read_bytes()for p in payload.rglob('*')if p.is_file()}
    return export(folder,contract['queue_source_sha256'],science_success=True,plan_value=declared,extra=extra,
        science_overrides={'completed_jobs':48,'crossjob_model_scores':scores})
