#!/usr/bin/env python3
"""Independent LOCAL final PENCIL audit. Missing verified export means waiting."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,importlib.util,json,math,statistics,sys
import numpy as np
from archive_contract import BASE,PREFIX,YT,InvalidEvidence,require,read,identity,sha,valid_sha,verify_export
from trace_audit import load_trace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
GOLDEN_QUEUE=ROOT/'work/vldb_pencil_queue_v3'
DEFAULT_EXPORT=ROOT/'outputs/seal_systems_final_20261125/pencil_bridge_v3_final_readonly_v1'
DEFAULT_OUTPUT=ROOT/'outputs/seal_systems_final_20261125/pencil_bridge_v3_independent_audit_v1'
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
ENV_PHASES=('base_environment_before','create_isolated_env','install_offline_extras','isolated_environment','isolated_environment_after','base_environment_after')

def frozen_protocol(contract):
    require(sha(GOLDEN_QUEUE/'pencil_queue_v3.py')==contract['queue_source_sha256']and sha(GOLDEN_QUEUE/'SOURCE_MANIFEST.json')==contract['queue_manifest_sha256'],
        'local golden queue source/manifest changed')
    manifest=read(GOLDEN_QUEUE/'SOURCE_MANIFEST.json')
    for item in manifest['files']:require(identity(GOLDEN_QUEUE/item['path'])=={k:item[k]for k in('bytes','sha256')},'local frozen source file changed')
    spec=importlib.util.spec_from_file_location('audit_only_frozen_qv2',GOLDEN_QUEUE/'pencil_queue_v3.py')
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    q.HERE=Path(BASE)/'pencil_queue_v3_code';q.BRIDGE=q.HERE/'bridge/vldb_pencil_bridge_v2'
    jobs=q.make_jobs(Path(BASE),Path(BASE)/'pencil_env_v3/bin/python')
    require(len(jobs)==48 and sum(j['kind']=='benchmark'for j in jobs)==36,'actual frozen48-job plan required')
    return q,jobs

def source_bindings(plan,contract,q):
    bindings=plan['source_bindings'];local=GOLDEN_QUEUE;remote=Path(BASE)/'pencil_queue_v3_code';wanted={}
    names=set(q.PINS)|{'pencil_queue_v3.py','runtime_probe_v3.py','README.txt','protocol_spec.json','bridge/vldb_pencil_bridge_v2/SOURCE_MANIFEST.json'}
    bridge=read(local/'bridge/vldb_pencil_bridge_v2/SOURCE_MANIFEST.json')
    names.update('bridge/vldb_pencil_bridge_v2/'+item['path']for item in bridge['files'])
    for name in names:wanted[str(remote/name)]=identity(local/name)
    L=q.layout(Path(BASE));cpp=contract['cpp']
    wanted[str(L['cpp_module'])]={'bytes':8499,'sha256':cpp['module_sha256']}
    wanted[str(L['cpp_receipt'])]={'bytes':cpp['build_receipt_bytes'],'sha256':cpp['build_receipt_sha256']}
    lib=bindings.get(str(L['cpp_library']),{});require(type(lib.get('bytes'))is int and 0<lib['bytes']<=4<<20,'bounded actual Linux library binding required')
    wanted[str(L['cpp_library'])]={'bytes':lib['bytes'],'sha256':cpp['library_sha256']}
    require(bindings==wanted,'plan must bind exact frozen queue/bridge/upstream/C++/wheel sources')
    return wanted

def validate_plan(plan,contract,q,jobs):
    require(plan.get('recovery_admission_sha256')==contract['recovery_admission']['sha256'],'root-pinned recovery admission differs')
    require(plan.get('kind')=='pencil_queue_v3'and plan.get('jobs')==jobs,'saved argv/job protocol differs from actual frozen queue')
    require(plan.get('protocol_spec')==read(GOLDEN_QUEUE/'protocol_spec.json'),'frozen48-job protocol specification differs')
    require(plan.get('bridge_source_package_sha256')==q.SOURCE_PACKAGE_SHA and plan.get('external_sampled_memory_guard_gib')==180 and
        plan.get('per_job_timeout_seconds')==1800,'declared source/budget policy changed')
    require(0<plan['max_admission_hours']<=8 and plan['max_admission_hours']<plan['max_total_hours_including_wait']<=12,'finite original queue time budgets required')
    source_bindings(plan,contract,q)

def clean_exit(exit,gpu=None):
    require(type(exit.get('returncode'))is int and exit.get('returncode')==0 and exit.get('timed_out')is False and exit.get('owned_tree_quiescent')is True and
        exit.get('orphan_cleanup')is False and exit.get('guard_abort_reason')is None,'failed/guarded/orphan phase is not scientific timing evidence')
    timeout=exit.get('timeout_seconds');require(type(timeout)in(int,float)and math.isfinite(timeout)and 0<timeout<=1800,'finite numeric per-phase1800s bound required')
    if gpu is not None:require(exit.get('exactly_one_visible_gpu')is gpu,'declared GPU phase visibility differs')

def guards(path):
    require(path.is_file()and not path.is_symlink()and path.stat().st_size<=8<<20,'whole sampled guard log required')
    errors=0;rows=0;maximum=None
    for line in path.read_text().splitlines():
        row=json.loads(line);rows+=1
        if 'sampling_error'in row:errors+=1;require(errors<3,'three consecutive external sampling failures');continue
        errors=0;decision=row['decision'];require(decision.get('abort_reason')is None and decision.get('memory_budget_bytes')==180*2**30,'guard abort/budget contradiction')
        value=decision.get('own_pss_estimate_bytes')
        if value is not None:
            require(type(value)is int and 0<=value<=180*2**30,'valid bounded PSS estimate');maximum=value if maximum is None else max(maximum,value)
    require(rows>0,'at least one owned-tree guard observation required')
    return {'samples':rows,'maximum_recorded_own_pss_estimate_bytes':maximum,'scope':'sampled proportional tree footprint, not unique RSS sum or true peak'}

def environment(payload,contract,plan,q):
    before=read(payload/'base_environment_before.json');after=read(payload/'base_environment_after.json')
    isolated=read(payload/'isolated_environment.json');post=read(payload/'isolated_environment_after.json')
    require(before==after and isolated==post,'base or isolated environment changed across scientific protocol')
    require(before['prefix']=='/opt/python'and before['base_prefix']=='/usr'and isolated['prefix']==BASE+'/pencil_env_v3','declared nested base/V2 origins differ')
    require(before['critical_versions']==isolated['critical_versions']and before['critical_modules']==isolated['critical_modules']and
        before['torch_sparse_binaries']==isolated['torch_sparse_binaries']and before['saint_wrapper']==isolated['saint_wrapper'],'original numeric/native versions/origins replaced')
    require(set(before['critical_modules'])=={'torch','numpy','scipy','numba','torch_sparse','torch_geometric'}and
        set(before['critical_versions'])=={'torch','numpy','scipy','numba','torch-sparse','torch-geometric'}and
        set(isolated['extra_modules'])=={'transformers','wandb','rich','tqdm'},'complete original numerical and isolated extra source origins required')
    require(before['torch']=='2.5.0+cu121'and before['numpy']=='1.24.4'and before['pyg']=='2.7.0'and before['torch_sparse_version'].split('+')[0]=='0.6.18','declared native stack differs')
    native=contract['native'];require(before['saint_wrapper']['sha256']==native['saint_wrapper_sha256']and
        before['saint_wrapper']['git_blob']==native['saint_wrapper_git_blob']and
        before['torch_sparse_binaries']['_saint_cpu.so']['sha256']==native['saint_cpu_sha256'],'root-admitted saint source/binary changed')
    require(isolated['visible_gpu']['count']==1,'one actual visible GPU required')
    optional_vision_policy(isolated['runtime_compatibility'],isolated,contract)
    route=isolated['base_site_route'];saved=read(payload/'base_site_binding.json');content=route['content'];paths=content.splitlines()
    require(paths==list(dict.fromkeys(before['site_packages']))and route['base_paths']==paths and content=='\n'.join(paths)+'\n'and
        route['isolated_site_precedes_all_base_paths']is True,'plain before-probe route with isolated precedence required')
    require(all(PurePosixPath(p).is_absolute()and PurePosixPath(p).name in('site-packages','dist-packages')and
        ('\n'not in p and '\r'not in p and '\x00'not in p)and(p.startswith('/opt/python/')or p.startswith('/usr/'))for p in paths),
        'no executable/untrusted base route lines allowed')
    route_path=BASE+'/pencil_env_v3/lib/python3.10/site-packages/pencil_base_site_v3.pth'
    route_info={'bytes':len(content.encode()),'sha256':hashlib.sha256(content.encode()).hexdigest()}
    require(route['path']==saved['path']==route_path and route['sha256']==saved['sha256']==route_info['sha256']and
        saved['bytes']==route_info['bytes']and saved['base_paths']==paths,'recorded .pth bytes/hash/paths inconsistent')
    bind=read(payload/'environment_runtime_bindings.json');initial=plan['source_bindings'];admission=read(payload/'admission.json')
    recovery=recovery_admission(admission,contract,plan)
    for path,value in{**initial,**admission['bindings']}.items():require(bind.get(path)==value,'runtime source/admission binding differs')
    L=q.layout(Path(BASE));require(bind.get(str(L['wheel_archive']))=={k:contract['wheel'][k]for k in('bytes','sha256')},
        'actual admitted full wheel archive binding changed')
    policy=isolated['runtime_compatibility'];require(bind.get(policy['import_utils_origin'],{}).get('sha256')==contract['optional_vision_import_utils_sha256']and
        bind.get(BASE+'/pencil_queue_v3_code/bridge/vldb_pencil_bridge_v2/graph_only_transformers_v2.py',{}).get('sha256')==contract['driver_sources']['graph_only_transformers_v2.py'],
        'loaded pinned optional-vision policy/HF origin is not runtime-bound')
    require(bind.get(route_path)==route_info,'runtime .pth binding changed')
    cfg=bind.get(BASE+'/pencil_env_v3/pyvenv.cfg',{});require(type(cfg.get('bytes'))is int and 0<cfg['bytes']<=16384 and valid_sha(cfg.get('sha256')),'bound nested venv configuration required')
    for item in isolated['critical_modules'].values():require(bind.get(item['origin'],{}).get('sha256')==item['source_sha256'],'numeric origin hash binding differs')
    for name,version in(('transformers','4.46.2'),('wandb','0.18.7'),('rich','14.2.0'),('tqdm','4.67.1')):
        item=isolated['extra_modules'][name];require(item['version']==version and isolated['packages'][name]==version and
            PurePosixPath(item['origin']).is_relative_to(BASE+'/pencil_env_v3')and
            bind.get(item['origin'],{}).get('sha256')==item['sha256'],'loaded pinned isolated extra source/version differs')
    native_root=str(PurePosixPath(isolated['critical_modules']['torch_sparse']['origin']).parent)
    for name,value in isolated['torch_sparse_binaries'].items():require(bind.get(native_root+'/'+name)==value,'native binary runtime binding differs')
    require(bind.get(native_root+'/saint.py',{}).get('sha256')==native['saint_wrapper_sha256'],'native wrapper runtime binding differs')
    extraction=read(payload/'wheel_extraction.json');require(extraction['files']==35 and extraction['manifest_sha256']==contract['wheel']['manifest_sha256']and
        0<extraction['uncompressed_bytes']<=64<<20,'exact matched offline wheel extraction required')
    verification=read(payload/'environment_verification.json');require(verification=={'status':'pass','base_unchanged':True,'numerical_and_native_origins_exact':True,'isolated_site_precedes_base_fallback':True},'full environment verification receipt required')
    for phase in ENV_PHASES:
        clean_exit(read(payload/(phase+'.exit.json')),phase.startswith('isolated_environment'));guards(payload/(phase+'.guard.jsonl'))
    return isolated,{'status':'pass','base_and_isolated_pre_post_exact':True,'plain_pth_and_loaded_numeric_extra_native_origins_bound':True,
        'root_native_saint_binary_exact':True,'offline_wheels':35,'visible_gpu':isolated['visible_gpu'],
        'root_pinned_recovery_admission':recovery,'explicit_optional_vision_policy':policy,
        'scope':'Exported hashes/origin receipts checked; remote environment files are not reopened locally.'}

def recovery_admission(actual,contract,plan):
    pin=contract['recovery_admission'];expected=pin['content'];path=Path(pin['local_path'])
    require(identity(path)=={k:pin[k]for k in('bytes','sha256')}and read(path)==expected,'independent root recovery admission anchor changed')
    require(expected['kind']=='pencil_v3_recovery_admission'and expected['base']==BASE and
        expected['old_environment']['pod_absent']is True and expected['old_environment']['statefulset_replicas']==0 and
        expected['old_environment']['root_confirmed']is True and expected['old_environment']['old_controller_final_cleanup']=='UNKNOWN',
        'root attestation must retain deleted old pod and unknown old outer-controller cleanup')
    require(actual['admitted']is True and actual['kind']=='root_pinned_recovery_admission'and
        actual['receipt_sha256']==plan['recovery_admission_sha256']==pin['sha256']and actual['replacement_hostname']==expected['replacement_hostname']and
        actual['prior_archives']==expected['prior_archives']and actual['bindings']=={BASE+'/RECOVERY_ADMISSION.json':{k:pin[k]for k in('bytes','sha256')}},
        'actual new-pod admission differs from root-pinned recovery archive/host/failure evidence')
    required={'pipeline':True,'native':True,'pencil_failed_v2':False};prior=actual['prior_archives']
    require(len(prior)==3 and{v['role']for v in prior}==set(required)and all(v['scientific_success']is required[v['role']]and
        v['all_internal_hashes_verified']is True and valid_sha(v['sha256'])for v in prior),'prior archive outcomes were relabelled')
    require(next(v for v in prior if v['role']=='pencil_failed_v2')['completed_jobs']==0,'old failed PENCIL must remain0 completed jobs')
    return {'status':'pass','root_receipt_sha256':pin['sha256'],'replacement_hostname':actual['replacement_hostname'],
        'original_outer_controller_cleanup':'UNKNOWN; no synthetic legacy COMPLETE/PID evidence','prior_failed_v2_preserved':True}

def optional_vision_policy(value,isolated,contract):
    origin=str(PurePosixPath(isolated['extra_modules']['transformers']['origin']).parent/'utils/import_utils.py')
    expected={'version':'graph_only_transformers_optional_vision_v2','transformers_version':'4.46.2','import_utils_origin':origin,
        'import_utils_sha256':contract['optional_vision_import_utils_sha256'],'policy_source_sha256':contract['driver_sources']['graph_only_transformers_v2.py'],
        'installed_torchvision_version':isolated['packages'].get('torchvision'),'torchvision_available_before':isolated['packages'].get('torchvision')is not None,
        'torchvision_available_for_graph_worker':False,'torchvision_imported':False,'upstream_source_modified':False,'base_package_modified':False,
        'fake_operator_registered':False,'scope':'Only unused optional vision discovery in pinned Transformers; graph sampler, sparse operator, BERT parameters/forward, precision and input tensors unchanged.'}
    require(value==expected,'explicit pinned graph-only optional vision policy/source/origin/version differs')

def native_environment(raw,isolated,n,m):
    setup=raw['native_setup'];env=raw['audit_environment']
    require(raw['runtime_compatibility']==env['runtime_compatibility']==isolated['runtime_compatibility'],
        'actual job optional-vision policy differs from isolated probe/semantic environment')
    expected={k:v for k,v in setup.items()if k not in('native_setup_seconds','native_setup_memory')}
    require(env['native']==expected and env['torch_threads']==1 and type(env['torch_interop_threads'])is int and env['torch_interop_threads']>0 and
        env['thread_environment']=={k:'1'for k in THREADS},'stable native/software/thread provenance differs')
    require(env['python']==isolated['python']and env['platform']==isolated['platform'],'driver environment host/python differs')
    require(setup['torch']==isolated['torch']and setup['torch_sparse']==isolated['torch_sparse_version']and setup['pyg']==isolated['pyg']and
        setup['wrapper_sha256']==isolated['saint_wrapper']['sha256']and setup['wrapper_git_blob']==isolated['saint_wrapper']['git_blob'],'driver native versions/wrapper differ')
    require({v['name']:v['sha256']for v in setup['extension_binaries']}=={k:v['sha256']for k,v in isolated['torch_sparse_binaries'].items()},'driver native binary wheel differs')
    require(all(setup['software_versions'][k]==isolated['packages'][k]for k in('numpy','numba','scipy','transformers','wandb')),'driver software versions differ')
    require(setup['int64_csr_bytes']==8*(n+1+m)and setup['public_coo_row_bytes']==setup['stored_value_bytes']==8*m and
        setup['symmetric_full_CSR_verified']is True,'full native CSR/COO/value representations/setup scope differs')
    admission=raw['native_admission'];require(admission.get('status')=='pass'and admission.get('actual_native_cases')==7 and
        admission.get('before_full_graph_allocation')is True,'actual native7-case admission before full graph allocation required')

def input_binding(raw,trace,contract,fixture):
    require(raw['source_files']==contract['driver_sources'],'frozen bridge driver/core/config source changed')
    upstream=raw['upstream'];pin=contract['upstream']
    require(upstream['commit']==pin['commit']and upstream['manifest_sha256']==pin['manifest_sha256']and
        upstream['all_source_files_verified']==53,'actual pinned upstream53-file proof required')
    require(raw['input_binding']==trace['manifest']['metadata']['binding'],'trace graph/features/cohort binding changed')
    if fixture:return
    expected=contract['input_contract'];binding=raw['input_binding']
    require(binding['n']==expected['n']and binding['m']==expected['m']and binding['input_contract_sha256']==contract['input_contract_sha256'],
        'full strict-clean citation2 contract changed')
    for name,value in expected['payloads'].items():
        actual=binding['payloads'][name];require(actual['sha256']==value['sha256']and('bytes'not in value or actual['bytes']==value['bytes']),'actual frozen graph/features/validation payload changed')
    meta=raw['input_metadata'];proof=meta['payload_binding_verification'];require(proof['status']=='pass'and proof['quality_plan_checked']is True and
        proof['payloads']==binding['payloads'],'actual original quality-plan payload proof required')
    strict=meta['strict_manifest']['content'];strict=strict.get('dataset_provenance',{}).get('content',strict)
    require(strict.get('variant')=='strict_clean'and strict.get('quality_dataset')is True,'strict-clean quality graph variant required')
    files={v['path']:v for v in strict['files']}
    for name in('graph_starts','graph_columns','features'):
        v=expected['payloads'][name];require(files[v['path']]['sha256']==v['sha256']and files[v['path']]['bytes']==v['bytes'],'strict manifest payload differs')
    for split in('valid','test'):
        heldout=strict['clean_heldout_audits']['train_csr'][split];require(heldout['forward']==heldout['reverse']==0,'bidirectional heldout-clean proof required')
    require(meta['positive_file']['sha256']==expected['payloads']['valid_positive']['sha256']and meta['negative_file']['sha256']==expected['payloads']['valid_negative']['sha256'],
        'frozen512/1000 validation source rows changed')

def model_contract(model,fixture=False):
    require(model['checkpoint']is None and model['initialization_seed']==42 and model['tiny_fixture_model']is fixture and
        model['features']==128 and model['feature_fusion']=='early'and model['encoding']=='adjacency_row','frozen random PENCIL model configuration required')
    settings=model['settings'];wanted={'hidden_size':32 if fixture else 512,'num_hidden_layers':1 if fixture else 3,
        'num_attention_heads':4 if fixture else 8,'intermediate_size':64 if fixture else 2048,'max_position_embeddings':2048,'from_pretrained':False}
    require(all(settings[k]==v for k,v in wanted.items()),'native model architecture changed')

def logits_fp(fp,count):
    require(fp.get('kind')=='tensor'and fp.get('dtype')=='torch.float32'and fp.get('shape')==[count]and fp.get('stride')==[1]and valid_sha(fp.get('sha256')),
        'ordered complete float32-export model logit fingerprint required')

def semantic_receipt(raw,job,trace,arms):
    result=raw['result'];fixture=job['name']=='tiny_audit';q=7 if fixture else 1024;batch=job.get('batch',128)
    require(result['model_forward_checked']is True and result['max_logit_abs_tolerance']==0,'mandatory zero-tolerance model-forward audit required')
    model_contract(result['model'],fixture)
    gates=result['semantic_gates'];require(len(gates)==16 and{(v['arm'],v['training_wrapper'])for v in gates}=={(a,t)for a in arms for t in(False,True)},
        'all8 arms in both unchanged native wrapper modes required')
    for gate in gates:
        require(gate['queries']==q and all(gate.get(k)is True for k in('raw_csr_slots_order_exact','postmask_data_exact','wrapped_and_collated_inputs_exact','native_postraw_cpu_rng_restored_before_candidate_wrapper')),
            'full ordered edges/slots/values/Data/target-mask/wrapper equivalence gate failed')
        require(gate['reference']=='actual online native public sampler and wrapper','actual online public native reference required')
    checks=result['batch_checks'];expected=[]
    for training in(False,True):
        for start in range(0,q,batch):
            count=min(batch,q-start);expected.append((training,start+count-1,count))
    require(len(checks)==len(expected),'all full-cohort batch shapes/wrapper modes must be audited')
    for check,(training,through,count)in zip(checks,expected):
        require(check['training_wrapper']is training and check['through_query_index']==through and check['batch_size']==count and check['all_input_tensors_exact']is True,
            'audit omitted/reordered model-input batch or changed shape')
        require(set(check['logits'])==set(arms),'all8 native/candidate logit fingerprints required')
        reference=check['logits']['torch_sparse_public']['hash'];logits_fp(reference,count)
        for value in check['logits'].values():
            require(value['max_abs']==0 and value['bitwise_equal']is True and value['hash']==reference,'ordered model logits differ within audit batch')
    require(raw['trace_manifest_sha256']==sha(Path(job['_trace_path'])/'manifest.json'),'semantic receipt belongs to another frozen trace')
    return {'queries':q,'batch_checks':len(checks),'arms':len(arms),'training_wrapper_audited':True,'ordered_raw_slots_values_and_mask_receipts':True}

def expected_configuration(job,arms):
    """Independent CLI/default decoder, not the driver's parser or validator."""
    bridge=BASE+'/pencil_queue_v3_code/bridge/vldb_pencil_bridge_v2'
    values={'command':job['kind'],'execute':False,'fixture':False,'graph_path':None,'features':None,
        'valid_positive':None,'valid_negative':None,'strict_manifest':None,'quality_plan':None,
        'input_contract':bridge+'/input_contract.json','trace_dir':str(PurePosixPath(bridge).parent/'pencil_bridge_trace'),
        'output':None,'upstream':bridge+'/upstream_pencil','upstream_manifest':bridge+'/upstream_manifest.json',
        'bert_config':bridge+'/bert_config','arms':','.join(arms[:6]),'arm':'torch_sparse_public',
        'cpp_module':None,'cpp_library':None,'cpp_build_receipt':None,'max_assoc_gib':1.0,'max_memory_gib':32.0,
        'max_public_coo_gib':1.0,'max_trace_mib':512.0,'max_query_neighbors':20000000,'eval_seed':20261126,
        'batch_size':128,'num_workers':0,'training_wrapper':False,'forward':False,'device':'cpu','precision':'fp32',
        'model_seed':42,'checkpoint':None,'tiny_model':False,'max_logit_abs':0.0,'audit_receipt':None,
        'pipeline_mode':'replay','warmup':1,'repeats':3,'file_endian':'little'}
    require(job['argv'][3]==job['kind'],'actual saved CLI command differs from job kind')
    tail=iter(job['argv'][4:]);seen=set()
    for token in tail:
        require(token.startswith('--'),'unexpected positional CLI value')
        key=token[2:].replace('-','_');require(key in values and key not in seen,'unknown/duplicate CLI option');seen.add(key)
        default=values[key]
        if type(default)is bool:values[key]=True
        else:
            try:value=next(tail)
            except StopIteration:raise InvalidEvidence('missing CLI option value')
            values[key]=int(value)if type(default)is int else float(value)if type(default)is float else value
    require(values['output']==job['output'],'CLI output differs from predeclared output')
    return values

def configuration(raw,job,contract,arms):
    expected=expected_configuration(job,arms)
    require(raw.get('configuration')==expected and raw.get('command')==job['kind']and raw.get('schema_version')==1,
        'actual driver configuration differs from full frozen argv/defaults')
    require(raw.get('quality_claim')=='none','driver must not imply a quality comparison')
    require(raw.get('source_files')==contract['driver_sources'],'driver source fingerprints changed')
    cpp=contract['cpp'];wanted={k:cpp[k]for k in('module_sha256','source_sha256','library_sha256','build_receipt_sha256')}
    needs_cpp=job['kind']=='audit'or(job['kind']=='benchmark'and job['arm'].startswith('cpp_'))
    require(('cpp_artifacts'in raw)is needs_cpp,'C++ artifact evidence presence differs from actually requested route')
    if needs_cpp:require(raw['cpp_artifacts']==wanted,'actual C++ source/build/library artifacts changed')
    if job['kind']=='preflight':require(raw.get('status')=='preflight'and raw.get('preflight_passed')is True,'real preflight did not pass')
    else:require(raw.get('status')=='complete','driver scientific phase is incomplete')
    return expected

def finite_number(value,positive=False):
    return type(value)in(int,float)and math.isfinite(value)and(value>0 if positive else value>=0)
def close_number(a,b):return finite_number(a)and finite_number(b)and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15)
def stage_summary(summary,count):
    keys={'n','mean_s','p50_s','p95_s','min_s','max_s'}
    require(set(summary)==keys and summary['n']==count and all(finite_number(summary[k])for k in keys-{'n'}),'complete finite stage summary in seconds required')
    require(summary['min_s']<=summary['p50_s']<=summary['p95_s']<=summary['max_s']and
        (summary['min_s']<=summary['mean_s']<=summary['max_s']or close_number(summary['mean_s'],summary['min_s'])or
            close_number(summary['mean_s'],summary['max_s'])),'stage quantile/mean order invalid')
def exact_summary(values):
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'mean_s':float(a.mean()),'p50_s':float(np.percentile(a,50)),
        'p95_s':float(np.percentile(a,95)),'min_s':float(a.min()),'max_s':float(a.max())}
def cpp_initializations(rows,job,contract,worker_pids):
    is_cpp=job['arm'].startswith('cpp_');require(bool(rows)is is_cpp,'C++ per-process initialization evidence presence differs')
    if not is_cpp:return
    require(len(rows)==1 and len(rows[0])==1,'one actual process-private C++ initialization required per loader')
    value=rows[0][0];mode='reference_full_clear'if job['arm']=='cpp_full_clear'else'persistent_touched_reset';meta=value['metadata']
    require(type(value['pid'])is int and value['pid']in worker_pids and value['mode']==mode and value['assoc_budget_bytes']==2**30,
        'actual C++ state owner/mode/budget differs')
    n=contract['input_contract']['n'];m=contract['input_contract']['m']
    require(meta['owner_pid']==value['pid']and meta['mode']==mode and meta['nodes']==n and meta['stored_entries']==m and
        meta['version']=='persistent_assoc_touched_reset_v1'and meta['assoc_bytes_during_each_call']==8*n and
        meta['persistent_assoc_bytes_per_worker']==(8*n if mode=='persistent_touched_reset'else 0)and
        meta['library_sha256']==contract['cpp']['library_sha256']and finite_number(meta['initialization_seconds']),
        'C++ map cost/source/owner initialization metadata inconsistent')
    require(meta['initialization_policy']==('once per worker'if mode=='persistent_touched_reset'else'full-N allocation/init inside every native call'),
        'C++ initialization policy changed')

def benchmark_receipt(raw,job,payload,contract,audits):
    result=raw['result'];name=f"audit_bf16_b{job['batch']}_w{job['workers']}";audit=audits[name]
    require(result['audit_receipt']=={'path':str(PurePosixPath(job['output']).parent/(name+'.json')),'sha256':sha(payload/(name+'.json'))},
        'benchmark does not bind exact completed matching forward audit')
    require(raw['audit_environment']==audit['audit_environment']and raw['input_binding']==audit['input_binding']and
        raw['trace_manifest_sha256']==audit['trace_manifest_sha256']and result['model']==audit['result']['model'],
        'benchmark changed environment/input/trace/model since matching semantic audit')
    require(result['arm']==job['arm']and result['pipeline_mode']==job['mode']and result['batch_size']==job['batch']and
        result['num_workers']==job['workers']and result['prefetch_factor']==(1 if job['workers']else None),'actual loader route/shape/workers changed')
    model_contract(result['model']);sweeps=result['sweeps'];require(len(sweeps)==4,'one full warmup and3 measured sweeps required')
    stages={'sampler_seconds','induction_seconds','raw_data_seconds','wrapper_seconds','S_verification_seconds'}
    for repetition,row in enumerate(sweeps):
        require(row['repetition']==repetition and row['warmup']is(repetition==0)and row['queries']==1024 and
            finite_number(row['elapsed_seconds'],True),'all4 ordered full1024-query sweeps required')
        require(close_number(row['queries_per_second'],1024/row['elapsed_seconds']),'reported throughput does not equal actual queries/wall time')
        require(set(row['stages'])==stages,'all nested stage summaries required')
        for value in row['stages'].values():stage_summary(value,1024)
        require(finite_number(row['collator_seconds_total'])and finite_number(row['h2d_forward_score_copy_seconds_total']),
            'finite nonnegative collator/forward stage totals required')
        if job['mode']=='replay':require(row['stages']['sampler_seconds']['max_s']==0,'REPLAY must explicitly bypass sampler')
        logits_fp(row['scores'],1024)
        workers=row['worker_memory'];require(len(workers)==1 and len({v['pid']for v in workers})==1 and
            all(type(v['pid'])is int and v['pid']>0 and type(v['maxrss_bytes'])is int and v['maxrss_bytes']>=0 and
                type(v['max_pss_bytes'])is int and v['max_pss_bytes']>=0 for v in workers),'one actual dataset execution process memory receipt required')
        cpp_initializations(row['cpp_initializations'],job,contract,{v['pid']for v in workers})
        gpu=row['gpu'];require(set(gpu)=={'max_allocated_bytes','max_reserved_bytes'}and all(type(v)is int and v>=0 for v in gpu.values())and
            gpu['max_allocated_bytes']<=gpu['max_reserved_bytes'],'actual CUDA cumulative high-water memory bytes required')
    expected=exact_summary([r['elapsed_seconds']for r in sweeps[1:]])
    require(set(result['elapsed_summary'])==set(expected)and all(result['elapsed_summary'][k]==v if k=='n'else close_number(result['elapsed_summary'][k],v)for k,v in expected.items()),
        'measured elapsed summary differs from three actual complete sweeps')
    median=expected['p50_s']
    return {'job':job['name'],'mode':job['mode'],'batch':job['batch'],'workers':job['workers'],'arm':job['arm'],
        'measured_seconds':[r['elapsed_seconds']for r in sweeps[1:]],'median_seconds':median,'queries_per_second_from_median_seconds':1024/median,
        'median_of_3_sweep_induction_p50_seconds':statistics.median(r['stages']['induction_seconds']['p50_s']for r in sweeps[1:]),
        'measured_per_query_stage_summary_receipts':[r['stages']for r in sweeps[1:]],
        'median_collator_total_seconds':statistics.median(r['collator_seconds_total']for r in sweeps[1:]),
        'median_h2d_forward_score_copy_total_seconds':statistics.median(r['h2d_forward_score_copy_seconds_total']for r in sweeps[1:]),
        'gpu_cumulative_high_water_bytes_including_warmup':[r['gpu']for r in sweeps],
        'cpp_process_initialization_receipts':sweeps[0]['cpp_initializations'],
        'score_fingerprint':sweeps[0]['scores'],'all_sweep_score_fingerprints':[r['scores']for r in sweeps],
        'model':result['model']}

def cross_scores(rows,arms,replay_arms):
    grouped={};expected={('online',b,w):set(arms)for b in(128,512)for w in(0,1)};expected[('replay',128,0)]=set(replay_arms)
    for row in rows:
        key=(row['mode'],row['batch'],row['workers']);bucket=grouped.setdefault(key,{})
        require(row['arm']not in bucket,'duplicate benchmark arm');bucket[row['arm']]=row
    require(set(grouped)==set(expected),'all5 predeclared separate shape/worker/mode groups required')
    report=[]
    for key,bucket in grouped.items():
        require(set(bucket)==expected[key],'missing native/candidate arm in full measured group')
        ref=bucket['torch_sparse_public'];fp=ref['score_fingerprint'];logits_fp(fp,1024)
        require(all(row['model']==ref['model']and len(row['all_sweep_score_fingerprints'])==4 and
            all(value==fp for value in row['all_sweep_score_fingerprints'])for row in bucket.values()),
            'ordered full1024 scores differ across measured arms or warmup/repeats within matching group')
        report.append({'mode':key[0],'precision':'bf16','batch':key[1],'workers':key[2],'arms':list(bucket),
            'all_4_sweeps_all_arms_scores_exact':True,'score_fingerprint':fp})
    ratios=[]
    for key,bucket in grouped.items():
        for row in bucket.values():
            row['ratios_of_median_wall_time']={name:bucket[name]['median_seconds']/row['median_seconds']for name in
                ('torch_sparse_public','torch_sparse_direct','cpp_touched_reset')if name in bucket}
    online=grouped[('online',128,0)];replay=grouped[('replay',128,0)]
    for arm,row in replay.items():
        same=row['score_fingerprint']==online[arm]['score_fingerprint']and row['model']==online[arm]['model']
        ratios.append({'arm':arm,'online_and_replay_score_fingerprints_exact':same,
            'online_over_replay_median_wall_time':online[arm]['median_seconds']/row['median_seconds']if same else None})
    return report,ratios

def inventory(payload,jobs,science,contract,arms):
    path=payload/'completed_jobs.json';completed=read(path)if path.exists()else{'jobs':[],'count':0}
    count=completed['count'];require(type(count)is int and 0<=count<=len(jobs)and count==len(completed['jobs'])==science['completed_jobs'],
        'completed-job inventory disagrees with scientific terminal')
    reports=[];raws={}
    for index,job in enumerate(jobs):
        rawpath=payload/(job['name']+'.json');exitpath=payload/(job['name']+'.exit.json')
        raw=read(rawpath)if rawpath.exists()else None;ex=read(exitpath)if exitpath.exists()else None
        if index<count:
            saved=completed['jobs'][index]
            require(saved['name']==job['name']and saved['kind']==job['kind']and raw is not None and ex is not None and
                saved['output']==identity(rawpath)and saved['exit']==ex,'completed inventory is not exact executed-job prefix with whole output/exit hashes')
            clean_exit(ex,job['gpu']);guard=guards(payload/(job['name']+'.guard.jsonl'))
            log=payload/(job['name']+'.log');require(log.is_file()and not log.is_symlink()and log.stat().st_size<=4<<20,'whole bounded executed-job log required')
            configuration(raw,job,contract,arms);raws[job['name']]=raw
            reports.append({'job':job['name'],'kind':job['kind'],'status':'completed_admitted','guard':guard,'output':identity(rawpath)})
        else:
            require(index==count or(raw is None and ex is None),'job evidence exists after queue first uncompleted phase')
            status='not_started'
            if ex is not None or raw is not None:
                status='guarded_or_failed'if raw is None or raw.get('status')in('failed','guarded_skip')else'executed_but_not_admitted'
            reports.append({'job':job['name'],'kind':job['kind'],'status':status,'driver_status':raw.get('status')if raw else None,
                'exit':ex,'driver_error':raw.get('error')if raw else None})
    if science['success']:require(count==48,'scientific success requires all48 exact predeclared completed jobs')
    return reports,raws

def capture_receipt(raw,trace,path,queries):
    result=raw['result'];require(result['queries']==queries and result['trace_manifest_sha256']==sha(path/'manifest.json')and
        result['trace_files']==trace['manifest']['files']and result['trace']==raw['configuration']['trace_dir'],
        'capture output does not bind complete exact frozen trace')
    meta=trace['manifest']['metadata'];require(meta['upstream']==raw['upstream']and meta['native']==raw['native_setup']and
        meta['cohort']==raw['input_metadata'],'capture trace source/native/cohort receipts differ from actual capture')
    for key in('native_sampler_summary','native_induction_summary'):stage_summary(result[key],queries)
    for summary_key,record_key in(('native_sampler_summary','native_sampler_seconds'),('native_induction_summary','native_public_induction_seconds')):
        expected=exact_summary([v[record_key]for v in trace['records']]);require(all(result[summary_key][k]==v if k=='n'else close_number(result[summary_key][k],v)for k,v in expected.items()),
            'capture summary differs from every actual trace-record time')

def scientific_audit(context,contract,q,jobs,raws):
    payload=context['payload'];isolated,env_report=environment(payload,contract,context['plan'],q)
    tiny_path=payload/'tiny_trace';real_path=payload/'real_trace'
    tiny=load_trace(tiny_path,7,True);real=load_trace(real_path,1024,False)
    traces={True:(tiny,tiny_path),False:(real,real_path)};semantic=[];audits={};bench=[];real_environment=None
    for planned in jobs:
        job=dict(planned);raw=raws[job['name']];fixture=job['name'].startswith('tiny');trace,trace_path=traces[fixture]
        input_binding(raw,trace,contract,fixture)
        if job['kind']=='preflight':continue
        native_environment(raw,isolated,9 if fixture else contract['input_contract']['n'],12 if fixture else contract['input_contract']['m'])
        if not fixture:
            if real_environment is None:real_environment=raw['audit_environment']
            require(raw['audit_environment']==real_environment,'stable actual native/software/thread origins differ across real protocol jobs')
        if job['kind']=='capture':capture_receipt(raw,trace,trace_path,7 if fixture else 1024)
        elif job['kind']=='audit':
            job['_trace_path']=trace_path;semantic.append({'job':job['name'],**semantic_receipt(raw,job,trace,q.ARMS)})
            require(raw['result']['model']['bert_config_sha256']==contract['driver_sources']['bert_config/config.json'],'model consumed another offline BERT config')
            audits[job['name']]=raw
        elif job['kind']=='benchmark':bench.append(benchmark_receipt(raw,job,payload,contract,audits))
    groups,crossmode=cross_scores(bench,q.ARMS,q.REPLAY_ARMS)
    require(len(bench)==36 and len(semantic)==9,'complete36 throughput jobs and9 tiny/real semantic audits required')
    for row in bench:row.pop('all_sweep_score_fingerprints');row.pop('model')
    return {'status':'pass','environment':env_report,'traces':{'tiny':tiny['report'],'real':real['report']},'semantic_audits':semantic,
        'measured_score_groups':groups,'primary_online_bf16_rows':[v for v in bench if v['mode']=='online'],
        'supplemental_replay_bf16_rows':[v for v in bench if v['mode']=='replay'],'supplemental_online_replay_diagnostics':crossmode,
        'limitations':[
            'Random frozen model, not a trained-checkpoint quality, MRR, ranking, SOTA, training or HPO comparison.',
            'Full native CSR/COO/value representations remain resident in every arm; no PENCIL memory-saving result.',
            'Audit runs sequentially; declared worker setting binds subsequent throughput admission.',
            'Raw/induction stage times are nested: raw includes sampler, induction, transpose, target masking and features; do not add them.',
            'Induction includes physical operator plus SparseTensor/value packing; it is not an isolated kernel-only timing.',
            'Wall throughput includes memory/smaps instrumentation, S verification, IPC, H2D, forward and score copy; worker overlap prevents adding CPU stage sums to wall.',
            'One excluded full warmup and three complete measured sweeps; ratios use medians of full-sweep times, with no per-query paired speedup or confidence-interval claim.',
            'GPU maximum is cumulative since reset before warmup, not an individual measured-sweep peak.',
            'PSS guard is sampled and cannot establish a true peak or global memory-saving claim.',
            'Real graph edge/value arrays were not exported. Local audit verifies source/bindings and full native equivalence receipts, not a new full-graph oracle execution.',
            'Only within-group hardware/model/precision/batch/workers/mode comparisons are eligible; historical different-host timings are not speed ratios.'
        ]}

def build_report(context,contract):
    q,jobs=frozen_protocol(contract);validate_plan(context['plan'],contract,q,jobs)
    rows,raws=inventory(context['payload'],jobs,context['science'],contract,q.ARMS)
    success=context['science']['success']and context['done']['success']
    report={'schema_version':1,'kind':'independent_local_pencil_v3_audit','input_export':str(context['folder']),
        'status':'scientific_checks_passed'if success else'persisted_scientific_failure_no_performance_analysis',
        'transport_integrity':context['integrity'],'controller_success':context['done']['success'],'scientific_success':context['science']['success'],
        'scientific_failure':context['science'].get('failure'),'completed_jobs':len(raws),'planned_jobs':48,'inventory':rows,
        'independent_scope':'NumPy/stdlib local verifier; frozen queue imported only to derive actual declared jobs. No remote operations, Torch, native replay, training or source mutations.'}
    report['scientific_checks']=scientific_audit(context,contract,q,jobs,raws)if success else{
        'status':'withheld','reason':'Transport integrity is distinct from scientific success. Failed/guarded/not-started phases remain explicit; no throughput headline or speedup analysis.'}
    return report

def text_report(report):
    lines=['Independent local PENCIL V3 audit',f"Status: {report['status']}",
        f"Persisted transport integrity: {report['transport_integrity']['status']}",
        f"Controller success: {report['controller_success']}; scientific success: {report['scientific_success']}",
        f"Admitted completed jobs: {report['completed_jobs']}/{report['planned_jobs']}"]
    checks=report['scientific_checks']
    if checks['status']!='pass':lines.extend([checks['reason'],f"Failure: {report['scientific_failure']}"])
    else:
        lines.extend(['','Primary ONLINE BF16: instrumented full-pipeline throughput; 1024 queries; median of3 measured full sweeps.',
            'batch workers arm median_seconds queries/second speed_ratio_vs_public'])
        for row in checks['primary_online_bf16_rows']:
            lines.append(f"{row['batch']} {row['workers']} {row['arm']} {row['median_seconds']:.9g} {row['queries_per_second_from_median_seconds']:.9g} {row['ratios_of_median_wall_time']['torch_sparse_public']:.6g}")
        lines.extend(['','Supplemental REPLAY BF16 (sampler bypassed):'])
        for row in checks['supplemental_replay_bf16_rows']:lines.append(f"{row['arm']} {row['median_seconds']:.9g}s {row['queries_per_second_from_median_seconds']:.9g} queries/s")
        lines.extend(['','Interpretation limits:',*['- '+value for value in checks['limitations']]])
    lines.extend(['','Job inventory:',*['- '+v['job']+': '+v['status']for v in report['inventory']]])
    return '\n'.join(lines)+'\n'

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',type=Path,default=DEFAULT_EXPORT)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--root-launch',type=Path);a=p.parse_args(argv)
    if not(a.export/'export.json').exists():
        print(json.dumps({'status':'waiting_for_verified_export','export':str(a.export),'scientific_analysis_performed':False,'output_created':False}));return 0
    export=a.export.resolve();output=a.output.resolve();protected=[export,GOLDEN_QUEUE.resolve(),HERE.resolve()]
    require(not output.exists()and all(not output.is_relative_to(v)and not v.is_relative_to(output)for v in protected),
        'fresh independent output outside export/frozen/auditor source trees required')
    contract=read(HERE/'FROZEN_CONTRACT.json');context=verify_export(export,contract['queue_source_sha256'],a.root_launch)
    report=build_report(context,contract)
    output.mkdir(parents=True);(output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'summary.txt').write_text(text_report(report))
    print(json.dumps({'status':report['status'],'output':str(output),'completed_jobs':report['completed_jobs'],'planned_jobs':48}));return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except(InvalidEvidence,KeyError,IndexError,TypeError,ValueError,OSError,EOFError,OverflowError)as e:
        print(json.dumps({'status':'invalid_evidence','error_type':type(e).__name__,'message':str(e),'performance_analysis_eligible':False}),file=sys.stderr);raise SystemExit(2)
