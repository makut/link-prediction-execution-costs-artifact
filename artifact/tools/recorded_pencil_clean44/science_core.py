"""Derived numerical verifier. Original V3 files remain unchanged.
Only explicit new clean-source/path/query-memory boundary edits; no producer validators.
"""
from pathlib import Path,PurePosixPath
import json,math,statistics
import numpy as np
from archive_contract import BASE,REMOTE_BRIDGE,InvalidEvidence,require,read,identity,sha,valid_sha,canonical
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
MEMORY_POLICY={"version":"explicit_query_memory_instrumentation_v1","query_memory":"disabled",
    "per_query_memory_samples_enabled":False,
    "parent_memory_observations":"before and after each sweep, outside the sweep wall timer; not a true peak",
    "worker_memory_scope":"unobserved in driver; null fields, external owned-tree guard remains required",
    "external_guard_requirement":"unchanged owned-tree guard algorithm with explicit coordinator budget below the pod limit; separate coordinator receipts establish enforcement",
    "memory_saving_claim":False}
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
        errors=0;decision=row['decision'];require(decision.get('abort_reason')is None and decision.get('memory_budget_bytes')==96*2**30,'guard abort/budget contradiction')
        value=decision.get('own_pss_estimate_bytes')
        if value is not None:
            require(type(value)is int and 0<=value<=96*2**30,'valid bounded PSS estimate');maximum=value if maximum is None else max(maximum,value)
    require(rows>0,'at least one owned-tree guard observation required')
    return {'samples':rows,'maximum_recorded_own_pss_estimate_bytes':maximum,'scope':'sampled proportional tree footprint, not unique RSS sum or true peak'}

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
    meta=raw['input_metadata'];proof=meta['payload_binding_verification'];require(proof['status']=='pass'and proof['quality_plan_checked']is False and
        proof['payloads']==binding['payloads'],'exact payload proof without unavailable private quality plan required')
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
    bridge=REMOTE_BRIDGE
    values={'command':job['kind'],'execute':False,'fixture':False,'graph_path':None,'features':None,
        'valid_positive':None,'valid_negative':None,'strict_manifest':None,'quality_plan':None,
        'input_contract':bridge+'/input_contract.json','trace_dir':str(PurePosixPath(bridge).parent/'pencil_bridge_trace'),
        'output':None,'upstream':bridge+'/upstream_pencil','upstream_manifest':bridge+'/upstream_manifest.json',
        'bert_config':bridge+'/bert_config','arms':','.join(arms[:6]),'arm':'torch_sparse_public',
        'cpp_module':None,'cpp_library':None,'cpp_build_receipt':None,'max_assoc_gib':1.0,'max_memory_gib':32.0,
        'max_public_coo_gib':1.0,'max_trace_mib':512.0,'max_query_neighbors':20000000,'eval_seed':20261126,
        'batch_size':128,'num_workers':0,'training_wrapper':False,'forward':False,'device':'cpu','precision':'fp32',
        'model_seed':42,'checkpoint':None,'tiny_model':False,'max_logit_abs':0.0,'audit_receipt':None,
        'pipeline_mode':'replay','query_memory':'inline_smaps','warmup':1,'repeats':3,'file_endian':'little'}
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
        require(type(row['repetition'])is int and row['repetition']==repetition and row['warmup']is(repetition==0)and type(row['queries'])is int and row['queries']==1024 and
            finite_number(row['elapsed_seconds'],True),'all4 ordered full1024-query sweeps required')
        require(close_number(row['queries_per_second'],1024/row['elapsed_seconds']),'reported throughput does not equal actual queries/wall time')
        require(set(row['stages'])==stages,'all nested stage summaries required')
        for value in row['stages'].values():stage_summary(value,1024)
        require(finite_number(row['collator_seconds_total'])and finite_number(row['h2d_forward_score_copy_seconds_total']),
            'finite nonnegative collator/forward stage totals required')
        logits_fp(row['scores'],1024)
        require(canonical(result['memory_measurement_policy'])==canonical(MEMORY_POLICY) and canonical(row['memory_measurement_policy'])==canonical(MEMORY_POLICY),
            'explicit disabled query-memory policy required at result and sweep')
        workers=row['worker_memory'];require(len(workers)==1 and len({v['pid']for v in workers})==1 and
            all(type(v['pid'])is int and v['pid']>0 and v['maxrss_bytes']is None and v['max_pss_bytes']is None and
                v['memory_measurement']=='disabled_unobserved; not zero or a worker peak' for v in workers),
            'one actual process identity with null unobserved worker memory required; never fake zero')
        for key in ('parent_memory_before_sweep','parent_memory_after_sweep'):
            observed=row[key];require(isinstance(observed,dict) and all(type(v)is int and v>=0 for v in observed.values()) and
                all(k in observed for k in ('maxrss_bytes','rss_bytes','pss_bytes')),
                'actual nonnegative Linux parent observations before/after outside wall required')
        require(row['parent_memory']==row['parent_memory_after_sweep'],'parent memory alias differs from after-wall observation')
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
        'model':result['model'],'query_memory':'disabled','worker_memory_observed':False,
        'parent_memory_outside_wall':[{k:r[k] for k in ('parent_memory_before_sweep','parent_memory_after_sweep')}for r in sweeps]}

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
