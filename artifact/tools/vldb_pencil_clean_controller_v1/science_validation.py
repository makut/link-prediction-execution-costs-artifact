"""Exact V3 per-job science gates; cross-score matrix explicitly restricted to all32ONLINE rows."""
import math
from contracts import require,read,sha,clean as clean_exit,BRIDGE
ARMS=("torch_sparse_public","torch_sparse_direct","compiled_scan_int64","compiled_pairwise_int64","compiled_scan_int32","compiled_pairwise_int32","cpp_full_clear","cpp_touched_reset")
REPLAY_ARMS=("torch_sparse_public","torch_sparse_direct","cpp_touched_reset","compiled_pairwise_int32")

def validate_job(job,exit):
    require(clean_exit(exit),'owned job failed, timed out, memory-guarded or not quiescent: '+job['name'])
    raw=read(job['output'])
    if job['kind']=='preflight':require(raw.get('status')=='preflight'and raw.get('preflight_passed')is True,'preflight not admitted');return raw
    require(raw.get('status')=='complete'and raw.get('command')==job['kind'],'driver did not complete declared phase')
    policy=raw.get('runtime_compatibility',{})
    require(policy.get('version')=='graph_only_transformers_optional_vision_v2'and policy.get('transformers_version')=='4.46.2'and
        policy.get('import_utils_sha256')=='9d7f63effbd7c8dcfe69e7f34cae0d164db9cca2d5f4d887405ae0f3e2ba455a'and
        policy.get('policy_source_sha256')==sha(BRIDGE/'graph_only_transformers_v2.py')and
        policy.get('torchvision_available_for_graph_worker')is False and policy.get('torchvision_imported')is False and
        policy.get('fake_operator_registered')is False and policy.get('base_package_modified')is False and policy.get('upstream_source_modified')is False and
        raw.get('audit_environment',{}).get('runtime_compatibility')==policy,'explicit pinned graph-only optional-vision policy missing/changed')
    if job['kind']=='audit':
        gates=raw['result']['semantic_gates'];require(len(gates)==16 and
            {(g['arm'],g['training_wrapper'])for g in gates}=={(arm,mode)for arm in ARMS for mode in(False,True)}and
            all(g['raw_csr_slots_order_exact']and g['postmask_data_exact']and g['wrapped_and_collated_inputs_exact']for g in gates),
            'all8arms eval/train semantic gates required')
        require(raw['result']['model_forward_checked']is True and raw['result']['max_logit_abs_tolerance']==0,'mandatory zero-tolerance native model logit audit')
        queries=7 if job['name']=='tiny_audit'else 1024
        require(all(g['queries']==queries for g in gates),'semantic audit cohort incomplete')
        config=raw['configuration'];batch=job.get('batch',128)
        require(config['batch_size']==batch and config['num_workers']==job.get('workers',0)and
            config['precision']==job.get('precision','fp32')and config['device']==('cuda'if job['gpu']else'cpu'),
            'declared audit execution configuration changed')
        require(len(raw['result']['batch_checks'])==2*math.ceil(queries/batch),'all eval/train full batch shapes must be checked')
        require(all(check['all_input_tensors_exact']and set(check['logits'])==set(ARMS)and
            all(v['max_abs']==0 and v['bitwise_equal']is True for v in check['logits'].values())for check in raw['result']['batch_checks']),
            'all native FP32/BF16 model logits must match exactly')
    elif job['kind']=='benchmark':
        config=raw['configuration']
        expected={'batch_size':job['batch'],'num_workers':job['workers'],'precision':'bf16','device':'cuda',
            'pipeline_mode':job['mode'],'arm':job['arm'],'model_seed':42,'tiny_model':False,'checkpoint':None,
            'eval_seed':20261126,'max_logit_abs':0}
        require(all(config.get(k)==v for k,v in expected.items()),'measured runtime/model configuration changed')
        sweeps=raw['result']['sweeps'];require(len(sweeps)==4 and[s['warmup']for s in sweeps]==[True,False,False,False]and
            all(s['queries']==1024 and s['scores']is not None and math.isfinite(s['elapsed_seconds'])and s['elapsed_seconds']>0 for s in sweeps),
            'complete1024query warmup+3timed score reports required')
    elif job['kind']=='capture':require(raw['result']['queries']==(7 if job['name'].startswith('tiny')else 1024),'fixed capture cohort changed')
    return raw

def crossjob_scores(jobs):
    grouped={};models={}
    for job in jobs:
        if job['kind']!='benchmark':continue
        raw=read(job['output']);key=(job['precision'],job['batch'],job['workers'],job['mode'])
        model=raw['result']['model']
        require(model.get('checkpoint')is None and model.get('initialization_seed')==42 and model.get('tiny_fixture_model')is False,
            'measured model seed/checkpoint/size changed')
        if key in models:require(models[key]==model,'measured model configuration differs within score group')
        models[key]=model
        grouped.setdefault(key,{})[job['arm']]=[s['scores']for s in raw['result']['sweeps']]
    expected={('bf16',b,w,'online'):set(ARMS)for b in(128,512)for w in(0,1)}
    require(set(grouped)==set(expected),'all4predeclared ONLINE measured configuration groups required')
    reports=[]
    for key,arms in grouped.items():
        require(set(arms)==expected[key],'full arm set missing from measured score group')
        reference=arms['torch_sparse_public'][0]
        require(reference.get('kind')=='tensor'and reference.get('shape',[0])[0]==1024,'complete1024model scores required')
        require(all(len(scores)==4 and all(s==reference for s in scores)for scores in arms.values()),'measured model score fingerprints differ across arms/repeats: '+str(key))
        reports.append({'precision':key[0],'batch':key[1],'workers':key[2],'mode':key[3],'arms':list(arms),
            'warmup_and_3repeats_all_model_scores_exact':True,'score_fingerprint':reference})
    return {'status':'pass','groups':reports,'scope':'1024measured model scores exact within each precision/batch/workers/mode group; shapes are not pooled; no quality metric.'}
