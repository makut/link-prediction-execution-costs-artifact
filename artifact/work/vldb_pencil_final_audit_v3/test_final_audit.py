import copy,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import audit_pencil_v3_v1 as a
import synthetic_fixtures as f
from archive_contract import InvalidEvidence,verify_export,read,identity,sha
from trace_audit import load_trace

@pytest.fixture(scope='module')
def frozen():
    contract=read(a.HERE/'FROZEN_CONTRACT.json');q,jobs=a.frozen_protocol(contract);return contract,q,jobs

def test_frozen_actual_job_matrix_and_source_contract(frozen):
    contract,q,jobs=frozen;a.validate_plan(f.plan(contract,q,jobs),contract,q,jobs)
    assert len(jobs)==48 and sum(j['kind']=='benchmark'for j in jobs)==36
    assert sum(j['kind']=='audit'for j in jobs)==9
    assert {j['arm']for j in jobs if j.get('mode')=='online'}==set(q.ARMS)

@pytest.mark.parametrize('change',['argv','missing_job','wheel','source','budget','protocol'])
def test_saved_protocol_source_budget_mutations_rejected(frozen,change):
    contract,q,jobs=frozen;plan=f.plan(contract,q,jobs)
    if change=='argv':plan['jobs'][20]['argv'].append('--wrong')
    elif change=='missing_job':plan['jobs'].pop()
    elif change=='wheel':plan['source_bindings'][str(q.layout(Path(a.BASE))['wheel_archive'])]={'bytes':1,'sha256':'0'*64}
    elif change=='source':plan['source_bindings'][a.BASE+'/pencil_queue_v3_code/pencil_queue_v3.py']['sha256']='0'*64
    elif change=='budget':plan['max_total_hours_including_wait']=13
    else:plan['protocol_spec']={}
    with pytest.raises(InvalidEvidence):a.validate_plan(plan,contract,q,jobs)

def test_full_cli_defaults_all48_jobs_are_decoded_without_driver_import(frozen):
    contract,q,jobs=frozen
    for job in jobs:a.configuration(f.raw(job,contract,q.ARMS),job,contract,q.ARMS)

@pytest.mark.parametrize('key,value',[('precision','fp32'),('batch_size',8192),('num_workers',2),('eval_seed',1),('model_seed',13),
    ('checkpoint','/tmp/weights'),('pipeline_mode','replay'),('warmup',0),('repeats',1),('file_endian','big'),('max_memory_gib',64)])
def test_measured_configuration_mutations_rejected(frozen,key,value):
    contract,q,jobs=frozen;job=next(j for j in jobs if j['kind']=='benchmark');raw=f.raw(job,contract,q.ARMS);raw['configuration'][key]=value
    with pytest.raises(InvalidEvidence):a.configuration(raw,job,contract,q.ARMS)

def test_tiny_trace_independent_slot_value_target_mask_oracle_and_equal_roots(tmp_path):
    path=f.trace(tmp_path/'tiny');trace=load_trace(path,7,True)
    assert trace['pairs'][0].tolist()==[3,0]and trace['roots'][0].tolist()!=[0,1]
    assert trace['roots'][4][0]==trace['roots'][4][1]and trace['roots'][6].tolist()==[0,0]
    assert trace['records'][0]['raw']['csr_edge_slots']!=trace['records'][0]['stored_values']
    assert trace['report']['tiny_ordered_raw_slots_values_and_target_mask_independently_recomputed']

@pytest.mark.parametrize('change',['file_hash','stored_values','slot_order','target_mask','rng','root','sampler','seed','idx'])
def test_tiny_trace_semantic_and_rng_mutations_rejected_even_with_updated_file_manifest(tmp_path,change):
    path=f.trace(tmp_path/'tiny');records=read(path/'records.json')
    if change=='file_hash':(path/'nodes.npy').write_bytes(b'corrupt')
    elif change=='stored_values':records['records'][0]['stored_values']=copy.deepcopy(records['records'][0]['raw']['csr_edge_slots'])
    elif change=='slot_order':records['records'][0]['raw']['csr_edge_slots']['hash']='0'*64
    elif change=='target_mask':records['records'][1]['data']['edge_index']['sha256']='0'*64
    elif change=='rng':records['records'][0]['post_raw_rng']['np_position']=1
    elif change=='root':
        roots=np.load(path/'root_positions.npy');roots[0]=[0,1];np.save(path/'root_positions.npy',roots,allow_pickle=False)
    elif change in('sampler','seed'):
        manifest=read(path/'manifest.json');manifest['metadata']['eval_seed']=1 if change=='seed'else 20261126
        if change=='sampler':manifest['metadata']['sampler']['replacement']=True
        f.put(path/'manifest.json',manifest)
    else:records['records'][0]['data']['idx']=1
    if change in('stored_values','slot_order','target_mask','rng','idx'):f.put(path/'records.json',records)
    if change!='file_hash':f.refresh_trace(path)
    with pytest.raises((InvalidEvidence,ValueError)):load_trace(path,7,True)

def test_full_real_shaped_trace_has1024_queries_and_original512_group_order(tmp_path):
    path=f.trace(tmp_path/'real',False);trace=load_trace(path,1024,False)
    assert trace['queries']==1024 and trace['labels'][:512].sum()==512 and trace['labels'][512:].sum()==0
    assert trace['report']['real_graph_oracle_reexecuted']is False

@pytest.mark.parametrize('batch,precision,workers',[(128,'fp32',0),(128,'bf16',1),(512,'fp32',1),(512,'bf16',0)])
def test_complete_real_semantic_batch_coverage_both_wrappers(tmp_path,frozen,batch,precision,workers):
    contract,q,jobs=frozen;path=f.trace(tmp_path/'real',False);trace=load_trace(path,1024,False)
    job=copy.deepcopy(next(j for j in jobs if j['name']==f'audit_{precision}_b{batch}_w{workers}'));job['_trace_path']=path
    raw=f.semantic(job,path,contract,q.ARMS);a.configuration(raw,job,contract,q.ARMS)
    report=a.semantic_receipt(raw,job,trace,q.ARMS);assert report['batch_checks']==2*(1024//batch)

@pytest.mark.parametrize('change',['omitted_batch','reordered_batch','shape','score','slot_gate','value_gate','training','rng_gate'])
def test_semantic_gates_shape_slot_values_scores_and_full_partition_rejected(tmp_path,frozen,change):
    contract,q,jobs=frozen;path=f.trace(tmp_path/'tiny');trace=load_trace(path,7,True);job=dict(jobs[1],_trace_path=path);raw=f.semantic(job,path,contract,q.ARMS)
    result=raw['result']
    if change=='omitted_batch':result['batch_checks'].pop()
    elif change=='reordered_batch':result['batch_checks'].reverse()
    elif change=='shape':result['batch_checks'][0]['batch_size']=6
    elif change=='score':result['batch_checks'][0]['logits'][q.ARMS[-1]]['hash']['sha256']='0'*64
    elif change=='slot_gate':result['semantic_gates'][0]['raw_csr_slots_order_exact']=False
    elif change=='value_gate':result['semantic_gates'][0]['postmask_data_exact']=False
    elif change=='training':result['semantic_gates'][0]['training_wrapper']=True
    else:result['semantic_gates'][0]['native_postraw_cpu_rng_restored_before_candidate_wrapper']=False
    with pytest.raises(InvalidEvidence):a.semantic_receipt(raw,job,trace,q.ARMS)

@pytest.mark.parametrize('arm',['torch_sparse_public','compiled_pairwise_int32','cpp_full_clear','cpp_touched_reset'])
def test_benchmark_recomputed_qps_medians_and_cpp_cost_receipts(tmp_path,frozen,arm):
    contract,q,jobs=frozen;job=next(j for j in jobs if j.get('mode')=='online'and j.get('arm')==arm);raw,audits=f.benchmark(job,tmp_path,contract,q.ARMS)
    row=a.benchmark_receipt(raw,job,tmp_path,contract,audits)
    assert row['median_seconds']==4 and row['queries_per_second_from_median_seconds']==256
    assert row['measured_seconds']==[3.,4.,5.]  # Warmup100s must never enter medians.
    assert len(row['measured_per_query_stage_summary_receipts'])==3 and len(row['gpu_cumulative_high_water_bytes_including_warmup'])==4
    assert row['median_of_3_sweep_induction_p50_seconds']==.01 and row['median_collator_total_seconds']==.1

@pytest.mark.parametrize('change',['qps','elapsed_units','stage_units','missing_stage','score_shape','warmup','repeat','audit_hash','environment','cpp_owner','cpp_memory'])
def test_benchmark_misreported_units_counts_sources_and_cpp_costs_rejected(tmp_path,frozen,change):
    contract,q,jobs=frozen;job=next(j for j in jobs if j.get('mode')=='online'and j.get('arm')=='cpp_touched_reset');raw,audits=f.benchmark(job,tmp_path,contract,q.ARMS)
    result=raw['result'];sweep=result['sweeps'][1]
    if change=='qps':sweep['queries_per_second']*=1000
    elif change=='elapsed_units':result['elapsed_summary']['p50_s']*=1000
    elif change=='stage_units':sweep['stages']['induction_seconds']['p95_s']=-1
    elif change=='missing_stage':sweep['stages'].pop('wrapper_seconds')
    elif change=='score_shape':sweep['scores']['shape']=[1023]
    elif change=='warmup':sweep['warmup']=True
    elif change=='repeat':sweep['repetition']=4
    elif change=='audit_hash':result['audit_receipt']['sha256']='0'*64
    elif change=='environment':raw['audit_environment']={'changed':True}
    elif change=='cpp_owner':sweep['cpp_initializations'][0][0]['metadata']['owner_pid']=456
    else:sweep['cpp_initializations'][0][0]['metadata']['persistent_assoc_bytes_per_worker']=0
    with pytest.raises(InvalidEvidence):a.benchmark_receipt(raw,job,tmp_path,contract,audits)

def all_score_rows(frozen):
    contract,q,jobs=frozen;rows=[]
    for job in jobs:
        if job['kind']!='benchmark':continue
        score=f.fp(1024,str(job['batch']));rows.append({'mode':job['mode'],'batch':job['batch'],'workers':job['workers'],'arm':job['arm'],
            'model':f.model(),'median_seconds':4.,'score_fingerprint':score,'all_sweep_score_fingerprints':[copy.deepcopy(score)for _ in range(4)]})
    return rows

def test_all36_measured_scores_exact_in5_separate_groups_without_pooling_batch_shapes(frozen):
    contract,q,jobs=frozen;rows=all_score_rows(frozen);groups,cross=a.cross_scores(rows,q.ARMS,q.REPLAY_ARMS)
    assert len(groups)==5 and len(cross)==4 and all(v['online_and_replay_score_fingerprints_exact']for v in cross)
    assert groups[0]['score_fingerprint']!=next(v for v in groups if v['batch']==512)['score_fingerprint']

@pytest.mark.parametrize('change',['measured_score','warmup_score','missing_arm','missing_group','model','score_dtype'])
def test_measured_score_or_full_group_omission_rejected(frozen,change):
    contract,q,jobs=frozen;rows=all_score_rows(frozen)
    if change=='measured_score':rows[-1]['all_sweep_score_fingerprints'][2]['sha256']='0'*64
    elif change=='warmup_score':rows[0]['all_sweep_score_fingerprints'][0]['sha256']='0'*64
    elif change=='missing_arm':rows.pop()
    elif change=='missing_group':rows=[v for v in rows if v['mode']!='replay']
    elif change=='model':rows[1]['model']['initialization_seed']=43
    else:rows[0]['score_fingerprint']['dtype']='torch.bfloat16'
    with pytest.raises(InvalidEvidence):a.cross_scores(rows,q.ARMS,q.REPLAY_ARMS)

def test_transport_valid_failed_science_is_retained_without_performance(tmp_path,frozen):
    contract,q,jobs=frozen;folder=f.export(tmp_path/'export',contract['queue_source_sha256'],plan_value=f.plan(contract,q,jobs))
    context=verify_export(folder,contract['queue_source_sha256']);report=a.build_report(context,contract)
    assert report['transport_integrity']['status']=='pass'and report['scientific_checks']['status']=='withheld'
    assert report['status']=='persisted_scientific_failure_no_performance_analysis'and report['completed_jobs']==0
    assert len(report['inventory'])==48 and all(v['status']=='not_started'for v in report['inventory'])
    assert 'primary_online_bf16_rows'not in report['scientific_checks']

@pytest.mark.parametrize('change',['tar_bytes','extracted_bytes','internal_manifest','extra_extracted','remote_marker','remote_bool_as_int','controller_identity','pointer','controller_state'])
def test_archive_export_hash_extraction_terminal_and_identity_tampering_rejected(tmp_path,frozen,change):
    contract,q,jobs=frozen;folder=f.export(tmp_path/'export',contract['queue_source_sha256'],bad_internal=change=='internal_manifest')
    if change=='tar_bytes':
        with(folder/'final_snapshot.tgz').open('ab')as stream:stream.write(b'changed')
    elif change=='extracted_bytes':(folder/'extracted'/a.PREFIX/'plan.json').write_text('{}')
    elif change=='extra_extracted':(folder/'extracted/extra.json').write_text('{}')
    elif change in('remote_marker','remote_bool_as_int','controller_identity','controller_state'):
        remote=folder/'remote_receipts/results/pencil_bridge_v3_20261125';name='SCIENTIFIC_TERMINAL.json'if change in('remote_marker','remote_bool_as_int')else'COMPLETE.json';value=read(remote/name)
        if change=='remote_marker':value['success']=True
        elif change=='remote_bool_as_int':value['success']=0
        elif change=='controller_identity':value['controller_identity']['pid']=999
        else:value['timing_lock_released']=False
        f.put(remote/name,value)
    elif change=='pointer':value=read(folder/'export.json');value['pointer']['new_YT_file_nodes']=2;f.put(folder/'export.json',value)
    with pytest.raises(InvalidEvidence):verify_export(folder,contract['queue_source_sha256'])

@pytest.mark.parametrize('science_overrides',[{'success':0},{'completed_jobs':False}])
def test_scientific_boolean_and_counter_types_must_be_literal_even_when_all_hashes_match(tmp_path,frozen,science_overrides):
    contract,q,jobs=frozen;folder=f.export(tmp_path/'export',contract['queue_source_sha256'],science_overrides=science_overrides)
    with pytest.raises(InvalidEvidence,match='literal boolean/integer'):verify_export(folder,contract['queue_source_sha256'])

def test_missing_actual_export_waits_without_creating_output_or_importing_torch(tmp_path):
    output=tmp_path/'not_created';result=subprocess.run([sys.executable,str(a.HERE/'audit_pencil_v3_v1.py'),'--export',str(tmp_path/'missing'),'--output',str(output)],capture_output=True,text=True)
    assert result.returncode==0 and json.loads(result.stdout)['status']=='waiting_for_verified_export'and not output.exists()

def test_observed_failure_and_guarded_job_inventory_are_not_success(tmp_path,frozen):
    contract,q,jobs=frozen;first=jobs[0];f.put(tmp_path/(first['name']+'.json'),{'status':'guarded_skip','error':{'message':'synthetic memory guard'}})
    failed=f.good_exit();failed['guard_abort_reason']='synthetic guard';failed['returncode']=1;f.put(tmp_path/(first['name']+'.exit.json'),failed)
    rows,raws=a.inventory(tmp_path,jobs,{'completed_jobs':0,'success':False},contract,q.ARMS)
    assert rows[0]['status']=='guarded_or_failed'and rows[1]['status']=='not_started'and not raws
    with pytest.raises(InvalidEvidence):a.inventory(tmp_path,jobs,{'completed_jobs':0,'success':True},contract,q.ARMS)

def test_completed_inventory_hash_and_prefix_are_mandatory(tmp_path,frozen):
    contract,q,jobs=frozen;job=jobs[0];raw=f.raw(job,contract,q.ARMS);path=f.put(tmp_path/(job['name']+'.json'),raw);ex=f.good_exit()
    f.put(tmp_path/(job['name']+'.exit.json'),ex);f.good_guard(tmp_path/(job['name']+'.guard.jsonl'));(tmp_path/(job['name']+'.log')).write_text('whole log\n')
    completed={'count':1,'jobs':[{'name':job['name'],'kind':job['kind'],'output':identity(path),'exit':ex}]};f.put(tmp_path/'completed_jobs.json',completed)
    rows,raws=a.inventory(tmp_path,jobs,{'completed_jobs':1,'success':False},contract,q.ARMS);assert rows[0]['status']=='completed_admitted'
    completed['jobs'][0]['name']=jobs[1]['name'];f.put(tmp_path/'completed_jobs.json',completed)
    with pytest.raises(InvalidEvidence):a.inventory(tmp_path,jobs,{'completed_jobs':1,'success':False},contract,q.ARMS)

@pytest.mark.parametrize('change',['abort','pss','sampling','returncode','orphan'])
def test_external_guard_and_process_failure_rejected(tmp_path,change):
    path=tmp_path/'guard.jsonl';f.good_guard(path)
    if change=='abort':value=read(path);value['decision']['abort_reason']='limit';path.write_text(json.dumps(value)+'\n')
    elif change=='pss':value=read(path);value['decision']['own_pss_estimate_bytes']=181*2**30;path.write_text(json.dumps(value)+'\n')
    elif change=='sampling':path.write_text('\n'.join(json.dumps({'sampling_error':'error'})for _ in range(3)))
    else:
        ex=f.good_exit();ex['returncode']=1 if change=='returncode'else 0;ex['orphan_cleanup']=change=='orphan'
        with pytest.raises(InvalidEvidence):a.clean_exit(ex)
        return
    with pytest.raises(InvalidEvidence):a.guards(path)

def test_nonfinite_json_is_not_accepted_as_evidence(tmp_path):
    path=tmp_path/'nonfinite.json';path.write_text('{"value":NaN}')
    with pytest.raises(InvalidEvidence):read(path)

@pytest.mark.parametrize('field,value',[('returncode',False),('timeout_seconds',True),('timeout_seconds',float('inf')),('timeout_seconds',float('nan'))])
def test_exit_boolean_status_or_nonfinite_timeout_is_not_success(field,value):
    ex=f.good_exit();ex[field]=value
    with pytest.raises(InvalidEvidence):a.clean_exit(ex)

def test_no_observed_pss_is_reported_as_unknown_not_zero(tmp_path):
    path=tmp_path/'guard.jsonl';path.write_text(json.dumps({'decision':{'abort_reason':None,'memory_budget_bytes':180*2**30,'own_pss_estimate_bytes':None}})+'\n')
    assert a.guards(path)['maximum_recorded_own_pss_estimate_bytes']is None

def test_constant_stage_summary_allows_float_mean_roundoff_but_not_wrong_scale():
    value=a.exact_summary([.001]*7);a.stage_summary(value,7)
    value['mean_s']*=2
    with pytest.raises(InvalidEvidence):a.stage_summary(value,7)

def test_v3_full_environment_plain_route_original_origins_recovery_and_policy_bound(tmp_path,frozen):
    contract,q,jobs=frozen;plan=f.plan(contract,q,jobs);isolated=f.environment(tmp_path,contract,q,plan)
    observed,report=a.environment(tmp_path,contract,plan,q)
    assert observed==isolated and report['status']=='pass'and report['root_pinned_recovery_admission']['prior_failed_v2_preserved']is True
    assert report['explicit_optional_vision_policy']['torchvision_available_for_graph_worker']is False

@pytest.mark.parametrize('change',['base_origin','post_base','post_isolated','executable_pth','route_precedence','missing_numeric','extra_base_origin',
    'hf_origin_binding','policy_source_binding','wheel_binding','missing_policy','policy_version','fake_operator','root_recovery_hash','old_failure_relabel'])
def test_v3_environment_source_route_policy_and_root_admission_mutations_rejected(tmp_path,frozen,change):
    contract,q,jobs=frozen;plan=f.plan(contract,q,jobs);isolated=f.environment(tmp_path,contract,q,plan)
    if change in('base_origin','post_base'):
        path=tmp_path/'base_environment_after.json';value=read(path)
        if change=='base_origin':value['critical_modules']['numpy']['origin']='/different/numpy.py'
        else:value['changed']=True
        f.put(path,value)
    elif change=='post_isolated':value=read(tmp_path/'isolated_environment_after.json');value['changed']=True;f.put(tmp_path/'isolated_environment_after.json',value)
    elif change in('hf_origin_binding','policy_source_binding','wheel_binding'):
        path=tmp_path/'environment_runtime_bindings.json';value=read(path)
        target=isolated['runtime_compatibility']['import_utils_origin']if change=='hf_origin_binding'else(
            a.BASE+'/pencil_queue_v3_code/bridge/vldb_pencil_bridge_v2/graph_only_transformers_v2.py'if change=='policy_source_binding'else str(q.layout(Path(a.BASE))['wheel_archive']))
        value[target]['sha256']='0'*64;f.put(path,value)
    elif change in('root_recovery_hash','old_failure_relabel'):
        path=tmp_path/'admission.json';value=read(path)
        if change=='root_recovery_hash':value['receipt_sha256']='0'*64
        else:next(v for v in value['prior_archives']if v['role']=='pencil_failed_v2')['scientific_success']=True
        f.put(path,value)
    else:
        if change=='executable_pth':isolated['base_site_route']['content']='import os\n'
        elif change=='route_precedence':isolated['base_site_route']['isolated_site_precedes_all_base_paths']=False
        elif change=='missing_numeric':isolated['critical_modules'].pop('torch')
        elif change=='extra_base_origin':isolated['extra_modules']['transformers']['origin']='/opt/python/lib/python3.10/site-packages/transformers/__init__.py'
        elif change=='missing_policy':isolated.pop('runtime_compatibility')
        elif change=='policy_version':isolated['runtime_compatibility']['transformers_version']='5.0'
        else:isolated['runtime_compatibility']['fake_operator_registered']=True
        f.put(tmp_path/'isolated_environment.json',isolated);f.put(tmp_path/'isolated_environment_after.json',isolated)
    with pytest.raises((InvalidEvidence,KeyError)):a.environment(tmp_path,contract,plan,q)

def test_v3_root_recovery_anchor_keeps_original_outer_cleanup_unknown_and_old_pencil0(frozen):
    contract,q,jobs=frozen;plan=f.plan(contract,q,jobs);actual=f.recovery(contract)
    result=a.recovery_admission(actual,contract,plan)
    assert result['status']=='pass'and result['original_outer_controller_cleanup'].startswith('UNKNOWN')
    actual['replacement_hostname']='another-pod'
    with pytest.raises(InvalidEvidence):a.recovery_admission(actual,contract,plan)

def test_v3_optional_policy_source_and_hf_loaded_origin_are_exact(frozen):
    contract,q,jobs=frozen;isolated={'extra_modules':{'transformers':{'origin':a.BASE+'/pencil_env_v3/lib/python3.10/site-packages/transformers/__init__.py'}},
        'packages':{'torchvision':'0.17.2+cu121'}};value=f.policy(contract,isolated)
    a.optional_vision_policy(value,isolated,contract)
    value['import_utils_origin']='/opt/python/lib/python3.10/site-packages/transformers/utils/import_utils.py'
    with pytest.raises(InvalidEvidence):a.optional_vision_policy(value,isolated,contract)

def test_v3_complete_synthetic48_job_export_exercises_all_scientific_glue_without_real_graph_or_scores(tmp_path,frozen):
    contract,q,jobs=frozen;folder=f.complete_scientific_export(tmp_path/'export',contract,q,jobs)
    context=verify_export(folder,contract['queue_source_sha256']);report=a.build_report(context,contract)
    assert report['status']=='scientific_checks_passed'and report['completed_jobs']==48
    checks=report['scientific_checks'];assert checks['status']=='pass'and len(checks['primary_online_bf16_rows'])==32 and len(checks['supplemental_replay_bf16_rows'])==4
    assert len(checks['semantic_audits'])==9 and len(checks['measured_score_groups'])==5
    assert all(v['median_seconds']==4 and v['queries_per_second_from_median_seconds']==256 for v in checks['primary_online_bf16_rows'])
