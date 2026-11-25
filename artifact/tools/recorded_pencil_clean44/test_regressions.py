"""Original meaningful numerical/semantic regressions rebound to new auditor."""
import copy,json,types
from pathlib import Path
import numpy as np
import pytest
import audit_clean_v1 as a
import synthetic_fixtures as f
from archive_contract import InvalidEvidence,read,identity,sha
from trace_audit import load_trace
@pytest.fixture(scope='module')
def frozen():
    contract=read(a.HERE/'FROZEN_CONTRACT.json');config,launch,admission,data=f.launch_admission(contract)
    contract['cpp'].update(library_sha256='e'*64,build_receipt_sha256='f'*64)
    return contract,types.SimpleNamespace(ARMS=a.ARMS),f.plan(config)['jobs']
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
    path=tmp_path/'guard.jsonl';path.write_text(json.dumps({'decision':{'abort_reason':None,'memory_budget_bytes':96*2**30,'own_pss_estimate_bytes':None}})+'\n')
    assert a.guards(path)['maximum_recorded_own_pss_estimate_bytes']is None

def test_constant_stage_summary_allows_float_mean_roundoff_but_not_wrong_scale():
    value=a.exact_summary([.001]*7);a.stage_summary(value,7)
    value['mean_s']*=2
    with pytest.raises(InvalidEvidence):a.stage_summary(value,7)
