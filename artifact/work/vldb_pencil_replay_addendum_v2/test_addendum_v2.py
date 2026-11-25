"""One-predicate correction tests; synthetic exports are not measurements."""
from pathlib import Path
import copy,hashlib,inspect,json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_addendum_v2 as v
sys.path.insert(0,str(v.CORE))
import synthetic_fixtures as f
from archive_contract import InvalidEvidence,verify_export

@pytest.fixture
def amended():
    core=v.install(v.load_core());contract=core.read(v.CORE/'FROZEN_CONTRACT.json');q,jobs=core.frozen_protocol(contract)
    return core,contract,q,jobs

def materialize(raw,core):
    for row in raw['result']['sweeps']:row['stages']['sampler_seconds']=core.exact_summary([9e-6]*1024)
    return raw

def test_actual_source_callchain_and_one_shot_callspy_receipt_are_admitted():
    assert len(v.verify_callchain())==4
    proof=v.verify_callspy()
    assert proof['replay_real_calls']==0 and proof['online_real_calls']==1
    assert proof['positive_replay_materialization_timer_permitted'] is True
    assert v.correction_receipt()['original_failed_attempt']['returncode']==2

def test_derivation_changes_exactly_one_predicate_and_no_other_benchmark_body():
    core=v.load_core();source=inspect.getsource(core.benchmark_receipt)
    assert source.count(v.OLD)==1
    derived=source.replace(v.OLD,v.NEW)
    assert derived.replace(v.NEW,v.OLD)==source
    assert derived.count('validate_replay_materialization')==1
    v.install(core)
    assert core.benchmark_receipt.__code__.co_filename=='<explicit_replay_timer_addendum_v2>'

def test_nonzero_replay_materialization_is_rejected_by_frozen_core_and_admitted_by_explicit_addendum(tmp_path,amended):
    core,contract,q,jobs=amended
    job=next(j for j in jobs if j.get('mode')=='replay'and j.get('arm')=='cpp_touched_reset')
    raw,audits=f.benchmark(job,tmp_path,contract,q.ARMS);materialize(raw,core)
    with pytest.raises(InvalidEvidence,match='REPLAY must explicitly bypass sampler'):
        v.load_core().benchmark_receipt(raw,job,tmp_path,contract,audits)
    row=core.benchmark_receipt(raw,job,tmp_path,contract,audits)
    assert row['median_seconds']==4 and row['queries_per_second_from_median_seconds']==256

@pytest.mark.parametrize('change',['qps','median','negative_stage','missing_stage','score_shape','warmup','audit_hash','environment','cpp_owner'])
def test_all_other_replay_false_success_conditions_remain_rejected(tmp_path,amended,change):
    core,contract,q,jobs=amended;job=next(j for j in jobs if j.get('mode')=='replay'and j.get('arm')=='cpp_touched_reset')
    raw,audits=f.benchmark(job,tmp_path,contract,q.ARMS);materialize(raw,core);row=raw['result']['sweeps'][1]
    if change=='qps':row['queries_per_second']*=1000
    elif change=='median':raw['result']['elapsed_summary']['p50_s']*=1000
    elif change=='negative_stage':row['stages']['sampler_seconds']['min_s']=-1
    elif change=='missing_stage':row['stages'].pop('wrapper_seconds')
    elif change=='score_shape':row['scores']['shape']=[1023]
    elif change=='warmup':row['warmup']=True
    elif change=='audit_hash':raw['result']['audit_receipt']['sha256']='0'*64
    elif change=='environment':raw['audit_environment']={'changed':True}
    else:row['cpp_initializations'][0][0]['metadata']['owner_pid']=456
    with pytest.raises((InvalidEvidence,ValueError)):
        core.benchmark_receipt(raw,job,tmp_path,contract,audits)

def test_complete48_job_synthetic_export_with_materialization_checks_all_original_gates(tmp_path,monkeypatch,amended):
    core,contract,q,jobs=amended;original=f.benchmark
    def make(job,*args,**kwargs):
        raw,audits=original(job,*args,**kwargs)
        if job.get('mode')=='replay':materialize(raw,core)
        return raw,audits
    monkeypatch.setattr(f,'benchmark',make)
    folder=f.complete_scientific_export(tmp_path/'export',contract,q,jobs)
    context=verify_export(folder,contract['queue_source_sha256']);report=core.build_report(context,contract)
    assert report['status']=='scientific_checks_passed'and report['completed_jobs']==48
    checks=report['scientific_checks']
    assert len(checks['primary_online_bf16_rows'])==32 and len(checks['supplemental_replay_bf16_rows'])==4
    assert len(checks['measured_score_groups'])==5 and len(checks['semantic_audits'])==9
    assert report['explicit_auditor_correction']['all_other_original_gates_unchanged'] is True
    assert report['explicit_auditor_correction']['actual_scientific_reruns']==0
    assert all(row['median_seconds']==4 for row in checks['supplemental_replay_bf16_rows'])

@pytest.mark.parametrize('key,value',[('replay_real_calls',1),('replay_real_calls',False),('online_real_calls',0),
    ('frozenS_int64_values_exact',False),('replay_materialization_seconds',0),('runtime_source_sha256','0'*64),
    ('scientific_jobs_run',True),('GPU_work',True)])
def test_callspy_false_claims_refused(monkeypatch,key,value):
    record=copy.deepcopy(v.read(v.HERE/'ACTUAL_SAMPLER_HOOK_CALLSPY.json'));record[key]=value
    monkeypatch.setattr(v,'read',lambda _:record)
    with pytest.raises(ValueError,match='callspy receipt invalid'):v.verify_callspy()

def test_first_failed_attempt_and_frozen_manifest_remain_unchanged():
    receipt=v.correction_receipt()
    assert v.sha(v.CORE/'SOURCE_MANIFEST.json')==v.CORE_MANIFEST
    for key in ('original_failed_attempt','original_failure_stderr'):
        row=receipt[key];assert v.sha(row['path'])==row['sha256']

def test_new_addendum_source_tree_cannot_be_used_as_output(tmp_path):
    output=v.HERE/'never_create_actual_output'
    with pytest.raises(ValueError,match='outside the new derived'):
        v.main(['--export',str(tmp_path/'absent'),'--output',str(output)])
    assert not output.exists()
