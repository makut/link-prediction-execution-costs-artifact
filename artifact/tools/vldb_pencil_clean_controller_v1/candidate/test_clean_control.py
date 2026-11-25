"""New instrumentation/plan contracts; tiny synthetic data, no scientific run."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,subprocess,sys,types
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent;BRIDGE=HERE/'bridge/vldb_pencil_bridge_clean_v1'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(BRIDGE))
import plan_only as plan
import pencil_bridge as driver

def arguments(tmp_path):
    return argparse.Namespace(out=tmp_path/'out',citation2_root=tmp_path/'c2',validation_root=tmp_path/'val',
        cpp_library=tmp_path/'cpp/native_assoc_v1.so',cpp_build_receipt=tmp_path/'cpp/build_receipt_v1.json',
        quality_plan=None,python=Path('/usr/bin/python3'),external_memory_guard_gib=96.,pod_memory_limit_gib=122.,accelerator_description='candidate A10080')

def test_all44_jobs_fixed_before_results_and_all32_arm_configurations_present(tmp_path):
    value=plan.make_plan(arguments(tmp_path))
    assert value['execution_authorized']is False and value['scientific_jobs']==44
    assert len(value['jobs'])==44 and value['external_guard_required']is True
    assert all(j['argv'][-2:]==['--query-memory','disabled']for j in value['jobs'])
    assert all(j.get('mode')!='replay'for j in value['jobs'])
    assert sum(j['kind']=='audit'for j in value['jobs'])==9
    assert {(j['arm'],j['batch'],j['workers'])for j in value['jobs']if j['kind']=='benchmark'}=={
        (a,b,w)for a in plan.ARMS for b in(128,512)for w in(0,1)}
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('guard,limit',[(180.,122.),(float('nan'),122.),(96.,0.)])
def test_guard_must_be_explicit_finite_and_fit_the_declared_pod_limit(tmp_path,guard,limit):
    args=arguments(tmp_path);args.external_memory_guard_gib=guard;args.pod_memory_limit_gib=limit
    with pytest.raises(ValueError):plan.make_plan(args)

def test_planner_has_no_execute_switch_or_output_side_effect(tmp_path):
    a=arguments(tmp_path);argv=[]
    for name in('citation2_root','validation_root','cpp_library','cpp_build_receipt','out'):
        argv+=['--'+name.replace('_','-'),str(getattr(a,name))]
    argv+=['--external-memory-guard-gib','96','--pod-memory-limit-gib','122']
    with pytest.raises(SystemExit)as error:plan.main([*argv,'--execute'])
    assert error.value.code==2 and not a.out.exists()

def test_actual_default_cli_imports_no_numerical_packages_or_graph_data(tmp_path):
    a=arguments(tmp_path);argv=[]
    for name in('citation2_root','validation_root','cpp_library','cpp_build_receipt','out'):
        argv+=['--'+name.replace('_','-'),str(getattr(a,name))]
    argv+=['--external-memory-guard-gib','96','--pod-memory-limit-gib','122']
    code="import sys;import plan_only;assert plan_only.main(sys.argv[1:])==0;assert not any(n in sys.modules for n in ('numpy','torch','transformers'))"
    result=subprocess.run([sys.executable,'-B','-c',code,*argv],env={**os.environ,'PYTHONPATH':str(HERE)},capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout);assert value['scientific_jobs']==44 and value['output_created']is False
    assert not a.out.exists()and not a.citation2_root.exists()

def test_disabled_query_memory_never_calls_smaps_and_retains_exact_S_check(monkeypatch):
    args=types.SimpleNamespace(query_memory='disabled',pipeline_mode='online')
    nodes=np.array([0,2],dtype=np.int64)
    trace=types.SimpleNamespace(nodes=lambda _:nodes)
    stages={'nodes':nodes,**{name:0.001 for name in('sampler_seconds','induction_seconds','raw_data_seconds','wrapper_seconds')}}
    state=types.SimpleNamespace(wrapped=lambda *a:('wrapped',None,stages),facade=types.SimpleNamespace(initializations=[]))
    item=driver.PipelineDataset(args,None,None,None,trace);item.state=state;item.pid=os.getpid()
    monkeypatch.setattr(driver,'memory',lambda:(_ for _ in()).throw(AssertionError('inline smaps must not run')))
    wrapped,meta=item[0]
    assert wrapped=='wrapped'and meta['memory']is None and meta['query_memory_mode']=='disabled'
    assert meta['S_verification_seconds']>=0
    stages['nodes']=np.array([0,3],dtype=np.int64)
    with pytest.raises(AssertionError,match='S changed'):item[0]

def test_explicit_inline_mode_retains_actual_memory_call(monkeypatch):
    calls=[];observation={'pss_bytes':123,'maxrss_bytes':456}
    monkeypatch.setattr(driver,'memory',lambda:calls.append('memory')or observation)
    assert driver.measure_query_memory(types.SimpleNamespace(query_memory='inline_smaps'))is observation
    assert calls==['memory']
    with pytest.raises(ValueError):driver.measure_query_memory(types.SimpleNamespace(query_memory='unknown'))

def test_disabled_worker_memory_is_unobserved_null_and_not_zero():
    stages=[{'pid':7,'memory':None,'query_memory_mode':'disabled'}]
    assert driver.summarize_worker_memory(stages,'disabled')[0]['max_pss_bytes']is None
    assert driver.summarize_worker_memory(stages,'disabled')[0]['maxrss_bytes']is None
    stages[0]['memory']={'pss_bytes':0,'maxrss_bytes':0}
    with pytest.raises(ValueError,match='disabled'):driver.summarize_worker_memory(stages,'disabled')

def test_inline_unobserved_pss_stays_unknown_not_fake_zero():
    stages=[{'pid':7,'memory':{'pss_bytes':None,'maxrss_bytes':123}}]
    result=driver.summarize_worker_memory(stages,'inline_smaps')
    assert result[0]['max_pss_bytes']is None and result[0]['maxrss_bytes']==123

def test_parser_default_preserves_inline_and_clean_plan_requires_explicit_disable(tmp_path):
    args=driver.parser().parse_args(['benchmark','--output',str(tmp_path/'x')])
    assert args.query_memory=='inline_smaps'
    args=driver.parser().parse_args(['benchmark','--query-memory','disabled','--output',str(tmp_path/'x')])
    assert args.query_memory=='disabled'

@pytest.mark.parametrize('change',['memory_mode','old_source'])
def test_benchmark_cannot_reuse_old_mode_or_old_source_semantic_receipt(tmp_path,monkeypatch,change):
    trace_dir=tmp_path/'trace';trace_dir.mkdir();(trace_dir/'manifest.json').write_text('{}')
    receipt=tmp_path/'audit.json'
    args=types.SimpleNamespace(audit_receipt=receipt,trace_dir=trace_dir,arm='torch_sparse_public',
        batch_size=128,num_workers=0,eval_seed=20261126,max_logit_abs=0,query_memory='disabled',forward=False)
    config={key:getattr(args,key)for key in('batch_size','num_workers','eval_seed','max_logit_abs','query_memory')}
    raw={'status':'complete','command':'audit','trace_manifest_sha256':driver.sha_file(trace_dir/'manifest.json'),
        'result':{'semantic_gates':[{'arm':args.arm,'training_wrapper':False,'wrapped_and_collated_inputs_exact':True}]},
        'configuration':config,'audit_environment':{'same':True},'source_files':driver.source_files()}
    if change=='memory_mode':config['query_memory']='inline_smaps'
    else:raw['source_files']={'old_source':'not_current_source'}
    receipt.write_text(json.dumps(raw));monkeypatch.setattr(driver,'audit_environment',lambda _:raw['audit_environment'])
    with pytest.raises(ValueError,match='query_memory'if change=='memory_mode'else'source changed'):
        driver.verify_audit_receipt(args,types.SimpleNamespace(),None)

def test_only_instrumentation_functions_changed_and_all_other_frozen_ASTs_identical():
    lineage=json.loads((HERE/'metadata/FROZEN_PRODUCER_INPUTS.json').read_text())
    original=HERE/'metadata/ORIGINAL_PENCIL_BRIDGE.py'
    allowed={'benchmark','verify_audit_receipt','parser','PipelineDataset.__getitem__'}
    def functions(tree):
        values={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        for cls in[n for n in tree.body if isinstance(n,ast.ClassDef)]:
            values.update({cls.name+'.'+n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)})
        return values
    before=functions(ast.parse(original.read_text()));after=functions(ast.parse((BRIDGE/'pencil_bridge.py').read_text()))
    assert {name for name in before if ast.dump(before[name],include_attributes=False)!=ast.dump(after[name],include_attributes=False)}==allowed
    for name,row in lineage['files'].items():
        if name=='bridge/vldb_pencil_bridge_clean_v1/pencil_bridge.py':continue
        p=HERE/name
        assert p.stat().st_size==row['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']

def test_parent_memory_observations_are_outside_wall_start_elapsed_interval():
    tree=ast.parse((BRIDGE/'pencil_bridge.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='benchmark')
    loop=next(n for n in ast.walk(fn)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='repetition')
    assignment={n.targets[0].id:i for i,n in enumerate(loop.body)if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    assert assignment['parent_memory_before']<assignment['start']<assignment['elapsed']<assignment['parent_memory_after']

def test_actual_tiny_online_BERT_pipeline_records_disabled_memory_honestly(tmp_path,monkeypatch):
    """Actual PENCIL/BERT classes; explicitly simulated sparse API, no GPU."""
    torch=pytest.importorskip('torch');pytest.importorskip('transformers')
    import test_pencil_adapter as fixture
    import pencil_runtime as runtime
    native,_,_,_=fixture.fake_native()
    native.provenance={'test_double':'independent sparse API simulator; NOT native sparse execution'}
    upstream=runtime.load_upstream(BRIDGE/'upstream_pencil',BRIDGE/'upstream_manifest.json')
    monkeypatch.setattr(torch.ops.torch_sparse,'ego_k_hop_sample_adj',fixture.fake_ego,raising=False)
    monkeypatch.setattr(driver,'setup',lambda *_:(upstream,native,{'test_double':True,'actual_native_execution':False}))
    common=['--fixture','--execute','--trace-dir',str(tmp_path/'trace'),'--batch-size','2',
            '--query-memory','disabled','--pipeline-mode','online']
    assert driver.main(['capture',*common,'--output',str(tmp_path/'capture.json')])==0
    assert driver.main(['audit',*common,'--arms','torch_sparse_public,compiled_scan_int32,compiled_pairwise_int64',
        '--forward','--tiny-model','--training-wrapper','--output',str(tmp_path/'audit.json')])==0
    observed=[];original=driver.measure_query_memory
    def observe(args):observed.append(args.query_memory);return original(args)
    monkeypatch.setattr(driver,'measure_query_memory',observe)
    assert driver.main(['benchmark',*common,'--arm','compiled_scan_int32','--forward','--tiny-model',
        '--audit-receipt',str(tmp_path/'audit.json'),'--warmup','1','--repeats','1','--output',str(tmp_path/'bench.json')])==0
    result=json.loads((tmp_path/'bench.json').read_text())['result']
    assert observed==['disabled']*14
    assert result['memory_measurement_policy']['per_query_memory_samples_enabled']is False
    for sweep in result['sweeps']:
        assert sweep['queries']==7 and sweep['scores']is not None
        assert sweep['parent_memory_before_sweep']['maxrss_bytes']>0
        assert sweep['parent_memory_after_sweep']['maxrss_bytes']>0
        assert all(row['max_pss_bytes']is None and row['maxrss_bytes']is None for row in sweep['worker_memory'])
