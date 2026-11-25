import copy,hashlib,json,subprocess,sys
from pathlib import Path
import pytest
import audit_clean_v1 as a
import synthetic_fixtures as f
from plan_contract import make_jobs,controller_configuration
from evidence_checks import cpp_build,public_inputs,strict_manifest,platform
from archive_contract import canonical,decode
@pytest.fixture(scope='module')
def contract():return a.read(a.HERE/'FROZEN_CONTRACT.json')
@pytest.fixture
def full(tmp_path,contract):return f.complete_payload(tmp_path/'payload',contract)
def context(payload,tmp_path,contract):
    folder,launch,admission=f.package(payload,tmp_path/'export',contract);return a.verify_export(folder,contract,launch,admission)
def test_complete_independent44_job_transport_science_and_memory_glue(full,tmp_path,contract):
    ctx=context(full,tmp_path,contract);report=a.build_report(ctx,contract)
    assert report['status']=='scientific_checks_passed'and report['completed_jobs']==44
    checks=report['scientific_checks'];assert len(checks['primary_online_bf16_rows'])==32 and len(checks['semantic_audits'])==9 and len(checks['measured_score_groups'])==4
    assert all(v['query_memory']=='disabled'and v['worker_memory_observed']is False for v in checks['primary_online_bf16_rows'])
    assert report['outer_final_pid_exit']=='UNKNOWN'and report['transport_integrity']['attribute_is_inside_tar_hash']is False
    assert 'Supplemental REPLAY'not in a.text_report(report)
@pytest.mark.parametrize('change',['argv','missing_job','query_memory','budget','setup','protocol','after','unrecognized_key'])
def test_complete_controller_plan_mutations_fail_before_performance(full,tmp_path,contract,change):
    value=a.read(full/'plan.json')
    if change=='argv':value['jobs'][20]['argv'].append('--wrong')
    elif change=='missing_job':value['jobs'].pop()
    elif change=='query_memory':value['jobs'][20]['argv'][-1]='inline_smaps'
    elif change=='budget':value['controller_protocol']['active_hardware']['owned_tree_guard_gib']=180
    elif change=='setup':value['setup_jobs'][0]['gpu']=False
    elif change=='protocol':value['protocol']['model_seed']=43
    elif change=='after':value['after_jobs']=[]
    else:value['fake']=True
    f.put(full/'plan.json',value);ctx=context(full,tmp_path,contract)
    with pytest.raises(a.InvalidEvidence):a.build_report(ctx,contract)
@pytest.mark.parametrize('change',['worker_zero','worker_rss','missing_parent','missing_pss','parent_bool','parent_alias','inline_policy','policy_zero','mode','audit_hash','score','cpp_library','cpp_flags','strict19','qualityplan','guard180','completed_hash','notfull'])
def test_resigned_whole_archive_still_rejects_bad_science(full,tmp_path,contract,change):
    jobs=a.read(full/'plan.json')['jobs'];job=next(j for j in jobs if j.get('arm')=='cpp_touched_reset');path=full/(job['name']+'.json');raw=a.read(path);row=raw['result']['sweeps'][1]
    if change=='worker_zero':row['worker_memory'][0]['max_pss_bytes']=0
    elif change=='worker_rss':row['worker_memory'][0]['maxrss_bytes']=1
    elif change=='missing_parent':row.pop('parent_memory_before_sweep')
    elif change=='missing_pss':row['parent_memory_before_sweep'].pop('pss_bytes')
    elif change=='parent_bool':row['parent_memory_before_sweep']['rss_bytes']=False
    elif change=='parent_alias':row['parent_memory']['rss_bytes']=999
    elif change=='inline_policy':raw['result']['memory_measurement_policy']['query_memory']='inline_smaps'
    elif change=='policy_zero':raw['result']['memory_measurement_policy']['per_query_memory_samples_enabled']=0
    elif change=='mode':raw['configuration']['query_memory']='inline_smaps'
    elif change=='audit_hash':raw['result']['audit_receipt']['sha256']='0'*64
    elif change=='score':row['scores']['sha256']='0'*64
    elif change=='cpp_library':(full/'cpp_build/native_assoc_v1.so').write_bytes(b'changed')
    elif change=='cpp_flags':v=a.read(full/'cpp_build/build_receipt_v1.json');v['argv'][1]='-std=c++11';f.put(full/'cpp_build/build_receipt_v1.json',v)
    elif change=='strict19':raw['input_metadata']['strict_manifest']['content']['files'].pop()
    elif change=='qualityplan':raw['input_metadata']['payload_binding_verification']['quality_plan_checked']=True
    elif change=='guard180':v=a.read(full/(job['name']+'.guard.jsonl'));v['decision']['memory_budget_bytes']=180*(1<<30);(full/(job['name']+'.guard.jsonl')).write_text(json.dumps(v)+'\n')
    elif change=='completed_hash':v=a.read(full/'completed_jobs.json');v['jobs'][0]['output']['sha256']='0'*64;f.put(full/'completed_jobs.json',v)
    else:raw['result']['sweeps'][2]['queries']=1
    f.put(path,raw)
    if change!='completed_hash':f.refresh_completed(full,jobs)
    ctx=context(full,tmp_path,contract)
    with pytest.raises((a.InvalidEvidence,KeyError)):a.build_report(ctx,contract)
@pytest.mark.parametrize('key,value',[('scientific_success',0),('completed_jobs',False),('stop_requested',0),('all_planned_science_job_exits_clean',1),('outer_final_pid_exit',0),('owned_children_quiescent_before_publication',1)])
def test_literal_terminal_types_even_with_repacked_tar_and_attribute(full,tmp_path,contract,key,value):
    terminal=a.read(full/'SCIENTIFIC_TERMINAL.json');terminal[key]=value;f.put(full/'SCIENTIFIC_TERMINAL.json',terminal)
    with pytest.raises(a.InvalidEvidence):context(full,tmp_path,contract)
@pytest.mark.parametrize('key,value',[('gpu_count',True),('cpu_request',17),('pod_uid','notuuid'),('owned_tree_guard_gib',180),('hostname','oldpod'),('image',''),('authorized_to_execute',1),('observed_unix',-4000)])
def test_actual_root_admission_literal_hardware_freshness_identity(full,contract,key,value):
    value_root=a.read(full/'ROOT_ADMISSION.json');value_root[key]=value
    with pytest.raises((a.InvalidEvidence,ValueError)):a.root_admission(value_root,a.read(full/'LAUNCH.json'),contract)
@pytest.mark.parametrize('change',['tar','extracted','extra','attribute','attribute_sha','anchor','admission_anchor','missing_anchor','outer_claim','duplicate_json'])
def test_transport_attribute_root_and_archive_tampering(full,tmp_path,contract,change):
    folder,launch,admission=f.package(full,tmp_path/'export',contract)
    if change=='tar':(folder/'final_snapshot.tgz').write_bytes((folder/'final_snapshot.tgz').read_bytes()+b'x')
    elif change=='extracted':(folder/'extracted/tiny_capture.json').write_text('{}')
    elif change=='extra':(folder/'extracted/EXTRA').write_text('extra')
    elif change=='attribute':v=a.read(folder/'persistence_attribute.json');v['archive_bytes']+=1;f.put(folder/'persistence_attribute.json',v)
    elif change=='attribute_sha':v=a.read(folder/'export.json');v['attribute_canonical_json_sha256']='0'*64;f.put(folder/'export.json',v)
    elif change=='anchor':v=a.read(launch);v['identity']['pid']=999;f.put(launch,v)
    elif change=='admission_anchor':admission.write_bytes(admission.read_bytes()+b' ')
    elif change=='missing_anchor':launch=None
    elif change=='outer_claim':v=a.read(folder/'persistence_attribute.json');v['outer_final_pid_exit']='exit0';f.put(folder/'persistence_attribute.json',v);e=a.read(folder/'export.json');e['attribute_canonical_json_sha256']=hashlib.sha256(canonical(v).encode()).hexdigest();f.put(folder/'export.json',e)
    else:(folder/'export.json').write_text('{"status":"x","status":"verified_existing_persisted_archive"}')
    with pytest.raises(a.InvalidEvidence):a.verify_export(folder,contract,launch,admission)
@pytest.mark.parametrize('change',['remove','hash','subset','count','strict','identity_type'])
def test_all19_and_two_frozen_payload_full_identity_mutations(full,contract,change):
    config=controller_configuration(a.read(full/'LAUNCH.json'),contract);value=a.read(full/'INPUT_IDENTITY.json');keys=list(value['actual_files']);first=value['actual_files'][keys[0]]
    if change=='remove':value['actual_files'].pop(keys[0])
    elif change=='hash':first['sha256']='0'*64
    elif change=='subset':value['actual_files'][config['validation_root']+'/valid_edge_neg.npy']['sha256']='0'*64
    elif change=='count':value['numeric_payloads']=18
    elif change=='strict':value['strict_manifest']['sha256']='invalid'
    else:first['inode']=False
    f.put(full/'INPUT_IDENTITY.json',value)
    with pytest.raises(a.InvalidEvidence):public_inputs(full,contract,config)
@pytest.mark.parametrize('change',['post','torch','numpy','scipy','numba','cpu','gpu','saint','hf_policy','cgroup','setup_exit'])
def test_fresh_platform_and_actual_setup_mutations(full,contract,change):
    declared=a.read(full/'plan.json');config=controller_configuration(a.read(full/'LAUNCH.json'),contract);v=a.read(full/'platform_before.json')
    if change=='post':after=copy.deepcopy(v);after['visible_gpu']['uuid']='GPU-CHANGED';f.put(full/'platform_after.json',after)
    else:
        if change in('torch','numpy','scipy','numba'):v['packages'][change]='bad'
        elif change=='cpu':v['cpu_observation']['cpu_affinity']=[False]
        elif change=='gpu':v['visible_gpu']['count']=True
        elif change=='saint':v['torch_sparse_binaries']['_saint_cpu.so']['sha256']='0'*64
        elif change=='hf_policy':v['runtime_compatibility']['upstream_source_modified']=True
        elif change=='cgroup':f.put(full/'initial_cgroup.json',{'max_bytes':True})
        else:ex=a.read(full/'platform_before.exit.json');ex['returncode']=1;f.put(full/'platform_before.exit.json',ex)
        f.put(full/'platform_before.json',v);f.put(full/'platform_after.json',v)
    with pytest.raises(a.InvalidEvidence):platform(full,contract,declared,config)
def score_rows(contract):
    config,*_=f.launch_admission(contract);rows=[]
    for j in make_jobs(config):
        if j['kind']=='benchmark':
            score=f.fp(1024,str(j['batch']));rows.append({'mode':j['mode'],'batch':j['batch'],'workers':j['workers'],'arm':j['arm'],'model':f.model(),'median_seconds':4.,'score_fingerprint':score,'all_sweep_score_fingerprints':[copy.deepcopy(score)for _ in range(4)]})
    return rows
def test_all32_four_shapes_and_no_batch_pooling(contract):
    groups=a.cross_scores(score_rows(contract));assert len(groups)==4 and groups[0]['score_fingerprint']!=groups[-1]['score_fingerprint']
@pytest.mark.parametrize('change',['warmup','measured','arm','group','replay','model','dtype'])
def test_allscore_and_matrix_mutations(contract,change):
    rows=score_rows(contract)
    if change in('warmup','measured'):rows[0]['all_sweep_score_fingerprints'][0 if change=='warmup'else 2]['sha256']='0'*64
    elif change=='arm':rows.pop()
    elif change=='group':rows=[r for r in rows if r['batch']!=512]
    elif change=='replay':rows[0]['mode']='replay'
    elif change=='model':rows[0]['model']['initialization_seed']=43
    else:rows[0]['score_fingerprint']['dtype']='torch.bfloat16'
    with pytest.raises(a.InvalidEvidence):a.cross_scores(rows)
def test_failed_science_persists_without_performance(full,tmp_path,contract):
    jobs=a.read(full/'plan.json')['jobs']
    for job in jobs:
        for suffix in('.json','.exit.json','.log','.guard.jsonl'):(full/(job['name']+suffix)).unlink()
    f.put(full/'tiny_capture.json',{'status':'failed','error':'SYNTHETIC fail'});ex=f.good_exit();ex['returncode']=2;f.put(full/'tiny_capture.exit.json',ex);f.put(full/'completed_jobs.json',{'jobs':[],'count':0})
    terminal=a.read(full/'SCIENTIFIC_TERMINAL.json');terminal.update(scientific_success=False,all_planned_science_job_exits_clean=False,completed_jobs=0,failure={'type':'SyntheticFailure'});f.put(full/'SCIENTIFIC_TERMINAL.json',terminal)
    report=a.build_report(context(full,tmp_path,contract),contract);assert report['scientific_success']is False and report['scientific_checks']['status']=='withheld'and report['inventory'][0]['status']=='guarded_or_failed'and 'primary_online_bf16_rows'not in report['scientific_checks']
def test_missing_export_cli_waits_and_creates_nothing(tmp_path):
    output=tmp_path/'never';result=subprocess.run([sys.executable,str(a.HERE/'audit_clean_v1.py'),'--export',str(tmp_path/'missing'),'--output',str(output),'--root-launch',str(tmp_path/'none'),'--root-admission',str(tmp_path/'none')],text=True,capture_output=True)
    assert result.returncode==0 and json.loads(result.stdout)['status']=='waiting_for_verified_export'and not output.exists()
def test_duplicate_json_and_nonfinite_refused():
    for text in ('{"x":1,"x":2}','{"x":NaN}'):
        with pytest.raises(a.InvalidEvidence):decode(text)
def test_canonical_utf8_uses_python_default_ensure_ascii():
    assert canonical({'x':'я'})=='{"x":"\\u044f"}'

def test_independent_matrix_equals_actual_frozen_default_plan_counterparty(contract):
    # Counterparty is used only in this synthetic test; production auditor never
    # imports/calls the producer plan, parser, source AST hashes or validators.
    from plan_contract import GOLDEN
    config,launch,admission,data=f.launch_admission(contract)
    result=subprocess.run([sys.executable,'-B',str(GOLDEN/'controller.py'),'--out',config['out'],'--python',config['python']],capture_output=True,text=True,check=True)
    observed=json.loads(result.stdout.replace(str(GOLDEN),a.REMOTE_CODE).replace(str(Path(a.BASE).resolve()),a.BASE));expected=f.plan(config)
    assert observed==expected and len(observed['jobs'])==44 and sum(v['kind']=='benchmark'for v in observed['jobs'])==32

def test_auditor_sources_parse_python310_and_no_heavy_or_producer_imports():
    import ast
    for path in a.HERE.glob('*.py'):
        tree=ast.parse(path.read_text(),feature_version=(3,10))
        if path.name.startswith('test_')or path.name=='synthetic_fixtures.py':continue
        modules=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):modules.extend(v.name for v in n.names)
            elif isinstance(n,ast.ImportFrom):modules.append(n.module or'')
        assert all(not v.startswith(('torch','transformers','controller','pencil_bridge','science_validation'))for v in modules)
