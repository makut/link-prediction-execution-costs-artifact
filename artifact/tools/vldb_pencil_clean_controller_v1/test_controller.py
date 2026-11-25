"""Local synthetic controller regressions; no cluster, YT or native GPU calls."""
from pathlib import Path
import argparse,copy,hashlib,io,json,signal,subprocess,sys,tarfile,time,types,uuid
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import contracts as c
import controller as control
import ownership as own
import publish_once as publication
from runner import Runner,StopRequested

def arguments(tmp_path):
    return argparse.Namespace(base=tmp_path,out=tmp_path/'out',prepared_root=tmp_path/'prepared',validation_root=tmp_path/'validation',
        python=Path(sys.executable),max_hours=4.,memory_guard_gib=96.,pod_memory_limit_gib=256.,target=c.TARGET,
        admission=None,admission_sha256=None,source_manifest_sha256='a'*64,source_packet_sha256='b'*64,source_packet=None)

def test_plan_all44_jobs_has_active_H100_contract_and_no_favorable_selection(tmp_path):
    args=arguments(tmp_path);control.validate_args(args);value=control.make_plan(args)
    assert value['scientific_jobs']==44 and len(value['jobs'])==44
    assert value['inherited_candidate_GPU_target_reference']=='one visible H100 per job'
    assert 'H10080Single'in value['protocol']['GPU_target']
    jobs=[j for j in value['jobs']if j['kind']=='benchmark'];assert len(jobs)==32
    assert {(j['arm'],j['batch'],j['workers'])for j in jobs}=={(a,b,w)for a in control.science.ARMS for b in(128,512)for w in(0,1)}
    assert all(j['argv'][-2:]==['--query-memory','disabled']for j in value['jobs'])
    assert all('--quality-plan'not in j['argv']for j in value['jobs'])
    assert sum(j['kind']=='audit'for j in value['jobs'])==9
    assert not args.out.exists()

def test_actual_offline_cli_imports_no_Torch_numpy_and_writes_no_outputs(tmp_path):
    out=tmp_path/'new';code="import controller,sys;assert controller.main(sys.argv[1:])==0;assert not any(n in sys.modules for n in ('torch','numpy','transformers'))"
    result=subprocess.run([sys.executable,'-B','-c',code,'--base',str(tmp_path),'--out',str(out)],
        cwd=HERE,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout);assert value['scientific_jobs']==44 and not out.exists()

def test_supplied_venv_interpreter_symlink_is_not_resolved_to_base_python(tmp_path):
    target=tmp_path/'base_python';target.write_text('synthetic interpreter')
    interpreter=tmp_path/'venv/bin/python';interpreter.parent.mkdir(parents=True);interpreter.symlink_to(target)
    args=arguments(tmp_path);args.python=interpreter;control.validate_args(args)
    assert args.python==interpreter.absolute()and args.python!=target.resolve()
    assert control.make_plan(args)['jobs'][0]['argv'][0]==str(interpreter.absolute())

def test_platform_CPU_observation_records_quota_without_claiming_dedicated_cores(tmp_path,monkeypatch):
    import platform_probe
    proc=tmp_path/'proc';group=tmp_path/'cgroup';proc.mkdir();group.mkdir()
    (proc/'cpuinfo').write_text('processor :0\nmodel name : Synthetic CPU\nprocessor :1\nmodel name : Synthetic CPU\n')
    (group/'cpu.max').write_text('1700000 100000\n')
    monkeypatch.setattr(platform_probe.os,'sched_getaffinity',lambda _:{0,1,2},raising=False)
    row=platform_probe.cpu_observation(proc,group)
    assert row['models_from_bounded_cpuinfo']==['Synthetic CPU']and row['cgroup_cpu_max']=='1700000 100000'
    assert row['cpu_affinity']==[0,1,2]and 'not dedicated'in row['scope']

@pytest.mark.parametrize('field,value',[('max_hours',float('nan')),('max_hours',4.01),('max_hours',0.1),('memory_guard_gib',180.),('pod_memory_limit_gib',512.)])
def test_finite_fixed_budgets_refused_before_output(tmp_path,field,value):
    args=arguments(tmp_path);setattr(args,field,value)
    with pytest.raises(ValueError):control.validate_args(args)
    assert not args.out.exists()

def root_admission(args):
    return {'kind':'PENCIL_CLEAN_CONSUMER_V1_ROOT_ADMISSION','authorized_to_execute':True,'fresh_owned_statefulset':True,
        'namespace':'m-utushkin','pod_name':c.POD,'statefulset_name':c.STS,'hostname':c.POD,
        'pod_uid':str(uuid.uuid4()),'statefulset_uid':str(uuid.uuid4()),'base':str(args.base.resolve()),
        'source_manifest_sha256':args.source_manifest_sha256,'source_packet_sha256':args.source_packet_sha256,
        'owned_tree_guard_gib':96,'pod_memory_limit_gib':256,'gpu_count':1,'gpu_name_contains':'H100',
        'cpu_request':27,'memory_request_gib':205,'observed_unix':100.,'image':'synthetic-root-observed-test-image'}

@pytest.mark.parametrize('mutation',['authorization','hostname','pod_uid','source','stale','GPU','guard'])
def test_fresh_root_admission_requires_actual_new_identity_and_explicit_authority(tmp_path,mutation):
    args=arguments(tmp_path);value=root_admission(args);path=tmp_path/'admission.json';path.write_text(json.dumps(value))
    assert c.verify_admission(path,c.sha(path),args,hostname=c.POD,now=101.)['pod_name']==c.POD
    if mutation=='authorization':value['authorized_to_execute']=False
    elif mutation=='hostname':value['hostname']='old-host'
    elif mutation=='pod_uid':value['pod_uid']='not-a-UUID'
    elif mutation=='source':value['source_manifest_sha256']='0'*64
    elif mutation=='stale':value['observed_unix']=-5000
    elif mutation=='GPU':value['gpu_name_contains']='A100'
    else:value['owned_tree_guard_gib']=180
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):c.verify_admission(path,c.sha(path),args,hostname=c.POD,now=101.)

def test_root_source_pin_and_entire_source_files_cannot_be_rebound(tmp_path,monkeypatch):
    (tmp_path/'source.py').write_text('source\n');row={'path':'source.py',**c.identity(tmp_path/'source.py')}
    manifest=tmp_path/'SOURCE_MANIFEST.json';manifest.write_text(json.dumps({'files':[row]}));pin=c.sha(manifest)
    monkeypatch.setattr(c,'HERE',tmp_path);assert c.verify_sources(pin)['files']==1
    (tmp_path/'source.py').write_text('changed\n')
    with pytest.raises(ValueError,match='source changed'):c.verify_sources(pin)
    row.update(c.identity(tmp_path/'source.py'));manifest.write_text(json.dumps({'files':[row]}))
    with pytest.raises(ValueError,match='root-pinned'):c.verify_sources(pin)

def test_admission_and_source_manifest_archive_copies_preserve_whole_original_JSON_bytes(tmp_path):
    original=tmp_path/'original.json';original.write_bytes(b'{ "z": 1, "a": 2 }\n')
    target=tmp_path/'archived.json';pin=c.sha(original)
    c.copy_bound_json(original,target,pin)
    assert target.read_bytes()==original.read_bytes()and c.sha(target)==pin
    with pytest.raises(FileExistsError):c.copy_bound_json(original,target,pin)
    original.write_bytes(b'{"changed":true}\n')
    with pytest.raises(ValueError,match='bound JSON bytes changed'):c.copy_bound_json(original,tmp_path/'new.json',pin)

def synthetic_public_inputs(tmp_path,monkeypatch):
    prepared=tmp_path/'prepared';prepared.mkdir();validation=tmp_path/'validation';validation.mkdir();rows=[]
    for number in range(19):
        name='numeric_'+str(number)+'.bin';(prepared/name).write_bytes(('synthetic numeric '+str(number)).encode())
        rows.append({'path':name,**c.identity(prepared/name)})
    contract={'payloads':{}}
    for name,key in(('valid_edge.npy','valid_positive'),('valid_edge_neg.npy','valid_negative')):
        (validation/name).write_bytes(('synthetic frozen '+name).encode());contract['payloads'][key]={'sha256':c.sha(validation/name)}
    strict={'variant':'strict_clean','quality_dataset':True,'files':rows,
        'clean_heldout_audits':{'train_csr':{split:{'forward':0,'reverse':0}for split in('valid','test')}}}
    (prepared/'manifest.json').write_text(json.dumps(strict));original=c.read
    def read(path,*args):
        if str(path).endswith('metadata/citation2_prepared_expected.json'):return {'files':rows}
        if str(path).endswith('input_contract.json'):return contract
        return original(path,*args)
    monkeypatch.setattr(c,'read',read);return prepared,validation

@pytest.mark.parametrize('mutation',['unselected_numeric','cohort','strict_manifest'])
def test_whole19_numeric_and_exact_frozen_cohort_identity_not_just_used_prefix(tmp_path,monkeypatch,mutation):
    prepared,validation=synthetic_public_inputs(tmp_path,monkeypatch)
    receipt=c.verify_inputs(prepared,validation);assert receipt['numeric_payloads']==19 and len(receipt['actual_files'])==21
    assert c.verify_input_continuity(receipt)
    if mutation=='unselected_numeric':(prepared/'numeric_18.bin').write_bytes(b'changed unselected bytes')
    elif mutation=='cohort':(validation/'valid_edge_neg.npy').write_bytes(b'changed frozen candidate rows')
    else:
        value=json.loads((prepared/'manifest.json').read_text());value['clean_heldout_audits']['train_csr']['test']['reverse']=1
        (prepared/'manifest.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):c.verify_inputs(prepared,validation)

def test_whole_archive_includes_nested_cache_named_science_but_excludes_only_top_cache(tmp_path):
    out=tmp_path/'out';(out/'nested/cache').mkdir(parents=True);(out/'cache/numba').mkdir(parents=True)
    (out/'raw.json').write_text('{"actual":"whole"}');(out/'nested/cache/science.npy').write_bytes(b'whole-science-bytes')
    (out/'cache/numba/derived-cache.bin').write_bytes(b'not-a-science-input');archive=tmp_path/'out.tgz'
    result=control.archive_payload(out,archive,lambda:None);assert result['whole_readback_verified']is True
    with tarfile.open(archive,'r:gz')as reader:
        assert set(reader.getnames())=={'raw.json','nested/cache/science.npy','SHA256_MANIFEST.json'}
        assert reader.extractfile('nested/cache/science.npy').read()==b'whole-science-bytes'
    with pytest.raises(ValueError,match='existing coarse'):control.archive_payload(out,archive,lambda:None)

def test_archive_never_follows_output_symlinks_or_truncates_source_files(tmp_path):
    out=tmp_path/'out';out.mkdir();(tmp_path/'external').write_bytes(b'outside')
    (out/'link').symlink_to(tmp_path/'external')
    with pytest.raises(ValueError,match='symlink'):control.archive_payload(out,tmp_path/'new.tgz',lambda:None)
    assert not(tmp_path/'new.tgz').exists()

class FakeClient:
    def __init__(self):self.exists_flag=False;self.data=b'';self.attribute=None;self.calls=[];self.bad_data=False;self.bad_attribute=False
    def exists(self,path):self.calls.append(('exists',path));return self.exists_flag
    def create(self,kind,path,**kw):
        assert not self.exists_flag and kw=={'recursive':False,'ignore_existing':False};self.exists_flag=True;self.calls.append(('create',path))
    def write_file(self,path,stream):self.calls.append(('write',path));self.data=stream.read()
    def read_file(self,path):self.calls.append(('read',path));return io.BytesIO(self.data+b'corrupt'if self.bad_data else self.data)
    def set(self,path,value):self.calls.append(('set',path));self.attribute=copy.deepcopy(value)
    def get(self,path):
        self.calls.append(('get',path));value=copy.deepcopy(self.attribute)
        if self.bad_attribute:value['canonical_receipt_json']+='corrupt'
        return value

def publication_context(archive):return {'target':c.TARGET,'archive':c.identity(archive),'outer_final_pid_exit':'UNKNOWN',
    'scientific_terminal_sha256':'a'*64,'scientific_success':False,'completed_jobs':3,'planned_jobs':44,
    'source_manifest_sha256':'b'*64,'source_packet_sha256':'c'*64,'launch_sha256':'d'*64,'all_planned_science_job_exits_clean':False}

def test_one_file_one_attribute_whole_readback_preserves_scientific_failure_and_unknown_outer_exit(tmp_path):
    archive=tmp_path/'coarse.tgz';archive.write_bytes(b'whole actual coarse bytes');client=FakeClient()
    result=publication.publish(client,archive,publication_context(archive))
    assert [name for name,_ in client.calls]==['exists','create','write','read','set','get']
    assert result['remote_file_nodes_created']==result['file_writes']==result['attribute_writes']==1
    value=json.loads(result['attribute']['canonical_receipt_json'])
    assert value['scientific_success']is False and value['outer_final_pid_exit']=='UNKNOWN'
    assert 'pre-publication'in value['scope']and result['attribute_is_inside_archive_hash']is False
    assert len(publication.canonical(result['attribute']).encode())<=32<<10
    with pytest.raises(ValueError,match='already exists'):publication.publish(client,archive,publication_context(archive))
    assert sum(name=='write'for name,_ in client.calls)==1

@pytest.mark.parametrize('mutation',['data','attribute','target','oversize'])
def test_persistence_corruption_or_unsupported_destination_never_reports_success(tmp_path,mutation):
    archive=tmp_path/'coarse.tgz';archive.write_bytes(b'evidence');client=FakeClient();context=publication_context(archive)
    if mutation=='data':client.bad_data=True
    elif mutation=='attribute':client.bad_attribute=True
    elif mutation=='target':context['target']='//unauthorized/path'
    else:context['oversized']='x'*(25<<10)
    with pytest.raises(ValueError):publication.publish(client,archive,context)
    assert sum(name=='create'for name,_ in client.calls)<=1 and sum(name=='set'for name,_ in client.calls)<=1
    if mutation in('target','oversize'):assert client.calls==[]

def test_attribute_readback_can_normalize_only_identical_UTF8_YSON_strings(tmp_path):
    archive=tmp_path/'coarse.tgz';archive.write_bytes(b'whole evidence');client=FakeClient();original=client.get
    def get(path):return {k.encode():v.encode()if isinstance(v,str)else v for k,v in original(path).items()}
    client.get=get;result=publication.publish(client,archive,publication_context(archive))
    assert result['status']=='ONE_COARSE_FILE_AND_ATTRIBUTE_READBACK_VERIFIED'

class FakeSupervisor:
    def __init__(self):self.groups={};self.signals=[];self.interrupt=False
    def owned_groups(self,owner):return self.groups
    def signal_confirmed(self,groups,sig):self.signals.append(sig);return []
    def run_owned(self,argv,log,**kwargs):
        if self.interrupt:raise StopRequested('synthetic stop while child phase active')
        kwargs['groups_reader'](123)
        return {'returncode':0,'timed_out':False,'orphan_cleanup':False,'owned_tree_quiescent':True,'timeout_seconds':kwargs['timeout']}
class FakeMemory:
    def __init__(self):self.limits=[]
    def sample_tree(self,owner):return {'tree_pss_bytes':1,'pss_complete':True,'membership_stable':True,'max_process_rss_bytes':1}
    def cgroup_memory(self):return {'max_bytes':256*c.GIB,'current_bytes':1}
    def budget_decision(self,sample,limit,charged):self.limits.append(limit);return {'abort_reason':None,'memory_budget_bytes':limit}

def test_runner_uses_explicit96_guard_and_keeps_genuine_exit_receipt(tmp_path):
    supervisor=FakeSupervisor();memory=FakeMemory();runner=Runner(tmp_path,time.monotonic()+4000,supervisor=supervisor,memory=memory)
    result=runner.run({'name':'synthetic_phase','argv':['never-executed'],'gpu':False})
    assert c.clean(result)and memory.limits==[96*c.GIB]
    assert c.read(tmp_path/'synthetic_phase.exit.json')['memory_budget_gib']==96

def test_stop_prevents_next_science_and_cleans_owned_children_without_inventing_exit0(tmp_path):
    supervisor=FakeSupervisor();supervisor.interrupt=True;runner=Runner(tmp_path,time.monotonic()+4000,supervisor=supervisor,memory=FakeMemory())
    with pytest.raises(StopRequested):runner.run({'name':'interrupted_phase','argv':['not-executed'],'gpu':False})
    receipt=c.read(tmp_path/'interrupted_phase.exit.json');assert receipt['returncode']is None and receipt['interrupted']is True
    assert signal.SIGTERM in supervisor.signals and signal.SIGKILL in supervisor.signals
    with pytest.raises(StopRequested):runner.request_stop(signal.SIGTERM,None)
    assert runner.stop_deadline-time.monotonic()<=110
    with pytest.raises(StopRequested):runner.run({'name':'science_after_stop','argv':['forbidden']})
    with pytest.raises(ValueError,match='only bounded persistence'):runner.run({'name':'science_after_stop','argv':['forbidden']},final=True)

def test_nested_cleanup_waits_are_clamped_to_remaining_stop_deadline_but_still_send_KILL(tmp_path):
    now=[0.];supervisor=FakeSupervisor();supervisor.groups={7:[{'pid':8}]}
    runner=Runner(tmp_path,100.,supervisor=supervisor,memory=FakeMemory(),clock=lambda:now[0],sleep=lambda delay:now.__setitem__(0,now[0]+delay))
    runner.stopped=True;runner.stop_deadline=.25
    assert runner.cleanup_owned(10)is False and now[0]<=.25
    assert signal.SIGKILL in supervisor.signals
    before=now[0];assert runner.cleanup_owned(10)is False and now[0]==before
    assert supervisor.signals.count(signal.SIGKILL)==2

def test_first_stop_during_local_finalization_records_deadline_without_repacking_or_science(tmp_path):
    runner=Runner(tmp_path,time.monotonic()+4000,supervisor=FakeSupervisor(),memory=FakeMemory())
    runner.finalizing=True
    assert runner.request_stop(signal.SIGTERM,None)is None
    assert runner.stopped is True and runner.stop_deadline-time.monotonic()<=110
    with pytest.raises(StopRequested):runner.run({'name':'no_science','argv':['forbidden']})
    with pytest.raises(StopRequested):runner.request_stop(signal.SIGTERM,None)

def test_first_stop_during_active_publisher_interrupts_it_without_remote_retry(tmp_path):
    runner=Runner(tmp_path,time.monotonic()+4000,supervisor=FakeSupervisor(),memory=FakeMemory())
    runner.finalizing=True;runner.active='publish_once'
    with pytest.raises(StopRequested):runner.request_stop(signal.SIGTERM,None)
    assert runner.stopped is True

def test_partial_initialization_always_releases_only_its_own_quiescent_lock(tmp_path,monkeypatch):
    args=arguments(tmp_path);admission=root_admission(args);supervisor=FakeSupervisor()
    def fail_memory(path,name):
        if 'owned_supervision'in str(path):return supervisor
        raise RuntimeError('intentional memory-helper import failure')
    monkeypatch.setattr(own,'load',fail_memory);resource=own.Ownership(args,admission,time.monotonic()+4000)
    try:
        with pytest.raises(RuntimeError,match='import failure'):resource.acquire()
        assert resource.lock_path.exists()
    finally:resource.close()
    assert resource.quiescent is True and not resource.lock_path.exists()and resource.lock.closed

def test_stop_during_helper_import_has_handler_before_lease_and_reaches_cleanup(tmp_path,monkeypatch):
    args=arguments(tmp_path);resource=own.Ownership(args,root_admission(args),time.monotonic()+4000);events=[]
    monkeypatch.setattr(own.signal,'signal',lambda sig,handler:events.append((sig,handler))or 'previous')
    def stop_during_load(*args):resource.request_stop(signal.SIGTERM,None)
    monkeypatch.setattr(own,'load',stop_during_load)
    try:
        with pytest.raises(StopRequested,match='initialization'):resource.acquire()
        assert resource.stopped is True and len(resource.handlers)==2
    finally:resource.close()
    assert not resource.lock_path.exists()and sum(value=='previous'for _,value in events)==2

def test_unquiescent_or_replaced_lock_is_retained_and_handlers_restore(tmp_path,monkeypatch):
    args=arguments(tmp_path);admission=root_admission(args);supervisor=FakeSupervisor();memory=FakeMemory();events=[]
    monkeypatch.setattr(own,'load',lambda path,name:supervisor if 'owned_supervision'in str(path)else memory)
    monkeypatch.setattr(own.signal,'signal',lambda sig,handler:events.append((sig,handler))or 'previous')
    resource=own.Ownership(args,admission,time.monotonic()+4000);resource.acquire()
    resource.runner.cleanup_owned=lambda *_:False
    resource.close()
    assert resource.lock_path.exists()and resource.lock.closed and resource.quiescent is False
    assert sum(handler=='previous'for _,handler in events)==2
    with pytest.raises(FileExistsError):own.Ownership(args,admission,time.monotonic()+4000).acquire()
    resource.lock_path.unlink()  # Synthetic test cleanup, never production stale-lock removal.

def test_execute_outer_finally_closes_ownership_when_acquire_partially_fails(tmp_path,monkeypatch):
    args=arguments(tmp_path);args.admission=tmp_path/'admission.json';args.admission_sha256='f'*64
    args.source_packet=tmp_path/'source.tgz';args.source_packet.write_bytes(b'synthetic package');args.source_packet_sha256=c.sha(args.source_packet)
    events=[]
    class FailedOwnership:
        runner=None;quiescent=None
        def __init__(self,*args):pass
        def acquire(self):events.append('acquire');raise OSError('initialization failed after lease acquisition')
        def record_initialization_failure(self,error):events.append('failure')
        def close(self):events.append('close');self.quiescent=True
    monkeypatch.setattr(control.sys,'platform','linux');monkeypatch.setattr(control,'Ownership',FailedOwnership)
    monkeypatch.setattr(control,'verify_sources',lambda _:None);monkeypatch.setattr(control,'verify_admission',lambda *a:root_admission(args))
    assert control.execute(args,{'jobs':[]})==2
    assert events==['acquire','failure','close']

def test_stop_arriving_in_final_ownership_close_cannot_return_exit0(tmp_path,monkeypatch):
    args=arguments(tmp_path);args.admission=tmp_path/'admission.json';args.admission_sha256='f'*64
    args.source_packet=tmp_path/'source.tgz';args.source_packet.write_bytes(b'synthetic package');args.source_packet_sha256=c.sha(args.source_packet)
    class LateStopOwnership:
        stopped=False;runner=types.SimpleNamespace(stopped=False);quiescent=None
        def __init__(self,*args):pass
        def acquire(self):pass
        def record_initialization_failure(self,error):raise AssertionError('unexpected failure')
        def close(self):self.stopped=True;self.runner.stopped=True;self.quiescent=True
    monkeypatch.setattr(control.sys,'platform','linux');monkeypatch.setattr(control,'Ownership',LateStopOwnership)
    monkeypatch.setattr(control,'verify_sources',lambda _:None);monkeypatch.setattr(control,'verify_admission',lambda *a:root_admission(args))
    monkeypatch.setattr(control,'_execute_owned',lambda *a:0)
    assert control.execute(args,{'jobs':[]})==143

def score_jobs(tmp_path):
    jobs=[];fingerprint={'kind':'tensor','shape':[1024],'sha256':'e'*64}
    for batch in(128,512):
        for workers in(0,1):
            for arm in control.science.ARMS:
                path=tmp_path/f'{batch}_{workers}_{arm}.json';path.write_text(json.dumps({'result':{
                    'model':{'checkpoint':None,'initialization_seed':42,'tiny_fixture_model':False},
                    'sweeps':[{'scores':fingerprint}for _ in range(4)]}}))
                jobs.append({'kind':'benchmark','precision':'bf16','batch':batch,'workers':workers,'mode':'online','arm':arm,'output':str(path)})
    return jobs

@pytest.mark.parametrize('mutation',['none','missing_arm','different_score'])
def test_all32_original_full_score_fingerprints_required_in_four_unpooled_groups(tmp_path,mutation):
    jobs=score_jobs(tmp_path)
    if mutation=='none':assert len(control.science.crossjob_scores(jobs)['groups'])==4
    elif mutation=='missing_arm':
        with pytest.raises(ValueError,match='arm set'):control.science.crossjob_scores(jobs[:-1])
    else:
        path=Path(jobs[-1]['output']);raw=json.loads(path.read_text());raw['result']['sweeps'][3]['scores']['sha256']='0'*64;path.write_text(json.dumps(raw))
        with pytest.raises(ValueError,match='fingerprints differ'):control.science.crossjob_scores(jobs)

@pytest.mark.parametrize('mutation',['none','fakezero','wrongmode','library','policy'])
def test_new_disabled_memory_schema_and_actual_probe_bindings_are_mandatory(mutation):
    names=('bridge_core.py','pencil_runtime.py','pencil_bridge.py','graph_only_transformers_v2.py',
           'upstream_manifest.json','input_contract.json','bert_config/config.json','bert_config/source.json')
    policy={'source':'synthetic exact probe policy'};packages={n:'same'for n in('numpy','numba','scipy','transformers','wandb')}
    admitted={'runtime_compatibility':policy,'torch':'targetTorch','torch_sparse_version':'targetSparse','pyg':'targetPyG',
        'saint_wrapper':{'sha256':'wrapper','git_blob':'blob'},'torch_sparse_binaries':{'native.so':{'sha256':'library'}},'packages':packages}
    raw={'source_files':{n:c.sha(c.BRIDGE/n)for n in names},'configuration':{'query_memory':'disabled'},
        'command':'benchmark','runtime_compatibility':dict(policy),'audit_environment':{'runtime_compatibility':dict(policy)},
        'native_setup':{'torch':'targetTorch','torch_sparse':'targetSparse','pyg':'targetPyG','wrapper_sha256':'wrapper','wrapper_git_blob':'blob',
            'extension_binaries':[{'name':'native.so','sha256':'library'}],'software_versions':dict(packages)},
        'result':{'memory_measurement_policy':{'query_memory':'disabled','per_query_memory_samples_enabled':False},'sweeps':[]}}
    for _ in range(4):raw['result']['sweeps'].append({'memory_measurement_policy':dict(raw['result']['memory_measurement_policy']),
        'worker_memory':[{'pid':7,'max_pss_bytes':None,'maxrss_bytes':None,'memory_measurement':'disabled_unobserved'}],
        'parent_memory_before_sweep':{'maxrss_bytes':100},'parent_memory_after_sweep':{'maxrss_bytes':110}})
    if mutation=='none':control.bind_driver(raw,admitted);return
    if mutation=='fakezero':raw['result']['sweeps'][1]['worker_memory'][0]['max_pss_bytes']=0
    elif mutation=='wrongmode':raw['configuration']['query_memory']='inline_smaps'
    elif mutation=='library':raw['native_setup']['extension_binaries'][0]['sha256']='changed'
    else:raw['runtime_compatibility']['source']='different'
    with pytest.raises(ValueError):control.bind_driver(raw,admitted)
