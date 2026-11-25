"""Tiny local wrapper/path/ownership fixtures only; no real GPU or graph claims."""
from pathlib import Path
import ast,contextlib,hashlib,io,json,os,signal,sys,types
import pytest
import run as r
import common as c
import ownership as own
import runner as rr
import platform_probe as p
from local_archive import archive_payload
UUID='GPU-00000000-0000-0000-0000-000000000001'
def args(tmp_path,extra=()):
    return r.parser().parse_args(['--out',str(tmp_path/'results with spaces'),'--prepared-root',str(tmp_path/'input α'),'--validation-root',str(tmp_path/'frozen512 β'),'--coordination-dir',str(tmp_path/'coordination shared'),'--gpu-uuid',UUID,*extra])
def setup_args(tmp_path,extra=()):
    a=args(tmp_path,extra);r.validate_args(a);return a

def test_default_plan_is_complete44_generic_and_does_not_create_paths_or_import_torch(tmp_path):
    a=setup_args(tmp_path);before=set(sys.modules);plan=r.make_plan(a)
    assert not a.out.exists()and not a.coordination_dir.exists()and 'torch'not in set(sys.modules)-before
    assert plan['scientific_jobs']==44 and plan['throughput_jobs']==32 and plan['semantic_audits']==9
    jobs=plan['jobs'];assert len(jobs)==44 and sum(j['kind']=='audit'for j in jobs)==9
    assert {(j['arm'],j['batch'],j['workers'])for j in jobs if j['kind']=='benchmark'}=={(arm,b,w)for arm in r.science.ARMS for b in(128,512)for w in(0,1)}
    for j in jobs:
        assert j['argv'][-2:]==['--query-memory','disabled']and j['output'].startswith(str(a.out))
        assert j.get('mode')!='replay'
    assert all(j['name'].startswith('audit_')for j in jobs[4:12])
    assert len({j['name']for j in jobs})==44 and 'pod_uid'not in json.dumps(plan)and 'jupiter.yt'not in json.dumps(plan)

def test_lexical_venv_symlink_is_preserved_with_space_and_unicode_paths(tmp_path):
    venv=tmp_path/'environment α'/'bin';venv.mkdir(parents=True);python=venv/'python';python.symlink_to(sys.executable)
    a=setup_args(tmp_path,['--python',str(python)]);plan=r.make_plan(a)
    assert str(a.python)==str(python)and a.python.is_symlink()and all(j['argv'][0]==str(python)for j in plan['jobs'])
    assert str(a.prepared_root)+'/node_feat.npy'in plan['jobs'][2]['argv']
@pytest.mark.parametrize('change',['overlap','existing','guard','time','nan','uuid','coordination'])
def test_invalid_path_resource_and_gpu_admission_refused(tmp_path,change):
    a=args(tmp_path)
    if change=='overlap':a.out=a.prepared_root/'result'
    elif change=='existing':a.out.mkdir()
    elif change=='guard':a.memory_guard_gib=256
    elif change=='time':a.max_hours=5
    elif change=='nan':a.minimum_gpu_memory_gib=float('nan')
    elif change=='uuid':a.gpu_uuid='GPU-fake'
    else:a.coordination_dir=a.out/'locks'
    with pytest.raises((ValueError,FileNotFoundError)):r.validate_args(a)

def test_source_manifest_pin_and_member_tamper_with_small_fixture(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();file=source/'core.py';file.write_text('SYNTHETIC=1\n');c.save(source/'SOURCE_MANIFEST.json',{'files':[{'path':'core.py',**c.identity(file)}]});monkeypatch.setattr(c,'HERE',source);pin=c.sha(source/'SOURCE_MANIFEST.json');assert c.verify_sources(pin)['files']==1
    file.write_text('SYNTHETIC=2\n')
    with pytest.raises(ValueError):c.verify_sources(pin)

def test_tiny_synthetic19_plus2_inputs_have_full_hashes_and_actual_paths(tmp_path,monkeypatch):
    prepared=tmp_path/'prepared α';validation=tmp_path/'validation β';prepared.mkdir();validation.mkdir();rows=[]
    for i in range(19):
        path=prepared/f'payload_{i}.bin';path.write_bytes(bytes([i]));rows.append({'path':path.name,**c.identity(path)})
    payloads={}
    for name,key in(('valid_edge.npy','valid_positive'),('valid_edge_neg.npy','valid_negative')):
        path=validation/name;path.write_bytes(b'SYNTHETIC_ARRAY_NOT_NUMPY');payloads[key]={'sha256':c.sha(path)}
    strict={'variant':'strict_clean','quality_dataset':True,'files':rows,'clean_heldout_audits':{'train_csr':{s:{'forward':0,'reverse':0}for s in('valid','test')}}};c.save(prepared/'manifest.json',strict);original=c.read
    def read(path,*a,**k):
        if str(path).endswith('metadata/citation2_prepared_expected.json'):return {'files':rows}
        if str(path).endswith('bridge/input_contract.json'):return {'payloads':payloads}
        return original(path,*a,**k)
    monkeypatch.setattr(c,'read',read);receipt=c.verify_inputs(prepared,validation);assert receipt['numeric_payloads']==19 and len(receipt['actual_files'])==21
    assert c.verify_input_continuity(receipt)
    (prepared/'payload_0.bin').write_bytes(b'CHANGED')
    with pytest.raises(ValueError):c.verify_input_continuity(receipt)

def test_whole_local_archive_retains_nested_cache_and_failed_raw(tmp_path):
    out=tmp_path/'out';out.mkdir();(out/'nested/cache').mkdir(parents=True);(out/'nested/cache/evidence.txt').write_text('KEEP');(out/'cache').mkdir();(out/'cache/temporary').write_text('omit');(out/'failed.json').write_text('{"status":"failed"}')
    archive=tmp_path/'snapshot.tgz';receipt=archive_payload(out,archive,lambda:None)
    import tarfile
    with tarfile.open(archive)as tf:names=tf.getnames()
    assert receipt['whole_readback_verified']is True and 'nested/cache/evidence.txt'in names and 'failed.json'in names and 'cache/temporary'not in names

def fake_ownership(tmp_path,monkeypatch):
    a=setup_args(tmp_path,['--source-manifest-sha256','a'*64]);a.coordination_dir.mkdir();supervisor=types.SimpleNamespace(owned_groups=lambda owner:[],signal_confirmed=lambda groups,sig:None)
    memory=object();monkeypatch.setattr(own,'process_start_ticks',lambda:456);monkeypatch.setattr(own,'load',lambda path,name:supervisor if 'supervision'in str(path)else memory)
    return a,own.Ownership(a,rr.time.monotonic()+3600)

def test_shared_flock_and_exact_active_marker_release_only_after_quiescence(tmp_path,monkeypatch):
    a,resource=fake_ownership(tmp_path,monkeypatch)
    resource.acquire();marker=a.coordination_dir/'PUBLIC_SYSTEMS_ACTIVE.json';token=c.read(marker)
    assert token['role']=='pencil44'and token['start_ticks']==456 and marker.exists()
    resource.close();assert not marker.exists()and(a.coordination_dir/'PUBLIC_SYSTEMS_RUNNING.lock').exists()

def test_unknown_owned_tree_keeps_shared_marker_and_blocks_next_runner(tmp_path,monkeypatch):
    a,resource=fake_ownership(tmp_path,monkeypatch);resource.acquire();resource.runner.cleanup_owned=lambda seconds:False;resource.close();marker=a.coordination_dir/'PUBLIC_SYSTEMS_ACTIVE.json';assert marker.exists()
    second=own.Ownership(a,rr.time.monotonic()+3600)
    try:
        with pytest.raises(FileExistsError):second.acquire()
    finally:second.close()
    assert marker.exists()

def test_marker_replacement_is_not_unlinked(tmp_path,monkeypatch):
    a,resource=fake_ownership(tmp_path,monkeypatch);resource.acquire();c.save(a.coordination_dir/'PUBLIC_SYSTEMS_ACTIVE.json',{'foreign':'token'});resource.close();assert c.read(a.coordination_dir/'PUBLIC_SYSTEMS_ACTIVE.json')=={'foreign':'token'}

def test_partial_lock_initialization_failure_restores_handlers(tmp_path,monkeypatch):
    a,resource=fake_ownership(tmp_path,monkeypatch);old={s:signal.getsignal(s)for s in(signal.SIGINT,signal.SIGTERM)};monkeypatch.setattr(own,'load',lambda *a:(_ for _ in()).throw(RuntimeError('SYNTHETIC helper failure')))
    try:
        with pytest.raises(RuntimeError):resource.acquire()
    finally:resource.close()
    assert all(signal.getsignal(s)==handler for s,handler in old.items())and not(a.coordination_dir/'PUBLIC_SYSTEMS_ACTIVE.json').exists()

@pytest.mark.parametrize('gpu',[False,True])
def test_child_cuda_visibility_preserves_supplied_actual_uuid(tmp_path,monkeypatch,gpu):
    captured={};out=tmp_path/'out';out.mkdir()
    def popen(argv,**kw):captured.update(kw['env']);return types.SimpleNamespace(pid=123)
    monkeypatch.setattr(rr.subprocess,'Popen',popen)
    def supervised(argv,log,**kw):
        kw['popen'](argv,env={});Path(log).write_text('SYNTHETIC');return {'returncode':0,'owned_tree_quiescent':True,'orphan_cleanup':False,'timed_out':False,'timeout_seconds':kw['timeout']}
    supervisor=types.SimpleNamespace(run_owned=supervised);rnr=rr.Runner(out,rr.time.monotonic()+3600,supervisor=supervisor,memory=object(),gpu_visibility=UUID);rnr.size_and_disk=lambda:None
    rnr.run({'name':'tiny_synthetic','gpu':gpu,'argv':['NOT_EXECUTED']});assert captured['CUDA_VISIBLE_DEVICES']==(UUID if gpu else'')

def test_stop_during_local_archive_records_stop_and_no_new_child(tmp_path):
    rnr=rr.Runner(tmp_path,rr.time.monotonic()+3600,supervisor=object(),memory=object(),gpu_visibility=UUID);rnr.finalizing=True;rnr.request_stop();assert rnr.stopped and rnr.stop_deadline is not None
    with pytest.raises(ValueError):rnr.run({'name':'forbidden','argv':['NO'],'gpu':False},final=True)

def test_failure_retains_raw_and_whole_local_archive_without_performance_claim(tmp_path,monkeypatch):
    a=setup_args(tmp_path,['--source-manifest-sha256','a'*64]);plan=r.make_plan(a);called=[]
    class Fake:
        stopped=False;finalizing=False
        supervisor=types.SimpleNamespace(enable_subreaper=lambda:None,process_identity=lambda p:{'pid':p,'start_ticks':1})
        memory=types.SimpleNamespace(cgroup_memory=lambda:{'max_bytes':256*c.GIB})
        def cleanup_owned(self,*x):return True
        def remaining(self,**x):return 3600
        def check(self,**x):pass
        def run(self,job):
            called.append(job['name']);ex={'returncode':2 if job['name']=='tiny_capture'else 0,'timed_out':False,'owned_tree_quiescent':True,'orphan_cleanup':False,'guard_abort_reason':None};c.save(a.out/(job['name']+'.exit.json'),ex)
            if job['name']=='platform_before':c.save(a.out/'platform_before.json',{'SYNTHETIC':True})
            if job['name']=='tiny_capture':c.save(a.out/'tiny_capture.json',{'status':'failed','error':'SYNTHETIC failure'})
            return ex
    resource=types.SimpleNamespace(runner=Fake(),out_created=False);monkeypatch.setattr(r,'copy_bound_json',lambda *x:c.save(a.out/'SOURCES.json',{'SYNTHETIC':True}));monkeypatch.setattr(r,'verify_sources',lambda *x:None);monkeypatch.setattr(r,'verify_inputs',lambda *x:{'SYNTHETIC':True});monkeypatch.setattr(r,'verify_input_continuity',lambda *x:None);monkeypatch.setattr(r,'gpu_idle',lambda *x:None)
    code=r.execute_owned(a,plan,resource);assert code==2 and called==['platform_before','build_cpp','tiny_capture'];terminal=c.read(a.out/'SCIENTIFIC_TERMINAL.json');assert terminal['scientific_success']is False and terminal['completed_jobs']==0 and terminal['failure']['phase']=='tiny_capture'
    assert(a.out/'tiny_capture.json').exists()and a.out.with_name(a.out.name+'.tgz').exists()and c.read(a.out/'COMPLETE.json')['success']is False

def test_public_wrapper_sources_have_no_remote_calls_and_parse_python310():
    for path in c.HERE.glob('*.py'):
        ast.parse(path.read_text(),feature_version=(3,10))
        if path.name=='test_portable.py':continue
        assert 'yt.wrapper'not in path.read_text()and 'kubectl'not in path.read_text()and 'tele.corp'not in path.read_text()
