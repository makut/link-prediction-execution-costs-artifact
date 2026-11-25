import contextlib,copy,io,json,os,subprocess,sys,tarfile
from pathlib import Path
from types import SimpleNamespace
import pytest
import pencil_queue_v3 as q

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n');return path
def policy():return {'version':'graph_only_transformers_optional_vision_v2','transformers_version':'4.46.2',
    'import_utils_sha256':'9d7f63effbd7c8dcfe69e7f34cae0d164db9cca2d5f4d887405ae0f3e2ba455a',
    'torchvision_available_for_graph_worker':False,'torchvision_imported':False,'fake_operator_registered':False,
    'base_package_modified':False,'upstream_source_modified':False,'policy_source_sha256':q.sha(q.BRIDGE/'graph_only_transformers_v2.py')}
def clean():return {'returncode':0,'timed_out':False,'owned_tree_quiescent':True,'orphan_cleanup':False,'guard_abort_reason':None}

def test_exact_reviewed_pins_and_untouched_bridge():
    for name,pin in q.PINS.items():assert q.sha(q.HERE/name)==pin
    manifest=q.BRIDGE/'SOURCE_MANIFEST.json';assert q.sha(manifest)==q.BRIDGE_MANIFEST_SHA
    for item in q.read(manifest)['files']:
        assert q.info(q.BRIDGE/item['path'])=={'bytes':item['bytes'],'sha256':item['sha256']}

def test_protocol_exact_48_jobs_8_audits_32_online_4_replay(tmp_path):
    jobs=q.make_jobs(tmp_path,tmp_path/'env/bin/python');assert len(jobs)==48
    audits=[j for j in jobs if j['kind']=='audit'and j['name']!='tiny_audit']
    assert {(j['precision'],j['batch'],j['workers'])for j in audits}=={(p,b,w)for p in('fp32','bf16')for b in(128,512)for w in(0,1)}
    bench=[j for j in jobs if j['kind']=='benchmark'];assert len(bench)==36
    assert sum(j['mode']=='online'for j in bench)==32 and sum(j['mode']=='replay'for j in bench)==4
    assert all(j['precision']=='bf16'and j['gpu']and j['timeout_seconds']==1800 for j in bench)
    for job in bench:
        argv=job['argv'];assert argv[argv.index('--warmup')+1]=='1'and argv[argv.index('--repeats')+1]=='3'
        assert argv[argv.index('--batch-size')+1]==str(job['batch'])and argv[argv.index('--num-workers')+1]==str(job['workers'])
        assert '--tiny-model'not in argv and 'train'not in argv
    assert [j['name']for j in jobs[:4]]==['tiny_capture','tiny_audit','real_preflight','real_capture']

def test_original_wheel_archive_prefix_unpacks_into_fresh_v3_destination_without_renaming_archive_members(tmp_path,monkeypatch):
    archive=tmp_path/'wheels.tgz';files=[];payloads={}
    for i in range(35):
        name=f'fixture_{i}.whl';data=f'wheel{i}'.encode();payloads[name]=data
        files.append({'path':name,'bytes':len(data),'sha256':q.hashlib.sha256(data).hexdigest()})
    payloads['WHEEL_MANIFEST.json']=json.dumps({'files':files}).encode()
    with tarfile.open(archive,'w:gz')as tf:
        for name,data in payloads.items():
            member=tarfile.TarInfo(q.WHEEL_MEMBER_PREFIX+'/'+name);member.size=len(data);tf.addfile(member,io.BytesIO(data))
    monkeypatch.setattr(q,'WHEEL_ARCHIVE_SHA',q.sha(archive));destination=tmp_path/(q.WHEEL_MEMBER_PREFIX+'_v3')
    result=q.unpack_wheels(archive,destination)
    assert result['files']==35 and all((destination/name).read_bytes()==data for name,data in payloads.items())
    assert not(tmp_path/q.WHEEL_MEMBER_PREFIX).exists()

def benchmark_fixtures(root):
    jobs=[j for j in q.make_jobs(root,root/'env/bin/python')if j['kind']=='benchmark']
    fp={'kind':'tensor','shape':[1024,1],'dtype':'torch.float32','stride':[1,1],'sha256':'literal'}
    for job in jobs:
        config={'batch_size':job['batch'],'num_workers':job['workers'],'precision':'bf16','device':'cuda',
            'pipeline_mode':job['mode'],'arm':job['arm'],'model_seed':42,'tiny_model':False,'checkpoint':None,
            'eval_seed':20261126,'max_logit_abs':0}
        put(Path(job['output']),{'status':'complete','command':'benchmark','configuration':config,
            'runtime_compatibility':policy(),'audit_environment':{'runtime_compatibility':policy()},
            'result':{'model':{'checkpoint':None,'initialization_seed':42,'tiny_fixture_model':False,'settings':{'hidden_size':512}},
            'sweeps':[{'scores':fp,'queries':1024,
            'elapsed_seconds':1,'warmup':i==0}for i in range(4)]}})
    return jobs

def test_all_1024_score_hashes_cross_arm_and_repeat_gated_with_separate_shapes(tmp_path):
    jobs=benchmark_fixtures(tmp_path);result=q.crossjob_scores(jobs)
    assert result['status']=='pass'and len(result['groups'])==5
    job=jobs[1];raw=q.read(job['output']);raw['result']['sweeps'][2]['scores']['sha256']='different';put(Path(job['output']),raw)
    with pytest.raises(RuntimeError,match='differ across arms/repeats'):q.crossjob_scores(jobs)

def test_missing_arm_or_incomplete_score_cohort_rejected(tmp_path):
    jobs=benchmark_fixtures(tmp_path)
    with pytest.raises(RuntimeError,match='arm set'):q.crossjob_scores(jobs[:-1])
    raw=q.read(jobs[0]['output']);raw['result']['sweeps'][0]['scores']['shape'][0]=1023;put(Path(jobs[0]['output']),raw)
    with pytest.raises(RuntimeError,match='complete1024'):q.crossjob_scores(jobs)

def test_score_groups_do_not_pool_shapes_but_bind_model_configuration(tmp_path):
    jobs=benchmark_fixtures(tmp_path)
    for job in jobs:
        if job['batch']==512:
            raw=q.read(job['output'])
            for sweep in raw['result']['sweeps']:sweep['scores']['sha256']='different-for-b512'
            put(Path(job['output']),raw)
    assert q.crossjob_scores(jobs)['status']=='pass'
    job=jobs[1];raw=q.read(job['output']);raw['result']['model']['settings']['hidden_size']=256;put(Path(job['output']),raw)
    with pytest.raises(RuntimeError,match='model configuration differs'):q.crossjob_scores(jobs)

def test_benchmark_requires_four_clean_sweeps_and_exact1024(tmp_path):
    job=benchmark_fixtures(tmp_path)[0];q.validate_job(job,clean())
    raw=q.read(job['output']);raw['result']['sweeps'][3]['queries']=1023;put(Path(job['output']),raw)
    with pytest.raises(RuntimeError,match='complete1024'):q.validate_job(job,clean())
    bad=clean();bad['orphan_cleanup']=True
    with pytest.raises(RuntimeError,match='not quiescent'):q.validate_job(job,bad)

def test_archive_keeps_whole_verbose_raw_and_trace_no_sources_wheels_or_graph(tmp_path):
    out=tmp_path/'out';put(out/'SCIENTIFIC_TERMINAL.json',{'success':False});put(out/'verbose_batches.json',{'all_batches':['literal']*50})
    (out/'real_trace').mkdir();(out/'real_trace/nodes.npy').write_bytes(b'literal bounded trace fixture')
    target=tmp_path/'final.tgz';receipt=q.pack_once(out,target)
    with tarfile.open(target)as tf:
        m=json.load(tf.extractfile('SHA256_MANIFEST.json'));assert receipt['files']==len(m['files'])==3
        for rel,item in m['files'].items():
            data=tf.extractfile(rel).read()
            assert len(data)==item['bytes']and q.hashlib.sha256(data).hexdigest()==item['sha256']
    (out/'source.py').write_text('# forbidden')
    with pytest.raises(RuntimeError,match='whole result'):q.pack_once(out,tmp_path/'bad.tgz')

def test_archive_never_truncates_oversized_raw_or_follows_symlinks(tmp_path):
    out=tmp_path/'out';out.mkdir();raw=out/'verbose.json'
    with raw.open('wb')as f:f.truncate(32*q.MIB+1)
    with pytest.raises(RuntimeError,match='never truncate'):q.pack_once(out,tmp_path/'bad.tgz')
    assert raw.stat().st_size==32*q.MIB+1
    raw.unlink();(out/'alias.json').symlink_to(tmp_path/'missing')
    with pytest.raises(RuntimeError,match='symlink'):q.pack_once(out,tmp_path/'bad2.tgz')

class FakeYT:
    def __init__(self,existing=False,corrupt=False):self.existing,self.corrupt,self.created,self.payload=existing,corrupt,[],b''
    def exists(self,path):return self.existing
    def Transaction(self,**kw):return contextlib.nullcontext()
    def create(self,kind,path):self.created.append((kind,path))
    def write_file(self,path,stream):self.payload=stream.read()
    def read_file(self,path):return io.BytesIO(self.payload+(b'corrupt'if self.corrupt else b''))

def test_exact_one_final_yt_archive_readback_and_no_overwrite(tmp_path):
    p=tmp_path/'final.tgz';p.write_bytes(b'whole fixture archive');client=FakeYT()
    assert q.upload_once(p,q.YT_TARGET,client)['readback_verified']and client.created==[('file',q.YT_TARGET)]
    for other in(FakeYT(existing=True),FakeYT(corrupt=True)):
        with pytest.raises(RuntimeError):q.upload_once(p,q.YT_TARGET,other)
    with pytest.raises(RuntimeError,match='exact one'):q.upload_once(p,q.YT_TARGET+'.other',FakeYT())

def test_external_reviewed_memory_policy_abort_and_gpu_environment(tmp_path,monkeypatch):
    calls=[];observed={}
    def fake_popen(*args,**kw):observed.update(kw['env']);return None
    monkeypatch.setattr(q.subprocess,'Popen',fake_popen)
    def run_owned(argv,log,**kw):
        kw['groups_reader'](os.getpid())
        kw['popen'](['literal nonexecuted fixture'],env={})
        return clean()
    supervisor=SimpleNamespace(owned_groups=lambda pid:{},run_owned=run_owned)
    memory=SimpleNamespace(sample_tree=lambda pid:{},cgroup_memory=lambda:{},
        budget_decision=lambda sample,limit,charged:calls.append(limit)or{'abort_reason':'literal memory fixture'})
    result=q.guarded_run(supervisor,memory,['fixture'],tmp_path/'job.log',tmp_path,1800,gpu=True)
    assert calls==[180*q.GIB]and result['guard_abort_reason']=='literal memory fixture'and result['exactly_one_visible_gpu']
    assert observed['CUDA_VISIBLE_DEVICES']=='0'and observed['HF_HUB_OFFLINE']=='1'and observed['WANDB_MODE']=='disabled'

def test_three_guard_sampling_failures_fail_closed(tmp_path,monkeypatch):
    tick=[0]
    def clock():tick[0]+=3;return tick[0]
    monkeypatch.setattr(q.time,'monotonic',clock)
    def run_owned(argv,log,**kw):
        for i in range(3):kw['groups_reader'](os.getpid())
        return clean()
    def broken(pid):raise OSError('fixture')
    supervisor=SimpleNamespace(owned_groups=lambda pid:{},run_owned=run_owned)
    result=q.guarded_run(supervisor,SimpleNamespace(sample_tree=broken),['fixture'],tmp_path/'job.log',tmp_path,30)
    assert result['guard_abort_reason']=='three consecutive external V3 sampling failures'

def test_preflight_does_not_create_env_results_or_launch(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(q,'verify_sources',lambda base,**kw:{})
    assert q.main(['--base',str(tmp_path),'--recovery-admission-sha256','a'*64])==0
    assert json.loads(capsys.readouterr().out)['total_jobs']==48 and not(tmp_path/'results').exists()
    for args in(['--max-hours','13'],['--admission-hours','nan'],['--max-hours','8','--admission-hours','8']):
        with pytest.raises(RuntimeError,match='TOTAL'):q.main(args)

def mocked_main(tmp_path,monkeypatch,failure=None):
    base=tmp_path/'base';base.mkdir();L=q.layout(base);monkeypatch.setattr(q,'BASE',base)
    monkeypatch.setattr(q,'verify_sources',lambda b,**kw:{})
    monkeypatch.setattr(q,'unpack_wheels',lambda *a:{'fixture':True})
    monkeypatch.setattr(q,'cleanup_owned',lambda s:True)
    def route(venv,before):
        p=venv/'lib/python3.10/site-packages/pencil_base_site_v3.pth';p.parent.mkdir(parents=True);p.write_text('/opt/python/lib/python3.10/site-packages\n')
        (venv/'pyvenv.cfg').write_text('include-system-site-packages = true\n');calls.append('write_plain_pth')
        return {'path':str(p),**q.info(p),'base_paths':['/opt/python/lib/python3.10/site-packages']}
    monkeypatch.setattr(q,'add_base_site',route)
    monkeypatch.setattr(q,'verify_environment_binding',lambda *a:{'status':'pass'})
    monkeypatch.setattr(q,'bind_environment',lambda *a:{})
    monkeypatch.setattr(q.signal,'signal',lambda *a:None)
    def acquire(path,token):
        with path.open('x')as f:json.dump(token,f)
    def release(path,token,quiescent):
        q.require(quiescent and q.read(path)==token,'foreign lock preserved');path.unlink()
    gate=SimpleNamespace(acquire_lock=acquire,release_lock=release)
    supervisor=SimpleNamespace(enable_subreaper=lambda:None,process_identity=lambda pid:{'start_ticks':123},owned_groups=lambda pid:{})
    monkeypatch.setattr(q,'load_module',lambda path,name:gate if 'main_gate'in name else supervisor)
    monkeypatch.setattr(q,'admission',lambda *a,**k:{'admitted':True,'bindings':{}})
    jobs=[{'name':'fixture_'+str(i),'kind':'preflight','argv':['fixture','--output',str(L['out']/('fixture_'+str(i)+'.json'))],
        'output':str(L['out']/('fixture_'+str(i)+'.json')),'gpu':False}for i in range(48)]
    monkeypatch.setattr(q,'make_jobs',lambda *a:jobs);monkeypatch.setattr(q,'validate_job',lambda *a:None)
    def scores(jobs):
        if failure=='scores':raise RuntimeError('measured model score mismatch fixture')
        return {'status':'pass'}
    monkeypatch.setattr(q,'crossjob_scores',scores)
    calls=[]
    def run(supervisor,memory,argv,log,cwd,timeout,**kw):
        lock=base/'results/ADDITIONAL_TIMING_RUNNING.lock';assert lock.exists();assert 0<timeout<=1800
        assert(L['out']/'plan.json').exists();calls.append(log.stem)
        log.write_text('literal fixture phase\n')
        if '--persist-worker'in argv:
            assert q.read(L['out']/'SCIENTIFIC_TERMINAL.json')['controller_jobs_finished']is True
            pointer={'yt_path':q.YT_TARGET,**q.info(L['archive']),'readback_verified':True,'new_YT_file_nodes':1}
            if failure=='pointer':pointer['sha256']='incorrect'
            put(L['out']/'archive_pointer.json',pointer)
            if failure=='lock':put(lock,{'foreign':True})
            if failure=='persist_exit':return {**clean(),'returncode':1}
        elif '--output'in argv:
            target=Path(argv[argv.index('--output')+1]);put(target,{'fixture':True})
        return clean()
    monkeypatch.setattr(q,'guarded_run',run)
    rc=q.main(['--base',str(base),'--execute','--recovery-admission-sha256','a'*64]);return rc,q.read(L['out']/'COMPLETE.json'),calls,L

def test_full_control_orchestration_holds_lock_through_install_jobs_pack_and_persist(tmp_path,monkeypatch):
    rc,complete,calls,L=mocked_main(tmp_path,monkeypatch)
    assert rc==0 and complete['success']and complete['persistence_complete']and complete['timing_lock_released']
    assert calls.index('create_isolated_env')<calls.index('write_plain_pth')<calls.index('install_offline_extras')<calls.index('fixture_0')<calls.index('persistence')
    assert calls.index('fixture_47')<calls.index('isolated_environment_after')<calls.index('base_environment_after')
    assert not(L['out'].parent/'ADDITIONAL_TIMING_RUNNING.lock').exists()

@pytest.mark.parametrize('failure',['scores','pointer','persist_exit','lock'])
def test_final_success_requires_all_score_gates_valid_pointer_no_failures_and_owned_lock_release(tmp_path,monkeypatch,failure):
    rc,complete,calls,L=mocked_main(tmp_path,monkeypatch,failure)
    assert rc==1 and complete['success']is False and complete['failure']is not None
    if failure in('pointer','persist_exit'):assert complete['persistence_complete']is False and complete['archive_pointer']is None
    if failure=='lock':assert complete['controller_finished']is False and complete['timing_lock_released']is False

def test_admission_timeout_never_creates_env_unpacks_wheels_launches_or_persists(tmp_path,monkeypatch):
    base=tmp_path/'base';base.mkdir();monkeypatch.setattr(q,'BASE',base)
    monkeypatch.setattr(q,'verify_sources',lambda b,**kw:{})
    monkeypatch.setattr(q,'admission',lambda *a,**k:None)
    monkeypatch.setattr(q,'cleanup_owned',lambda s:True)
    monkeypatch.setattr(q.signal,'signal',lambda *a:None)
    supervisor=SimpleNamespace(enable_subreaper=lambda:None,process_identity=lambda pid:{'start_ticks':123},owned_groups=lambda pid:{})
    monkeypatch.setattr(q,'load_module',lambda *a:supervisor)
    tick=iter([0.,30000.]);monkeypatch.setattr(q.time,'monotonic',lambda:next(tick,30000.))
    def forbidden(*a,**k):raise AssertionError('no environment, job, archive or persistence before admission')
    monkeypatch.setattr(q,'unpack_wheels',forbidden);monkeypatch.setattr(q,'guarded_run',forbidden);monkeypatch.setattr(q,'pack_once',forbidden)
    assert q.main(['--base',str(base),'--execute','--recovery-admission-sha256','a'*64])==1
    L=q.layout(base);complete=q.read(L['out']/'COMPLETE.json')
    assert complete['success']is False and complete['persistence_complete']is False
    assert not L['venv'].exists()and not L['wheels'].exists()and not L['archive'].exists()
    assert not(base/'results/ADDITIONAL_TIMING_RUNNING.lock').exists()

def test_version3_paths_leave_v1_unused_and_protocol_unchanged(tmp_path):
    L=q.layout(tmp_path);assert L['venv'].name=='pencil_env_v3'and L['out'].name=='pencil_bridge_v3_20261125'
    jobs=q.make_jobs(tmp_path,L['venv']/'bin/python');assert len(jobs)==48 and sum(j['kind']=='benchmark'for j in jobs)==36
    assert all('pencil_env_v1'not in' '.join(j['argv'])and'pencil_queue_v1_code'not in' '.join(j['argv'])for j in jobs)

def test_actual_nested_venv_plain_route_restores_base_and_isolated_shadow_precedence(tmp_path):
    base=tmp_path/'opt/python';nested=tmp_path/'pencil_env_v3'
    subprocess.run([sys.executable,'-m','venv','--without-pip',str(base)],check=True,capture_output=True,timeout=30)
    base_python=base/'bin/python';site_relative=Path(f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
    base_site=base/site_relative
    (base_site/'pencil_base_only.py').write_text('value = "base-only"\n')
    (base_site/'pencil_shared.py').write_text('value = "base"\n')
    original={str(p):q.info(p)for p in base_site.glob('pencil_*.py')}
    subprocess.run([str(base_python),'-m','venv','--without-pip','--system-site-packages',str(nested)],check=True,capture_output=True,timeout=30)
    python=nested/'bin/python';own=nested/site_relative
    missing=subprocess.check_output([str(python),'-c','import importlib.util; print(importlib.util.find_spec("pencil_base_only"))'],text=True,timeout=15)
    assert missing.strip()=='None'
    (own/'pencil_shared.py').write_text('value = "isolated"\n')
    route=q.add_base_site(nested,{'site_packages':[str(base_site)]},allowed_roots=(base,),site_relative=site_relative)
    code='import json,sys,pencil_base_only,pencil_shared;print(json.dumps([pencil_base_only.value,pencil_shared.value,sys.path]))'
    actual=json.loads(subprocess.check_output([str(python),'-c',code],text=True,timeout=15))
    assert actual[:2]==['base-only','isolated']and actual[2].index(str(own))<actual[2].index(str(base_site))
    assert Path(route['path']).read_text()==str(base_site)+'\n'and'import 'not in Path(route['path']).read_text()
    q.assert_bindings(original)
    with pytest.raises(RuntimeError,match='preserve existing'):q.add_base_site(nested,{'site_packages':[str(base_site)]},allowed_roots=(base,),site_relative=site_relative)

def test_route_rejects_nonexisting_untrusted_or_executable_lines(tmp_path):
    env=tmp_path/'env';(env/'lib/python3.10/site-packages').mkdir(parents=True)
    for value in('/tmp/not-original/site-packages','import os\n/tmp/site-packages','/opt/python/missing/site-packages'):
        with pytest.raises(RuntimeError):q.add_base_site(env,{'site_packages':[value]})

def test_base_numerical_origins_and_native_binaries_must_match_before_and_after():
    before={'critical_versions':{'numpy':'1.24.4'},'critical_modules':{'numpy':{'origin':'/opt/python/numpy/__init__.py','source_sha256':'a'}},
        'torch_sparse_binaries':{'native.so':{'sha256':'b','bytes':1}},'saint_wrapper':{'sha256':'c'}}
    isolated={**copy.deepcopy(before),'base_site_route':{'isolated_site_precedes_all_base_paths':True}}
    assert q.verify_environment_binding(before,isolated,copy.deepcopy(before))['status']=='pass'
    bad=copy.deepcopy(isolated);bad['critical_modules']['numpy']['origin']='/newenv/numpy/__init__.py'
    with pytest.raises(RuntimeError,match='numerical'):q.verify_environment_binding(before,bad,before)
    bad=copy.deepcopy(isolated);bad['torch_sparse_binaries']['native.so']['sha256']='different'
    with pytest.raises(RuntimeError,match='native binaries'):q.verify_environment_binding(before,bad,before)
    bad=copy.deepcopy(isolated);bad['base_site_route']['isolated_site_precedes_all_base_paths']=False
    with pytest.raises(RuntimeError,match='precedence'):q.verify_environment_binding(before,bad,before)
    with pytest.raises(RuntimeError,match='base environment changed'):q.verify_environment_binding(before,isolated,{**before,'changed':True})

def recovery_fixture(tmp_path,monkeypatch):
    base=tmp_path/'base';base.mkdir();old={'pod_absent':True,'statefulset_replicas':0,'root_confirmed':True}
    roles={'pipeline':True,'native':True,'pencil_failed_v2':False}
    prior=[{'role':role,'yt_path':'//home/ucp-research/projects/graph_baselines/seal_revision_20261124/'+role+'.tgz',
        'bytes':100,'sha256':'a'*64,'all_internal_hashes_verified':True,'scientific_success':success,**({'completed_jobs':0}if not success else{})}
        for role,success in roles.items()]
    value={'schema_version':1,'kind':'pencil_v3_recovery_admission','base':str(base),'replacement_hostname':'new-owned-pod',
        'old_environment':old,'prior_archives':prior};path=put(base/'RECOVERY_ADMISSION.json',value)
    monkeypatch.setattr(q,'RECOVERY_ADMISSION_SHA',q.sha(path));monkeypatch.setattr(q.socket,'gethostname',lambda:'new-owned-pod')
    gate=SimpleNamespace(LOCKS=('GPU_SUITE_RUNNING.lock','PIPELINE_SUITE_RUNNING.lock','ADDITIONAL_TIMING_RUNNING.lock'))
    return base,value,path,gate

def test_root_pinned_recovery_uses_actual_replacement_host_without_legacy_pid_or_complete_files(tmp_path,monkeypatch):
    base,value,path,gate=recovery_fixture(tmp_path,monkeypatch)
    result=q.admission(base,gate,None,scanner=lambda base:[])
    assert result['admitted']and result['kind']=='root_pinned_recovery_admission'and result['bindings']=={str(path):q.info(path)}
    assert result['prior_archives'][-1]['scientific_success']is False and result['prior_archives'][-1]['completed_jobs']==0
    assert not(base/'results/native_controls_v1_20261125/COMPLETE.json').exists()
    assert not(base/'native_controls_v1_code').exists()

@pytest.mark.parametrize('change',['hash','host','base','kind','pod_present','replicas','unconfirmed','missing_role','archive_hash','unverified','native_failure','pencil_relabel','pencil_jobs'])
def test_recovery_attestation_hash_host_archive_outcome_mutations_fail_closed(tmp_path,monkeypatch,change):
    base,value,path,gate=recovery_fixture(tmp_path,monkeypatch)
    if change=='hash':value['modified']=True
    elif change=='host':value['replacement_hostname']='other-pod'
    elif change=='base':value['base']='/other'
    elif change=='kind':value['kind']='legacy'
    elif change=='pod_present':value['old_environment']['pod_absent']=False
    elif change=='replicas':value['old_environment']['statefulset_replicas']=1
    elif change=='unconfirmed':value['old_environment']['root_confirmed']=False
    elif change=='missing_role':value['prior_archives'].pop()
    elif change=='archive_hash':value['prior_archives'][0]['sha256']='invalid'
    elif change=='unverified':value['prior_archives'][0]['all_internal_hashes_verified']=False
    elif change=='native_failure':value['prior_archives'][1]['scientific_success']=False
    elif change=='pencil_relabel':value['prior_archives'][2]['scientific_success']=True
    else:value['prior_archives'][2]['completed_jobs']=1
    put(path,value)
    if change!='hash':monkeypatch.setattr(q,'RECOVERY_ADMISSION_SHA',q.sha(path))
    with pytest.raises(RuntimeError):q.admission(base,gate,None,scanner=lambda base:[])

@pytest.mark.parametrize('name',['GPU_SUITE_RUNNING.lock','PIPELINE_SUITE_RUNNING.lock','ADDITIONAL_TIMING_RUNNING.lock'])
def test_each_foreign_timing_lock_blocks_without_deletion_and_only_exact_additional_owner_recheck_allowed(tmp_path,monkeypatch,name):
    base,value,path,gate=recovery_fixture(tmp_path,monkeypatch);lock=put(base/'results'/name,{'foreign':True})
    assert q.admission(base,gate,None,scanner=lambda base:[])is None and lock.exists()
    if name=='ADDITIONAL_TIMING_RUNNING.lock':assert q.admission(base,gate,None,scanner=lambda base:[],owned_lock=lock)['admitted']

def test_actual_foreign_timing_process_blocks_recovery_even_with_valid_root_attestation(tmp_path,monkeypatch):
    base,value,path,gate=recovery_fixture(tmp_path,monkeypatch)
    assert q.admission(base,gate,None,scanner=lambda base:[{'pid':123,'entrypoint':'foreign benchmark'}])is None

def test_pth_original_numeric_extra_and_optional_hf_sources_are_all_runtime_bound(tmp_path,monkeypatch):
    env=tmp_path/'env';native=tmp_path/'original/torch_sparse';native.mkdir(parents=True)
    cfg=env/'pyvenv.cfg';cfg.parent.mkdir();cfg.write_text('unchanged')
    route=env/'route.pth';route.write_text('/opt/python/site-packages\n')
    init=native/'__init__.py';init.write_text('# original numeric module')
    binary=native/'native.so';binary.write_bytes(b'original binary');wrapper=native/'saint.py';wrapper.write_text('# original wrapper')
    hf=env/'transformers/__init__.py';hf.parent.mkdir();hf.write_text('# isolated HF')
    utils=hf.parent/'utils/import_utils.py';utils.parent.mkdir()
    utils.write_bytes(b'fixture pinned external HF API source')
    actual_sha=q.sha
    def digest(path):
        if Path(path)==utils and utils.read_bytes()==b'fixture pinned external HF API source':return '9d7f63effbd7c8dcfe69e7f34cae0d164db9cca2d5f4d887405ae0f3e2ba455a'
        return actual_sha(path)
    monkeypatch.setattr(q,'sha',digest)
    observed={'base_site_route':{'path':str(route)},'critical_modules':{'torch_sparse':{'origin':str(init),'source_sha256':q.sha(init)}},
        'extra_modules':{'transformers':{'origin':str(hf),'sha256':q.sha(hf)}},'torch_sparse_binaries':{'native.so':q.info(binary)},
        'saint_wrapper':{'sha256':q.sha(wrapper)},'runtime_compatibility':{**policy(),'import_utils_origin':str(utils)}}
    bound=q.bind_environment(observed,env);q.assert_bindings(bound)
    assert str(utils)in bound and str(q.BRIDGE/'graph_only_transformers_v2.py')in bound
    route.write_text('changed')
    with pytest.raises(RuntimeError,match='changed'):q.assert_bindings(bound)
    route.write_text('/opt/python/site-packages\n');utils.write_text('changed HF API')
    with pytest.raises(RuntimeError,match='optional-vision API origin'):q.bind_environment(observed,env)

@pytest.mark.parametrize('field',['version','policy_source_sha256','import_utils_sha256','torchvision_available_for_graph_worker','torchvision_imported','fake_operator_registered'])
def test_completed_job_requires_explicit_same_pinned_optional_vision_policy(tmp_path,field):
    job=benchmark_fixtures(tmp_path)[0];raw=q.read(job['output']);raw['runtime_compatibility'][field]='changed';put(Path(job['output']),raw)
    with pytest.raises(RuntimeError,match='optional-vision'):q.validate_job(job,clean())

