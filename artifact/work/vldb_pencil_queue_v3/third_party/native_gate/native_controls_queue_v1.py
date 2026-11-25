#!/usr/bin/env python3
"""Finite post-recovery native CPU controls. Default preflight; no runtime edits."""
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,math,os,signal,subprocess,sys,tarfile,time

HERE=Path(__file__).resolve().parent
BASE=Path('/home/jovyan/seal_revision_20261124')
YT_TARGET='//home/ucp-research/projects/graph_baselines/seal_revision_20261124/native_controls_v1_20261125.tgz'
GIB=2**30
PINS={'third_party/reviewed_supervision_v3.py':'ec07bddc13f69cbbc1c5ba63a3ede390cfea254bcc02b811422338f5b9604aaf',
      'third_party/memory_guard_v3.py':'e5bc616b518c17d6804c72f376872155514b902b490f43639526680ca5a516c8',
      'evidence/main_execution_launch.json':'1c50f58a4e1b87794d2381f3de451270565023b2c7361edf8a4297d404087e63'}
LIBRARY_SHA='2a8b892a63a84061ec03fc8e7d6c5d616ecb351720a1e4f236cae0374999f967'
MANIFEST_SHA='bcddcf02ef108660f9a32d359011a13fe33716e4b392a9d61ef4890acfb575fd'
HELPER_SHA='525e88c904fd03e10fc617853c8d733e88af90487cbbdd9122ced97962df112c'
RECORDS={'friendster':(9040463,'bed119047100f89080081fe638ed60816d0256bfeea9750418668078cdb2d975'),
         'vk':(8987220,'ce8e1065f2a740b0ee16bee5fb9e35fc37364b2d152be411089197c0f7483b04')}
MAIN_SHA='512ef52c58e7c6f7a00266c91070a21c9d8374d572ef6a91796bbecd6b65e37e'
BACKUP_SHA='65227c2c567daac52c3fb0a4708c14798763a7a1cbeca506661ec16daaa16541'
MAIN_START_TICKS=527240237
LOCKS=('GPU_SUITE_RUNNING.lock','PIPELINE_SUITE_RUNNING.lock','ADDITIONAL_TIMING_RUNNING.lock')

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def info(path):
    p=Path(path);require(p.is_file()and not p.is_symlink(),'regular immutable file required: '+str(p))
    return {'bytes':p.stat().st_size,'sha256':sha(p)}
def read(path):
    p=Path(path);require(p.stat().st_size<=16<<20,'bounded JSON required')
    d=json.loads(p.read_text());require(isinstance(d,dict),'JSON object required');return d
def save(path,value):
    p=Path(path);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def assert_bindings(bindings):
    require(all(info(p)==v for p,v in bindings.items()),'bound source/records/library/admission changed')

def verify_sources(base):
    stage=base/'corrected_native_v1';native=stage/'work/vldb_corrected_native'
    for name,pin in PINS.items():require(sha(HERE/name)==pin,'reviewed source/launch pin changed: '+name)
    manifest=native/'source_manifest_v1.json';require(sha(manifest)==MANIFEST_SHA,'corrected source manifest pin changed')
    paths=[p for p in HERE.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix in('.py','.json','.md','.txt')]
    paths.append(manifest)
    for item in read(manifest)['files']:
        p=stage/item['path'];require(p.resolve().is_relative_to(stage.resolve()),'manifest path escapes stage')
        require(info(p)=={'bytes':item['bytes'],'sha256':item['sha256']},'corrected native source binding changed');paths.append(p)
    library=stage/'linux_build/native_assoc_v1.so';require(sha(library)==LIBRARY_SHA,'actual admitted Linux library changed');paths.append(library)
    for graph,(size,pin)in RECORDS.items():
        p=stage/'records'/(graph+'_exact_int32.json');require(info(p)=={'bytes':size,'sha256':pin},'original exact record identity changed');paths.append(p)
    for name,pin in [('run_recovery_execution_v1.py',MAIN_SHA),('backup_v5.py',BACKUP_SHA)]:
        p=base/name;require(sha(p)==pin,'main recovery/persistence source changed');paths.append(p)
    return {str(p):info(p)for p in paths}

def main_process_quiescent(base,launch,supervisor,proc=Path('/proc')):
    require(launch.get('pid')==257 and launch['sources']['run_recovery_execution_v1.py']==MAIN_SHA,'unexpected main launch identity')
    p=proc/'257'
    if not p.exists():return True
    identity=supervisor.process_identity(257,proc);require(identity is not None,'main PID exists but identity unreadable')
    require(identity['start_ticks']==MAIN_START_TICKS,'PID257 reused or launch/start identity mismatch; no inference or signals')
    if identity['state']=='Z':return True
    argv=(p/'cmdline').read_bytes().split(b'\0')
    require(str(base/'run_recovery_execution_v1.py').encode()in argv,'main PID257 entrypoint changed')
    return False

def active_phases(base,proc=Path('/proc')):
    names={'run_pipeline_suite_v4.py','pipeline_benchmark_v4.py','runner.py','recovery_input_reader_v1.py',
           'run_recovery_execution_v1.py','pencil_bridge.py','replay_worker_v1.py','run_parent_replay_v1.py',
           'prepare_native_only_v1.py','external_operator_benchmark.py'}
    found=[]
    for p in proc.iterdir():
        if not p.name.isdigit()or int(p.name)==os.getpid():continue
        try:
            state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
            args=[x.decode(errors='replace')for x in(p/'cmdline').read_bytes().split(b'\0')if x]
            if state=='Z'or str(base)not in ' '.join(args):continue
            if any(Path(x).name in names for x in args)or(any(Path(x).name=='backup_v5.py'for x in args)and'--once'in args):
                found.append({'pid':int(p.name),'entrypoints':[Path(x).name for x in args if Path(x).name in names or Path(x).name=='backup_v5.py']})
        except(OSError,ValueError,IndexError):continue
    return found

def recovery_admission(base,supervisor,*,scanner=active_phases,process_check=main_process_quiescent,owned_lock=None):
    r=base/'results';out=r/'recovery_execution_v1';marker=out/'COMPLETE.json'
    if not marker.exists():return None
    m=read(marker);require(m.get('complete')is True and m.get('quality_success')is True and type(m.get('pipeline_success'))is bool,'invalid main scientific terminal')
    terminal=r/'PIPELINE_SUITE_COMPLETE.json';p=read(terminal)
    require(p.get('controller_finished')is True and type(p.get('success'))is bool,'invalid pipeline terminal')
    require(m['pipeline_terminal_sha256']==sha(terminal)and m['pipeline_success']==p['success'],'pipeline terminal binding changed')
    require((p['success']and m['pipeline_returncode']==0 and p['terminal_status']=='complete')or
            (not p['success']and m['pipeline_returncode']!=0 and p['terminal_status']in('completed_with_failures','failed_precondition')),'pipeline terminal/exit contradiction')
    log=out/'persist_pipeline_final.log';require(log.stat().st_size<=8<<20,'bounded persistence log')
    pointers=[]
    for line in log.read_text().splitlines():
        try:d=json.loads(line)
        except ValueError:continue
        if isinstance(d,dict)and d.get('role')=='pipeline'and d.get('anchor_kind')=='final':pointers.append(d)
    require(len(pointers)==1 and pointers[0].get('terminal_in_snapshot')is True and pointers[0].get('terminal_sha256')==sha(terminal),'pipeline durable final receipt missing/changed')
    persisted=read(out/'persist_pipeline_final_exit.json')
    require(persisted['returncode']==0 and not persisted['timed_out']and persisted['owned_tree_quiescent']and not persisted.get('orphan_cleanup'),'pipeline persistence phase not clean')
    launch=read(HERE/'evidence/main_execution_launch.json')
    if not process_check(base,launch,supervisor):return None
    for name in LOCKS:
        lock=r/name
        if lock.exists()and lock!=owned_lock:return None
    if scanner(base):return None
    files=[marker,terminal,log,out/'persist_pipeline_final_exit.json']
    return {'complete':True,'scientifically_terminal_main':True,'pipeline_success':p['success'],
            'pipeline_final_archive':pointers[0],'bindings':{str(p):info(p)for p in files},
            'scope':'Main controller PID257 is absent/zombie with launch identity checked if present; pipeline final persisted; no observed active phase or foreign timing lock.'}

def acquire_lock(path,token):
    with path.open('x')as f:f.write(json.dumps(token,sort_keys=True)+'\n')
def release_lock(path,token,quiescent):
    require(quiescent,'never clear timing lock before owned tree quiescence')
    require(path.is_file()and not path.is_symlink()and read(path)==token,'owned timing lock changed')
    path.unlink()

def cleanup_owned(supervisor,timeout=15):
    deadline=time.monotonic()+timeout
    for sig in(signal.SIGTERM,signal.SIGKILL):
        groups=supervisor.owned_groups(os.getpid());supervisor.signal_confirmed(groups,sig)
        until=min(deadline,time.monotonic()+5)
        while time.monotonic()<until and supervisor.owned_groups(os.getpid()):time.sleep(.1)
    return not supervisor.owned_groups(os.getpid())

def guarded_run(supervisor,memory,argv,log,cwd,timeout):
    abort={'reason':None};last=[0.];errors=[0]
    guard=log.with_suffix('.guard.jsonl')
    def clock():return time.monotonic()+(10**9 if abort['reason']else 0)
    def groups(owner):
        found=supervisor.owned_groups(owner)
        if time.monotonic()-last[0]>=2:
            last[0]=time.monotonic()
            try:
                sample=memory.sample_tree(owner);decision=memory.budget_decision(sample,180*GIB,memory.cgroup_memory())
                abort['reason']=abort['reason']or decision['abort_reason'];errors[0]=0
                row={'unix':time.time(),'tree_pss_bytes':sample.get('tree_pss_bytes'),'pss_complete':sample.get('pss_complete'),
                     'membership_stable':sample.get('membership_stable'),'max_process_rss_bytes':sample.get('max_process_rss_bytes'),'decision':decision}
            except Exception as e:
                errors[0]+=1;row={'unix':time.time(),'sampling_error':type(e).__name__+': '+str(e)[:300]}
                if errors[0]>=3:abort['reason']='three consecutive external V3 sampling failures'
            with guard.open('a')as f:f.write(json.dumps(row)+'\n')
            if guard.stat().st_size>8<<20:abort['reason']='bounded guard-log8MiB limit'
        if log.exists()and log.stat().st_size>4<<20:abort['reason']='bounded stdout-log4MiB limit'
        return found
    def popen(*args,**kwargs):
        kwargs['env'].update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
        return subprocess.Popen(*args,**kwargs)
    try:result=supervisor.run_owned(argv,log,cwd=cwd,timeout=timeout,clock=clock,popen=popen,groups_reader=groups)
    except BaseException:
        cleanup_owned(supervisor);raise
    result.update(guard_abort_reason=abort['reason'],guard_policy='unchanged stable_pss_individual_rss_cgroup90_v3 externally sampled at2s;180GiB; CPU-only')
    save(log.with_suffix('.exit.json'),result);return result

def parent_argv(base,graph,out,marker):
    stage=base/'corrected_native_v1';code=stage/'work/vldb_corrected_native'
    return [sys.executable,'-u',str(code/'run_parent_replay_v1.py'),'--archive',str(stage/'records'/(graph+'_exact_int32.json')),
            '--count','128','--output-dir',str(out/graph),'--execute','--library',str(stage/'linux_build/native_assoc_v1.so'),
            '--native-dir',str(base/'native_controls_v1_data'/(graph+'_i64')),'--prepare-native',
            '--graph-path',str(base/('data/friendster/full_csr'if graph=='friendster'else'data/vk/train_csr')),
            '--admission-marker',str(marker),'--include-upstream','--external-helper',str(stage/'work/vldb_literature/external_operator_benchmark.py'),
            '--external-helper-sha256',HELPER_SHA,'--conversion-max-native-gib','28'if graph=='friendster'else'98',
            '--max-assoc-gib','0.5'if graph=='friendster'else'1.4','--max-peak-gib','48'if graph=='friendster'else'160',
            '--max-query-neighbors','20000000','--warmup','1','--repeats','3','--timeout-seconds','3600','--conversion-timeout-seconds','1800']

def graph_outcome(folder,exit):
    clean=exit['returncode']==0 and not exit['timed_out']and not exit.get('orphan_cleanup')and exit['owned_tree_quiescent']and exit.get('guard_abort_reason')is None
    marker=folder/'COMPLETE.json';complete=read(marker)if marker.exists()else{}
    arms=complete.get('arms',[])
    expected={'reference_full_clear','persistent_touched_reset','upstream_torch_sparse_direct'}
    valid=(clean and complete.get('complete')is True and complete.get('success')is True and len(arms)==3
           and{a['mode']for a in arms}==expected and all(a.get('eligible')is True for a in arms))
    if valid:
        for arm in arms:
            raw=read(folder/(arm['mode']+'.json'))
            require(raw.get('record_count')==128 and raw.get('all_executed_outputs_exact')is True,'parent/worker completeness contradiction')
    return {'status':'complete'if valid else'failed_or_guarded','scientifically_eligible':bool(valid),'exit':exit,
            'parent_terminal':complete,'failed_evidence_retained':True,'no_speed_claim_inferred':True}

def pack_once(out,sources,records,target):
    require(not target.exists(),'fresh final archive required')
    members={}
    for p in sorted(out.rglob('*')):
        require(not p.is_symlink(),'symlink result refused')
        if not p.is_file():continue
        partial_json=p.name.endswith(('.json.partial','.json.tmp'))
        require(p.suffix in('.json','.jsonl','.log')or partial_json,'no array/binary result in archive; interrupted JSON is diagnostic only')
        members['results/native_controls_v1/'+str(p.relative_to(out))]=p
    for name,p in sources.items():
        require(not any(part in('__pycache__','.pytest_cache')for part in Path(name).parts),'no cache sources')
        require(p.suffix in('.py','.cpp','.json','.md','.txt')or p.name.endswith('LICENSE'),'small textual source/provenance only')
        members['sources/'+name]=p
    for graph,p in records.items():members['original_records/'+graph+'_exact_int32.json']=p
    values={name:info(p)for name,p in members.items()}
    require(all(v['bytes']<=16<<20 for v in values.values()),'single runtime artifact16MiB limit')
    require(sum(v['bytes']for v in values.values())<=60<<20,'runtime payload60MiB leaves manifest/compression reserve')
    data=(json.dumps({'schema_version':1,'files':values,'scope':'One final native control archive; original bounded replay JSON, new results and textual sources only. No full graph, native array, checkpoint, environment dump or secret.'},indent=2)+'\n').encode()
    with tarfile.open(target,'x:gz')as tar:
        for name,p in sorted(members.items()):
            require(info(p)==values[name],'archive source changed')
            item=tarfile.TarInfo(name);item.size=values[name]['bytes'];item.mode=0o644
            with p.open('rb')as f:tar.addfile(item,f)
            require(info(p)==values[name],'archive source changed during read')
        item=tarfile.TarInfo('SHA256_MANIFEST.json');item.size=len(data);tar.addfile(item,io.BytesIO(data))
    require(target.stat().st_size<=64<<20,'compressed final archive64MiB limit')
    return {**info(target),'files':len(values),'manifest_sha256':hashlib.sha256(data).hexdigest()}

def upload_once(archive,target,client=None):
    require(target==YT_TARGET,'exactly one new native-controls YT path is allowed')
    expected=info(archive);require(expected['bytes']<=64<<20,'bounded upload')
    if client is None:
        import yt.wrapper as yt
        client=yt.YtClient(proxy='jupiter.yt.vk.team',config={'proxy':{'request_timeout':120000}})
    require(not client.exists(target),'existing YT node; no overwrite or second archive')
    with client.Transaction(timeout=600000):
        client.create('file',target)
        with archive.open('rb')as f:client.write_file(target,f)
        h=hashlib.sha256();size=0;stream=client.read_file(target)
        try:
            for block in iter(lambda:stream.read(1<<20),b''):
                size+=len(block);require(size<=64<<20,'bounded archive readback');h.update(block)
        finally:stream.close()
        require({'bytes':size,'sha256':h.hexdigest()}==expected,'YT final archive readback hash mismatch')
    return {'yt_path':target,**expected,'readback_verified':True,'new_YT_file_nodes':1}

def validate_pointer(pointer,archive):
    expected=info(archive)
    require(pointer.get('readback_verified')is True and pointer.get('yt_path')==YT_TARGET
            and pointer.get('new_YT_file_nodes')==1 and pointer.get('sha256')==expected['sha256']
            and pointer.get('bytes')==expected['bytes'],'durable final pointer mismatch')
    return pointer

def final_terminal(science,pointer,failure,quiescent,lock_owned,elapsed,controller_identity=None):
    finished=quiescent and not lock_owned
    persisted=pointer is not None
    success=bool(science.get('success')is True and persisted and failure is None and finished)
    return {'controller_finished':finished,'success':success,'scientific_complete':science.get('success')is True,
            'persistence_complete':persisted,'archive_pointer':pointer,'failure':failure,
            'timing_lock_released':not lock_owned,'owned_tree_quiescent':quiescent,'runtime_seconds':elapsed,
            'controller_identity':controller_identity}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=BASE);p.add_argument('--execute',action='store_true')
    p.add_argument('--max-hours',type=float,default=12);p.add_argument('--admission-hours',type=float,default=8)
    p.add_argument('--persist-worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--archive-sha256',help=argparse.SUPPRESS);p.add_argument('--archive-bytes',type=int,help=argparse.SUPPRESS);a=p.parse_args(argv)
    require(all(math.isfinite(v)and v>0 for v in(a.max_hours,a.admission_hours))and a.max_hours<=12 and a.admission_hours<=8,'finite global<=12h/admission<=8h required')
    base=a.base.resolve();out=base/'results/native_controls_v1_20261125';archive=base/'native_controls_v1_20261125.tgz'
    if a.execute:require(base==BASE,'execution is scoped to the canonical existing project BASE')
    if a.persist_worker:
        require(a.execute,'explicit persistence execution required')
        require(info(archive)=={'bytes':a.archive_bytes,'sha256':a.archive_sha256},'packed archive changed before persistence')
        receipt=upload_once(archive,YT_TARGET);save(out/'archive_pointer.json',receipt);return 0
    bindings=verify_sources(base)
    require(not out.exists()and not archive.exists(),'fresh native queue output/archive required; no automatic rerun')
    if not a.execute:print(json.dumps({'status':'preflight','bindings':len(bindings),'global_hours':a.max_hours,'admission_hours':a.admission_hours,'graph_order':['friendster','vk'],'no_execution':True}));return 0
    out.mkdir(parents=True);started=time.monotonic();deadline=started+a.max_hours*3600;wait_deadline=min(deadline-1200,started+a.admission_hours*3600)
    supervisor=load_module(HERE/'third_party/reviewed_supervision_v3.py','native_reviewed_supervision')
    memory=load_module(HERE/'third_party/memory_guard_v3.py','native_reviewed_memory');supervisor.enable_subreaper()
    identity=supervisor.process_identity(os.getpid());require(identity is not None,'queue identity unavailable')
    identity={'pid':os.getpid(),'start_ticks':identity['start_ticks'],'entrypoint':str(Path(__file__).resolve()),'source_sha256':sha(Path(__file__))}
    lock=base/'results/ADDITIONAL_TIMING_RUNNING.lock';token={'kind':'native_controls_queue_v1','pid':os.getpid(),'start_ticks':identity['start_ticks'],'started_unix':time.time()};owned=False;stages={};failure=None;pointer=None
    def status(phase,**kw):save(out/'status.json',{'phase':phase,'updated_unix':time.time(),**kw})
    def interrupted(sig,frame):raise KeyboardInterrupt('received signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted)
    save(out/'plan.json',{'kind':'native_controls_queue_v1','bindings':bindings,'controller_identity':identity,'max_hours':a.max_hours,'admission_hours':a.admission_hours,
         'graph_order':['friendster','vk'],'cases_per_graph':128,'arms':3,'warmup':1,'timed_repeats':3,'external_memory_guard_gib':180,
         'worker_timeout_seconds':3600,'conversion_timeout_seconds':1800,'timing':'no sampler/model/quality change; setup/conversion/map-init separate','no_existing_runtime_edits':True})
    try:
        admitted=None
        while time.monotonic()<wait_deadline:
            assert_bindings(bindings);admitted=recovery_admission(base,supervisor)
            if admitted:break
            status('waiting_for_main_pipeline_persistence_and_quiescence');time.sleep(min(15,max(0,wait_deadline-time.monotonic())))
        require(admitted is not None,'finite admission deadline reached without safe main quiescence')
        acquire_lock(lock,token);owned=True
        again=recovery_admission(base,supervisor,owned_lock=lock);require(again==admitted,'admission changed while acquiring timing lock')
        bindings.update(admitted['bindings']);save(out/'admission.json',admitted);save(out/'runtime_bindings.json',bindings)
        for graph in('friendster','vk'):
            assert_bindings(bindings);require(read(lock)==token,'exclusive timing lock lost')
            require(recovery_admission(base,supervisor,owned_lock=lock)==admitted,'same quiescent admission required before each graph')
            remaining=deadline-time.monotonic()-1200
            if remaining<120:
                stages[graph]={'status':'not_started_global_deadline','scientifically_eligible':False};continue
            command=parent_argv(base,graph,out,out/'admission.json');save(out/(graph+'_command.json'),{'argv':command,'outer_timeout_seconds':min(13000,remaining)})
            status('running',graph=graph);exit=guarded_run(supervisor,memory,command,out/(graph+'.log'),base,min(13000,remaining))
            require(exit['owned_tree_quiescent'],'owned child tree remains; retain lock and stop')
            assert_bindings(bindings);stages[graph]=graph_outcome(out/graph,exit);save(out/'arm_results.json',stages)
    except BaseException as e:
        failure={'type':type(e).__name__,'message':str(e)[:500]};status('failed',error=failure)
    finally:
        quiescent=cleanup_owned(supervisor)
        save(out/'SCIENTIFIC_TERMINAL.json',{'controller_jobs_finished':quiescent,'success':quiescent and not failure and len(stages)==2 and all(x['scientifically_eligible']for x in stages.values()),
             'stages':stages,'failure':failure,'owned_tree_quiescent':quiescent,'timing_lock_retained':owned,'runtime_seconds':time.monotonic()-started})
    try:
        require(quiescent and not supervisor.owned_groups(os.getpid()),'unsafe owned-tree state; no completion/persistence')
        assert_bindings(bindings)
        require(time.monotonic()<deadline-60,'global deadline before final persistence')
        source_paths={str(p.relative_to(HERE)):p for p in HERE.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc'}
        stage=base/'corrected_native_v1'
        for item in read(stage/'work/vldb_corrected_native/source_manifest_v1.json')['files']:
            source_paths['corrected_native/'+item['path']]=stage/item['path']
        source_paths['corrected_native/source_manifest_v1.json']=stage/'work/vldb_corrected_native/source_manifest_v1.json'
        result=pack_once(out,source_paths,{g:stage/'records'/(g+'_exact_int32.json')for g in RECORDS},archive)
        save(out/'archive_local_receipt.json',result)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--base',str(base),'--execute','--persist-worker',
                 '--archive-sha256',result['sha256'],'--archive-bytes',str(result['bytes'])]
        exit=guarded_run(supervisor,memory,command,out/'persistence.log',base,min(600,max(1,deadline-time.monotonic()-30)))
        require(exit['returncode']==0 and not exit['timed_out']and exit['owned_tree_quiescent']and not exit['orphan_cleanup']and exit['guard_abort_reason']is None,'final persistence phase failed')
        assert_bindings(bindings);require(info(archive)=={'bytes':result['bytes'],'sha256':result['sha256']},'packed archive changed during persistence')
        candidate=read(out/'archive_pointer.json');pointer=validate_pointer(candidate,archive)
    except BaseException as e:
        failure=failure or {'type':type(e).__name__,'message':str(e)[:500]};cleanup_owned(supervisor)
    finally:
        quiescent=cleanup_owned(supervisor)
        if owned and quiescent:
            try:release_lock(lock,token,True);owned=False
            except Exception as e:failure=failure or {'type':type(e).__name__,'message':str(e)[:500]}
    terminal=read(out/'SCIENTIFIC_TERMINAL.json')
    completed=final_terminal(terminal,pointer,failure,quiescent,owned,time.monotonic()-started,identity);success=completed['success']
    save(out/'COMPLETE.json',completed)
    status('complete'if success else'completed_with_failures',success=success);return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
