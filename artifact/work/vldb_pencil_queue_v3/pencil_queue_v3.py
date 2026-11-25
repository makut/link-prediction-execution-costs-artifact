#!/usr/bin/env python3
"""Finite isolated PENCIL systems queue. Default: read-only preflight, no launch."""
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,math,os,signal,socket,subprocess,sys,tarfile,time

HERE=Path(__file__).resolve().parent
BASE=Path('/home/jovyan/seal_revision_20261124')
RECOVERY_ADMISSION_SHA=None
YT_TARGET='//home/ucp-research/projects/graph_baselines/seal_revision_20261124/pencil_bridge_v3_20261125.tgz'
SOURCE_PACKAGE_SHA='d2b1370620f24f0bbb14ce50493fce9908495ce02b65ffd666ef31319296fd74'
BRIDGE_MANIFEST_SHA='ec7ad32ef3b9f3bf84053f6f0aca41a96f1375495b41ddec71d3f8a260a40523'
WHEEL_ARCHIVE_SHA='885ba0ca7b68bee01ab01c134ffa97ba2120509c7def677d1e8d0e7f7d92ee30'
WHEEL_MEMBER_PREFIX='pencil_bridge_wheelhouse_linux_cp310_matched'
CPP_LIBRARY_SHA='2a8b892a63a84061ec03fc8e7d6c5d616ecb351720a1e4f236cae0374999f967'
CPP_MODULE_SHA='4aa4f51fcc6621ec42c6890be42439ef98132cc0554efbde3295e1d1f5e29549'
CPP_SOURCE_SHA='67a9e651b7bad7889a73751bff3b33b4309fb2ca8ddfe2660acf851141045d20'
NATIVE_QUEUE_SHA='35b84f8178b5dc6804e7c9ba71859d714640a2e35751710f879d260fd5750617'
NATIVE_QUEUE_PID=7962
NATIVE_QUEUE_START_TICKS=527714219
GIB=2**30
MIB=2**20
MAX_ARCHIVE=128*MIB
PINS={'third_party/reviewed_supervision_v3.py':'ec07bddc13f69cbbc1c5ba63a3ede390cfea254bcc02b811422338f5b9604aaf',
      'evidence/pencil_venv_inheritance_probe_v1.json':'716287dde5ea5007b084304d33812d5a0e26e832f368aede1525905112ab5fbd',
      'evidence/native_queue_identity_v1.json':'234f26d7c7afecbef2ecce5716a03960fdd669d4e18b2a4912b1d9b021b6b15c',
      'third_party/memory_guard_v3.py':'e5bc616b518c17d6804c72f376872155514b902b490f43639526680ca5a516c8',
      'third_party/native_gate/native_controls_queue_v1.py':NATIVE_QUEUE_SHA,
      'third_party/native_gate/evidence/main_execution_launch.json':'1c50f58a4e1b87794d2381f3de451270565023b2c7361edf8a4297d404087e63'}
ARMS=('torch_sparse_public','torch_sparse_direct','compiled_scan_int64','compiled_pairwise_int64',
      'compiled_scan_int32','compiled_pairwise_int32','cpp_full_clear','cpp_touched_reset')
REPLAY_ARMS=('torch_sparse_public','torch_sparse_direct','cpp_touched_reset','compiled_pairwise_int32')
BRIDGE=HERE/'bridge/vldb_pencil_bridge_v2'

def require(value,message):
    if not value:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def info(path):
    p=Path(path);require(p.is_file()and not p.is_symlink(),'regular immutable file required: '+str(p))
    return {'bytes':p.stat().st_size,'sha256':sha(p)}
def read(path):
    p=Path(path);require(p.is_file()and not p.is_symlink()and p.stat().st_size<=32*MIB,'bounded regular JSON required')
    value=json.loads(p.read_text());require(isinstance(value,dict),'JSON object required');return value
def save(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def assert_bindings(bindings):require(all(info(p)==v for p,v in bindings.items()),'bound source/library/admission changed')

def layout(base):
    return {'out':base/'results/pencil_bridge_v3_20261125','archive':base/'pencil_bridge_v3_20261125.tgz',
            'venv':base/'pencil_env_v3','wheel_archive':base/'pencil_bridge_wheelhouse_linux_cp310_matched.tgz',
            'wheels':base/'pencil_bridge_wheelhouse_linux_cp310_matched_v3',
            'cpp_module':base/'corrected_native_v1/work/vldb_corrected_native/native_adapter_v1.py',
            'cpp_library':base/'corrected_native_v1/linux_build/native_assoc_v1.so',
            'cpp_receipt':base/'corrected_native_v1/linux_build/build_receipt_v1.json'}

def verify_sources(base,*,full=True):
    paths=[]
    for name,pin in PINS.items():
        p=HERE/name;require(sha(p)==pin,'reviewed source pin changed: '+name);paths.append(p)
    observed=read(HERE/'evidence/native_queue_identity_v1.json')
    require(observed['pid']==NATIVE_QUEUE_PID and observed['start_ticks']==NATIVE_QUEUE_START_TICKS and
        str(base/'native_controls_queue_v1_code/native_controls_queue_v1.py')in observed['argv'],'root-observed native identity pin changed')
    manifest=BRIDGE/'SOURCE_MANIFEST.json';require(sha(manifest)==BRIDGE_MANIFEST_SHA,'frozen bridge source manifest changed')
    for item in read(manifest)['files']:
        p=BRIDGE/item['path'];require(p.resolve().is_relative_to(BRIDGE.resolve()),'bridge path escaped source tree')
        require(info(p)=={'bytes':item['bytes'],'sha256':item['sha256']},'frozen bridge source changed');paths.append(p)
    paths.append(manifest)
    for p in(HERE/'pencil_queue_v3.py',HERE/'runtime_probe_v3.py',HERE/'README.txt',HERE/'protocol_spec.json'):
        paths.append(p)
    L=layout(base)
    for key,pin in(('cpp_module',CPP_MODULE_SHA),('cpp_library',CPP_LIBRARY_SHA)):
        require(sha(L[key])==pin,'root-admitted native artifact changed: '+key);paths.append(L[key])
    receipt=read(L['cpp_receipt']);require(receipt.get('kind')=='corrected_native_build_v1'and
        receipt.get('source_sha256')==CPP_SOURCE_SHA and receipt.get('library_sha256')==CPP_LIBRARY_SHA,'reviewed source/library build receipt changed')
    paths.append(L['cpp_receipt'])
    if full:
        require(info(L['wheel_archive'])['bytes']<64*MIB and sha(L['wheel_archive'])==WHEEL_ARCHIVE_SHA,'matched offline wheel archive changed')
        paths.append(L['wheel_archive'])
    return {str(p):info(p)for p in paths}

def scan_active(base,proc=Path('/proc')):
    names={'native_controls_queue_v1.py','pencil_queue_v1.py','pencil_queue_v2.py','pencil_queue_v3.py','pencil_bridge.py','run_parent_replay_v1.py','replay_worker_v1.py',
           'prepare_native_only_v1.py','run_pipeline_suite_v4.py','pipeline_benchmark_v4.py','runner.py','recovery_input_reader_v1.py',
           'run_recovery_execution_v1.py','external_operator_benchmark.py'}
    found=[]
    for p in proc.iterdir():
        if not p.name.isdigit()or int(p.name)==os.getpid():continue
        try:
            state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
            args=[x.decode(errors='replace')for x in(p/'cmdline').read_bytes().split(b'\0')if x]
            if state=='Z'or str(base)not in' '.join(args):continue
            entries=[Path(x).name for x in args if Path(x).name in names]
            if entries or('backup_v5.py'in[Path(x).name for x in args]and'--once'in args):
                found.append({'pid':int(p.name),'entrypoints':entries})
        except(OSError,ValueError,IndexError):continue
    return found

def valid_sha(value):return isinstance(value,str)and len(value)==64 and all(c in'0123456789abcdef'for c in value)

def admission(base,gate,supervisor,*,scanner=scan_active,owned_lock=None):
    """Root-pinned recovery attestation, actual replacement host and timing locks.

    Does not fabricate legacy COMPLETE/PID files or infer old controller cleanup.
    Root verified immutable prior archives and deletion of the original pod.
    """
    path=base/'RECOVERY_ADMISSION.json'
    require(valid_sha(RECOVERY_ADMISSION_SHA),'explicit root recovery admission SHA256 required')
    require(path.is_file()and not path.is_symlink()and path.stat().st_size<=2*MIB,'bounded regular root recovery admission required')
    require(sha(path)==RECOVERY_ADMISSION_SHA,'root-pinned recovery admission changed')
    value=read(path)
    require(value.get('schema_version')==1 and value.get('kind')=='pencil_v3_recovery_admission'and value.get('base')==str(base),
        'root recovery admission kind/base changed')
    require(value.get('replacement_hostname')==socket.gethostname(),'recovery receipt belongs to another replacement pod')
    old=value.get('old_environment',{})
    require(old.get('pod_absent')is True and old.get('statefulset_replicas')==0 and old.get('root_confirmed')is True,
        'root-confirmed old pod deletion and zero replicas required')
    prior=value.get('prior_archives',[]);required={'pipeline':True,'native':True,'pencil_failed_v2':False}
    require(len(prior)==3 and{v.get('role')for v in prior}==set(required),'all3 immutable prior archive roles required')
    for item in prior:
        require(item.get('all_internal_hashes_verified')is True and item.get('scientific_success')is required[item['role']]and
            type(item.get('bytes'))is int and item['bytes']>0 and valid_sha(item.get('sha256'))and
            isinstance(item.get('yt_path'),str)and item['yt_path'].startswith('//home/ucp-research/projects/graph_baselines/seal_revision_20261124/'),
            'root-verified immutable prior archive identity/outcome required')
        if item['role']=='pencil_failed_v2':require(item.get('completed_jobs')==0,'old PENCIL failure must remain0jobs, not success')
    # All original GPU/pipeline/additional locks must be absent; caller may own the exact
    # additional lock during rechecks. No stale-owner deletion or signaling.
    for name in gate.LOCKS:
        lock=base/'results'/name
        if os.path.lexists(lock):
            if owned_lock is not None and lock==owned_lock:continue
            return None
    if scanner(base):return None
    return {'admitted':True,'kind':'root_pinned_recovery_admission','receipt_sha256':RECOVERY_ADMISSION_SHA,
        'replacement_hostname':value['replacement_hostname'],'prior_archives':prior,'bindings':{str(path):info(path)},
        'scope':'Root-verified prior immutable archives and deleted original pod; actual replacement hostname, active process scan and all original GPU/pipeline/additional timing locks. No legacy terminal/PID fabrication or clean old-controller inference.'}

def unpack_wheels(archive,destination):
    require(not destination.exists(),'fresh offline wheel directory required')
    require(sha(archive)==WHEEL_ARCHIVE_SHA,'wheel archive pin changed')
    with tarfile.open(archive)as tf:
        members=tf.getmembers();require(len(members)==36,'expected35 wheels and manifest')
        total=0;names=set()
        for item in members:
            p=Path(item.name);require(item.isfile()and len(p.parts)==2 and p.parts[0]==WHEEL_MEMBER_PREFIX and
                p.parts[1]not in names and(p.suffix=='.whl'or p.name=='WHEEL_MANIFEST.json'),'bounded regular wheel member only')
            require(0<=item.size<=24*MIB,'single wheel24MiB bound');total+=item.size;names.add(p.parts[1])
        require(total<=64*MIB,'wheel extraction64MiB bound')
        destination.mkdir()
        for item in members:
            with tf.extractfile(item)as src,(destination/Path(item.name).name).open('xb')as out:
                while True:
                    block=src.read(1<<20)
                    if not block:break
                    out.write(block)
    m=read(destination/'WHEEL_MANIFEST.json');require(len(m['files'])==35,'wheel manifest count')
    for item in m['files']:
        p=destination/item['path'];require(p.parent==destination and info(p)=={'bytes':item['bytes'],'sha256':item['sha256']},'wheel content pin changed')
    return {'files':len(m['files']),'uncompressed_bytes':total,'manifest_sha256':sha(destination/'WHEEL_MANIFEST.json')}

def clean_exit(exit):
    return exit.get('returncode')==0 and not exit.get('timed_out')and exit.get('owned_tree_quiescent')is True and not exit.get('orphan_cleanup')and exit.get('guard_abort_reason')is None

def add_base_site(venv,before,*,allowed_roots=(Path('/opt/python'),Path('/usr')),site_relative=Path('lib/python3.10/site-packages')):
    """Plain fallback paths only; preserve isolated package precedence in a nested venv."""
    allowed=[]
    for value in before['site_packages']:
        require(isinstance(value,str)and '\n'not in value and '\r'not in value and '\x00'not in value,'plain path line required')
        path=Path(value).resolve()
        require(path.is_absolute()and path.is_dir()and path.name in('site-packages','dist-packages')and
            any(path.is_relative_to(root.resolve())for root in allowed_roots),'unexpected original base site path')
        require(not path.is_relative_to(venv.resolve()),'fallback must refer to original base, not new environment')
        if str(path)not in allowed:allowed.append(str(path))
    require(allowed,'original existing base site paths unavailable')
    site=venv/site_relative;require(site.is_dir()and not site.is_symlink(),'fresh nested venv site path required')
    target=site/'pencil_base_site_v3.pth';require(not os.path.lexists(target),'preserve existing route; no override')
    with target.open('x')as f:f.write('\n'.join(allowed)+'\n')
    return {'path':str(target),**info(target),'base_paths':allowed,
        'scope':'Plain existing before-probe paths only, appended by site after isolated site; no executable pth code or sitecustomize.'}

def verify_environment_binding(before,isolated,after):
    require(before==after,'base environment changed despite isolated v2 install')
    require(before['critical_versions']==isolated['critical_versions']and before['critical_modules']==isolated['critical_modules'],
        'isolated environment replaced original numerical version/module origin')
    require(before['torch_sparse_binaries']==isolated['torch_sparse_binaries']and before['saint_wrapper']==isolated['saint_wrapper'],
        'isolated environment replaced native binaries/wrapper')
    require(isolated['base_site_route']['isolated_site_precedes_all_base_paths']is True,'isolated package precedence not verified')
    return {'status':'pass','base_unchanged':True,'numerical_and_native_origins_exact':True,'isolated_site_precedes_base_fallback':True}

def bind_environment(isolated,venv):
    paths=[Path(isolated['base_site_route']['path']),venv/'pyvenv.cfg']
    policy=isolated['runtime_compatibility'];origin=Path(policy['import_utils_origin'])
    require(sha(origin)==policy['import_utils_sha256']=='9d7f63effbd7c8dcfe69e7f34cae0d164db9cca2d5f4d887405ae0f3e2ba455a'and
        origin==Path(isolated['extra_modules']['transformers']['origin']).parent/'utils/import_utils.py',
        'loaded pinned Transformers private optional-vision API origin changed')
    require(policy['policy_source_sha256']==sha(BRIDGE/'graph_only_transformers_v2.py'),'loaded graph-only policy source changed')
    paths.extend([origin,BRIDGE/'graph_only_transformers_v2.py'])
    for value in isolated['critical_modules'].values():
        p=Path(value['origin']);require(sha(p)==value['source_sha256'],'original numerical module source changed');paths.append(p)
    for value in isolated['extra_modules'].values():
        p=Path(value['origin']);require(sha(p)==value['sha256'],'loaded extra dependency source changed');paths.append(p)
    native=Path(isolated['critical_modules']['torch_sparse']['origin']).parent
    for name,value in isolated['torch_sparse_binaries'].items():
        p=native/name;require(p.parent==native and info(p)==value,'loaded native binary changed');paths.append(p)
    wrapper=native/'saint.py';require(sha(wrapper)==isolated['saint_wrapper']['sha256'],'loaded native wrapper changed');paths.append(wrapper)
    return {str(p):info(p)for p in paths}

def guarded_run(supervisor,memory,argv,log,cwd,timeout,*,gpu=False):
    abort={'reason':None};last=[0.];errors=[0];guard=log.with_suffix('.guard.jsonl')
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
            if guard.stat().st_size>8*MIB:abort['reason']='bounded guard-log8MiB limit'
        if log.exists()and log.stat().st_size>4*MIB:abort['reason']='bounded stdout-log4MiB limit'
        return found
    def popen(*args,**kwargs):
        kwargs['env'].update(CUDA_VISIBLE_DEVICES='0'if gpu else'',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',
            HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONNOUSERSITE='1')
        return subprocess.Popen(*args,**kwargs)
    try:result=supervisor.run_owned(argv,log,cwd=cwd,timeout=timeout,clock=clock,popen=popen,groups_reader=groups)
    except BaseException:
        cleanup_owned(supervisor);raise
    result.update(guard_abort_reason=abort['reason'],guard_policy='unchanged stable_pss_individual_rss_cgroup90_v3 sampled2s;180GiB',
                  exactly_one_visible_gpu=gpu)
    save(log.with_suffix('.exit.json'),result);return result

def cleanup_owned(supervisor):
    for sig in(signal.SIGTERM,signal.SIGKILL):
        supervisor.signal_confirmed(supervisor.owned_groups(os.getpid()),sig)
        until=time.monotonic()+5
        while time.monotonic()<until and supervisor.owned_groups(os.getpid()):time.sleep(.1)
    return not supervisor.owned_groups(os.getpid())

def real_inputs(base):
    return ['--graph-path',str(base/'data/citation2/prepared/train_csr'),
        '--features',str(base/'data/citation2/prepared/node_feat.npy'),
        '--valid-positive',str(base/'results/quality/subsets/valid_edge.npy'),
        '--valid-negative',str(base/'results/quality/subsets/valid_edge_neg.npy'),
        '--strict-manifest',str(base/'data/citation2/prepared/manifest.json'),
        '--quality-plan',str(base/'results/quality/plan.json')]

def make_jobs(base,python):
    L=layout(base);out=L['out'];tiny=out/'tiny_trace';trace=out/'real_trace';script=BRIDGE/'pencil_bridge.py'
    cpp=['--cpp-module',str(L['cpp_module']),'--cpp-library',str(L['cpp_library']),'--cpp-build-receipt',str(L['cpp_receipt'])]
    rows=[]
    def add(name,kind,tail,*,gpu=False,**metadata):
        output=out/(name+'.json');rows.append({'name':name,'kind':kind,'argv':[str(python),'-u',str(script),*tail,'--output',str(output)],
            'output':str(output),'gpu':gpu,'timeout_seconds':1800,**metadata})
    add('tiny_capture','capture',['capture','--fixture','--execute','--trace-dir',str(tiny)])
    add('tiny_audit','audit',['audit','--fixture','--execute','--trace-dir',str(tiny),'--arms',','.join(ARMS),*cpp,
        '--forward','--tiny-model','--training-wrapper','--batch-size','128'])
    add('real_preflight','preflight',['preflight',*real_inputs(base)])
    add('real_capture','capture',['capture','--execute',*real_inputs(base),'--trace-dir',str(trace)])
    for precision in('fp32','bf16'):
        for batch in(128,512):
            for workers in(0,1):
                name=f'audit_{precision}_b{batch}_w{workers}'
                add(name,'audit',['audit','--execute',*real_inputs(base),'--trace-dir',str(trace),'--arms',','.join(ARMS),*cpp,
                    '--forward','--training-wrapper','--device','cuda','--precision',precision,'--batch-size',str(batch),
                    '--num-workers',str(workers),'--max-logit-abs','0'],gpu=True,precision=precision,batch=batch,workers=workers)
    for mode in('online','replay'):
        combinations=[(b,w,ARMS)for b in(128,512)for w in(0,1)]if mode=='online'else[(128,0,REPLAY_ARMS)]
        for batch,workers,arms in combinations:
            for arm in arms:
                name=f'bench_{mode}_bf16_b{batch}_w{workers}_{arm}'
                add(name,'benchmark',['benchmark','--execute',*real_inputs(base),'--trace-dir',str(trace),*cpp,
                    '--audit-receipt',str(out/f'audit_bf16_b{batch}_w{workers}.json'),'--arm',arm,'--pipeline-mode',mode,
                    '--forward','--device','cuda','--precision','bf16','--batch-size',str(batch),'--num-workers',str(workers),
                    '--max-logit-abs','0','--warmup','1','--repeats','3'],gpu=True,mode=mode,precision='bf16',batch=batch,workers=workers,arm=arm)
    return rows

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
    expected[('bf16',128,0,'replay')]=set(REPLAY_ARMS)
    require(set(grouped)==set(expected),'all5predeclared measured configuration groups required')
    reports=[]
    for key,arms in grouped.items():
        require(set(arms)==expected[key],'full arm set missing from measured score group')
        reference=arms['torch_sparse_public'][0]
        require(reference.get('kind')=='tensor'and reference.get('shape',[0])[0]==1024,'complete1024model scores required')
        require(all(len(scores)==4 and all(s==reference for s in scores)for scores in arms.values()),'measured model score fingerprints differ across arms/repeats: '+str(key))
        reports.append({'precision':key[0],'batch':key[1],'workers':key[2],'mode':key[3],'arms':list(arms),
            'warmup_and_3repeats_all_model_scores_exact':True,'score_fingerprint':reference})
    return {'status':'pass','groups':reports,'scope':'1024measured model scores exact within each precision/batch/workers/mode group; shapes are not pooled; no quality metric.'}

def pack_once(out,target):
    require(not target.exists(),'fresh final archive required')
    files={}
    for p in sorted(out.rglob('*')):
        require(not p.is_symlink(),'symlink result refused')
        if not p.is_file():continue
        rel=p.relative_to(out)
        require(p.suffix in('.json','.jsonl','.log')or p.name.endswith(('.json.partial','.json.tmp'))or
            (p.suffix=='.npy'and rel.parts[0]in('tiny_trace','real_trace')),'only whole result/provenance/trace files allowed')
        value=info(p);require(value['bytes']<=32*MIB,'whole rawfile32MiB limit; never truncate or omit verbose audits')
        files[str(rel)]={'path':p,'info':value}
    require(sum(v['info']['bytes']for v in files.values())<=120*MIB,'whole raw result payload120MiB guard; no truncation/exclusion')
    values={'results/pencil_bridge_v3/'+name:v['info']for name,v in files.items()}
    data=(json.dumps({'schema_version':1,'files':values,'scope':'One final PENCIL result archive, complete raw receipts/logs/cohort trace and protocol/source/binary hashes only. No source/wheel objects, full graph payloads, environment tree, model checkpoint or training quality claim.'},indent=2)+'\n').encode()
    with tarfile.open(target,'x:gz')as tf:
        for name,item in files.items():
            p=item['path'];require(info(p)==item['info'],'raw result changed before pack')
            entry=tarfile.TarInfo('results/pencil_bridge_v3/'+name);entry.size=item['info']['bytes'];entry.mode=0o644
            with p.open('rb')as f:tf.addfile(entry,f)
            require(info(p)==item['info'],'raw result changed during pack')
        entry=tarfile.TarInfo('SHA256_MANIFEST.json');entry.size=len(data);tf.addfile(entry,io.BytesIO(data))
    require(target.stat().st_size<=MAX_ARCHIVE,'coarse final compressed archive128MiB bound')
    return {**info(target),'files':len(files),'raw_bytes':sum(v['bytes']for v in values.values()),'manifest_sha256':hashlib.sha256(data).hexdigest()}

def upload_once(archive,target,client=None):
    require(target==YT_TARGET,'only exact one new PENCIL YT archive path allowed')
    expected=info(archive);require(expected['bytes']<=MAX_ARCHIVE,'bounded final upload')
    if client is None:
        import yt.wrapper as yt
        client=yt.YtClient(proxy='jupiter.yt.vk.team',config={'proxy':{'request_timeout':120000}})
    require(not client.exists(target),'existing YT node; no overwrite/retry object')
    with client.Transaction(timeout=600000):
        client.create('file',target)
        with archive.open('rb')as f:client.write_file(target,f)
        h=hashlib.sha256();size=0;stream=client.read_file(target)
        try:
            for block in iter(lambda:stream.read(1<<20),b''):
                size+=len(block);require(size<=MAX_ARCHIVE,'bounded final archive readback');h.update(block)
        finally:stream.close()
        require({'bytes':size,'sha256':h.hexdigest()}==expected,'final YT archive readback mismatch')
    return {'yt_path':target,**expected,'readback_verified':True,'new_YT_file_nodes':1}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=BASE);p.add_argument('--execute',action='store_true')
    p.add_argument('--max-hours',type=float,default=12);p.add_argument('--admission-hours',type=float,default=8)
    p.add_argument('--recovery-admission-sha256',help='Explicit root SHA for RECOVERY_ADMISSION.json; original deleted pod terminals are never fabricated')
    p.add_argument('--persist-worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--archive-sha256',help=argparse.SUPPRESS);p.add_argument('--archive-bytes',type=int,help=argparse.SUPPRESS);a=p.parse_args(argv)
    require(all(math.isfinite(v)and v>0 for v in(a.max_hours,a.admission_hours))and a.max_hours<=12 and
        a.admission_hours<=8 and a.admission_hours<a.max_hours,'finite TOTAL<=12h including admission<=8h required')
    base=a.base.resolve();L=layout(base);out=L['out'];archive=L['archive']
    if a.persist_worker:
        require(a.execute and a.archive_sha256 is not None and a.archive_bytes is not None,'explicit hash-bound persistence execution required')
        expected={'sha256':a.archive_sha256,'bytes':a.archive_bytes};require(info(archive)==expected,'predeclared final archive changed before upload')
        candidate=upload_once(archive,YT_TARGET);require(info(archive)==expected,'predeclared final archive changed during upload')
        save(out/'archive_pointer.json',candidate);return 0
    global RECOVERY_ADMISSION_SHA
    RECOVERY_ADMISSION_SHA=a.recovery_admission_sha256
    require(valid_sha(RECOVERY_ADMISSION_SHA),'explicit root-pinned recovery admission SHA required')
    bindings=verify_sources(base,full=not a.execute);jobs=make_jobs(base,L['venv']/'bin/python')
    require(not out.exists()and not archive.exists()and not L['venv'].exists()and not L['wheels'].exists(),'fresh queue/results/env/wheels required; no rerun')
    if not a.execute:
        print(json.dumps({'status':'preflight','no_execution':True,'total_jobs':len(jobs),'benchmark_jobs':36,'semantic_audits':8,'bindings':len(bindings),'global_total_hours':a.max_hours,'wait_max_hours':a.admission_hours}));return 0
    require(base==BASE,'execution scoped to reviewed root-owned BASE')
    out.mkdir(parents=True);started=time.monotonic();deadline=started+a.max_hours*3600;wait_deadline=min(deadline-1200,started+a.admission_hours*3600)
    supervisor=load_module(HERE/'third_party/reviewed_supervision_v3.py','pencil_supervision_v3')
    memory=load_module(HERE/'third_party/memory_guard_v3.py','pencil_memory_v3')
    gate=load_module(HERE/'third_party/native_gate/native_controls_queue_v1.py','pencil_main_gate')
    supervisor.enable_subreaper();lock=base/'results/ADDITIONAL_TIMING_RUNNING.lock'
    identity=supervisor.process_identity(os.getpid());require(identity is not None,'own process identity required')
    identity={'pid':os.getpid(),'start_ticks':identity['start_ticks'],'entrypoint':str(Path(__file__).resolve()),'source_sha256':sha(Path(__file__))}
    token={'kind':'pencil_queue_v3','pid':os.getpid(),'start_ticks':identity['start_ticks'],'started_unix':time.time()}
    owned=False;failure=None;completed=[];pointer=None;checks=None;admitted=None
    def status(phase,**kw):save(out/'status.json',{'phase':phase,'updated_unix':time.time(),'completed_jobs':len(completed),**kw})
    def interrupt(sig,frame):raise KeyboardInterrupt('received signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupt)
    save(out/'plan.json',{'schema_version':1,'kind':'pencil_queue_v3','protocol_spec':read(HERE/'protocol_spec.json'),
        'source_bindings':bindings,'controller_identity':identity,'bridge_source_package_sha256':SOURCE_PACKAGE_SHA,'jobs':jobs,
        'max_total_hours_including_wait':a.max_hours,'max_admission_hours':a.admission_hours,'external_sampled_memory_guard_gib':180,
        'per_job_timeout_seconds':1800,'declared_workers_audit_note':'Audit itself runs sequentially; worker field is bound for subsequent loader throughput admission.',
        'recovery_admission_sha256':RECOVERY_ADMISSION_SHA,'scope':'PENCIL instrumented systems throughput and exact functional equivalence only. All native representations remain resident. No training/HPO/MRR/quality claim.'})
    def run(name,argv,*,gpu=False):
        assert_bindings(bindings);require(read(lock)==token,'exclusive timing lock lost')
        require(admission(base,gate,supervisor,owned_lock=lock)is not None,'predecessor or foreign timing quiescence changed')
        remaining=deadline-time.monotonic()-1200;require(remaining>=60,'finite global budget reached; no further job')
        exit=guarded_run(supervisor,memory,argv,out/(name+'.log'),base,min(1800,remaining),gpu=gpu)
        require(clean_exit(exit),'failed/guarded owned phase: '+name);return exit
    try:
        while time.monotonic()<wait_deadline:
            assert_bindings(bindings);admitted=admission(base,gate,supervisor)
            if admitted:break
            status('waiting_for_native_terminal_main_pipeline_and_process_quiescence');time.sleep(min(15,max(0,wait_deadline-time.monotonic())))
        require(admitted is not None,'finite admission deadline reached; no environment/job launch')
        gate.acquire_lock(lock,token);owned=True
        require(admission(base,gate,supervisor,owned_lock=lock)==admitted,'admission changed during exclusive lock acquisition')
        # Full58MB wheel hashing/extraction only after safe receipt/process/lock
        # admission and exclusive timing ownership; never in the wait loop.
        bindings.update(admitted['bindings']);bindings.update(verify_sources(base))
        save(out/'admission.json',admitted);save(out/'runtime_bindings.json',bindings)
        save(out/'wheel_extraction.json',unpack_wheels(L['wheel_archive'],L['wheels']))
        run('base_environment_before',[sys.executable,'-u',str(HERE/'runtime_probe_v3.py'),'--mode','base','--output',str(out/'base_environment_before.json')])
        run('create_isolated_env',[sys.executable,'-m','venv','--system-site-packages',str(L['venv'])])
        before=read(out/'base_environment_before.json')
        route=add_base_site(L['venv'],before);save(out/'base_site_binding.json',route)
        bindings[route['path']]=info(Path(route['path']));bindings[str(L['venv']/'pyvenv.cfg')]=info(L['venv']/'pyvenv.cfg')
        python=L['venv']/'bin/python'
        run('install_offline_extras',[str(python),'-m','pip','install','--no-index','--no-cache-dir','--find-links',str(L['wheels']),
            '-r',str(BRIDGE/'requirements-extras.txt'),'-c',str(BRIDGE/'constraints-linux-cp310.txt')])
        run('isolated_environment',[str(python),'-u',str(HERE/'runtime_probe_v3.py'),'--mode','isolated','--output',str(out/'isolated_environment.json')],gpu=True)
        isolated=read(out/'isolated_environment.json');verify_environment_binding(before,isolated,before)
        bindings.update(bind_environment(isolated,L['venv']));save(out/'environment_runtime_bindings.json',bindings)
        for job in jobs:
            status('running_'+job['name']);exit=run(job['name'],job['argv'],gpu=job['gpu'])
            checked=validate_job(job,exit)
            if job['kind']!='preflight':require(checked['runtime_compatibility']==isolated['runtime_compatibility'],'job optional-vision policy differs from admitted isolated probe')
            completed.append({'name':job['name'],'kind':job['kind'],'output':info(job['output']),'exit':exit})
            save(out/'completed_jobs.json',{'jobs':completed,'count':len(completed)})
        checks=crossjob_scores(jobs);save(out/'crossjob_model_scores.json',checks)
        run('isolated_environment_after',[str(python),'-u',str(HERE/'runtime_probe_v3.py'),'--mode','isolated',
            '--output',str(out/'isolated_environment_after.json')],gpu=True)
        require(isolated==read(out/'isolated_environment_after.json'),'isolated environment/source/route changed across protocol')
        run('base_environment_after',[sys.executable,'-u',str(HERE/'runtime_probe_v3.py'),'--mode','base','--output',str(out/'base_environment_after.json')])
        after=read(out/'base_environment_after.json');save(out/'environment_verification.json',verify_environment_binding(before,isolated,after))
        assert_bindings(bindings);require(len(completed)==48,'all48predeclared scientific/preflight jobs required')
    except BaseException as e:failure={'type':type(e).__name__,'message':str(e)[:700]}
    finally:
        quiescent=cleanup_owned(supervisor)
        save(out/'SCIENTIFIC_TERMINAL.json',{'controller_jobs_finished':quiescent,'success':quiescent and failure is None and checks is not None and len(completed)==48,
            'completed_jobs':len(completed),'failure':failure,'crossjob_model_scores':checks,'owned_tree_quiescent':quiescent,
            'timing_lock_retained':owned,'runtime_seconds':time.monotonic()-started})
    try:
        require(owned and quiescent and read(lock)==token,'safe exclusive lock and owned-tree quiescence required for final persistence')
        assert_bindings(bindings);require(time.monotonic()<deadline-60,'global deadline before final result persistence')
        packed=pack_once(out,archive);save(out/'archive_local_receipt.json',packed)
        exit=guarded_run(supervisor,memory,[sys.executable,'-u',str(Path(__file__).absolute()),'--base',str(base),'--execute','--persist-worker',
            '--archive-sha256',packed['sha256'],'--archive-bytes',str(packed['bytes'])],
            out/'persistence.log',base,min(600,max(1,deadline-time.monotonic()-30)))
        require(clean_exit(exit),'final persistence phase failed');candidate=read(out/'archive_pointer.json')
        require(candidate.get('readback_verified')is True and candidate['sha256']==sha(archive)and candidate['yt_path']==YT_TARGET and
            candidate.get('bytes')==archive.stat().st_size and candidate.get('new_YT_file_nodes')==1,
            'durable PENCIL final pointer mismatch')
        require(info(archive)=={'bytes':packed['bytes'],'sha256':packed['sha256']},'predeclared archive changed during persistence')
        assert_bindings(bindings);require(read(lock)==token,'exclusive lock changed during final persistence')
        pointer=candidate
    except BaseException as e:
        failure=failure or {'type':type(e).__name__,'message':str(e)[:700]};cleanup_owned(supervisor)
    finally:
        quiescent=cleanup_owned(supervisor)
        if owned and quiescent:
            try:gate.release_lock(lock,token,True);owned=False
            except BaseException as e:failure=failure or {'type':type(e).__name__,'message':str(e)[:700]}
    terminal=read(out/'SCIENTIFIC_TERMINAL.json')
    success=terminal['success']and failure is None and pointer is not None and quiescent and not owned
    save(out/'COMPLETE.json',{'controller_finished':quiescent and not owned,'success':success,'scientific_complete':terminal['success'],
        'persistence_complete':pointer is not None,'archive_pointer':pointer,'failure':failure,'timing_lock_released':not owned,
        'owned_tree_quiescent':not supervisor.owned_groups(os.getpid()),'controller_identity':identity,'runtime_seconds':time.monotonic()-started})
    status('complete'if success else'completed_with_failures',success=success);return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
