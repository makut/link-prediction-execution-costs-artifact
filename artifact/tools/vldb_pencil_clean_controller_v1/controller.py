#!/usr/bin/env python3
"""NEW finite LOCAL clean-consumer controller candidate; offline plan by default.

No data fetch, installation, Kubernetes API, login, self-release or restart.
Execution needs a fresh independently pinned root admission after review.
"""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,math,os,shutil,signal,socket,subprocess,sys,tarfile,time,traceback,uuid
sys.dont_write_bytecode=True
from contracts import HERE,BRIDGE,BASE,GIB,TARGET,require,read,save,sha,identity,snapshot,clean,copy_bound_json,verify_sources,verify_admission,verify_inputs,verify_input_continuity
from runner import Runner,StopRequested
from ownership import Ownership
import science_validation as science

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def make_plan(args):
    # Derive the same complete matrix directly from its frozen builder. Fresh
    # C++ outputs live inside OUT and are not treated as pre-existing inputs.
    source=HERE/'candidate/metadata/ORIGINAL_SCIENTIFIC_JOB_BUILDER.py'
    real=['--graph-path',str(args.prepared_root/'train_csr'),'--features',str(args.prepared_root/'node_feat.npy'),
        '--valid-positive',str(args.validation_root/'valid_edge.npy'),'--valid-negative',str(args.validation_root/'valid_edge_neg.npy'),
        '--strict-manifest',str(args.prepared_root/'manifest.json')]
    layout={'out':args.out,'cpp_module':BRIDGE/'corrected_native/native_adapter_v1.py',
        'cpp_library':args.out/'cpp_build/native_assoc_v1.so','cpp_receipt':args.out/'cpp_build/build_receipt_v1.json'}
    namespace={'Path':Path,'BRIDGE':BRIDGE,'ARMS':science.ARMS,'REPLAY_ARMS':science.REPLAY_ARMS,
        'layout':lambda _:layout,'real_inputs':lambda _:real}
    exec(compile(source.read_text(),str(source),'exec'),namespace)
    jobs=[{**j,'argv':[*j['argv'],'--query-memory','disabled']}for j in namespace['make_jobs'](args.base,args.python)if j.get('mode')!='replay']
    require(len(jobs)==44 and sum(j['kind']=='benchmark'for j in jobs)==32,'complete44-job/32ONLINE matrix required')
    value={'status':'ROOT_REVIEWABLE_CONTROLLER_PLAN','execution_authorized':False,'jobs':jobs,'scientific_jobs':44,
        'protocol':read(HERE/'candidate/PROTOCOL.json'),'source_job_builder_sha256':sha(source),'output_created':False}
    value['inherited_candidate_GPU_target_reference']=value['protocol']['GPU_target']
    value['protocol']['GPU_target']='one visible H10080Single per job; active controller27CPU205/256GiB contract'
    value['controller_protocol']=read(HERE/'PROTOCOL.json')
    value['setup_jobs']=[{'name':'platform_before','gpu':True,'timeout_seconds':120,
        'argv':[str(args.python),'-B',str(HERE/'platform_probe.py'),'--output',str(args.out/'platform_before.json')]},
        {'name':'build_cpp','gpu':False,'timeout_seconds':180,'argv':[str(args.python),'-B',str(BRIDGE/'corrected_native/build_native_v1.py'),'--output-dir',str(args.out/'cpp_build')]}]
    value['after_jobs']=[{'name':'platform_after','gpu':True,'timeout_seconds':120,
        'argv':[str(args.python),'-B',str(HERE/'platform_probe.py'),'--output',str(args.out/'platform_after.json')]}]
    return value

def validate_args(args):
    require(math.isfinite(args.max_hours)and .25<args.max_hours<=4,'finite total budget above15minutes and at most4hours required')
    require(args.memory_guard_gib==96 and args.pod_memory_limit_gib==256,'reviewed96GiB guard/256GiB pod contract required')
    for name in('base','out','prepared_root','validation_root'):
        setattr(args,name,getattr(args,name).resolve())
    # A venv interpreter is normally a symlink; resolving it would silently
    # launch the base interpreter and discard the supplied pinned HF environment.
    args.python=Path(os.path.abspath(args.python))
    require(not args.out.exists()and not args.out.is_symlink(),'fresh output required; failures are never resumed or overwritten')
    for value in(HERE,args.prepared_root,args.validation_root,args.python):
        require(not args.out.is_relative_to(value)and not value.is_relative_to(args.out),'output overlaps immutable source/input')
    require(args.target==TARGET,'one fixed coarse final YT target only')

def bind_driver(raw,admitted):
    names=('bridge_core.py','pencil_runtime.py','pencil_bridge.py','graph_only_transformers_v2.py',
           'upstream_manifest.json','input_contract.json','bert_config/config.json','bert_config/source.json')
    require(raw['source_files']=={name:sha(BRIDGE/name)for name in names},'fresh driver source bindings differ')
    require(raw['configuration']['query_memory']=='disabled','clean experiment requires explicit disabled query memory mode')
    if raw['command']=='preflight':return
    native=raw['native_setup'];policy=admitted['runtime_compatibility']
    require(raw['runtime_compatibility']==policy and raw['audit_environment']['runtime_compatibility']==policy,'actual driver HF policy differs from supplied-stack probe')
    require(native['torch']==admitted['torch']and native['torch_sparse']==admitted['torch_sparse_version']and native['pyg']==admitted['pyg']
        and native['wrapper_sha256']==admitted['saint_wrapper']['sha256']and native['wrapper_git_blob']==admitted['saint_wrapper']['git_blob'],'actual numerical/native wrapper differs')
    require({r['name']:r['sha256']for r in native['extension_binaries']}=={name:r['sha256']for name,r in admitted['torch_sparse_binaries'].items()},'native library changed after probe')
    require(all(native['software_versions'][name]==admitted['packages'][name]for name in('numpy','numba','scipy','transformers','wandb')),'driver numerical/HF versions differ')
    if raw['command']=='benchmark':
        result=raw['result'];require(result['memory_measurement_policy']['query_memory']=='disabled'
            and result['memory_measurement_policy']['per_query_memory_samples_enabled']is False,'new explicit memory policy missing')
        for sweep in result['sweeps']:
            require(sweep['memory_measurement_policy']==result['memory_measurement_policy'],'sweep memory policy changed')
            require(bool(sweep['worker_memory'])and all(row['max_pss_bytes']is None and row['maxrss_bytes']is None
                and 'unobserved'in row['memory_measurement']for row in sweep['worker_memory']),'disabled worker memory must be null/unobserved, never fakezero')
            require(isinstance(sweep['parent_memory_before_sweep'],dict)and isinstance(sweep['parent_memory_after_sweep'],dict),'actual outside-wall parent observations required')

def gpu_idle(uuid_value):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,timeout=10,check=True)
    require(len(result.stdout)<=64<<10,'bounded GPU process inventory required')
    import csv
    rows=list(csv.reader(result.stdout.splitlines(),skipinitialspace=True))
    require(all(len(row)==2 and row[0].startswith('GPU-')and row[1].isdigit()for row in rows),'unrecognized GPU process inventory')
    require(not any(row[0]==uuid_value for row in rows),'allocated GPU is busy; no foreign process will be signaled')

def archive_payload(out,archive,check):
    """Whole static evidence, excluding only declared cache directories."""
    out=Path(out);archive=Path(archive);require(not archive.exists(),'preserve existing coarse archive')
    files=[];total=0
    for path in sorted(out.rglob('*')):
        require(not path.is_symlink(),'output symlink refused during finalization')
        if not path.is_file()or path.relative_to(out).parts[0]=='cache':continue
        check();row={'path':str(path.relative_to(out)),**identity(path,check)};files.append(row);total+=row['bytes']
    require(total<=384<<20,'whole evidence exceeds384MiB; no truncation or partial success')
    manifest={'kind':'whole_clean_consumer_evidence_v1','files':files,'file_count':len(files),'payload_bytes':total,
        'excluded':'only declared cache/ directories; no raw,trace,guard,exit or failed-job evidence omitted'}
    save(out/'SHA256_MANIFEST.json',manifest,exclusive=True)
    temporary=archive.with_name(archive.name+'.partial-'+uuid.uuid4().hex[:8])
    with tarfile.open(temporary,'x:gz',compresslevel=1)as target:
        for row in files:
            check();path=out/row['path'];before=snapshot(path)
            target.add(path,arcname=row['path'],recursive=False)
            require(identity(path,check)=={k:row[k]for k in('bytes','sha256')}and snapshot(path)==before,'evidence changed during whole archive creation')
        target.add(out/'SHA256_MANIFEST.json',arcname='SHA256_MANIFEST.json',recursive=False)
    require(temporary.stat().st_size<=384<<20,'bounded coarse archive required')
    expected={row['path']:{k:row[k]for k in('bytes','sha256')}for row in files};expected['SHA256_MANIFEST.json']=identity(out/'SHA256_MANIFEST.json')
    observed={}
    with tarfile.open(temporary,'r|gz')as source:
        for member in source:
            check();require(member.isfile()and member.name in expected and member.name not in observed,'exact regular archive members required')
            h=hashlib.sha256();count=0;stream=source.extractfile(member)
            for block in iter(lambda:stream.read(4<<20),b''):check();h.update(block);count+=len(block)
            observed[member.name]={'bytes':count,'sha256':h.hexdigest()}
    require(observed==expected,'whole local coarse archive readback differs')
    temporary.replace(archive);return {**identity(archive,check),'file_count':len(expected),'whole_readback_verified':True}

def _execute_owned(args,plan,admission,resources):
    runner=resources.runner
    completed=[];success=False;failure=None;input_receipt=None;phase='initialization';published=None
    launch_sha=None
    try:
        args.out.mkdir();resources.out_created=True
        launch={'kind':'actual_clean_consumer_controller_launch','identity':runner.supervisor.process_identity(os.getpid()),'entrypoint':str(Path(__file__).resolve()),
            'source_sha256':sha(Path(__file__)),'source_manifest_sha256':args.source_manifest_sha256,'source_packet_sha256':args.source_packet_sha256,
            'root_admission_sha256':args.admission_sha256,'argv':sys.argv,'started_unix':time.time(),'hostname':socket.gethostname()}
        save(args.out/'LAUNCH.json',launch,exclusive=True);launch_sha=sha(args.out/'LAUNCH.json')
        save(args.out/'plan.json',plan,exclusive=True)
        copy_bound_json(args.admission,args.out/'ROOT_ADMISSION.json',args.admission_sha256,64<<10)
        copy_bound_json(HERE/'SOURCE_MANIFEST.json',args.out/'SOURCES.json',args.source_manifest_sha256)
        runner.supervisor.enable_subreaper()
        charged=runner.memory.cgroup_memory();require(charged['max_bytes']==256*GIB and 96*GIB<=charged['max_bytes']*9//10,'actual finite256GiB cgroup/96GiB safety guard required')
        save(args.out/'initial_cgroup.json',charged,exclusive=True)
        phase='full_public_input_identity'
        def check_science():require(runner.remaining()>600,'scientific phase exhausted final10minute reserve')
        input_receipt=verify_inputs(args.prepared_root,args.validation_root,check_science)
        save(args.out/'INPUT_IDENTITY.json',input_receipt,exclusive=True)
        for job in plan['setup_jobs']:
            phase=job['name'];verify_sources(args.source_manifest_sha256)
            require(clean(runner.run(job)),'fresh platform/native build phase failed')
        admitted=read(args.out/'platform_before.json');device_uuid=admitted['visible_gpu']['uuid']
        for job in plan['jobs']:
            phase=job['name'];verify_sources(args.source_manifest_sha256);verify_input_continuity(input_receipt)
            if job['gpu']:gpu_idle(device_uuid)
            exit=runner.run(job);raw=science.validate_job(job,exit);bind_driver(raw,admitted)
            completed.append({'name':job['name'],'kind':job['kind'],'output':identity(job['output']),'exit':exit})
            save(args.out/'completed_jobs.json',{'count':len(completed),'jobs':completed})
        phase='crossjob_model_scores';scores=science.crossjob_scores(plan['jobs'])
        save(args.out/'crossjob_scores.json',scores,exclusive=True)
        for job in plan['after_jobs']:
            phase=job['name'];gpu_idle(device_uuid);require(clean(runner.run(job)),'final platform probe failed')
        require(read(args.out/'platform_before.json')==read(args.out/'platform_after.json'),'fresh platform/native/numeric/HF environment changed')
        verify_input_continuity(input_receipt);verify_sources(args.source_manifest_sha256)
        require(len(completed)==44,'all44 scientific jobs required');success=True
    except BaseException as error:
        failure={'phase':phase,'type':type(error).__name__,'message':str(error)[:1000],'stop_requested':runner.stopped}
    finally:
        runner.finalizing=True;quiescent=runner.cleanup_owned(10)
        terminal={'kind':'PENCIL_CLEAN_CONSUMER_SCIENTIFIC_TERMINAL_V1','scientific_success':bool(success and quiescent),
            'completed_jobs':len(completed),'planned_jobs':44,'failure':failure,'stop_requested':runner.stopped,
            'owned_children_quiescent_before_publication':quiescent,'all_planned_science_job_exits_clean':len(completed)==44 and all(clean(r['exit'])for r in completed),
            'source_manifest_sha256':args.source_manifest_sha256,'source_packet_sha256':args.source_packet_sha256,
            'launch_sha256':launch_sha,'root_admission_sha256':args.admission_sha256,'outer_final_pid_exit':'UNKNOWN','finished_unix':time.time()}
        save(args.out/'SCIENTIFIC_TERMINAL.json',terminal,exclusive=True)
        try:
            require(quiescent,'owned child tree not quiescent; no immutable success archive')
            require(launch_sha is not None,'actual launch receipt unavailable; preserve local failed initialization without claiming durable launch provenance')
            def final_check():runner.check(final=True);require(runner.remaining(final=True)>15,'finalization reserve exhausted')
            phase='whole_local_archive';archive=args.out.with_name(args.out.name+'.tgz')
            archive_info=archive_payload(args.out,archive,final_check)
            context={'target':TARGET,'archive':{k:archive_info[k]for k in('bytes','sha256')},
                'scientific_terminal_sha256':sha(args.out/'SCIENTIFIC_TERMINAL.json'),'scientific_success':terminal['scientific_success'],
                'all_planned_science_job_exits_clean':terminal['all_planned_science_job_exits_clean'],
                'completed_jobs':len(completed),'planned_jobs':44,'source_manifest_sha256':args.source_manifest_sha256,
                'source_packet_sha256':args.source_packet_sha256,'controller_source_sha256':sha(Path(__file__)),
                'launch_sha256':launch_sha,'root_admission_sha256':args.admission_sha256,
                'pod_uid':admission['pod_uid'],'statefulset_uid':admission['statefulset_uid'],
                'stop_requested_at_context_creation':runner.stopped,'outer_final_pid_exit':'UNKNOWN'}
            save(args.out/'publication_context.json',context,exclusive=True)
            phase='one_coarse_file_persistence'
            job={'name':'publish_once','gpu':False,'timeout_seconds':60,'argv':[sys.executable,'-B',str(HERE/'publish_once.py'),
                '--archive',str(archive),'--context',str(args.out/'publication_context.json'),'--output',str(args.out/'PERSISTENCE.json')]}
            require(clean(runner.run(job,final=True)),'coarse persistence/attribute readback child failed or stopped')
            published=read(args.out/'PERSISTENCE.json')
        except BaseException as error:
            runner.cleanup_owned(8);save(args.out/'FINALIZATION_FAILED.json',{'phase':phase,'type':type(error).__name__,'message':str(error)[:1000],
                'partial_local_evidence_retained':True,'no_file_overwrite_or_second_remote_node':True,'stop_requested':runner.stopped},exclusive=True)
        finally:
            final_quiescent=runner.cleanup_owned(5)
            done={'kind':'PENCIL_CLEAN_CONSUMER_OUTER_COMPLETE_V1','controller_finished':True,'scientific_success':terminal['scientific_success'],
                'persistence_verified':published is not None,'success':terminal['scientific_success']and published is not None and not runner.stopped and final_quiescent,
                'owned_tree_quiescent':final_quiescent,'stop_requested':runner.stopped,'outer_final_pid_exit':'UNKNOWN',
                'persistence_attribute_canonical_json_sha256':published.get('attribute_canonical_json_sha256')if published else None,
                'scope':'This receipt precedes this process exit; no recursive claim of own final exit0. Durable tar+post-readback attribute are independently science-auditable.'}
            save(args.out/'COMPLETE.json',done,exclusive=True)
    return 143 if runner.stopped else 0 if done['success']and done['owned_tree_quiescent']else 2

def execute(args,plan):
    deadline=time.monotonic()+args.max_hours*3600
    require(sys.platform.startswith('linux'),'actual owned execution requires Linux')
    require(args.admission and args.admission_sha256 and args.source_manifest_sha256 and args.source_packet and args.source_packet_sha256,'independent root admission/source pins required')
    verify_sources(args.source_manifest_sha256)
    require(sha(args.source_packet)==args.source_packet_sha256,'actual staged source packet SHA differs')
    admission=verify_admission(args.admission,args.admission_sha256,args)
    args.out.parent.mkdir(parents=True,exist_ok=True);require(shutil.disk_usage(args.out.parent).free>=2*GIB,'twoGiB scratch reserve required')
    resources=Ownership(args,admission,deadline)
    result=2
    try:
        resources.acquire()
        result=_execute_owned(args,plan,admission,resources)
    except BaseException as error:
        resources.record_initialization_failure(error)
        result=143 if getattr(resources,'stopped',False)or(resources.runner and resources.runner.stopped)else 2
    finally:
        resources.close()
    if resources.quiescent is not True and result==0:result=2
    if getattr(resources,'stopped',False)or(resources.runner and resources.runner.stopped):result=143
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--base',type=Path,default=BASE);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepared-root',type=Path,default=BASE/'data/citation2/prepared')
    p.add_argument('--validation-root',type=Path,default=BASE/'clean_validation512')
    p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--max-hours',type=float,default=4)
    p.add_argument('--memory-guard-gib',type=float,default=96);p.add_argument('--pod-memory-limit-gib',type=float,default=256)
    p.add_argument('--target',default=TARGET);p.add_argument('--admission',type=Path);p.add_argument('--admission-sha256')
    p.add_argument('--source-manifest-sha256');p.add_argument('--source-packet',type=Path);p.add_argument('--source-packet-sha256')
    args=p.parse_args(argv);validate_args(args);plan=make_plan(args)
    if not args.execute:print(json.dumps(plan,indent=2));return 0
    return execute(args,plan)
if __name__=='__main__':raise SystemExit(main())
