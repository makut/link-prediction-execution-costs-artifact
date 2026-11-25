#!/usr/bin/env python3
"""Generic public exact44-job PENCIL clean consumer. Default prints plan only."""
from pathlib import Path
import argparse,csv,math,os,shutil,socket,subprocess,sys,time,uuid
sys.dont_write_bytecode=True
from common import *
from runner import Runner,StopRequested
from ownership import Ownership
from matrix import make_jobs
from local_archive import archive_payload
import science_validation as science

def validate_args(args):
    for name in('out','prepared_root','validation_root','coordination_dir'):setattr(args,name,getattr(args,name).resolve())
    args.python=Path(os.path.abspath(args.python)) # Preserve supplied venv symlink lexically.
    require(not args.out.exists()and not args.out.is_symlink(),'fresh output required; no resume or overwrite')
    for path in(HERE,args.prepared_root,args.validation_root):require(not args.out.is_relative_to(path)and not path.is_relative_to(args.out),'output overlaps immutable source/input')
    require(not args.coordination_dir.is_relative_to(args.out),'coordination directory must lie outside output')
    require(all(math.isfinite(v)and v>0 for v in(args.max_hours,args.memory_guard_gib,args.memory_limit_gib,args.minimum_gpu_memory_gib))and .25<args.max_hours<=4,'finite positive resources and15minute..4hour total bound required')
    require(args.memory_guard_gib<=.9*args.memory_limit_gib,'guard must fit below90percent of actual declared finite memory limit')
    require(args.gpu_uuid.startswith('GPU-')and str(uuid.UUID(args.gpu_uuid[4:]))==args.gpu_uuid[4:]and bool(args.gpu_name_contains),'explicit canonical whole GPU UUID/name admission required')

def make_plan(args):
    config={k:str(getattr(args,k))for k in('out','prepared_root','validation_root','python')};jobs=make_jobs(config)
    require(len(jobs)==44 and sum(j['kind']=='benchmark'for j in jobs)==32 and sum(j['kind']=='audit'for j in jobs)==9,'exact44/32/9 matrix required')
    probe=[str(args.python),'-B',str(HERE/'platform_probe.py'),'--gpu-uuid',args.gpu_uuid,'--gpu-name-contains',args.gpu_name_contains,'--minimum-gpu-memory-gib',str(args.minimum_gpu_memory_gib)]
    return {'kind':'PUBLIC_PORTABLE_PENCIL_CLEAN44_V1_PLAN','execution_requested':bool(args.execute),'output_created':False,'scientific_jobs':44,'throughput_jobs':32,'semantic_audits':9,'query_memory':'disabled','modes':['online'],'arms':list(science.ARMS),'jobs':jobs,
        'source_manifest_sha256':args.source_manifest_sha256,'setup_jobs':[{'name':'platform_before','gpu':True,'timeout_seconds':120,'argv':[*probe,'--output',str(args.out/'platform_before.json')]},{'name':'build_cpp','gpu':False,'timeout_seconds':180,'argv':[str(args.python),'-B',str(BRIDGE/'corrected_native/build_native_v1.py'),'--output-dir',str(args.out/'cpp_build')]}],
        'after_jobs':[{'name':'platform_after','gpu':True,'timeout_seconds':120,'argv':[*probe,'--output',str(args.out/'platform_after.json')]}],
        'resources':{'gpu_uuid_supplied_not_observed_yet':args.gpu_uuid,'gpu_name_contains':args.gpu_name_contains,'minimum_gpu_memory_gib':args.minimum_gpu_memory_gib,'memory_guard_gib':args.memory_guard_gib,'declared_actual_finite_cgroup_limit_gib':args.memory_limit_gib,'max_total_hours':args.max_hours,'final_reserve_seconds':600},
        'coordination':{'directory':str(args.coordination_dir),'persistent_flock':'PUBLIC_SYSTEMS_RUNNING.lock','shared_exclusive_active_marker':'PUBLIC_SYSTEMS_ACTIVE.json','role':'pencil44','stale_marker_policy':'refuse; never automatically unlink; retain if owned quiescence unknown'},
        'scope':'Fresh complete local exact44-job experiment. No archived result reuse, source/environment installation, internal hostname/admission, remote storage or publication.'}

def gpu_idle(gpu_uuid):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
    require(len(result.stdout)<=64<<10,'bounded GPU process inventory required');rows=list(csv.reader(result.stdout.splitlines(),skipinitialspace=True))
    require(all(len(r)==2 and r[0].startswith('GPU-')and r[1].isdigit()for r in rows),'recognized GPU process inventory required')
    require(not any(r[0]==gpu_uuid for r in rows),'supplied GPU is busy; no foreign process will be signaled')

def bind_driver(raw,platform):
    names=('bridge_core.py','pencil_runtime.py','pencil_bridge.py','graph_only_transformers_v2.py','upstream_manifest.json','input_contract.json','bert_config/config.json','bert_config/source.json')
    require(raw['source_files']=={n:sha(BRIDGE/n)for n in names}and raw['configuration']['query_memory']=='disabled','unchanged clean bridge source and explicit disabled query memory required')
    if raw['command']=='preflight':return
    native=raw['native_setup'];policy=platform['runtime_compatibility'];require(raw['runtime_compatibility']==raw['audit_environment']['runtime_compatibility']==policy,'fresh graph-only HF policy changed')
    require(native['torch']==platform['torch']and native['torch_sparse']==platform['torch_sparse_version']and native['pyg']==platform['pyg']and native['wrapper_sha256']==platform['saint_wrapper']['sha256']and native['wrapper_git_blob']==platform['saint_wrapper']['git_blob'],'actual unchanged native source/versions differ')
    require({r['name']:r['sha256']for r in native['extension_binaries']}=={n:r['sha256']for n,r in platform['torch_sparse_binaries'].items()}and all(native['software_versions'][n]==platform['packages'][n]for n in('numpy','numba','scipy','transformers','wandb')),'actual numerical/native/HF library bindings differ')
    if raw['command']=='benchmark':
        result=raw['result'];policy=result['memory_measurement_policy'];require(policy['query_memory']=='disabled'and policy['per_query_memory_samples_enabled']is False and policy['memory_saving_claim']is False,'explicit disabled result policy required')
        for sweep in result['sweeps']:
            require(sweep['memory_measurement_policy']==policy and bool(sweep['worker_memory'])and all(r['max_pss_bytes']is None and r['maxrss_bytes']is None and r['memory_measurement']=='disabled_unobserved; not zero or a worker peak'for r in sweep['worker_memory']),'worker memory must remain null/unobserved, never fake zero')
            for key in('parent_memory_before_sweep','parent_memory_after_sweep'):require(isinstance(sweep[key],dict)and all(type(v)is int and v>=0 for v in sweep[key].values())and all(k in sweep[key]for k in('maxrss_bytes','rss_bytes','pss_bytes')),'actual Linux parent before/after observations required outside wall')

def execute_owned(args,plan,resources):
    runner=resources.runner;completed=[];success=False;failure=None;phase='initialization';archive_info=None
    try:
        args.out.mkdir();resources.out_created=True;runner.supervisor.enable_subreaper()
        save(args.out/'LAUNCH.json',{'kind':'ACTUAL_PUBLIC_PENCIL_CLEAN44_LOCAL_LAUNCH','argv':sys.argv,'identity':runner.supervisor.process_identity(os.getpid()),'hostname_observed':socket.gethostname(),'source_manifest_sha256':args.source_manifest_sha256,'started_unix':time.time()},exclusive=True)
        save(args.out/'plan.json',plan,exclusive=True);copy_bound_json(HERE/'SOURCE_MANIFEST.json',args.out/'SOURCES.json',args.source_manifest_sha256)
        cgroup=runner.memory.cgroup_memory();require(cgroup['max_bytes']==int(args.memory_limit_gib*GIB)and args.memory_guard_gib*GIB<=cgroup['max_bytes']*.9,'actual finite cgroup limit differs from supplied resource admission');save(args.out/'initial_cgroup.json',cgroup,exclusive=True)
        phase='full_public_input_identity'
        def check_science():require(runner.remaining()>600,'ten minute final reserve reached')
        inputs=verify_inputs(args.prepared_root,args.validation_root,check_science);save(args.out/'INPUT_IDENTITY.json',inputs,exclusive=True)
        for job in plan['setup_jobs']:
            phase=job['name'];verify_sources(args.source_manifest_sha256)
            if job['gpu']:gpu_idle(args.gpu_uuid)
            require(clean(runner.run(job)),'fresh platform/build phase failed')
        admitted=read(args.out/'platform_before.json')
        for job in plan['jobs']:
            phase=job['name'];verify_sources(args.source_manifest_sha256);verify_input_continuity(inputs)
            if job['gpu']:gpu_idle(args.gpu_uuid)
            ex=runner.run(job);raw=science.validate_job(job,ex);bind_driver(raw,admitted);completed.append({'name':job['name'],'kind':job['kind'],'output':identity(job['output']),'exit':ex});save(args.out/'completed_jobs.json',{'count':len(completed),'jobs':completed})
        phase='crossjob_scores';save(args.out/'crossjob_scores.json',science.crossjob_scores(plan['jobs']),exclusive=True)
        for job in plan['after_jobs']:
            phase=job['name'];gpu_idle(args.gpu_uuid);require(clean(runner.run(job)),'final platform probe failed')
        require(read(args.out/'platform_before.json')==read(args.out/'platform_after.json'),'platform/native/numerical/HF environment changed');verify_sources(args.source_manifest_sha256);verify_input_continuity(inputs);require(len(completed)==44,'all44 science jobs required');success=True
    except BaseException as error:failure={'phase':phase,'type':type(error).__name__,'message':str(error)[:1000],'stop_requested':runner.stopped}
    finally:
        runner.finalizing=True;quiescent=runner.cleanup_owned(10);terminal={'kind':'PUBLIC_PENCIL_CLEAN44_SCIENTIFIC_TERMINAL','scientific_success':bool(success and quiescent),'completed_jobs':len(completed),'planned_jobs':44,'failure':failure,'stop_requested':runner.stopped,'owned_children_quiescent':quiescent,'source_manifest_sha256':args.source_manifest_sha256,'outer_final_pid_exit':'UNKNOWN','finished_unix':time.time()};save(args.out/'SCIENTIFIC_TERMINAL.json',terminal,exclusive=True)
        try:
            require(quiescent,'owned child tree not quiescent; local raw evidence retained without immutable archive claim')
            def final_check():runner.check(final=True);require(runner.remaining(final=True)>15,'bounded finalization deadline reached')
            archive=args.out.with_name(args.out.name+'.tgz');archive_info=archive_payload(args.out,archive,final_check);save(args.out/'LOCAL_ARCHIVE.json',{'archive':str(archive),**archive_info,'scientific_success':terminal['scientific_success'],'scope':'Whole local prefinalization evidence/readback. This receipt is outside tar; own final PID exit remains UNKNOWN. No remote persistence.'},exclusive=True)
        except BaseException as error:save(args.out/'FINALIZATION_FAILED.json',{'type':type(error).__name__,'message':str(error)[:1000],'partial_local_evidence_retained':True},exclusive=True)
        finally:
            final_quiescent=runner.cleanup_owned(5);done={'kind':'PUBLIC_PENCIL_CLEAN44_LOCAL_COMPLETE','scientific_success':terminal['scientific_success'],'archive_verified':archive_info is not None,'owned_tree_quiescent':final_quiescent,'stop_requested':runner.stopped,'outer_final_pid_exit':'UNKNOWN','success':bool(terminal['scientific_success']and archive_info is not None and final_quiescent and not runner.stopped)};save(args.out/'COMPLETE.json',done,exclusive=True)
    return 143 if runner.stopped else 0 if done['success']else 2

def execute(args,plan):
    require(sys.platform.startswith('linux'),'actual public execution requires documented Linux environment');require(args.source_manifest_sha256,'execution requires independent release source-manifest SHA pin');verify_sources(args.source_manifest_sha256)
    require(args.python.is_file(),'supplied interpreter missing');args.out.parent.mkdir(parents=True,exist_ok=True);require(shutil.disk_usage(args.out.parent).free>=2*GIB,'twoGiB local scratch reserve required');args.coordination_dir.mkdir(parents=True,exist_ok=True)
    resources=Ownership(args,time.monotonic()+args.max_hours*3600);code=2
    try:resources.acquire();code=execute_owned(args,plan,resources)
    except BaseException as error:resources.record_initialization_failure(error);code=143 if resources.stopped or bool(resources.runner and resources.runner.stopped)else 2
    finally:resources.close()
    if code==0 and resources.quiescent is not True:code=2
    if resources.stopped or bool(resources.runner and resources.runner.stopped):code=143
    return code

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    for name in('out','prepared-root','validation-root','coordination-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--source-manifest-sha256');p.add_argument('--gpu-uuid',required=True);p.add_argument('--gpu-name-contains',default='H100');p.add_argument('--minimum-gpu-memory-gib',type=float,default=79)
    p.add_argument('--memory-guard-gib',type=float,default=96);p.add_argument('--memory-limit-gib',type=float,default=256);p.add_argument('--max-hours',type=float,default=4);return p

def main(argv=None):
    args=parser().parse_args(argv);validate_args(args);plan=make_plan(args)
    if not args.execute:print(json.dumps(plan,indent=2));return 0
    return execute(args,plan)
if __name__=='__main__':raise SystemExit(main())
