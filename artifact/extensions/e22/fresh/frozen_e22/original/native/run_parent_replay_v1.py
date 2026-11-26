#!/usr/bin/env python3
"""Prepare128/768 existing cases; execute only with explicit post-pipeline admission.

Each arm uses a fresh sequential process, preventing the persistent map from
being charged to the fresh-clear comparator. No downloads, GPU work or YT calls.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,random,shutil,signal,subprocess,sys,time
from prepare_replay_v1 import prepare,write_new,read,sha
from build_native_v1 import build

HERE=Path(__file__).resolve().parent
SOURCE_NAMES=('native_assoc_v1.cpp','native_adapter_v1.py','build_native_v1.py','fixtures_v1.py',
              'prepare_replay_v1.py','prepare_native_only_v1.py','replay_worker_v1.py','run_parent_replay_v1.py')

def save(path,value):
    p=Path(path);t=p.with_name(p.name+'.partial');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)

def admission(marker):
    if marker is None:raise ValueError('execution requires an explicit terminal post-pipeline admission marker')
    marker=Path(marker)
    if not marker.is_file()or marker.is_symlink()or marker.stat().st_size>2<<20:raise ValueError('invalid bounded admission marker')
    data=read(marker)
    if data.get('controller_finished')is not True and data.get('complete')is not True:raise ValueError('admission is not scientifically terminal')
    roots={marker.parent,marker.parent.parent}
    for root in roots:
        for name in('PIPELINE_SUITE_RUNNING.lock','GPU_SUITE_RUNNING.lock'):
            if(root/name).exists():raise ValueError('quality/pipeline lock remains; no overlapping execution')
    return {'path':str(marker.resolve()),'sha256':sha(marker),'content':data,
            'scope':'Read-only terminal/lock admission. Root schedules this control after all pipeline/persistence work is quiescent.'}

def owned_run(argv,log,timeout):
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'NUMBA_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    with Path(log).open('xb')as f:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
            env=env,cwd=HERE,start_new_session=True,close_fds=True)
        def stop_owned_child():
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:return child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                return child.wait(timeout=10)
        timed_out=False
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            code=stop_owned_child()
        except BaseException:
            stop_owned_child()
            raise
    return {'pid':child.pid,'returncode':code,'timed_out':timed_out,'timeout_seconds':timeout,
            'scope':'Only this fresh-session worker is signaled on timeout or Python interruption; workers do not spawn subprocesses. An external supervisor owns process-group cleanup for abrupt termination of this parent.'}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--count',type=int,choices=(128,768),required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--native-dir',type=Path);p.add_argument('--library',type=Path);p.add_argument('--build-dir',type=Path)
    p.add_argument('--admission-marker',type=Path);p.add_argument('--include-upstream',action='store_true');p.add_argument('--external-helper',type=Path);p.add_argument('--external-helper-sha256')
    p.add_argument('--prepare-native',action='store_true');p.add_argument('--graph-path',type=Path)
    p.add_argument('--conversion-max-native-gib',type=float,default=8);p.add_argument('--conversion-timeout-seconds',type=float,default=1800)
    p.add_argument('--max-assoc-gib',type=float,default=2);p.add_argument('--max-peak-gib',type=float,default=16)
    p.add_argument('--max-query-neighbors',type=int,default=20_000_000)
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=3);p.add_argument('--timeout-seconds',type=float,default=3600);p.add_argument('--order-seed',type=int,default=20261125)
    a=p.parse_args(argv)
    if not all(math.isfinite(x)and x>0 for x in(a.max_assoc_gib,a.max_peak_gib,a.timeout_seconds,a.conversion_max_native_gib,a.conversion_timeout_seconds))or max(a.timeout_seconds,a.conversion_timeout_seconds)>14400 or a.warmup<0 or a.repeats<=0 or a.max_query_neighbors<=0:p.error('finite bounded budgets/time and valid repetitions required')
    if a.output_dir.exists():raise FileExistsError('fresh parent output directory required; no automatic resume/overwrite')
    a.output_dir.mkdir(parents=True)
    replay=prepare(a.archive,a.count);replay_path=a.output_dir/'replay_plan.json';write_new(replay_path,replay)
    if any(r['sum_out_degrees']>a.max_query_neighbors for r in replay['selected']):raise ValueError('replay exceeds selected-row neighbor guard before setup')
    modes=['reference_full_clear','persistent_touched_reset']
    if a.include_upstream:modes.append('upstream_torch_sparse_direct')
    random.Random(a.order_seed).shuffle(modes)
    plan={'schema_version':1,'kind':'corrected_native_parent_v1','replay_plan_sha256':sha(replay_path),
        'source_hashes':{name:sha(HERE/name)for name in SOURCE_NAMES},'modes':modes,'mode_order_seed':a.order_seed,
        'configuration':{k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},
        'scope':'Matched standalone compiled full-clear/touched-reset controls; unmodified upstream optional separate arm. Existing S only; no sampler/model/quality claim.',
        'fairness':'Fresh sequential processes, one worker/arm, one CPU thread, matching input/order/warmups/repeats. Graph conversion/content verification and persistent initialization reported outside timers. No speed claim from plan-only output.'}
    save(a.output_dir/'parent_plan.json',plan)
    if not a.execute:print(json.dumps({'status':'prepared_only','cases':replay['count'],'modes':modes,'output':str(a.output_dir)}));return 0
    if a.native_dir is None or(a.library is None and a.build_dir is None):raise ValueError('execute requires native directory and explicit library or build directory')
    if a.library is not None and a.build_dir is not None:raise ValueError('choose existing library or fresh build, not both')
    gate=admission(a.admission_marker);save(a.output_dir/'admission.json',gate)
    if(a.include_upstream or a.prepare_native)and(not a.external_helper or not a.external_helper_sha256 or sha(a.external_helper)!=a.external_helper_sha256):raise ValueError('frozen unmodified external helper SHA required')
    if a.build_dir is not None:library,build_receipt=build(a.build_dir);save(a.output_dir/'build_receipt.json',build_receipt)
    else:library=a.library.resolve()
    bindings={str(HERE/name):value for name,value in plan['source_hashes'].items()}
    bindings.update({str(library):sha(library),str(replay_path):sha(replay_path),str(a.archive):sha(a.archive)})
    if a.include_upstream or a.prepare_native:bindings[str(a.external_helper)]=a.external_helper_sha256
    save(a.output_dir/'runtime_bindings.json',bindings);records=[]
    if shutil.disk_usage(a.output_dir).free<2**30:raise ValueError('need1GiB free result/log disk reserve')
    if a.include_upstream:
        command=[sys.executable,'-u',str(a.external_helper),'--fixture','--execute',
                 '--output',str(a.output_dir/'upstream_native_admission.json')]
        exit_data=owned_run(command,a.output_dir/'upstream_native_admission.log',120)
        save(a.output_dir/'upstream_native_admission_exit.json',exit_data)
        report=read(a.output_dir/'upstream_native_admission.json')if(a.output_dir/'upstream_native_admission.json').exists()else{}
        if exit_data['returncode']!=0 or exit_data['timed_out']or report.get('status')!='complete' or report.get('all_executed_arms_exact')is not True:
            raise RuntimeError('actual unmodified upstream admission failed before conversion/benchmarks')
        if any(sha(Path(path))!=expected for path,expected in bindings.items()):raise ValueError('bound code/input changed during upstream admission')
    if a.prepare_native:
        if a.graph_path is None:raise ValueError('explicit original GraphCSR path required for conversion')
        command=[sys.executable,'-u',str(HERE/'prepare_native_only_v1.py'),'--external-helper',str(a.external_helper),
            '--external-helper-sha256',a.external_helper_sha256,'--graph-path',str(a.graph_path),'--native-dir',str(a.native_dir),
            '--output',str(a.output_dir/'native_conversion_receipt.json'),'--execute','--library',str(library),
            '--admission-marker',str(a.admission_marker),'--max-native-gib',str(a.conversion_max_native_gib),
            '--max-peak-gib',str(a.max_peak_gib),'--max-assoc-gib',str(a.max_assoc_gib)]
        save(a.output_dir/'conversion_command.json',{'argv':command,'timeout_seconds':a.conversion_timeout_seconds})
        exit_data=owned_run(command,a.output_dir/'native_conversion.log',a.conversion_timeout_seconds)
        save(a.output_dir/'native_conversion_exit.json',exit_data)
        if exit_data['returncode']!=0 or exit_data['timed_out']:raise RuntimeError('bounded native conversion failed; outputs/partial directory preserved')
        if any(sha(Path(path))!=expected for path,expected in bindings.items()):raise ValueError('bound code/input changed during conversion')
    for mode in modes:
        if admission(a.admission_marker)['sha256']!=gate['sha256']:raise ValueError('admission changed')
        if shutil.disk_usage(a.output_dir).free<2**30:raise ValueError('need1GiB free result/log disk reserve')
        if any(sha(Path(path))!=expected for path,expected in bindings.items()):raise ValueError('runtime source/input changed')
        output=a.output_dir/(mode+'.json')
        command=[sys.executable,'-u',str(HERE/'replay_worker_v1.py'),'--archive',str(a.archive),'--replay-plan',str(replay_path),
            '--native-dir',str(a.native_dir),'--library',str(library),'--mode',mode,'--output',str(output),
            '--max-assoc-gib',str(a.max_assoc_gib),'--max-peak-gib',str(a.max_peak_gib),'--max-query-neighbors',str(a.max_query_neighbors),
            '--warmup',str(a.warmup),'--repeats',str(a.repeats)]
        if mode=='upstream_torch_sparse_direct':command+=['--external-helper',str(a.external_helper),'--external-helper-sha256',a.external_helper_sha256]
        save(a.output_dir/'status.json',{'stage':'running','mode':mode,'finished_arms':records})
        exit_data=owned_run(command,a.output_dir/(mode+'.log'),a.timeout_seconds);save(a.output_dir/(mode+'_exit.json'),exit_data)
        if any(sha(Path(path))!=expected for path,expected in bindings.items()):raise ValueError('bound source/input changed during worker')
        report=read(output)if output.exists()else{}
        eligible=(exit_data['returncode']==0 and not exit_data['timed_out']and report.get('status')=='complete'
            and report.get('record_count')==replay['count']and report.get('all_executed_outputs_exact')is True
            and report.get('replay_plan_sha256')==plan['replay_plan_sha256'])
        records.append({'mode':mode,'eligible':eligible,'exit':exit_data,'report_status':report.get('status','not_produced'),
                        'report_sha256':sha(output)if output.exists()else None})
        save(a.output_dir/'arm_results.json',{'arms':records})
    success=all(r['eligible']for r in records)
    save(a.output_dir/'COMPLETE.json',{'complete':True,'scientific_terminal':True,'success':success,
        'terminal_status':'complete'if success else'completed_with_failures','arms':records,
        'replay_plan_sha256':sha(replay_path),'runtime_bindings_sha256':sha(a.output_dir/'runtime_bindings.json'),
        'no_remote_or_YT_actions':True,'no_performance_advantage_inferred':True})
    save(a.output_dir/'status.json',{'stage':'complete','success':success});return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
