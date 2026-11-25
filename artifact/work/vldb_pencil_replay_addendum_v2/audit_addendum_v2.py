#!/usr/bin/env python3
"""Explicit derived audit: correct one REPLAY timer assumption, retain all gates.

The original frozen110-check verifier and first failed actual attempt stay
unchanged. No scientific execution, Torch import, native call or producer edit.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,inspect,json,math,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CORE=ROOT/'work/vldb_pencil_final_audit_v3'
BRIDGE=ROOT/'work/vldb_pencil_queue_v3/bridge/vldb_pencil_bridge_v2'
CORE_MANIFEST='22822f198620846ca0db4b6f71e74da1ff3229d65438eeec87f6a842e717313c'
PINS={'pencil_runtime.py':'ecd2ffdd38fc225a22e39d4a15fc1735de18e78e5045783f4f232a1880c82ab5',
      'pencil_bridge.py':'6167f42f3add94e50b12397d5403c17142d5749b50570576329ed597de75e07d'}
CALLCHAIN={('pencil_runtime.py','SamplerHook','__call__'):'f98fc1e97611fa7b74bcac516b8759860a2bf8cb0a7ce83bf49e71849e8d9d1f',
 ('pencil_runtime.py','QueryRuntime','raw'):'b04ffb53b5935f8f715649c5804232fb788330a27c9f5c4f18f61208f6b6f68c',
 ('pencil_runtime.py','QueryRuntime','wrapped'):'0f8e26bf9c0cd118d8358293c4268dbba96ab7df5a3dc90ad0fb2981efd2db15',
 ('pencil_bridge.py','PipelineDataset','__getitem__'):'d2eef9277ad7965bdffb56910a88adb8dd580d21d48da6aa26cc5e11a05efac4'}
OLD="if job['mode']=='replay':require(row['stages']['sampler_seconds']['max_s']==0,'REPLAY must explicitly bypass sampler')"
NEW="if job['mode']=='replay':validate_replay_materialization(row['stages']['sampler_seconds'],raw,job)"

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):
    def unique(pairs):
        result={}
        for key,value in pairs:require(key not in result,'duplicate JSON key');result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))

def verify_callchain():
    for name,pin in PINS.items():require(sha(BRIDGE/name)==pin,'frozen producer source differs')
    for (name,cls,method),pin in CALLCHAIN.items():
        tree=ast.parse((BRIDGE/name).read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls)
        fn=next(n for n in node.body if isinstance(n,ast.FunctionDef)and n.name==method)
        require(hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()==pin,'frozen REPLAY propagation/callchain differs')
    return {'.'.join(key):value for key,value in CALLCHAIN.items()}

def verify_callspy():
    record=read(HERE/'ACTUAL_SAMPLER_HOOK_CALLSPY.json')
    require(record['status']=='pass'and record['kind']=='ACTUAL_FROZEN_SAMPLER_HOOK_SYNTHETIC_CALLSPY'
        and record['runtime_source_sha256']==PINS['pencil_runtime.py']
        and record['runner_source_sha256']==sha(HERE/'callspy_once.py')
        and type(record['replay_real_calls'])is int and record['replay_real_calls']==0
        and type(record['online_real_calls'])is int and record['online_real_calls']==1
        and record['replay_real_raises_if_called']is True and record['frozenS_int64_values_exact']is True
        and type(record['replay_materialization_seconds'])in(int,float)
        and math.isfinite(record['replay_materialization_seconds'])and record['replay_materialization_seconds']>0
        and record['source_modified']is False and type(record['scientific_jobs_run'])is int
        and record['scientific_jobs_run']==0 and record['GPU_work']is False,
        'actual synthetic frozen-source callspy receipt invalid')
    return {'receipt_sha256':sha(HERE/'ACTUAL_SAMPLER_HOOK_CALLSPY.json'),'replay_real_calls':0,'online_real_calls':1,
        'positive_replay_materialization_timer_permitted':True,'scope':record['scope']}

def correction_receipt():
    value=read(HERE/'CORRECTION.json')
    require(value['original_core_manifest_sha256']==CORE_MANIFEST,'original frozen core pin differs')
    for key in('original_failed_attempt','original_failure_stderr'):
        row=value[key];path=Path(row['path'])
        require(path.stat().st_size==row['bytes']and sha(path)==row['sha256'],'first failed actual audit evidence changed')
    require(read(value['original_failed_attempt']['path'])['returncode']==2,'first invalid audit attempt must remain explicit')
    return {**value,'correction_receipt_sha256':sha(HERE/'CORRECTION.json'),
        'frozen_callchain':verify_callchain(),'actual_synthetic_callspy':verify_callspy(),
        'replay_sampler_seconds_interpretation':'frozenS materialization into int64 Torch tensor; native sampler bypassed',
        'all_other_original_gates_unchanged':True}

def load_core():
    require(sha(CORE/'SOURCE_MANIFEST.json')==CORE_MANIFEST,'frozen110-check core manifest differs')
    for row in read(CORE/'SOURCE_MANIFEST.json')['files']:
        path=CORE/row['path'];require(path.stat().st_size==row['bytes']and sha(path)==row['sha256'],'frozen110-check source changed')
    sys.path.insert(0,str(CORE))
    spec=importlib.util.spec_from_file_location('pencil_unchanged_core_for_replay_addendum_v2',CORE/'audit_pencil_v3_v1.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    return core

def validate_replay_materialization(summary,raw,job):
    require(job['mode']=='replay'and raw['configuration']['pipeline_mode']=='replay'
        and raw['result']['pipeline_mode']=='replay','REPLAY frozenS route not declared')
    # Original stage_summary has already checked count, units, finite values,
    # nonnegative times and ordered quantiles. No zero overhead is asserted.
    require(summary['n']==1024 and summary['min_s']>=0 and summary['max_s']>=summary['min_s'],
        'bounded finite REPLAY materialization summary required')

def install(core):
    source=inspect.getsource(core.benchmark_receipt)
    require(source.count(OLD)==1 and NEW not in source,'exact original single REPLAY predicate required')
    derived=source.replace(OLD,NEW)
    require(derived.replace(NEW,OLD)==source,'other benchmark gate body changed')
    core.__dict__['validate_replay_materialization']=validate_replay_materialization
    namespace={};exec(compile(derived,'<explicit_replay_timer_addendum_v2>','exec'),core.__dict__,namespace)
    core.benchmark_receipt=namespace['benchmark_receipt']
    original_build,original_text=core.build_report,core.text_report
    def build(context,contract):
        receipt=correction_receipt();report=original_build(context,contract)
        report['kind']='independent_local_pencil_v3_audit_explicit_replay_timer_addendum_v2'
        report['explicit_auditor_correction']=receipt
        report['explicit_auditor_correction']['benchmark_function_original_sha256']=hashlib.sha256(source.encode()).hexdigest()
        report['explicit_auditor_correction']['benchmark_function_derived_sha256']=hashlib.sha256(derived.encode()).hexdigest()
        if report['scientific_checks']['status']=='pass':report['scientific_checks']['limitations'].append(
            'REPLAY sampler_seconds times frozenS tensor materialization, not native sampler execution; first frozen audit rejected a false zero-timer assumption and is preserved separately.')
        return report
    core.build_report=build
    core.text_report=lambda report:original_text(report)+'\nExplicit correction V2: first failed frozen audit preserved; REPLAY sampler_seconds is frozenS materialization, not native sampling. All other original gates retained.\n'
    return core

def main(argv=None):
    core=install(load_core())
    guard=argparse.ArgumentParser(add_help=False);guard.add_argument('--output',type=Path,default=core.DEFAULT_OUTPUT)
    arguments,_=guard.parse_known_args(argv)
    output=arguments.output.resolve()
    require(not output.is_relative_to(HERE)and not HERE.is_relative_to(output),
        'output must be outside the new derived auditor source tree')
    correction_receipt()
    return core.main(argv)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status':'invalid_evidence','error_type':type(error).__name__,'message':str(error),'performance_analysis_eligible':False}),file=sys.stderr)
        raise SystemExit(2)
