#!/usr/bin/env python3
"""Reviewable candidate only: print the full44-job clean-consumer control.

This entrypoint cannot execute jobs, contact a cluster, install packages,
upload objects, create outputs, or select favorable configurations.
"""
from pathlib import Path
import argparse,hashlib,json,math,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BRIDGE=HERE/'bridge/vldb_pencil_bridge_clean_v1'
ARMS=('torch_sparse_public','torch_sparse_direct','compiled_scan_int64','compiled_pairwise_int64',
      'compiled_scan_int32','compiled_pairwise_int32','cpp_full_clear','cpp_touched_reset')
REPLAY_ARMS=('torch_sparse_public','torch_sparse_direct','cpp_touched_reset','compiled_pairwise_int32')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def make_plan(args):
    if not all(math.isfinite(x)and x>0 for x in(args.external_memory_guard_gib,args.pod_memory_limit_gib)):
        raise ValueError('finite positive explicit memory guard and pod limit required')
    if args.external_memory_guard_gib>.9*args.pod_memory_limit_gib:
        raise ValueError('owned-tree guard must fit below90percent of the declared pod memory limit')
    out=args.out.resolve();prepared=args.citation2_root.resolve();validation=args.validation_root.resolve()
    real=['--graph-path',str(prepared/'train_csr'),'--features',str(prepared/'node_feat.npy'),
          '--valid-positive',str(validation/'valid_edge.npy'),'--valid-negative',str(validation/'valid_edge_neg.npy'),
          '--strict-manifest',str(prepared/'manifest.json')]
    if args.quality_plan:real+=['--quality-plan',str(args.quality_plan.resolve())]
    layout={'out':out,'cpp_module':BRIDGE/'corrected_native/native_adapter_v1.py',
            'cpp_library':args.cpp_library.resolve(),'cpp_receipt':args.cpp_build_receipt.resolve()}
    source=HERE/'metadata/ORIGINAL_SCIENTIFIC_JOB_BUILDER.py'
    namespace={'Path':Path,'BRIDGE':BRIDGE,'ARMS':ARMS,'REPLAY_ARMS':REPLAY_ARMS,
               'layout':lambda _:layout,'real_inputs':lambda _:list(real)}
    exec(compile(source.read_text(),str(source),'exec'),namespace)
    frozen=namespace['make_jobs'](Path('/unused_candidate_base'),args.python)
    jobs=[]
    for original in frozen:
        if original.get('mode')=='replay':continue
        job={**original,'argv':[*original['argv'],'--query-memory','disabled']}
        jobs.append(job)
    assert len(jobs)==44 and sum(j['kind']=='benchmark' for j in jobs)==32
    assert {(j['arm'],j['batch'],j['workers']) for j in jobs if j['kind']=='benchmark'}=={
        (arm,b,w) for arm in ARMS for b in(128,512) for w in(0,1)}
    assert sum(j['kind']=='audit'for j in jobs)==9
    protected=[HERE,prepared,validation,args.cpp_library.resolve(),args.cpp_build_receipt.resolve()]
    if args.quality_plan:protected.append(args.quality_plan.resolve())
    if out.exists()or any(out.is_relative_to(p)or p.is_relative_to(out)for p in protected):
        raise ValueError('fresh output outside every source/input required')
    protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    protocol['required_guard']['actual_planned_owned_tree_budget_gib']=args.external_memory_guard_gib
    protocol['required_guard']['actual_planned_pod_memory_limit_gib']=args.pod_memory_limit_gib
    return {'status':'CANDIDATE_PLAN_ONLY_NO_EXECUTION','execution_authorized':False,
        'protocol':protocol,'accelerator_description_supplied_by_root':args.accelerator_description,
        'source_job_builder_sha256':sha(source),'jobs':jobs,'scientific_jobs':44,
        'external_guard_required':True,'controller_and_archive_writer':'root must review and supply; this planner has no execution capability',
        'inputs_not_opened':True,'output_created':False,'new_scientific_measurements':0}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in('citation2-root','validation-root','cpp-library','cpp-build-receipt','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--quality-plan',type=Path)
    p.add_argument('--external-memory-guard-gib',type=float,required=True)
    p.add_argument('--pod-memory-limit-gib',type=float,required=True)
    p.add_argument('--accelerator-description',default='root hardware decision pending; no allocation asserted')
    args=p.parse_args(argv);print(json.dumps(make_plan(args),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
