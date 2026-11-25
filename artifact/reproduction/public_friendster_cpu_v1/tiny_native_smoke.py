#!/usr/bin/env python3
"""Bounded local C++17 build + four literal fixtures in each standalone mode.

No full graph, sampling, Numba kernel, model, GPU, wheel install or timing result.
Linux target execution remains a separate smoke receipt from local Mac testing.
"""
from pathlib import Path
import argparse,importlib,platform,sys
sys.dont_write_bytecode=True
from portable_common import *
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--source-manifest-sha256',required=True)
    a=p.parse_args(argv);a.out=a.out.resolve();require(not a.out.exists()and not a.out.is_relative_to(HERE),'fresh tiny output outside source required')
    require(a.source_manifest_sha256==sha(HERE/'SOURCE_MANIFEST.json'),'reviewed portable source manifest pin required');verify_portable_sources();load_originals()
    sys.path.insert(0,str(FROZEN/'original/native'))
    builder=importlib.import_module('build_native_v1');worker=importlib.import_module('replay_worker_v1')
    require(Path(builder.__file__).resolve()==FROZEN/'original/native/build_native_v1.py'and Path(worker.__file__).resolve()==FROZEN/'original/native/replay_worker_v1.py','unchanged tiny build/admission sources required')
    a.out.mkdir(parents=True);library,build=builder.build(a.out/'cpp_build')
    proofs={mode:worker.admission(library,mode,1<<20)for mode in('reference_full_clear','persistent_touched_reset')}
    require(all(proof['status']=='pass'and proof['literal_cases']==4 for proof in proofs.values()),'all4 literal fixtures per mode required')
    import numpy,scipy
    receipt={'kind':'LOCAL_TINY_NATIVE_SOURCE_PORTABILITY_SMOKE','status':'PASS_TINY_ONLY','platform':platform.platform(),'python':sys.version,
      'source_manifest_sha256':a.source_manifest_sha256,'build':build,'admission':proofs,'numpy':numpy.__version__,'scipy':scipy.__version__,
      'full_graph_or_model_jobs':0,'GPU_calls':0,'YT_writes':0,'remote_actions':0,'scope':'Tiny standalone native correctness/build only; not full seven-arm target reproduction or performance evidence.'}
    save(a.out/'TINY_SMOKE.json',receipt);print(json.dumps({'status':receipt['status'],'receipt':str(a.out/'TINY_SMOKE.json'),'platform':receipt['platform']}));return 0
if __name__=='__main__':raise SystemExit(main())
