#!/usr/bin/env python3
"""Reuse the frozen external helper's bounded converter, without its benchmark.

Default is preflight. Explicit execution requires post-pipeline admission and
an actual compiled-control fixture before full graph conversion. No downloads.
"""
from pathlib import Path
import argparse,importlib.util,json,math,time
from prepare_replay_v1 import sha,write_new
from replay_worker_v1 import admission as native_admission
from run_parent_replay_v1 import admission as phase_admission

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--external-helper',type=Path,required=True);p.add_argument('--external-helper-sha256',required=True)
    p.add_argument('--graph-path',type=Path,required=True);p.add_argument('--native-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--library',type=Path);p.add_argument('--admission-marker',type=Path)
    p.add_argument('--max-native-gib',type=float,default=8);p.add_argument('--max-peak-gib',type=float,default=16);p.add_argument('--max-assoc-gib',type=float,default=2)
    p.add_argument('--chunk-entries',type=int,default=1_000_000);a=p.parse_args(argv)
    if not all(math.isfinite(v)and v>0 for v in(a.max_native_gib,a.max_peak_gib,a.max_assoc_gib))or a.chunk_entries<=0:p.error('finite positive budgets/chunk required')
    if a.output.exists():raise FileExistsError('fresh conversion receipt required')
    if sha(a.external_helper)!=a.external_helper_sha256:raise ValueError('frozen external converter source hash differs')
    spec=importlib.util.spec_from_file_location('frozen_external_converter',a.external_helper);external=importlib.util.module_from_spec(spec);spec.loader.exec_module(external)
    graph=external.inspect_graph(a.graph_path,'little');estimate=external.estimate_memory(graph,False,a.chunk_entries)
    external.enforce_budgets(graph,estimate,max_native_gib=a.max_native_gib,max_peak_gib=a.max_peak_gib,public_api=False,max_public_coo_gib=0)
    report={'schema_version':1,'kind':'corrected_native_conversion_only_v1','status':'preflight',
        'external_helper_sha256':a.external_helper_sha256,'graph':graph,'estimate':estimate,
        'scope':'Frozen converter only; no query benchmark, graph sampler, model, GPU or YT operation.'}
    if not a.execute:write_new(a.output,report);print(json.dumps({'status':'preflight','output':str(a.output)}));return 0
    if a.library is None:p.error('actual native library fixture required before conversion')
    report['admission']=phase_admission(a.admission_marker)
    report['actual_native_fixture']=native_admission(a.library,'persistent_touched_reset',int(a.max_assoc_gib*(2**30)))
    capacity=external.capacity_snapshot();external.enforce_capacity(estimate,capacity);report['capacity']=capacity
    start=time.perf_counter();meta=external.prepare_native(a.graph_path,a.native_dir,graph,a.chunk_entries)
    ptr,col,verified=external.open_native(a.native_dir,graph);del ptr,col
    if sha(a.external_helper)!=a.external_helper_sha256:raise ValueError('frozen converter source changed')
    report.update(status='complete',conversion=meta,runtime_native_content_verification=verified['runtime_content_verification'],
                  setup_wall_seconds=time.perf_counter()-start)
    write_new(a.output,report);print(json.dumps({'status':'complete','output':str(a.output)}));return 0

if __name__=='__main__':raise SystemExit(main())
