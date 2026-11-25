#!/usr/bin/env python3
"""Explicit new public E7/E12 admission around unchanged scientific functions."""
from pathlib import Path
import argparse,importlib.util,json,os,sys,time,traceback
sys.dont_write_bytecode=True
from admission import Admission,HERE,ORIGINAL,sha,read,require,save,snapshot,expected_payloads

def load_original(role):
    name,pin=ORIGINAL[role];path=HERE/'original'/name;require(sha(path)==pin,'original source identity differs')
    key='_public_frozen_'+role;spec=importlib.util.spec_from_file_location(key,path);module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module);return module
def configure(role,out):
    for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    os.environ['NUMBA_NUM_THREADS']='1'if role=='e7'else'8';os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')

def execute_e7(admission,module,out):
    original=module.load_inputs
    def public_inputs(args):
        admission.stable();exact=module.load_exact(admission.repo);graph=exact.load_graph(admission.prepared/'train_csr','little');cohort=admission.cohorts['valid']
        require((graph.n,graph.m)==(65608366,2889707416),'exact public training graph dimensions required')
        stats={p:{k:r[k]for k in('bytes','mtime_ns')}for p,r in admission.stats.items()if '/train_csr/'in p and Path(p).name in('edge_starts.bin','edge_ends.bin')}
        return exact,graph,cohort['ids'],cohort['positive'],dict(admission.small),dict(admission.sources),stats
    module.load_inputs=public_inputs
    try:return module.main(['--repo',str(admission.repo),'--prepared-root',str(admission.prepared),'--quality-root',str(admission.quality),'--out',str(out/'results'),
      '--expected-query-count','512','--seed','20261125','--cap','200','--max-neighbors','2000000','--max-nodes','10000','--max-control-induction-nodes','1000','--max-working-set-mib','4096','--max-wall-seconds','3600'])
    finally:module.load_inputs=original

def execute_e12(admission,module,out):
    original_gate,original_inputs=module.verify_test_gate,module.load_inputs
    def public_inputs(prepared,split,*,plan=None):
        admission.stable();require(split=='test','exact frozen test experiment only');arrays=admission.cohorts['test']['arrays'];graph=module.CSR(admission.prepared/'train_csr')
        require((graph.n,graph.m)==(65608366,2889707416),'exact public training graph dimensions required')
        provenance={'prepared_manifest_sha256':sha(admission.prepared/'manifest.json'),
          'query_files':{name:{'path':str(admission.prepared/'evaluation/test'/f'{name}.npy'),'sha256':sha(admission.prepared/'evaluation/test'/f'{name}.npy'),'bytes':(admission.prepared/'evaluation/test'/f'{name}.npy').stat().st_size}for name in arrays},
          'graph':{'path':str(admission.prepared/'train_csr'),'nodes':graph.n,'stored_entries':graph.m,'logical_csr_bytes':graph.logical_bytes,
            'offset_dtype':str(graph.starts.dtype),'index_dtype':str(graph.col.dtype),'mapping':'read-only mmap','identity_scope':'New public admission full23 numeric-file SHA, then actual stat continuity; no historical timestamp requirement.'}}
        return graph,arrays,read(admission.prepared/'manifest.json'),provenance
    module.verify_test_gate=lambda *_a,**_kw:admission.stable();module.load_inputs=public_inputs
    try:
        module.run(admission.prepared,'test',out/'results',query_ids_path=admission.cohorts['test']['query_ids_path'],quality_plan_path=admission.quality/'plan.json',
          frozen_path=admission.quality/'FROZEN_CHECKPOINTS.json',threads=8,max_operations=20000000000,memory_budget_gib=180,chunk_queries=64)
        return 0
    finally:module.verify_test_gate,module.load_inputs=original_gate,original_inputs

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',choices=('e7','e12'),required=True)
    for name in('prepared-root','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--quality-root',type=Path,default=HERE/'quality_reference');p.add_argument('--quality-manifest',type=Path,default=HERE/'QUALITY_MANIFEST.json')
    p.add_argument('--repo',type=Path,default=HERE/'scientific_repo')
    p.add_argument('--execute',action='store_true');args=p.parse_args(argv)
    require(not args.out.exists()and not args.out.is_symlink(),'fresh public output required')
    require(not args.out.resolve().is_relative_to(args.prepared_root.resolve())and not args.out.resolve().is_relative_to(args.quality_root.resolve())and not args.out.resolve().is_relative_to(args.repo.resolve())and not args.out.resolve().is_relative_to(HERE.resolve()),'output must be outside all immutable inputs')
    expected=expected_payloads()
    if not args.execute:
        print(json.dumps({'status':'PUBLIC_ADAPTER_PLAN_ONLY','experiment':args.experiment,'numeric_files_to_stream_hash':len(expected),'numeric_bytes_to_read':sum(r['bytes']for r in expected),
          'queries':512 if args.experiment=='e7'else 4096,'negatives_per_test_query':1000,'full_graph_copy':False,'original_metadata_changes':False,'out_created':False}));return 0
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.mkdir();configure(args.experiment,args.out)
    try:
        admission=Admission(args.prepared_root,args.quality_root,args.quality_manifest,args.repo).verify_all();save(args.out/'PUBLIC_ADMISSION.json',admission.receipt())
        original=load_original(args.experiment);code=execute_e7(admission,original,args.out)if args.experiment=='e7'else execute_e12(admission,original,args.out)
        admission.stable();require(code==0,'original scientific routine failed or reached deadline')
        terminal={'status':'PUBLIC_FRESH_SCIENCE_COMPLETE','experiment':args.experiment,'new_public_admission':True,'original_science_source_sha256':ORIGINAL[args.experiment][1],
          'replaced_operational_gates':['load_inputs']if args.experiment=='e7'else['verify_test_gate','load_inputs'],'historical_source_or_metadata_modified':False,
          'scientific_parameters_changed':False,'query_selection_changed':False,'new_method_claim':False,'model_loaded':False,'remote_actions':0,
          'results_directory':str(args.out/'results'),'admission_sha256':sha(args.out/'PUBLIC_ADMISSION.json'),'adapter_sha256':sha(Path(__file__)),'completed_unix':time.time()}
        save(args.out/'COMPLETE.json',terminal);print(json.dumps(terminal));return 0
    except BaseException as error:
        save(args.out/'FAILED.json',{'status':'PUBLIC_FRESH_ADMISSION_OR_SCIENCE_FAILED','type':type(error).__name__,'message':str(error)[:500],'complete':False})
        raise
if __name__=='__main__':raise SystemExit(main())
