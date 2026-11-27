#!/usr/bin/env python3
"""Tiny actual consumer gates: local CPU1 or remote CUDA1/8, no full graph."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from common import *
import environment as environment
from native_storage import promote_native, verify_native_files

def one(args,arm,workers):
 out=(args.out/('w'+str(workers)+'_'+arm)).resolve();out.mkdir()
 warmup=2*workers;consumed=warmup+4
 cfg={'fixture':True,'graph_path':None,'file_endian':'little','pairs_path':str(out/'pairs.npy'),
  'seed':20261124,'hops':2,'cap':3,'oversample':1.5,'auto_ratio':1.0,'max_neighbors':10000,'warmup_pairs':warmup,'steps':2,'prefetch_factor':1,
  'audit_pairs':2,'device':args.device,'torch_threads':1,'gpu_memory_budget_gib':24.0,'stock_index_dtype':'auto','batch_size':2,
  'num_workers':workers,'arm':arm,'out':str(out),'worker_receipts':str(out/'workers'),'native_root':str(out/'runtime/native'),
  'native_library':str(args.library.resolve()),'native_expected':args.fixture_native_expected,'native_reuse_receipt_sha256':args.fixture_native_reuse_sha256,'native_reuse_expected_device':args.fixture_native_reuse_root.stat().st_dev,'checkpoints':sorted({2,warmup,consumed}),'environment_contract':args.environment}
 if arm==ARMS[2]:cfg['native_root']=str(args.fixture_native_reuse_root)
 from dataset import v4
 v4.prepare_manifest(out/'pairs.npy',cfg,consumed+2*workers,64,None,'uniform')
 config=out/'config.json';save(config,cfg)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
 env.update(environment.exported(args.environment));environment.guard(args.environment,'qualification_before_Popen',out/'ENVIRONMENT_FAILURE.json')
 began=time.monotonic_ns();env['E25_LAUNCH_MONOTONIC_NS']=str(began)
 with(out/'process.log').open('xb')as stream:
  child=subprocess.Popen([sys.executable,'-B',str(HERE/'lifecycle_child.py'),'--config',str(config)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:code=child.wait(timeout=300)
  except BaseException:
   import signal
   try:os.killpg(child.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
   raise
 record={'pid':child.pid,'returncode':code,'actual_wait_returned':True,'launch_monotonic_ns':began,'wait_returned_monotonic_ns':time.monotonic_ns()}
 save(out/'exit.json',record);require(code==0,'actual tiny consumer failed: '+arm)
 environment.guard(args.environment,'qualification_after_wait',out/'ENVIRONMENT_FAILURE.json')
 report=read(out/'report.json')
 require(report['consumed_graphs']==consumed and report['measured_graphs']==4 and len(report['losses'])==2,'complete tiny training prefix required')
 require(len(report['worker_receipts'])==workers and all(w['status']=='closed'for w in report['worker_receipts']),'every actual spawn worker must initialize and normally close')
 allowed=('PASS',)if args.device=='cuda'else('PASS','NOT_APPLICABLE_PORTABLE_CPU1')
 require(all(report[k]['status']in allowed for k in('environment_start','environment_end'))
  and all(w[k]['status']in allowed for w in report['worker_receipts']for k in('environment_start','environment_end')),'actual fixture environment boundary guards required')
 return report,record

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--library',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
 parser.add_argument('--device',choices=('cpu','cuda'),required=True);parser.add_argument('--workers',type=int,nargs='+',choices=(1,8),default=[1])
 a=parser.parse_args();a.out=a.out.resolve();require(len(set(a.workers))==len(a.workers),'unique fixture worker counts required')
 require(a.device!='cpu'or a.workers==[1],'local portable CPU smoke is bounded to one worker')
 require(not a.out.exists(),'fresh qualification output required');a.out.mkdir()
 a.environment=environment.admit(environment.selected(),portable=a.device=='cpu',failure_path=a.out/'ENVIRONMENT_FAILURE.json')
 save(a.out/'ENVIRONMENT_ADMISSION.json',{'status':'PASS'if a.device=='cuda'else'PORTABILITY_ONLY','contract':a.environment})
 os.environ.setdefault('NUMBA_CACHE_DIR',str(a.out/'shared_numba_cache'))
 a.fixture_native_reuse_root=a.out/'runtime/native_reuse'
 fixture_cfg={'fixture':True,'graph_path':None,'file_endian':'little','native_root':str(a.fixture_native_reuse_root),'native_preparation_scope':'fixture_once_outside_applications'}
 promotion=promote_native(fixture_cfg);root=a.fixture_native_reuse_root
 expected={'nodes':promotion['nodes'],'stored_entries':promotion['stored_entries'],'rowptr_bytes':promotion['rowptr_bytes'],'col_bytes':promotion['col_bytes'],'rowptr_sha256':sha(root/'rowptr.i8'),'col_sha256':sha(root/'col.i8')}
 files=verify_native_files(root,expected);stats={}
 for name in files:
  p=root/name;p.chmod(0o444);st=p.stat();stats[name]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev}
 save(root/'READY.json',{'kind':'CLOSURE_V1_IMMUTABLE_NATIVE_REUSE','dataset_id':'fixture','nodes':expected['nodes'],'stored_entries':expected['stored_entries'],'expected_native':expected,'files':files,'file_stats':stats,'native_promotion':promotion,'scope':'Tiny fixture qualification only; outside all tiny application clocks.'})
 (root/'READY.json').chmod(0o444);a.fixture_native_expected=expected;a.fixture_native_reuse_sha256=sha(root/'READY.json')
 records=[];initials=[];comparisons=[]
 for workers in a.workers:
  pair=[]
  for arm in ARMS:
   report,record=one(a,arm,workers);pair.append(report);records.append(record);initials.append(report['model']['initial_state_sha256'])
  require(all(pair[0]['input_stream_sha256']==r['input_stream_sha256']and pair[0]['model']['initial_state_sha256']==r['model']['initial_state_sha256'] for r in pair[1:]),
   'tiny actual consumer inputs/model differ')
  if a.device=='cuda':require(all('H100'in r['device']['name']and r['model']['initial_state_sha256']==MODEL_SHA for r in pair),'original actual H100 fixture/model required')
  comparisons.append({'workers':workers,'consumed_graphs':pair[0]['consumed_graphs'],'all_actual_consumed_inputs_exact':True,'stream_sha256':pair[0]['input_stream_sha256']})
 require(len(set(initials))==1,'same initial model required across every tiny consumer')
 save(a.out/'QUALIFIED.json',{'success':True,'device':a.device,'workers':a.workers,'actual_consumers':len(records),'comparisons':comparisons,
  'environment_policy':environment.POLICY,'chosen_cpu_affinity':a.environment['chosen_cpu_affinity'],
  'environment_target_proof':not a.environment.get('portable_unsupported',False),
  'initial_model_sha256':initials[0],'source_manifest_sha256':sha(HERE/'SOURCE_MANIFEST.json'),'native_library':pin(a.library),
  'actual_child_exits':records,'new_full_Friendster_science':False,
  'scope':'Actual tiny consumer qualification only. CPU1 smoke is not Linux/H100 scientific evidence; CUDA8 uses enough batches to deliver input from every worker.'})
if __name__=='__main__':main()
