#!/usr/bin/env python3
"""Fresh E25 application. Parent launch clock includes imports and setup."""
from __future__ import annotations
import os,time
ENTRY_NS=time.monotonic_ns()
import argparse,hashlib,json,sys
from pathlib import Path
from common import HERE,GIB,MODEL_SHA,require,read,save,verify_sources
import environment as environment
import numpy as np
from dataset import LifecycleDataset,Collate,worker_init,v4

def promote_native(config):
 root=Path(config['native_root']);require(not root.exists(),'fresh owned native runtime required');root.mkdir(parents=True)
 began=time.monotonic_ns();graph=v4.load_graph(config);n,m=graph.n,graph.m
 require(sys.byteorder=='little','native little-endian input required')
 ptr=np.memmap(root/'rowptr.i8',dtype='<i8',mode='w+',shape=(n+1,))
 col=np.memmap(root/'col.i8',dtype='<i8',mode='w+',shape=(m,))
 for start in range(0,n,1000000):ptr[start:min(n,start+1000000)]=graph.starts[start:min(n,start+1000000)]
 ptr[n]=m
 for start in range(0,m,1000000):col[start:min(m,start+1000000)]=graph.indices[start:min(m,start+1000000)]
 ptr.flush();col.flush();del ptr,col,graph
 for name in('rowptr.i8','col.i8'):
  with(root/name).open('rb')as stream:os.fsync(stream.fileno())
 result={'nodes':n,'stored_entries':m,'begin_ns':began,'end_ns':time.monotonic_ns(),
  'rowptr_bytes':8*(n+1),'col_bytes':8*m,'dtype':'little-endian int64','sentinel_added':True,
  'full_integrity_readback_inside_application':False,'full_native_copy_once_per_app':True,'files_fsynced':True}
 save(root/'READY.json',result);return result

def update_stream(hasher,record):
 """Typed/shaped ordered actual CPU consumer inputs; no graph recomputation."""
 began=time.monotonic_ns();batch=record['batch']
 fields=[('query_indices',record['query_indices'])]
 for name in('node_id','edge_index','edge_weight','z','y','ptr','batch'):
  tensor=getattr(batch,name);require(tensor.device.type=='cpu','input hash must precede H2D')
  fields.append((name,tensor.detach().numpy()))
 for name,array in fields:
  array=np.ascontiguousarray(array)
  header=json.dumps({'field':name,'dtype':array.dtype.str,'shape':list(array.shape)},sort_keys=True,separators=(',',':')).encode()
  hasher.update(len(header).to_bytes(8,'little'));hasher.update(header);hasher.update(array.tobytes())
 return time.monotonic_ns()-began

def run(config):
 import torch
 from torch.utils.data import DataLoader
 from seal.models import DGCNN
 from seal.reproducibility import model_state_sha256
 import seal.models,seal.utils
 require(Path(seal.models.__file__).resolve()==HERE/'frozen_src/seal/models.py'
  and Path(seal.utils.__file__).resolve()==HERE/'frozen_src/seal/utils.py','actual frozen model/DRNL/PyG origins differ')
 launch=int(os.environ['E25_LAUNCH_MONOTONIC_NS']);require(0<launch<=ENTRY_NS,'genuine same-kernel parent launch clock required')
 imports_ready=time.monotonic_ns();verify_sources()
 out=Path(config['out']);phase_path=out/'phase.json'
 contract=config['environment_contract'];env_start=environment.guard(contract,'application_start',out/'ENVIRONMENT_FAILURE.json')
 def phase(name):save(phase_path,{'phase':name,'monotonic_ns':time.monotonic_ns(),'pid':os.getpid()})
 torch.set_num_threads(1);torch.manual_seed(config['seed'])
 if config['device']=='cuda':require(torch.cuda.is_available(),'CUDA required; no silent CPU fallback')
 phase('native_promotion'if config['arm']=='reference_union_native_persistent'else'model_setup')
 promotion=promote_native(config)if config['arm']=='reference_union_native_persistent'else None
 phase('model_setup');model_begin=time.monotonic_ns()
 device=torch.device('cuda:0'if config['device']=='cuda'else'cpu')
 gpu=v4.configure_gpu_budget(torch,device,config['gpu_memory_budget_gib'])
 if device.type=='cuda':torch.cuda.manual_seed_all(config['seed'])
 model=DGCNN(32,3,1000,281,train_dataset=None,dynamic_train=True,use_feature=False,node_embedding=None).to(device)
 initial=model_state_sha256(model)
 if not config['fixture']:require(initial==MODEL_SHA,'original initial DGCNN model differs')
 optimizer=torch.optim.Adam(model.parameters(),lr=1e-4);model.train()
 model_end=time.monotonic_ns();dataset=LifecycleDataset(config)
 loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=False,num_workers=config['num_workers'],collate_fn=Collate(),
  worker_init_fn=worker_init,drop_last=True,pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(config['seed']),
  multiprocessing_context='spawn',prefetch_factor=1,persistent_workers=True)
 iterator=None;stream=hashlib.sha256();hash_ns=0;checkpoints=[];semantic_prefixes={};audits=[];losses=[];worker_pids=set();wait_ns=0
 warmup=config['warmup_pairs']//config['batch_size'];total=warmup+config['steps'];steady_start=None;first_ready=None
 report={};phase('worker_setup_and_first_batch')
 def synchronize():
  if device.type=='cuda':torch.cuda.synchronize()
 def step(record,measured):
  batch=record['batch'].to(device,non_blocking=device.type=='cuda')
  optimizer.zero_grad(set_to_none=True)
  logits=model(batch.z,batch.edge_index,batch.batch,None,batch.edge_weight,None).view(-1)
  loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,batch.y.float())
  loss.backward();optimizer.step()
  if measured:losses.append(loss.detach())
  return batch.num_graphs
 iterator_begin=time.monotonic_ns()
 try:
  iterator=iter(loader);iterator_created=time.monotonic_ns()
  for batch_index in range(total):
   wait=time.monotonic_ns();record=next(iterator);wait_ns+=time.monotonic_ns()-wait
   if first_ready is None:first_ready=time.monotonic_ns();phase('warmup')
   expected=np.arange(batch_index*config['batch_size'],(batch_index+1)*config['batch_size'],dtype='<i8')
   require(np.array_equal(record['query_indices'],expected),'actual query stream reordered or omitted')
   hash_ns+=update_stream(stream,record);audits.extend(record['audits']);worker_pids.update(record['worker_pids'])
   require(step(record,batch_index>=warmup)==config['batch_size'],'fixed full training batch required')
   consumed=(batch_index+1)*config['batch_size']
   if consumed in config.get('semantic_input_prefixes',config['checkpoints']):semantic_prefixes[str(consumed)]=stream.hexdigest()
   if consumed in config['checkpoints']:
    synchronize();now=time.monotonic_ns()
    labels={32:'first_trained_batch',256:'warmup_complete',576:'measured_prefix320',1280:'measured_prefix1024',3456:'measured_prefix3200',6656:'measured_prefix6400'}
    checkpoints.append({'label':labels.get(consumed,'fixture_consumed'+str(consumed)),
     'total_trained_graphs':consumed,'measured_graphs':max(0,consumed-config['warmup_pairs']),
     'monotonic_ns':now,'launch_to_complete_seconds':(now-launch)/1e9,'input_stream_sha256':stream.hexdigest(),
     'input_hash_seconds_included':hash_ns/1e9})
   if batch_index+1==warmup:
    if consumed not in config['checkpoints']:synchronize()
    steady_start=time.monotonic_ns()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    phase('checkpointed_measured_window')
  require(checkpoints[-1]['total_trained_graphs']==total*config['batch_size'],'final consumed prefix must synchronize')
  training_end=checkpoints[-1]['monotonic_ns'];phase('shutdown')
  shutdown_begin=time.monotonic_ns();iterator._shutdown_workers();iterator=None;shutdown_end=time.monotonic_ns()
  phase('reporting')
  values=[float(loss.cpu())for loss in losses];require(all(np.isfinite(v)for v in values),'finite training losses required')
  final=model_state_sha256(model)
  env_end=environment.guard(contract,'application_end',out/'ENVIRONMENT_FAILURE.json')
  workers=[read(p)for p in sorted(Path(config['worker_receipts']).glob('worker*.json'))]
  require(len(workers)==config['num_workers']and all(w['status']=='closed'for w in workers),'every spawn worker must emit its normal close receipt')
  require({w['pid']for w in workers}==worker_pids,'worker identity/count differs from consumed batches')
  report={'status':'complete','config':config,'clock':'same Linux monotonic_ns; parent before Popen',
   'environment_start':env_start,'environment_end':env_end,
   'launch_monotonic_ns':launch,'entry_monotonic_ns':ENTRY_NS,'imports_ready_ns':imports_ready,
   'model_setup_begin_ns':model_begin,'model_setup_end_ns':model_end,'native_promotion':promotion,
   'iterator_begin_ns':iterator_begin,'iterator_created_ns':iterator_created,'first_batch_ready_ns':first_ready,
   'launch_to_first_batch_ready_seconds':(first_ready-launch)/1e9,'iterator_to_first_batch_ready_seconds':(first_ready-iterator_begin)/1e9,
   'checkpoints':checkpoints,'steady_start_ns':steady_start,'training_end_ns':training_end,
   'semantic_input_prefixes':semantic_prefixes,
   'checkpointed_measured_seconds':(training_end-steady_start)/1e9,'measured_graphs':config['steps']*config['batch_size'],
   'consumed_graphs':total*config['batch_size'],'input_stream_sha256':stream.hexdigest(),'input_hash_seconds_included':hash_ns/1e9,
   'loader_wait_seconds':wait_ns/1e9,'shutdown_begin_ns':shutdown_begin,'shutdown_end_ns':shutdown_end,
   'worker_shutdown_seconds':(shutdown_end-shutdown_begin)/1e9,'worker_receipts':workers,
   'completed_extractions_including_unconsumed_prefetch':sum(w['completed_graphs']for w in workers),
   'model':{**v4.MODEL_CONFIG,'parameters':sum(p.numel()for p in model.parameters()),'initial_state_sha256':initial,
    'final_state_sha256_descriptive_only':final},'losses':values,'excluded_warmup_prefix_audits':audits,
   'gpu_memory_guard':gpu,'gpu_reserved_peak_gib':torch.cuda.max_memory_reserved()/GIB if device.type=='cuda'else None,
   'device':{'type':device.type,'name':torch.cuda.get_device_name()if device.type=='cuda'else'cpu'},
   'scope':'Configured short/long app genuinely terminates; interior checkpoint prefixes are continuous, not independent lifetimes. Full CPU input SHA includes observer cost. Not original E23 timing.'}
  return report
 finally:
  if iterator is not None:iterator._shutdown_workers()

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
 config=read(args.config);out=Path(config['out'])
 try:report=run(config)
 except BaseException as error:
  save(out/'FAILED.json',{'type':type(error).__name__,'message':str(error)[:2000]});raise
 save(out/'report.json',report)
if __name__=='__main__':main()
