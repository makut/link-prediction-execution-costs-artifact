"""E25 operational hooks around the byte-identical V4 Dataset.__getitem__."""
from pathlib import Path
import importlib.util,json,os,sys,time
from multiprocessing.util import Finalize
import numpy as np
from common import HERE,NATIVE_ARMS,require,save
import environment as environment
CODE=HERE/'frozen_src'
sys.path.insert(0,str(CODE))
from scripts.seal_revision import pipeline_benchmark_v4 as v4
require(Path(v4.__file__).resolve()==CODE/'scripts/seal_revision/pipeline_benchmark_v4.py'
 and Path(v4.exact.__file__).resolve()==CODE/'scripts/seal_revision/exact_benchmark.py'
 and Path(v4.exact.kernels.__file__).resolve()==CODE/'seal/graph_csr_kernels.py','actual frozen V4/exact/kernel origins differ')
_ORIGINAL_INDUCE=v4.induce_selected
_NATIVE=None

def native_module():
 spec=importlib.util.spec_from_file_location('e25_frozen_native_adapter',HERE/'native_core/native_adapter_v1.py')
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def induce_selected(graph,nodes,config,selected_nnz):
 if config['arm'] not in NATIVE_ARMS:return _ORIGINAL_INDUCE(graph,nodes,config,selected_nnz)
 require(selected_nnz<=config['max_neighbors'],'native selected-row induction exceeds original neighbor guard')
 require(_NATIVE is not None and _NATIVE.owner_pid==os.getpid(),'native state must be created in this spawn worker')
 result=_NATIVE.induce(np.ascontiguousarray(nodes,dtype=np.int64))
 try:
  require(_NATIVE.live_touched==0,'native touched state not reset after successful induction')
  adjacency=_NATIVE._e25_module.pack_local(result.rows,result.cols,len(nodes),root_positions=(0,1))
 finally:result.close()
 return adjacency,None
v4.induce_selected=induce_selected

class LifecycleDataset(v4.PipelineDataset):
 def __init__(self,config):
  super().__init__(config);self.completed=0;self.worker_id=None;self.receipt=None;self.finalizer=None
 def initialize_worker(self,worker_id):
  global _NATIVE
  import torch
  began=time.monotonic_ns();self.worker_id=worker_id
  env_start=environment.guard(self.config['environment_contract'],'worker_start',Path(self.config['worker_receipts'])/('worker'+str(worker_id)+'.ENVIRONMENT_FAILURE.json'))
  self.graph=v4.load_graph(self.config);self.pairs=np.load(self.config['pairs_path'],mmap_mode='r',allow_pickle=False)
  self.receipt={'pid':os.getpid(),'worker_id':worker_id,'arm':self.config['arm'],'native_storage_policy':self.config.get('native_storage_policy'),'setup_begin_ns':began,'environment_start':env_start,
   'original_graph_mmap_end_ns':time.monotonic_ns(),'initialization_count':0,'completed_graphs':0,
   'torch_cpu_threads':torch.get_num_threads(),'thread_environment':{k:os.environ.get(k)for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')}}
  require(self.receipt['torch_cpu_threads']==1 and all(v=='1'for v in self.receipt['thread_environment'].values()),'actual one-thread CPU worker environment required')
  if self.config['arm'] in NATIVE_ARMS:
   root=Path(self.config['native_root']);meta=json.loads((root/'READY.json').read_text());n,m=meta['nodes'],meta['stored_entries']
   if self.config['arm']=='reference_union_native_reuse':
    require((root/'rowptr.i8').stat().st_dev==(root/'col.i8').stat().st_dev==root.stat().st_dev==self.config['native_reuse_expected_device'],'worker native reuse files differ from actual admitted PVC device')
   ptr=np.memmap(root/'rowptr.i8',dtype='<i8',mode='r',shape=(n+1,))
   col=np.memmap(root/'col.i8',dtype='<i8',mode='r',shape=(m,))
   require(n==self.graph.n and m==self.graph.m,'native/original graph dimensions differ')
   module=native_module();state_begin=time.monotonic_ns()
   _NATIVE=module.NativeCSR(ptr,col,self.config['native_library'],mode='persistent_touched_reset',max_assoc_bytes=1<<30)
   _NATIVE._e25_module=module
   self.receipt.update(initialization_count=1,native_state_begin_ns=state_begin,native_state_end_ns=time.monotonic_ns(),
    native_initialization=_NATIVE.initialization,native_arrays={'rowptr_bytes':ptr.nbytes,'col_bytes':col.nbytes,'readonly':not ptr.flags.writeable and not col.flags.writeable})
  else:self.receipt['compiled_setup_policy']='no prewarm; natural first-use JIT/cache load'
  self.receipt['setup_end_ns']=time.monotonic_ns();self.write_receipt('ready')
  self.finalizer=Finalize(self,self.close_worker,exitpriority=10)
 def write_receipt(self,status):
  self.receipt.update(status=status,completed_graphs=self.completed,observed_ns=time.monotonic_ns())
  save(Path(self.config['worker_receipts'])/('worker'+str(self.worker_id)+'.json'),self.receipt)
 def close_worker(self):
  global _NATIVE
  before=time.monotonic_ns()
  if _NATIVE is not None:
   require(_NATIVE.live_touched==0,'native state dirty at worker shutdown')
   _NATIVE.close();_NATIVE=None
  self.receipt.update(close_begin_ns=before,close_end_ns=time.monotonic_ns(),native_closed=True)
  self.receipt['environment_end']=environment.guard(self.config['environment_contract'],'worker_end',Path(self.config['worker_receipts'])/('worker'+str(self.worker_id)+'.ENVIRONMENT_FAILURE.json'))
  self.receipt['compiled_induction_signatures_at_close']={name:[str(s)for s in getattr(getattr(v4.exact.kernels,name),'nopython_signatures',())]
   for name in('_build_induced_edges_pairwise','_fill_neighbors_with_owner_numba','_build_induced_edges_neighbor_scan')}
  self.write_receipt('closed')
 def __getitem__(self,index):
  if self.worker_id is None:
   import torch
   info=torch.utils.data.get_worker_info();require(info is not None,'dataset must initialize inside a spawn worker')
   self.initialize_worker(info.id)
  first=self.completed==0
  if first:self.receipt['first_query_begin_ns']=time.monotonic_ns()
  record=super().__getitem__(index)  # Original sampler/RNG/ordered S/DRNL/PyG body.
  if first:self.receipt['first_query_end_ns']=time.monotonic_ns()
  self.completed+=1
  return (*record,{'query_index':int(index),'worker_pid':os.getpid()})

def worker_init(worker_id):
 import torch
 torch.set_num_threads(1)

class Collate:
 def __call__(self,records):
  out=v4.Collate()(records)
  out['query_indices']=np.asarray([r[4]['query_index']for r in records],dtype='<i8')
  out['worker_pids']=[r[4]['worker_pid']for r in records]
  return out
