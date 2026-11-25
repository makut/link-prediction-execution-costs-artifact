#!/usr/bin/env python3
"""One fresh, CPU-only replay worker. Source/native setup is outside query timers."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,platform,resource,sys,time
import numpy as np
from native_adapter_v1 import NativeCSR,pack_local,canonical_edges,drnl,digest,ordered_oracle
from prepare_replay_v1 import read,sha,load_selected
from fixtures_v1 import data as fixture_data,bfs_labels

GIB=2**30
MODES=('reference_full_clear','persistent_touched_reset','upstream_torch_sparse_direct')

def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def memory():
    x={'ru_maxrss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))}
    p=Path('/proc/self/smaps_rollup')
    if p.exists():
        for line in p.read_text().splitlines():
            key=line.split(':')[0]
            if key in('Rss','Pss','Private_Clean','Private_Dirty','Shared_Clean','Shared_Dirty'):x[key.lower()+'_bytes']=int(line.split()[1])*1024
    return x

def available():
    values=[]
    p=Path('/proc/meminfo')
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith('MemAvailable:'):values.append(int(line.split()[1])*1024)
    root=Path('/sys/fs/cgroup')
    if(root/'memory.max').exists()and(root/'memory.current').exists():
        limit=(root/'memory.max').read_text().strip()
        if limit!='max'and int(limit)<2**60:values.append(max(0,int(limit)-int((root/'memory.current').read_text())))
    return min(values)if values else None

def budget(plan,records,max_assoc_bytes,max_peak_bytes):
    n,m=plan['graph']['nodes'],plan['graph']['stored_entries']
    native=8*(n+1+m);assoc=8*n;largest=max(r['induced_directed_entries']for _,r in records)
    local=160*largest+256*max(len(r['nodes'])for _,r in records)
    peak=native+assoc+local+GIB
    if assoc>max_assoc_bytes:raise ValueError('declared per-worker association byte budget exceeded')
    if peak>max_peak_bytes:raise ValueError('declared native-map/association/local-work estimate exceeds peak budget')
    capacity=available()
    if capacity is not None and peak>.85*capacity:raise ValueError('conservative estimate exceeds85% current host/cgroup headroom')
    return {'native_int64_storage_and_potential_file_working_set_bytes':native,'association_bytes_per_worker':assoc,
        'local_output_upper_estimate_bytes':local,'conservative_peak_estimate_bytes':peak,'effective_available_bytes':capacity,
        'scope':'Estimate, not measured peak. Current cgroup headroom includes charged file cache; conservative guards can skip reclaimable-cache cases.'}

def open_native(folder,plan):
    folder=Path(folder);meta=read(folder/'manifest.json');g=meta['graph'];n,m=plan['graph']['nodes'],plan['graph']['stored_entries']
    if(g['n'],g['m'])!=(n,m):raise ValueError('native dimensions differ from archived replay')
    start=time.perf_counter();files=[]
    for name,size,key in(('rowptr.i8',8*(n+1),'rowptr_sha256'),('col.i8',8*m,'col_sha256')):
        p=folder/name;s=p.stat();before=(s.st_size,s.st_mtime_ns)
        if p.is_symlink()or s.st_size!=size or sha(p)!=meta[key]:raise ValueError('native payload size/hash differs: '+name)
        after=p.stat()
        if before!=(after.st_size,after.st_mtime_ns):raise ValueError('native payload changed during setup')
        files.append({'path':str(p),'bytes':size,'mtime_ns':s.st_mtime_ns,'sha256':meta[key]})
    ptr=np.memmap(folder/'rowptr.i8',dtype='<i8',mode='c',shape=(n+1,))
    col=np.memmap(folder/'col.i8',dtype='<i8',mode='c',shape=(m,))if m else np.empty(0,dtype=np.int64)
    if not ptr.dtype.isnative or not col.dtype.isnative or int(ptr[0])!=0 or int(ptr[-1])!=m:raise ValueError('native endian/CSR sentinel invalid')
    return ptr,col,{'original_conversion_manifest':meta,'content_verification_seconds':time.perf_counter()-start,'runtime_files':files,
                   'conversion_performed_by_this_worker':False,'scope':'Reuses an explicit int64 conversion; conversion and verification outside per-query timers.'}

def extension(path,expected_sha):
    if not path or not expected_sha or sha(path)!=expected_sha:raise ValueError('exact frozen external helper SHA required')
    spec=importlib.util.spec_from_file_location('frozen_external_helper',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch,_,software=module.load_operator(1);module.native_admission_smoke(torch,__import__('torch_sparse'),False)
    return torch,software

def admission(library,mode,assoc_bytes,torch=None):
    ptr,col,cases=fixture_data();count=0
    with NativeCSR(ptr,col,library,'persistent_touched_reset'if mode=='upstream_torch_sparse_direct'else mode,max_assoc_bytes=assoc_bytes)as state:
        for ids,roots in cases:
            nodes=np.asarray(ids,dtype=np.int64);expected=ordered_oracle(ptr,col,nodes)
            if torch is None:
                with state.induce(nodes)as out:arrays=out.copy_arrays()
            else:
                tensors=torch.ops.torch_sparse.saint_subgraph(torch.from_numpy(nodes),torch.from_numpy(ptr),torch.empty(0,dtype=torch.int64),torch.from_numpy(col))
                arrays=tuple(x.numpy()for x in tensors)
            if not all(np.array_equal(a,b)for a,b in zip(arrays,expected)):raise AssertionError('actual native fixture ordered/e_id mismatch')
            a=pack_local(arrays[0],arrays[1],len(nodes),roots)
            if not np.array_equal(drnl(a,roots),bfs_labels(canonical_edges(a),len(nodes),roots)):raise AssertionError('independent arbitrary-root DRNL fixture differs')
            count+=1
    return {'status':'pass','literal_cases':count,'ordered_raw_edges_and_eids':True,'arbitrary_root_packaging_and_independent_BFS_labels':True}

def one(record,ptr,col,mode,state,torch,warmup,repeats):
    nodes=np.asarray(record['nodes'],dtype=np.int64);roots=(0,1)
    if sum(int(ptr[u+1])-int(ptr[u])for u in nodes)!=record['sum_out_degrees']:raise ValueError('selected row degree sum differs')
    for u in nodes:
        lo,hi=int(ptr[u]),int(ptr[u+1])
        if not 0<=lo<=hi<=len(col):raise ValueError('invalid selected row bounds')
        row=col[lo:hi]
        for start in range(1,len(row),65536):
            stop=min(len(row),start+65536)
            if np.any(row[start:stop]<=row[start-1:stop-1]):raise ValueError('selected rows must be sorted and unique')
        if len(row)and(int(row[0])<0 or int(row[-1])>=len(ptr)-1):raise ValueError('selected neighbor outside graph')
        at=np.searchsorted(row,u)
        if at<len(row)and int(row[at])==u:raise ValueError('selected rows must be loop-free')
    # Validation/oracle is outside timers, as in the existing exact harness.
    expected=ordered_oracle(ptr,col,nodes);expected_hashes=[digest(a)for a in expected]
    idx=torch.from_numpy(nodes)if torch is not None else None
    empty=torch.empty(0,dtype=torch.int64)if torch is not None else None
    tensors_ptr,tensors_col=(torch.from_numpy(ptr),torch.from_numpy(col))if torch is not None else(None,None)
    raw_times=[];edge_times=[];drnl_times=[]
    for i in range(warmup+repeats):
        result=None;tensors=None;start=time.perf_counter()
        try:
            if torch is None:
                result=state.induce(nodes);arrays=(result.rows,result.cols,result.edge_ids);raw=result.native_seconds
            else:
                native_start=time.perf_counter();tensors=torch.ops.torch_sparse.saint_subgraph(idx,tensors_ptr,empty,tensors_col)
                raw=time.perf_counter()-native_start;arrays=tuple(v.numpy()for v in tensors)
            actual=pack_local(arrays[0],arrays[1],len(nodes),roots);edge_end=time.perf_counter()
            labels=drnl(actual,roots);labels_end=time.perf_counter()
            if any(digest(a)!=b for a,b in zip(arrays,expected_hashes)):raise AssertionError('ordered rows/cols/original e_id differ')
            if digest(canonical_edges(actual))!=record['canonical_edges_sha256']or digest(labels)!=record['drnl_sha256']:raise AssertionError('archived edge or DRNL oracle differs')
            if actual.nnz!=record['induced_directed_entries']or actual[0,1]or actual[1,0]:raise AssertionError('target removal/entry count differs')
            if state is not None and state.live_touched!=0:raise AssertionError('association map not reset')
            if i>=warmup:raw_times.append(raw);edge_times.append(edge_end-start);drnl_times.append(labels_end-edge_end)
        finally:
            if result is not None:result.close()
            tensors=None
    return {'status':'ok','raw_operator_seconds':raw_times,'edge_seconds':edge_times,'drnl_seconds':drnl_times,
        'exact_ordered_rows_cols_and_original_eids':True,'ordered_output_sha256':expected_hashes,
        'exact_edges':True,'exact_drnl':True,'root_positions':[0,1]}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--replay-plan',type=Path,required=True);p.add_argument('--native-dir',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-assoc-gib',type=float,default=2);p.add_argument('--max-peak-gib',type=float,default=16);p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-query-neighbors',type=int,default=20_000_000)
    p.add_argument('--external-helper',type=Path);p.add_argument('--external-helper-sha256');a=p.parse_args(argv)
    if not all(math.isfinite(x)and x>0 for x in(a.max_assoc_gib,a.max_peak_gib))or a.warmup<0 or a.repeats<=0 or a.max_query_neighbors<=0:p.error('finite positive budgets/repeats required')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError('fresh worker output required')
    report={'schema_version':1,'kind':'corrected_native_replay_worker_v1','status':'initializing','mode':a.mode,
        'configuration':{k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},'provenance':{'python':sys.version,'platform':platform.platform(),'worker_sha256':sha(Path(__file__)),'adapter_sha256':sha(Path(__file__).with_name('native_adapter_v1.py')),'library_sha256':sha(a.library)},'records':[],
        'timing_contract':'edge_seconds includes invocation, native fresh output clones, NumPy views, target removal and local CSR. DRNL is separate; validation/hashes/setup/conversion/init are outside.',
        'uncertainty':'Repeated-cache measurements on existing frozen S; no independent-seed confidence interval or model-quality claim.'}
    save(a.output,report);state=None
    try:
        plan=read(a.replay_plan);records=load_selected(plan,a.archive);report['replay_plan_sha256']=sha(a.replay_plan);report['archive_sha256']=sha(a.archive)
        if any(r['sum_out_degrees']>a.max_query_neighbors for _,r in records):raise ValueError('predeclared replay exceeds explicit selected-row neighbor guard')
        report['budget']=budget(plan,records,int(a.max_assoc_gib*GIB),int(a.max_peak_gib*GIB))
        torch=None
        if a.mode=='upstream_torch_sparse_direct':torch,software=extension(a.external_helper,a.external_helper_sha256);report['upstream_software']=software
        report['actual_native_admission']=admission(a.library,a.mode,int(a.max_assoc_gib*GIB),torch)
        ptr,col,meta=open_native(a.native_dir,plan);report['native_setup']=meta;report['memory_before_state_init']=memory()
        if torch is None:
            state=NativeCSR(ptr,col,a.library,a.mode,int(a.max_assoc_gib*GIB));report['state_initialization']=state.initialization
        else:report['state_initialization']={'persistent_assoc_bytes_per_worker':0,'assoc_bytes_during_each_call':8*plan['graph']['nodes'],'policy':'unmodified upstream full-N association allocation/init inside each call'}
        report['memory_after_state_init']=memory();report['status']='running';save(a.output,report)
        for index,record in records:
            measured=one(record,ptr,col,a.mode,state,torch,a.warmup,a.repeats)
            report['records'].append({'archive_record_index':index,'node_set_sha256':record['node_set_sha256'],**measured});save(a.output,report)
        report['status']='complete';report['all_executed_outputs_exact']=True;report['record_count']=len(records);report['memory_final']=memory();save(a.output,report);return 0
    except Exception as error:
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)});save(a.output,report);return 1
    finally:
        if state is not None:state.close()

if __name__=='__main__':raise SystemExit(main())
