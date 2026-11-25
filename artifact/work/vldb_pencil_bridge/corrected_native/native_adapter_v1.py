"""Per-worker adapter for matched standalone full-clear / touched-reset controls.

Raw outputs retain upstream CSR scan order and original edge IDs. Target removal,
arbitrary-root labeling and SciPy packaging are separate operations below.
"""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import hashlib,json,os,time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

VERSION='persistent_assoc_touched_reset_v1'
MODES={'reference_full_clear':0,'persistent_touched_reset':1}
I64P=C.POINTER(C.c_int64)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def digest(array):return hashlib.sha256(np.ascontiguousarray(array,dtype='<i8').tobytes()).hexdigest()

def require_array(value,name):
    if not isinstance(value,np.ndarray)or value.ndim!=1 or value.dtype!=np.dtype('int64')or not value.flags.c_contiguous:
        raise ValueError(name+' must already be a contiguous native int64 vector; graph conversion is never implicit')
    return value

class ResultStruct(C.Structure):
    _fields_=[('rows',I64P),('cols',I64P),('edge_ids',I64P),('length',C.c_int64)]

class NativeResult:
    """Owns native clones. Views are valid until close(); copy_arrays detaches."""
    def __init__(self,library,result,native_seconds):
        self._library,self._result=library,result;self.native_seconds=native_seconds;self._closed=False
        size=int(result.length)
        self.rows=np.ctypeslib.as_array(result.rows,shape=(size,))if size else np.empty(0,dtype=np.int64)
        self.cols=np.ctypeslib.as_array(result.cols,shape=(size,))if size else np.empty(0,dtype=np.int64)
        self.edge_ids=np.ctypeslib.as_array(result.edge_ids,shape=(size,))if size else np.empty(0,dtype=np.int64)
        for a in(self.rows,self.cols,self.edge_ids):a.flags.writeable=False
    def copy_arrays(self):
        if self._closed:raise RuntimeError('native result is closed')
        return self.rows.copy(),self.cols.copy(),self.edge_ids.copy()
    def close(self):
        if not self._closed:self._library.lpp_native_result_free_v1(C.byref(self._result));self._closed=True
    def __enter__(self):
        if self._closed:raise RuntimeError('native result is closed')
        return self
    def __exit__(self,*_):self.close()
    def __del__(self):self.close()

class NativeCSR:
    def __init__(self,rowptr,col,library_path,mode='persistent_touched_reset',max_assoc_bytes=2*(2**30)):
        self.rowptr=require_array(rowptr,'rowptr');self.col=require_array(col,'col')
        self.n=len(rowptr)-1;self.m=len(col);self.mode=mode;self.owner_pid=os.getpid();self._state=C.c_void_p();self._closed=False
        if self.n<=0 or mode not in MODES:raise ValueError('invalid graph or mode')
        if type(max_assoc_bytes)is not int or not 0<max_assoc_bytes<2**63:raise ValueError('finite integer association byte budget required')
        self.library_path=Path(library_path).resolve();self.library=C.CDLL(str(self.library_path))
        lib=self.library;lib.lpp_native_version_v1.restype=C.c_char_p
        if lib.lpp_native_version_v1().decode()!=VERSION:raise ValueError('wrong native ABI version')
        lib.lpp_native_create_v1.argtypes=[C.c_int64,C.c_int,C.c_int64,C.POINTER(C.c_void_p),C.c_char_p,C.c_size_t]
        lib.lpp_native_create_v1.restype=C.c_int
        lib.lpp_native_destroy_v1.argtypes=[C.c_void_p];lib.lpp_native_destroy_v1.restype=None
        lib.lpp_native_persistent_bytes_v1.argtypes=[C.c_void_p];lib.lpp_native_persistent_bytes_v1.restype=C.c_int64
        lib.lpp_native_live_touched_v1.argtypes=[C.c_void_p];lib.lpp_native_live_touched_v1.restype=C.c_int64
        lib.lpp_native_result_free_v1.argtypes=[C.POINTER(ResultStruct)];lib.lpp_native_result_free_v1.restype=None
        lib.lpp_native_induce_v1.argtypes=[C.c_void_p,I64P,C.c_int64,I64P,C.c_int64,I64P,C.c_int64,C.c_int64,C.POINTER(ResultStruct),C.c_char_p,C.c_size_t]
        lib.lpp_native_induce_v1.restype=C.c_int
        error=C.create_string_buffer(1024);start=time.perf_counter()
        code=lib.lpp_native_create_v1(self.n,MODES[mode],max_assoc_bytes,C.byref(self._state),error,len(error))
        elapsed=time.perf_counter()-start
        if code:self._closed=True;raise RuntimeError(error.value.decode())
        self.initialization={'version':VERSION,'mode':mode,'nodes':self.n,'stored_entries':self.m,'owner_pid':self.owner_pid,
            'initialization_seconds':elapsed,'persistent_assoc_bytes_per_worker':int(lib.lpp_native_persistent_bytes_v1(self._state)),
            'assoc_bytes_during_each_call':8*self.n,'initialization_policy':'once per worker'if MODES[mode]else'full-N allocation/init inside every native call',
            'library':str(self.library_path),'library_sha256':sha(self.library_path),
            'scope':'Initialization is outside query timers. Persistent 8N map is not free or shared between workers.'}
    def induce(self,nodes,*,fixture_fail_after_touches=-1):
        if self._closed:raise RuntimeError('native state is closed')
        if os.getpid()!=self.owner_pid:raise RuntimeError('construct a separate association state inside each worker after fork')
        if len(nodes)>10000:raise ValueError('selected set exceeds10000node guard')
        a=np.asarray(nodes)
        if a.dtype!=np.dtype('int64')or a.ndim!=1 or not a.flags.c_contiguous:raise ValueError('nodes must be contiguous native int64')
        if len(a)==0:raise ValueError('empty selected set')
        result=ResultStruct();error=C.create_string_buffer(1024);start=time.perf_counter()
        code=self.library.lpp_native_induce_v1(self._state,self.rowptr.ctypes.data_as(I64P),len(self.rowptr),
            self.col.ctypes.data_as(I64P),self.m,a.ctypes.data_as(I64P),len(a),fixture_fail_after_touches,
            C.byref(result),error,len(error))
        elapsed=time.perf_counter()-start
        if code:raise RuntimeError(error.value.decode())
        return NativeResult(self.library,result,elapsed)
    @property
    def live_touched(self):
        if self._closed:raise RuntimeError('native state is closed')
        return int(self.library.lpp_native_live_touched_v1(self._state))
    def close(self):
        if not getattr(self,'_closed',True):
            if hasattr(self,'library')and bool(self._state):self.library.lpp_native_destroy_v1(self._state)
            self._state=C.c_void_p()
            self._closed=True
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def __del__(self):self.close()

def validate_roots(nodes,root_positions):
    roots=tuple(root_positions)
    if len(roots)!=2 or any(type(i)is not int or not 0<=i<len(nodes) for i in roots)or roots[0]==roots[1]:
        raise ValueError('two distinct in-range integer root positions required')
    return roots

def pack_local(rows,cols,size,root_positions=(0,1)):
    u,v=validate_roots(range(size),root_positions)
    keep=~(((rows==u)&(cols==v))|((rows==v)&(cols==u)))
    return sparse.csr_matrix((np.ones(int(keep.sum()),dtype=np.int8),(rows[keep],cols[keep])),shape=(size,size))

def canonical_edges(a):
    r,c=a.nonzero();order=np.lexsort((c,r));return np.stack((r[order],c[order]),axis=1).astype(np.int64)

def drnl(a,root_positions=(0,1)):
    u,v=validate_roots(range(a.shape[0]),root_positions)
    order=np.asarray([u,v,*[i for i in range(a.shape[0])if i not in(u,v)]],dtype=np.int64)
    x=a[order,:][:,order];n=x.shape[0]
    keep_dst,keep_src=np.r_[0,np.arange(2,n)],np.arange(1,n)
    ds=shortest_path(x[keep_dst,:][:,keep_dst],directed=False,unweighted=True,indices=0)
    dt=shortest_path(x[keep_src,:][:,keep_src],directed=False,unweighted=True,indices=0)
    ds,dt=np.insert(ds,1,0),np.insert(dt,0,0);z=np.zeros(n,dtype=np.int64);valid=np.isfinite(ds)&np.isfinite(dt)
    total=ds[valid]+dt[valid];half,mod=np.floor_divide(total,2),np.mod(total,2)
    z[valid]=(1+np.minimum(ds[valid],dt[valid])+half*(half+mod-1)).astype(np.int64);z[:2]=1
    result=np.empty(n,dtype=np.int64);result[order]=z;return result

def ordered_oracle(rowptr,col,nodes):
    mapping={int(v):i for i,v in enumerate(nodes)};rows=[];cols=[];ids=[]
    for i,v in enumerate(nodes):
        for j in range(int(rowptr[v]),int(rowptr[v+1])):
            at=mapping.get(int(col[j]))
            if at is not None:rows.append(i);cols.append(at);ids.append(j)
    return tuple(np.asarray(v,dtype=np.int64)for v in(rows,cols,ids))
