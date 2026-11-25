"""Independent NumPy-only trace checks and literal tiny-graph raw-slot oracle."""
import hashlib,json
from pathlib import Path
import numpy as np
from archive_contract import require,read,identity,valid_sha

NAMES={'pairs.npy','labels.npy','node_ptr.npy','nodes.npy','root_positions.npy','records.json',
       'rng_python_words.npy','rng_numpy_words.npy','rng_torch_cpu.npy','rng_scalars.npy'}

def array_hash(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def numpy_fp(a):return {'kind':'numpy','dtype':a.dtype.str,'shape':list(a.shape),'hash':array_hash(a)}
def tensor_fp(a,dtype):
    a=np.ascontiguousarray(a);step=1;stride=[]
    for dim in reversed(a.shape):stride.insert(0,step);step*=max(1,dim)
    return {'kind':'tensor','dtype':dtype,'shape':list(a.shape),'stride':stride,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
def check_fp(fp,kind,dtype,shape):
    require(isinstance(fp,dict)and fp.get('kind')==kind and fp.get('dtype')==dtype and fp.get('shape')==list(shape),'fingerprint dtype/shape/kind mismatch')
    require(valid_sha(fp.get('hash'if kind=='numpy'else'sha256')),'valid payload fingerprint required')
    if kind=='tensor':require(isinstance(fp.get('stride'),list)and len(fp['stride'])==len(shape)and all(type(v)is int and v>=0 for v in fp['stride']),'tensor stride metadata required')

def load_trace(path,expected_queries,fixture=False):
    path=Path(path);manifest=read(path/'manifest.json');require(manifest.get('status')=='complete','complete trace required')
    meta=manifest['metadata'];require(meta.get('fixture')is fixture and meta.get('eval_seed')==20261126 and
        meta.get('query_seed_policy')=='SeedSequence([base,0,query_index])'and meta.get('sampler')=={
            'actual_native_operator':'ego_k_hop_sample_adj','depth':1,'neighbors_per_endpoint':40,'replacement':False,'torch_threads':1},
        'frozen fixture/seed/native sampler semantics changed')
    files=manifest['files'];require(len(files)==len(NAMES)and{v['name']for v in files}==NAMES,'complete unique trace+CPU RNG file set required')
    require(sum(v['bytes']for v in files)<=512<<20,'bounded trace files')
    for item in files:
        require(Path(item['name']).name==item['name']and identity(path/item['name'])=={k:item[k]for k in('bytes','sha256')},'immutable trace file changed')
    arrays={n:np.load(path/n,mmap_mode='r',allow_pickle=False)for n in NAMES if n.endswith('.npy')}
    pairs,labels,ptr,ids,roots=[arrays[n]for n in('pairs.npy','labels.npy','node_ptr.npy','nodes.npy','root_positions.npy')]
    q=expected_queries;require(pairs.dtype==np.int64 and pairs.shape==(q,2)and labels.dtype==np.int64 and labels.shape==(q,),
        'exact query pairs/label array shapes required')
    require(ptr.dtype==np.int64 and ptr.shape==(q+1,)and ptr[0]==0 and ptr[-1]==len(ids)and np.all(np.diff(ptr)>=1),
        'complete nonempty ragged node sets required')
    require(ids.dtype==np.int32 and ids.ndim==1 and roots.dtype==np.int64 and roots.shape==(q,2),'literal ID/root dtypes required')
    schema=manifest['metadata']['post_raw_rng_schema'];d=schema['torch_cpu_bytes']
    require(schema['py_version']==3 and schema['np_algorithm']=='MT19937'and type(d)is int and 0<d<=65536,'captured CPU RNG schema required')
    shapes={'rng_python_words.npy':((q,625),np.uint32),'rng_numpy_words.npy':((q,624),np.uint32),
            'rng_torch_cpu.npy':((q,d),np.uint8),'rng_scalars.npy':((q,5),np.float64)}
    for name,(shape,dtype)in shapes.items():require(arrays[name].dtype==dtype and arrays[name].shape==shape,'CPU RNG array shape/dtype mismatch')
    scalars=arrays['rng_scalars.npy'];require(np.isfinite(scalars).all()and np.isin(scalars[:,0],[0,1]).all()and np.isin(scalars[:,3],[0,1]).all(),
        'valid CPU RNG scalar states required')
    require(np.all((scalars[:,2]>=0)&(scalars[:,2]<=624))and np.all(scalars[:,2]==np.floor(scalars[:,2]))and
        np.all(arrays['rng_python_words.npy'][:,-1]<=624),'captured Python/NumPy RNG position is invalid')
    records=read(path/'records.json')['records'];require(len(records)==q,'one native raw fingerprint record per query required')
    n=9 if fixture else 2927963;sets=[]
    for i in range(q):
        S=np.asarray(ids[int(ptr[i]):int(ptr[i+1])],dtype=np.int64);k=len(S);root=np.asarray(roots[i]);rec=records[i]
        require(1<=k<=82 and np.all(np.diff(S)>0)and np.all(S>=0)and np.all(S<n),'literal sorted unique bounded S required')
        require(np.all(root>=0)and np.all(root<k)and np.array_equal(S[root],pairs[i]),'arbitrary/equal root positions must match original query')
        require(rec['query_index']==i and rec['nodes_hash']==array_hash(S),'query index/S fingerprint mismatch')
        require(type(rec['sum_selected_degrees'])is int and rec['sum_selected_degrees']>=0 and all(
            type(rec[key])in(int,float)and np.isfinite(rec[key])and rec[key]>=0 for key in
            ('native_sampler_seconds','native_public_induction_seconds','native_raw_data_seconds')),'finite native raw-stage seconds and selected-degree count required')
        raw=rec['raw'];require(set(raw)=={'rows','cols','csr_edge_slots'},'ordered raw rows/columns/CSR slots required')
        E=raw['rows']['shape'][0]
        require(type(E)is int and E>=0 and E%2==0,'symmetric loop-free raw entry count required')
        require(E<=rec['sum_selected_degrees'],'induced entry count exceeds selected-degree work')
        for value in raw.values():check_fp(value,'numpy','<i8',(E,))
        check_fp(rec['stored_values'],'numpy','<i8',(E,))
        data=rec['data'];require(set(data)=={'num_nodes','root_n_id','seed_node','y','edge_index','x','id','idx'},'complete native Data field set required')
        require(data['num_nodes']==k and data['idx']==i,'native node/query metadata mismatch')
        require(data['root_n_id']==tensor_fp(root,'torch.int64')and data['seed_node']==tensor_fp(np.asarray(pairs[i]),'torch.int64')and
            data['id']==tensor_fp(S,'torch.int64')and data['y']==tensor_fp(np.asarray([labels[i]],dtype=np.int64),'torch.int64'),
            'native roots/global IDs/labels/order fingerprint mismatch')
        check_fp(data['x'],'tensor','torch.float32',(k,128))
        shape=data['edge_index']['shape'];require(len(shape)==2 and shape[0]==2 and type(shape[1])is int and shape[1]>=0,'postmask directed edge tensor shape required')
        post=shape[1];check_fp(data['edge_index'],'tensor','torch.int64',(2,post))
        require(E-post in(0,2),'native both-direction target removal count inconsistent')
        if not fixture and(labels[i]==1 or root[0]==root[1]):require(E==post,'clean heldout/self query must not remove a graph target edge')
        s=scalars[i];rng={'py_version':3,'py_words':numpy_fp(np.asarray(arrays['rng_python_words.npy'][i])),
            'py_gauss':float(s[1])if s[0]else None,'np_algorithm':'MT19937','np_words':numpy_fp(np.asarray(arrays['rng_numpy_words.npy'][i])),
            'np_position':int(s[2]),'np_has_gauss':int(s[3]),'np_cached_gauss':float(s[4]),'torch_cpu':numpy_fp(np.asarray(arrays['rng_torch_cpu.npy'][i]))}
        require(rec['post_raw_rng']==rng,'captured post-native CPU RNG fingerprints disagree with actual exported arrays')
        sets.append(S)
    if fixture:check_tiny(pairs,labels,sets,roots,records)
    else:
        require(q==1024 and np.array_equal(labels,np.r_[np.ones(512,dtype=np.int64),np.zeros(512,dtype=np.int64)]),
            '512 frozen positives then512 fixed candidate0 negatives required')
        require(np.array_equal(pairs[:512,0],pairs[512:,0]),'original per-source query order changed')
        meta=manifest['metadata']['cohort'];require(meta['sources']==512 and meta['original_negatives_per_source']==1000 and meta['queries']==1024 and
            meta['selection_depends_on_timing_or_scores']is False and meta['pairs_hash']==array_hash(pairs)and meta['labels_hash']==array_hash(labels),
            'predeclared original-group cohort/hash contract changed')
    return {'manifest':manifest,'records':records,'queries':q,'pairs':pairs,'labels':labels,'sets':sets,'roots':roots,
        'report':{'queries':q,'literal_sorted_S_and_arbitrary_equal_roots_verified':True,'node_id_label_and_CPU_RNG_hashes_independently_verified':True,
            'raw_slot_value_and_postmask_shape_fingerprints_complete':True,'tiny_ordered_raw_slots_values_and_target_mask_independently_recomputed':fixture,
            'real_graph_oracle_reexecuted':False,'scope':'Real edge/value tensors are not exported: their exactness is supported by full-cohort native audit receipts and pinned source, not new graph reconstruction.'}}

def check_tiny(pairs,labels,sets,roots,records):
    edges=[(0,1),(0,2),(1,3),(2,4),(3,4),(6,7)];arcs=sorted(edges+[(v,u)for u,v in edges]);slots={e:i for i,e in enumerate(arcs)}
    expected_pairs=np.asarray([[3,0],[0,1],[8,5],[8,0],[0,0],[7,6],[8,8]],dtype=np.int64)
    require(np.array_equal(pairs,expected_pairs)and np.array_equal(labels,[0,1,0,0,0,1,0]),'all7 predeclared native tiny queries required')
    features=(np.arange(9*128,dtype=np.float32).reshape(9,128)%97)/97
    for i,(u,v)in enumerate(pairs):
        S=sets[i];expected_S=sorted({int(u),int(v)}|{b for a,b in arcs if a in(u,v)})
        require(S.tolist()==expected_S,'native tiny depth1/40 sampler must retain full literal neighborhood')
        mapping={int(node):j for j,node in enumerate(S)};raw=[(mapping[a],mapping[b],slot)for slot,(a,b)in enumerate(arcs)if a in mapping and b in mapping]
        rows=np.asarray([x[0]for x in raw],dtype=np.int64);cols=np.asarray([x[1]for x in raw],dtype=np.int64)
        ids=np.asarray([x[2]for x in raw],dtype=np.int64);values=np.asarray([slots[(arcs[int(x)][1],arcs[int(x)][0])]for x in ids],dtype=np.int64)
        rec=records[i];require(rec['raw']=={k:numpy_fp(a)for k,a in zip(('rows','cols','csr_edge_slots'),(rows,cols,ids))}and
            rec['stored_values']==numpy_fp(values),'literal tiny native directed order/CSR slots/separate stored values mismatch')
        ru,rv=roots[i];post=sorted((int(c),int(r))for r,c in zip(rows,cols)if(r,c)not in((ru,rv),(rv,ru)))
        matrix=np.asarray(post,dtype=np.int64).reshape(-1,2).T.copy()
        require(rec['data']['edge_index']==tensor_fp(matrix,'torch.int64')and rec['data']['x']==tensor_fp(features[S],'torch.float32'),
            'literal tiny target removal/transpose order/features mismatch')
