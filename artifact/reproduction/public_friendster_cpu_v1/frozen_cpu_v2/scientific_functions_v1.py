"""Exact extracted original functions; only operational module imports adapted."""
from cpu_common_v1 import *
def original_identity(graph_path,graph):
    files={}
    for name,size in(('edge_starts.bin',8*graph.n),('edge_ends.bin',8+4*graph.m)):
        p=graph_path/name;s=p.stat();require(not p.is_symlink()and s.st_size==size,'original GraphCSR size/type changed')
        files[str(p)]={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
    return files

def verify_same_selection(plan,existing):
    # Metadata path is local to each preparation; every scientific field matches.
    a,b=read(existing),dict(plan)
    a['archive']=dict(a['archive']);b['archive']=dict(b['archive'])
    a['archive'].pop('original_local_path',None);b['archive'].pop('original_local_path',None)
    require(a==b,'selection differs from preceding native128 replay')

def one(record,index,graph,np,exact,*,warmup=1,repeats=3):
    require(warmup==1 and repeats==3,'frozen1warmup3repeats required')
    nodes=np.asarray(record['nodes'],dtype=np.int64)
    degree=graph.degree_sum(nodes)
    require(degree==record['sum_out_degrees'],'frozen selected degree changed')
    require(degree<=20_000_000,'selected-row neighbor guard20M exceeded')
    require(degree*256+nodes.size*256<=6*GIB,'local workspace6GiB guard exceeded')
    require(nodes.size*(nodes.size-1)//2<=20_000_000,'fixed pairwise work20M guard exceeded')
    for u in nodes:graph.validate_row(int(u))
    oracle=exact.oracle_induced(graph,nodes)
    require((oracle!=oracle.T).nnz==0,'undirected query oracle required')
    edges,labels=exact.canonical_edges(oracle),exact.drnl_labels(oracle)
    require(exact.digest(edges)==record['canonical_edges_sha256']and exact.digest(labels)==record['drnl_sha256'],
            'original frozen target-removed CSR/DRNL oracle differs')
    require(oracle.nnz==record['induced_directed_entries']and not oracle[0,1]and not oracle[1,0],'target removal/count differs')
    order=list(METHODS);np.random.default_rng(20261125+index).shuffle(order)
    results={}
    for mode in order:
        results[mode]=exact.measure(lambda mode=mode:exact.production_induced(graph,nodes,mode),edges,labels,warmup,repeats)
    return {'archive_record_index':index,'node_set_sha256':record['node_set_sha256'],
            'canonical_edges_sha256':record['canonical_edges_sha256'],'drnl_sha256':record['drnl_sha256'],
            'sum_out_degrees':record['sum_out_degrees'],'induced_directed_entries':record['induced_directed_entries'],
            'method_order':order,'methods':results,'exact_archived_oracles':True}
