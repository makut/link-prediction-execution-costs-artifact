"""Tiny literals and independent ordered-edge / BFS-label fixtures."""
from collections import deque
import numpy as np
from scipy import sparse

def data():
    edges=[(0,1),(1,0),(0,2),(2,3),(3,1),(4,2),(3,4),(6,7)]
    a=sparse.csr_matrix((np.ones(len(edges),dtype=np.int8),tuple(zip(*edges))),shape=(9,9));a.sort_indices()
    cases=[([4,0,8,2,1,3],(1,4)),([8,7,6,0],(0,1)),([0,1],(0,1)),([6,7,8],(0,1))]
    return a.indptr.astype(np.int64),a.indices.astype(np.int64),cases

def bfs_labels(edge_pairs,size,roots):
    neighbors=[set()for _ in range(size)]
    for u,v in edge_pairs:neighbors[u].add(v);neighbors[v].add(u)
    def distances(source,removed):
        result={source:0};queue=deque([source])
        while queue:
            u=queue.popleft()
            for v in neighbors[u]:
                if v!=removed and v not in result:result[v]=result[u]+1;queue.append(v)
        return result
    u,v=roots;du,dv=distances(u,v),distances(v,u);result=[]
    for x in range(size):
        if x in roots:result.append(1)
        elif x not in du or x not in dv:result.append(0)
        else:
            half,mod=divmod(du[x]+dv[x],2)
            result.append(1+min(du[x],dv[x])+half*(half+mod-1))
    return np.asarray(result,dtype=np.int64)
