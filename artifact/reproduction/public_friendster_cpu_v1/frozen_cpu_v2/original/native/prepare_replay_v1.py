#!/usr/bin/env python3
"""Select existing archived records only. Never sample graph vertices or edges."""
from pathlib import Path
import argparse,collections,hashlib,json
import numpy as np
from native_adapter_v1 import digest

MAX_JSON=64<<20

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):
    path=Path(path)
    if path.is_symlink()or path.stat().st_size>MAX_JSON:raise ValueError('regular bounded JSON required')
    return json.loads(path.read_text(),parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))

def valid_record(r,n):
    ids=r['nodes']
    if not isinstance(ids,list)or not 2<=len(ids)<=10000 or any(type(i)is not int or not 0<=i<n for i in ids)or len(set(ids))!=len(ids):
        raise ValueError('invalid frozen ordered S')
    if r.get('pair',ids[:2])!=ids[:2]or r.get('virtual_graph_nodes',n)!=n or r.get('added_isolates',0):
        raise ValueError('native replay requires original unpadded graph and endpoint-first archived S')
    if digest(np.asarray(ids,dtype=np.int64))!=r['node_set_sha256']:raise ValueError('frozen S hash differs')
    for key in ('canonical_edges_sha256','drnl_sha256'):
        x=r[key]
        if not isinstance(x,str)or len(x)!=64 or any(c not in '0123456789abcdef'for c in x):raise ValueError('invalid oracle hash')
    if type(r['sum_out_degrees'])is not int or r['sum_out_degrees']<0 or type(r['induced_directed_entries'])is not int or r['induced_directed_entries']<0:
        raise ValueError('invalid archived work/output counts')

def choose(records,count,archive_sha,seed=20261125):
    if type(count)is not int or not 0<count<=len(records):raise ValueError('invalid replay count')
    if count==len(records):return list(range(count)),'all original records in archive order'
    groups=collections.defaultdict(list)
    for i,r in enumerate(records):groups[(r.get('sampler'),r.get('cap'),r.get('workload'))].append(i)
    ordered=[]
    for key in sorted(groups,key=repr):
        ordered.append(sorted(groups[key],key=lambda i:hashlib.sha256(f'{archive_sha}:{seed}:{i}'.encode()).hexdigest()))
    picked=[];round_index=0
    while len(picked)<count:
        for group in ordered:
            if round_index<len(group):picked.append(group[round_index])
            if len(picked)==count:break
        round_index+=1
    return sorted(picked),'deterministic round-robin strata selection; original record order retained; no S resampling'

def prepare(archive,count,seed=20261125):
    archive=Path(archive);data=read(archive);graph=data['provenance']['graph'];n,m=graph['nodes'],graph['stored_entries']
    records=data['records']
    for row in records:valid_record(row,n)
    archive_sha=sha(archive);indices,policy=choose(records,count,archive_sha,seed)
    keys=('node_set_sha256','canonical_edges_sha256','drnl_sha256','sum_out_degrees','induced_directed_entries','pair','sampler','cap','workload','pair_index')
    selected=[{'archive_record_index':i,**{key:records[i][key]for key in keys if key in records[i]}}for i in indices]
    counts=collections.Counter(repr((records[i].get('sampler'),records[i].get('cap'),records[i].get('workload')))for i in indices)
    return {'schema_version':1,'kind':'corrected_native_frozen_replay_v1',
        'archive':{'original_local_path':str(archive.resolve()),'bytes':archive.stat().st_size,'sha256':archive_sha,
                   'record_count':len(records),'stock_index_dtype':data.get('configuration',{}).get('stock_index_dtype')},
        'graph':{'nodes':n,'stored_entries':m},'count':len(indices),'selection_seed':seed,'selection_policy':policy,
        'stratum_counts':dict(counts),'selected':selected,
        'scope':'Existing original ordered S and oracle hashes only. Archive widths repeat S and are not independent workloads.'}

def load_selected(plan,archive):
    data=read(archive)
    if sha(archive)!=plan['archive']['sha256']or Path(archive).stat().st_size!=plan['archive']['bytes']:raise ValueError('archive identity differs from replay plan')
    graph=data['provenance']['graph']
    if (graph['nodes'],graph['stored_entries'])!=(plan['graph']['nodes'],plan['graph']['stored_entries']):raise ValueError('graph dimensions differ')
    result=[];seen=set()
    for frozen in plan['selected']:
        i=frozen['archive_record_index']
        if type(i)is not int or i in seen or not 0<=i<len(data['records']):raise ValueError('invalid replay index')
        seen.add(i);row=data['records'][i];valid_record(row,graph['nodes'])
        if any(row.get(k)!=v for k,v in frozen.items()if k!='archive_record_index'):raise ValueError('archived record fingerprint differs')
        result.append((i,row))
    if len(result)!=plan['count']:raise ValueError('replay count differs')
    return result

def write_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists()or path.with_suffix(path.suffix+'.partial').exists():raise FileExistsError('preserve existing replay plan')
    temporary=path.with_suffix(path.suffix+'.partial');temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.rename(path)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--count',type=int,choices=(128,768),required=True);p.add_argument('--seed',type=int,default=20261125);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=prepare(a.archive,a.count,a.seed);write_new(a.output,plan);print(json.dumps({'status':'prepared_existing_S_only','count':plan['count'],'output':str(a.output)}))
