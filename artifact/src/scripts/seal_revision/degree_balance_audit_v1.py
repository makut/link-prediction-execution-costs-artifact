"""Predeclared CPU-only first-hop endpoint geometry; no model/score access in phases1–2."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import sys
import time
import numpy as np

VERSION = 1
COUNTS = {'val': 512, 'test': 4096}
SEEDS = tuple(20261128 + i for i in range(16))
BIN_EDGES = (1, 2, 4, 16, 64)
BIN_LABELS = {-1:'min_degree_zero',0:'[1,2)',1:'[2,4)',2:'[4,16)',3:'[16,64)',4:'[64,infinity)'}
CAP, OVERSAMPLE, PER_VERTEX_K = 200, 1.5, 150
MAX_NEIGHBORS, WORKING_BYTES, MAX_WALL = 20_000_000, 4*2**30, 1800
CORE_SOURCES = ('seal/graph_csr_kernels.py','seal/graph_csr_extraction.py')
EXPECTED_JOBS = {(v,s) for v in ('union','pv') for s in (42,43,44)}


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class Guarded(RuntimeError): pass
class Deadline(RuntimeError): pass


class Budget:
    def __init__(self): self.started=time.monotonic()
    def check(self, degree_sum=0):
        if time.monotonic()-self.started >= MAX_WALL: raise Deadline('30-minute compute deadline')
        if degree_sum > MAX_NEIGHBORS: raise Guarded('endpoint row entries exceed20M')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss=int(rss if sys.platform=='darwin' else rss*1024)
        # Conservative touched-row residency + bounded per-query NumPy workspace.
        if rss + int(degree_sum)*8 + (8<<20) > WORKING_BYTES: raise Guarded('4GiB process working-set/headroom guard')


def imbalance_bin(du,dv):
    require(int(du)>=0 and int(dv)>=0,'endpoint degrees must be nonnegative')
    lo,hi=min(int(du),int(dv)),max(int(du),int(dv))
    if lo==0: return -1,None
    ratio=hi/lo
    return min(len(BIN_EDGES)-1,int(np.searchsorted(BIN_EDGES,ratio,side='right')-1)),ratio


def uniform_union(m,k,cap=CAP):
    require(0<=k<=m and 0<=cap<=CAP,'invalid hypergeometric population or cap')
    draws=min(cap,m)
    expected=draws*k/m if m else 0.0
    if draws>m-k: return expected,0.0
    pzero=1.0
    for i in range(draws): pzero *= (m-k-i)/(m-i)
    return expected,pzero


def coverage_bound(degree,own_unique,cap=CAP,k=PER_VERTEX_K):
    require(cap>k and 0<=own_unique<=min(degree,k),'invalid PV bound anatomy')
    return min(own_unique,cap-k)


def unconditional_bound(degree,cap=CAP,k=PER_VERTEX_K):
    return 0 if degree==0 else min(degree,cap-k) if degree<=k else 1


def count_overlap(a,b):
    if len(a)>len(b): a,b=b,a
    total=0
    for lo in range(0,len(a),65536):
        chunk=a[lo:lo+65536];positions=np.searchsorted(b,chunk)
        good=positions<len(b)
        total += int(np.sum(b[positions[good]]==chunk[good]))
    return total


def count_members(values,row):
    if not len(values) or not len(row): return 0
    positions=np.searchsorted(row,values);good=positions<len(row)
    return int(np.sum(row[positions[good]]==values[good]))


def verify_kernel_anatomy(kernels):
    starts=np.array([0,4,4,4,4,4],dtype=np.int64);indices=np.array([1,2,3,4],dtype=np.int32)
    frontier=np.array([0],dtype=np.int64);duplicate_seed=None
    for seed in range(16):
        kernels.seed_numba_rng(seed)
        values,_=kernels._sample_neighbors_per_vertex_numba(starts,indices,4,frontier,3)
        require(len(values)==3 and count_members(values,indices)==3,'PV kernel draws invalid row values')
        if len(np.unique(values))<3: duplicate_seed=seed;break
    require(duplicate_seed is not None,'Expected with-replacement behavior was not observed; inspect frozen kernel')
    values,_=kernels._sample_neighbors_per_vertex_numba(starts,indices,4,frontier,4)
    require(np.array_equal(values,indices),'PV d<=k must copy every neighbor')
    return {'large_degree_draws':'with replacement, deduplicated after concatenation','small_degree_draws':'copy all when degree<=k',
            'duplicate_observed_d4_k3_seed':duplicate_seed,'copy_all_probe_passed':True,
            'universal_degree_bound_rejected':'min(degree,50) is not guaranteed for degree>150',
            'valid_conditional_bound':'retained endpoint neighbors >= min(unique own draws,200-150)',
            'nonisolated_first_hop_starvation_possible':False,
            'preconditions':'simple rows, held-out target absent in train, no self loops, two endpoints, cap200/k150'}


def pv_trial(graph,u,v,kernels,seed):
    np.random.seed(seed);kernels.seed_numba_rng(seed)
    nodes=np.array([u,v],dtype=np.int64)
    values,_=kernels._sample_neighbors_per_vertex_numba(graph.starts,graph.indices,graph.m,nodes,PER_VERTEX_K)
    rows=[graph.neighbors(u),graph.neighbors(v)];lengths=[min(len(row),PER_VERTEX_K) for row in rows]
    require(len(values)==sum(lengths),'PV raw draw count differs from frozen kernel anatomy')
    own=[np.unique(values[:lengths[0]]),np.unique(values[lengths[0]:])]
    for row,draw in zip(rows,own): require(count_members(draw,row)==len(draw),'PV draw outside its endpoint neighborhood')
    fringe=kernels._sorted_setdiff_int64(kernels._unique_sorted_int64(values),np.sort(nodes))
    if len(fringe)>CAP: fringe=np.sort(fringe[np.random.permutation(len(fringe))[:CAP]])
    coverage=[count_members(fringe,row) for row in rows]
    bounds=[coverage_bound(len(row),len(draw)) for row,draw in zip(rows,own)]
    require(all(c>=b for c,b in zip(coverage,bounds)),'PV conditional coverage bound violated')
    require(all(c>=unconditional_bound(len(row)) for c,row in zip(coverage,rows)),'PV unconditional coverage bound violated')
    return coverage,bounds,[len(x) for x in own]


def load_runtime(repo):
    sys.path.insert(0,str(repo))
    from scripts.seal_revision import exact_benchmark
    from seal import graph_csr_kernels
    require('torch' not in sys.modules,'Torch must not be imported')
    return exact_benchmark,graph_csr_kernels


def common_inputs(args):
    prepared,quality,repo=args.prepared_root.resolve(),args.quality_root.resolve(),args.repo.resolve()
    qpath=quality/'plan.json';mpath=prepared/'manifest.json';plan,manifest=read(qpath),read(mpath)
    require(read(prepared/'PREPARED_COMPLETE.json')['manifest_sha256']==sha(mpath)==plan['dataset_provenance']['sha256'],'prepared manifest identity changed')
    require(manifest['dataset']=='friendster' and manifest['variant']=='exact_colleague_80_10_10','wrong prepared Friendster variant')
    protocol=plan['protocol']
    require(protocol['dataset']=='friendster_colleague_80_10_10' and protocol['max_nodes_per_hop']==CAP
            and protocol['per_vertex_oversample']==OVERSAMPLE and protocol['num_hops']==2,'sampling protocol differs')
    require(protocol['validation_query_count']==COUNTS['val'] and protocol['test_query_count']==COUNTS['test'],'query counts differ')
    require(Path(plan['graph_root']).resolve()==prepared/'train_csr','quality train graph differs')
    stats={}
    for name in ('edge_starts.bin','edge_ends.bin'):
        path=prepared/'train_csr'/name;value={'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
        require(value==plan['graph_input_stats'][str(path)],'train graph metadata changed');stats[str(path)]=value
    sources={name:sha(repo/name) for name in CORE_SOURCES}
    for name,value in sources.items(): require(plan['code'][str(repo/name)]==value,'frozen production source changed')
    sources['scripts/seal_revision/exact_benchmark.py']=sha(repo/'scripts/seal_revision/exact_benchmark.py')
    return plan,manifest,{'quality_plan_sha256':sha(qpath),'prepared_manifest_sha256':sha(mpath),'graph_input_stats':stats,'source_sha256':sources,
                          'runtime_versions':{'python':sys.version,'numpy':np.__version__,'numba':importlib.metadata.version('numba')}}


def load_subset(args,plan,manifest,split):
    prefix='valid' if split=='val' else 'test';count=COUNTS[split];subset=plan['subsets'][split]
    require(subset['query_count']==count,'subset count changed')
    arrays={};hashes={}
    for name,suffix in [('query_ids','query_ids'),('positive','edge'),('warm_query','warm_query')]:
        path=args.quality_root/'subsets'/(prefix+'_'+suffix+'.npy')
        require(sha(path)==subset['outputs'][name]['sha256'],'frozen subset hash changed: '+name)
        arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False);hashes[name]=sha(path)
    ids,pos=arrays['query_ids'],arrays['positive']
    require(ids.shape==(count,) and pos.shape==(count,2) and arrays['warm_query'].shape==(count,),'query shapes changed')
    require(ids.dtype.kind in 'iu' and pos.dtype.kind in 'iu' and len(np.unique(ids))==count,'original query IDs must be unique integer rows')
    originals=[]
    for name in ('source_node','target_node','warm_query'):
        path=args.prepared_root/'evaluation'/prefix/(name+'.npy')
        expected=manifest['files']['evaluation/'+prefix+'/'+name+'.npy']['sha256']
        require(sha(path)==expected==subset['inputs'][name]['sha256'],'original row input changed')
        originals.append(np.load(path,mmap_mode='r',allow_pickle=False))
    require(ids.min()>=0 and ids.max()<len(originals[0]),'query ID outside original input')
    require(np.array_equal(pos[:,0],originals[0][ids]) and np.array_equal(pos[:,1],originals[1][ids])
            and np.array_equal(arrays['warm_query'],originals[2][ids]),'subset is not aligned to original query rows')
    return ids,pos,arrays['warm_query'],hashes


def predeclare(args,inputs):
    path=args.output_root/'plan.json';script_sha=sha(Path(__file__))
    if path.exists():
        plan=read(path);require(plan['inputs']==inputs and plan['analysis_code_sha256']==script_sha,'predeclared code/inputs changed')
        return plan
    require(not (args.quality_root/'FROZEN_CHECKPOINTS.json').exists(),'Initial predeclaration must precede joint freeze and held-out test')
    plan={'schema_version':1,'created_unix':time.time(),'predeclared_before_joint_freeze':True,'analysis_code_sha256':script_sha,'inputs':inputs,
          'validation_queries':COUNTS['val'],'test_queries':COUNTS['test'],'mc_seeds':list(SEEDS),'cap':CAP,'per_vertex_k':PER_VERTEX_K,
          'imbalance_bin_edges':[1,2,4,16,64,'infinity'],'cold_bin':'min_degree_zero',
          'hypothesis_status':'Hypothesis motivated by validation behavior; bins and held-out RR analysis fixed before joint freeze/test.',
          'heldout_analysis':'paired PV-minus-union query MRR within fixed bins; three seed means and sample SD; all cold queries retained; no p-values or causal claim',
          'limits':{'max_neighbors':MAX_NEIGHBORS,'working_set_bytes':WORKING_BYTES,'compute_wall_seconds':MAX_WALL},
          'no_model_or_score_access_in_geometry_phases':True}
    save(path,plan);return plan


def open_graph(args,manifest):
    exact,kernels=load_runtime(args.repo);graph=exact.load_graph(args.prepared_root/'train_csr','little')
    require(graph.n==manifest['num_nodes'] and graph.m==manifest['train_stored_entries'],'train dimensions changed')
    require(isinstance(graph.starts,np.memmap) and isinstance(graph.indices,np.memmap)
            and not graph.starts.flags.writeable and not graph.indices.flags.writeable,'read-only existing CSR mmap required')
    return graph,kernels


def validation_geometry(args):
    budget=Budget();plan,manifest,inputs=common_inputs(args);declared=predeclare(args,inputs)
    ids,pos,warm,subset_hashes=load_subset(args,plan,manifest,'val');graph,kernels=open_graph(args,manifest)
    anatomy=verify_kernel_anatomy(kernels);rows=[];deadline=False
    for index,(qid,pair) in enumerate(zip(ids,pos)):
        row={'query_id':int(qid),'endpoint_u':int(pair[0]),'endpoint_v':int(pair[1]),'provided_warm_query':bool(warm[index])}
        if deadline: row['status']='not_processed_deadline';rows.append(row);continue
        try:
            u,v=map(int,pair);require(u!=v,'positive endpoints must differ')
            du,dv=graph.bounds(u),graph.bounds(v);du,dv=du[1]-du[0],dv[1]-dv[0]
            bid,ratio=imbalance_bin(du,dv);row.update(degree_u=du,degree_v=dv,bin_id=bid,imbalance=ratio)
            budget.check(du+dv)
            graph.validate_row(u);graph.validate_row(v)
            require(not graph.has_edge(u,v) and not graph.has_edge(v,u),'validation target must be absent in train')
            nu,nv=graph.neighbors(u),graph.neighbors(v);overlap=count_overlap(nu,nv);population=du+dv-overlap
            low=0 if du<=dv else 1;expected,pzero=uniform_union(population,min(du,dv))
            cover=[];bounds=[];unique=[]
            for base_seed in SEEDS:
                budget.check()
                seed=int(np.random.SeedSequence([base_seed,int(qid)]).generate_state(1)[0])
                a,b,c=pv_trial(graph,u,v,kernels,seed);cover.append(a);bounds.append(b);unique.append(c)
            cover=np.asarray(cover);bounds=np.asarray(bounds)
            row.update(status='complete',overlap=overlap,union_population=population,low_endpoint='u' if low==0 else 'v',
                       union_expected_low_neighbors=expected,union_probability_zero_low_neighbors=pzero,
                       pv_mean_low_neighbors=float(cover[:,low].mean()),pv_probability_zero_low_neighbors=float((cover[:,low]==0).mean()),
                       pv_mean_endpoint_neighbors=cover.mean(0).tolist(),pv_unique_own_draws_min=np.min(unique,axis=0).tolist(),
                       pv_conditional_bound_min=bounds.min(0).tolist(),pv_unconditional_bound=[unconditional_bound(du),unconditional_bound(dv)],
                       bound_possible_violations=int(np.sum(cover<bounds)),
                       nonisolated_zero_coverage=int(np.sum((cover==0)&(np.array([du,dv])>0))))
            budget.check()
        except Guarded as error: row.update(status='guarded',reason=str(error))
        except Deadline as error: row.update(status='not_processed_deadline',reason=str(error));deadline=True
        rows.append(row)
    complete=[r for r in rows if r['status']=='complete'];bins=[]
    for bid,label in BIN_LABELS.items():
        group=[r for r in complete if r['bin_id']==bid]
        provided=[r for r in rows if r.get('bin_id')==bid]
        bins.append({'bin_id':bid,'label':label,'queries':len(group),'provided_queries':len(provided),
                     'guarded_queries':sum(r['status']=='guarded' for r in provided),**{key:float(np.mean([r[key] for r in group])) if group else None for key in
             ('union_expected_low_neighbors','union_probability_zero_low_neighbors','pv_mean_low_neighbors','pv_probability_zero_low_neighbors')}})
    result={'schema_version':1,'status':'complete' if len(complete)==len(ids) else 'partial','scope':'Validation positive-query first-hop geometry only; no MRR, training, GPU or timing claim',
            'analysis_plan_sha256':sha(args.output_root/'plan.json'),'quality_plan_sha256':inputs['quality_plan_sha256'],
            'subset_sha256':subset_hashes,'kernel_anatomy':anatomy,'records':rows,'bins':bins,'mc_trials_per_query':len(SEEDS),
            'queries_provided':len(ids),'queries_completed':len(complete),
            'endpoint_trials':len(complete)*len(SEEDS)*2,
            'nonisolated_endpoint_trials':sum(int(r['degree_u']>0)+int(r['degree_v']>0) for r in complete)*len(SEEDS),
            'queries_guarded':sum(r['status']=='guarded' for r in rows),
            'queries_deadline':sum(r['status']=='not_processed_deadline' for r in rows),
            'bound_possible_violations':sum(r.get('bound_possible_violations',0) for r in rows),
            'nonisolated_zero_coverage':sum(r.get('nonisolated_zero_coverage',0) for r in rows),
            'interpretation':'Uniform-union expectation is analytic; PV uses16 seeded actual-kernel Monte Carlo trials. First-hop coverage does not establish downstream quality causality.',
            'completed_unix':time.time()}
    require(sha(Path(__file__))==declared['analysis_code_sha256'],'analysis code changed during geometry')
    save(args.output_root/'validation_geometry.json',result);return result


def frozen_gate(quality,quality_plan_sha):
    path=quality/'FROZEN_CHECKPOINTS.json';frozen=read(path);plan=read(quality/'plan.json')
    require(sha(quality/'plan.json')==quality_plan_sha==frozen['plan_sha256'] and frozen['selection_uses_test'] is False,'joint freeze plan/selection binding changed')
    jobs=frozen['jobs'];planned={j['name']:j for j in plan['jobs']}
    require(len(jobs)==6 and len({j['name'] for j in jobs})==6 and {(j['variant'],j['seed']) for j in jobs}==EXPECTED_JOBS,'six jointly frozen jobs required')
    initials={}
    for job in jobs:
        require(job['name']==f"{job['variant']}_seed{job['seed']}" and job['name'] in planned
                and all(job[k]==planned[job['name']][k] for k in ('name','variant','seed','config_sha256')),'frozen job differs from plan')
        require(job['selection']['selection_split']=='validation'
                and job['selection']['metric']==plan['protocol']['implementation_metric_alias'] and job['training']['test_scored'] is False
                and job['training']['steps']==plan['protocol']['max_steps'],'incomplete or nonvalidation checkpoint freeze')
        require(all(isinstance(job[k],str) and len(job[k])==64 and set(job[k])<=set('0123456789abcdef')
                    for k in ('checkpoint_sha256','initial_model_sha256','config_sha256')),'frozen checkpoint identity missing')
        require(initials.get(job['seed'],job['initial_model_sha256'])==job['initial_model_sha256'],'paired initial states differ')
        initials[job['seed']]=job['initial_model_sha256']
    return frozen,sha(path)


def wait_freeze_test_geometry(args):
    require(math.isfinite(args.wait_freeze_hours) and 0<args.wait_freeze_hours<=8,'freeze wait must be finite, positive and at most8hours')
    declared=read(args.output_root/'plan.json')
    require(declared['analysis_code_sha256']==sha(Path(__file__)),'predeclared analysis code changed')
    deadline=time.monotonic()+args.wait_freeze_hours*3600
    while not (args.quality_root/'FROZEN_CHECKPOINTS.json').exists():
        require(time.monotonic()<deadline,'joint-freeze wait deadline reached')
        time.sleep(min(30,max(0,deadline-time.monotonic())))
    frozen,frozen_sha=frozen_gate(args.quality_root,declared['inputs']['quality_plan_sha256'])
    require(declared['created_unix']<=frozen['created_unix'],'analysis plan was not predeclared before freeze')
    plan,manifest,inputs=common_inputs(args);require(inputs==declared['inputs'],'geometry inputs changed after predeclaration')
    ids,pos,warm,hashes=load_subset(args,plan,manifest,'test');graph,_=open_graph(args,manifest);budget=Budget()
    degrees=np.empty((len(ids),2),dtype=np.int64);bins=np.empty(len(ids),dtype=np.int8);ratios=np.empty(len(ids),dtype=np.float64)
    for i,pair in enumerate(pos):
        budget.check()
        require(pair[0]!=pair[1],'test endpoints must differ')
        for side,node in enumerate(pair):
            lo,hi=graph.bounds(int(node));degrees[i,side]=hi-lo
        bins[i],ratio=imbalance_bin(*degrees[i]);ratios[i]=np.nan if ratio is None else ratio
    path=args.output_root/'test_geometry.npz'
    temporary=path.with_name('test_geometry.tmp.npz')
    np.savez_compressed(temporary,query_ids=np.asarray(ids),positive=np.asarray(pos),degree_u=degrees[:,0],degree_v=degrees[:,1],
                        bin_id=bins,imbalance=ratios,provided_warm_query=np.asarray(warm,dtype=bool))
    temporary.replace(path)
    result={'schema_version':1,'status':'complete','scope':'Frozen test positive query IDs/endpoints/train degrees/bins only; no scores, reciprocal ranks or model bytes read',
            'analysis_plan_sha256':sha(args.output_root/'plan.json'),'quality_plan_sha256':inputs['quality_plan_sha256'],
            'frozen_manifest_sha256':frozen_sha,'geometry_sha256':sha(path),'query_count':len(ids),'subset_sha256':hashes,
            'bin_counts':{BIN_LABELS[b]:int(np.sum(bins==b)) for b in BIN_LABELS},'cold_queries_retained':True,
            'provided_warm_matches_train_nonisolated':bool(np.array_equal(np.asarray(warm,dtype=bool),np.min(degrees,axis=1)>0)),
            'created_unix':time.time()}
    require(sha(args.quality_root/'FROZEN_CHECKPOINTS.json')==frozen_sha,'joint freeze changed during geometry')
    require(sha(Path(__file__))==declared['analysis_code_sha256'],'analysis code changed during geometry')
    save(args.output_root/'test_geometry.json',result);return result


def align_rr(geometry_ids,run_ids,rr):
    geometry_ids=np.asarray(geometry_ids);run_ids=np.asarray(run_ids);rr=np.asarray(rr)
    require(geometry_ids.ndim==run_ids.ndim==rr.ndim==1 and len(geometry_ids)==len(run_ids)==len(rr),'RR/ID shape mismatch')
    require(len(np.unique(geometry_ids))==len(geometry_ids) and len(np.unique(run_ids))==len(run_ids),'duplicate original query IDs')
    require(geometry_ids.dtype.kind in 'iu' and run_ids.dtype.kind in 'iu' and rr.dtype.kind=='f','integer query IDs and floating RR required')
    order=np.argsort(run_ids);sorted_ids=run_ids[order]
    require(np.array_equal(np.sort(geometry_ids),sorted_ids),'RR original query ID set differs')
    require(np.all(np.isfinite(rr)) and np.all((rr>0)&(rr<=1)),'invalid reciprocal ranks')
    return rr[order[np.searchsorted(sorted_ids,geometry_ids)]]


def seed_stats(values):
    return {'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1)),'seed_count':len(values)} if len(values)==3 else {'mean':None,'sample_sd':None,'seed_count':len(values)}


def summarize_arrays(bins,rr_by_job):
    require(set(rr_by_job)==EXPECTED_JOBS,'all six paired RR vectors required')
    rows=[]
    for bid,label in [(-2,'all_queries'),*BIN_LABELS.items()]:
        mask=np.ones(len(bins),dtype=bool) if bid==-2 else bins==bid;n=int(mask.sum());seeds=[]
        for seed in (42,43,44):
            a,b=rr_by_job[('union',seed)][mask],rr_by_job[('pv',seed)][mask]
            seeds.append({'seed':seed,'union_mrr':float(a.mean()) if n else None,'pv_mrr':float(b.mean()) if n else None,
                          'pv_minus_union_mrr':float((b-a).mean()) if n else None})
        rows.append({'bin_id':bid,'label':label,'query_count':n,'seeds':seeds,
                     **{key:seed_stats([r[key] for r in seeds]) if n else {'mean':None,'sample_sd':None,'seed_count':0}
                        for key in ('union_mrr','pv_mrr','pv_minus_union_mrr')}})
    return rows


def summarize_exports(args):
    root=args.export_root.resolve();quality=root/'results/quality_friendster';geometry=root/'results/degree_balance_v1'
    declared,meta=read(geometry/'plan.json'),read(geometry/'test_geometry.json')
    require(declared['analysis_code_sha256']==sha(Path(__file__)),'summarizer code differs from predeclaration')
    require(sha(geometry/'plan.json')==meta['analysis_plan_sha256'] and declared['inputs']['quality_plan_sha256']==meta['quality_plan_sha256'],'geometry predeclaration identity differs')
    require(sha(geometry/'test_geometry.npz')==meta['geometry_sha256'],'test geometry checksum changed')
    frozen,frozen_sha=frozen_gate(quality,meta['quality_plan_sha256']);require(frozen_sha==meta['frozen_manifest_sha256'],'geometry freeze differs from exported freeze')
    final_path=root/'results/FRIENDSTER_QUALITY_COMPLETE.json'
    require(final_path.exists(),'final Friendster quality export required')
    final=read(final_path);require(final['success'] is True and final['frozen_checkpoints_sha256']==frozen_sha,'Friendster final quality/freeze identity differs')
    finals={r['name']:r for r in final['results']};require(len(finals)==6,'six final RR artifacts required')
    require(declared['created_unix']<=frozen['created_unix'],'test geometry was not predeclared')
    with np.load(geometry/'test_geometry.npz',allow_pickle=False) as data:
        ids=np.array(data['query_ids']);bins=np.array(data['bin_id']);positive=np.array(data['positive'])
    require(len(ids)==COUNTS['test'] and bins.shape==ids.shape and positive.shape==(len(ids),2),'test geometry shapes differ')
    require(set(np.unique(bins))<=set(BIN_LABELS),'geometry has undeclared bins')
    plan=read(quality/'plan.json');expected_ids_sha=plan['subsets']['test']['outputs']['query_ids']['sha256']
    rr_by_job={};sources={}
    for job in frozen['jobs']:
        name=job['name'];run=quality/'runs'/name/'run1';ids_path,rr_path=run/'query_ids.npy',run/'query_rr.npy'
        require(sha(ids_path)==expected_ids_sha,'run original query ID file differs from frozen subset')
        artifacts=finals[name]['artifacts']
        require(sha(ids_path)==artifacts['original_query_ids']['sha256']
                and sha(rr_path)==artifacts['query_reciprocal_ranks']['sha256'],'RR artifacts differ from final export manifest')
        result=read(quality/('test_results_'+name+'.json'));require(result['frozen_manifest_sha256']==frozen_sha,'RR result freeze differs')
        row=next(r for r in result['results'] if r['name']==name)
        require(row['checkpoint_sha256']==job['checkpoint_sha256'] and row['original_query_ids_sha256']==expected_ids_sha,'RR checkpoint/query identity differs')
        aligned=align_rr(ids,np.load(ids_path,allow_pickle=False),np.load(rr_path,allow_pickle=False))
        require(np.isclose(aligned.mean(),row['test_metrics']['custom_sampled_mrr'],rtol=0,atol=1e-12),'RR mean differs from frozen test result')
        rr_by_job[(job['variant'],job['seed'])]=aligned;sources[name]={'query_ids_sha256':sha(ids_path),'query_rr_sha256':sha(rr_path)}
    result={'schema_version':1,'status':'complete','frozen_manifest_sha256':frozen_sha,'analysis_plan_sha256':meta['analysis_plan_sha256'],
            'rows':summarize_arrays(bins,rr_by_job),'sources':sources,'final_quality_summary_sha256':sha(final_path),'cold_queries_retained':True,
            'inference':'Descriptive three paired-seed means and sample SD on one fixed query cohort; no p-values, independence, equivalence or causal-quality claim.',
            'hypothesis_status':declared['hypothesis_status'],'created_unix':time.time()}
    save(args.output_root/'rr_by_degree_balance.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('validation_geometry','wait_freeze_test_geometry','summarize_exports'))
    parser.add_argument('--repo',type=Path,default=Path('/home/jovyan/seal_revision_20261124/code'))
    parser.add_argument('--prepared-root',type=Path,default=Path('/home/jovyan/seal_revision_20261124/data/friendster/prepared'))
    parser.add_argument('--quality-root',type=Path,default=Path('/home/jovyan/seal_revision_20261124/results/quality_friendster'))
    parser.add_argument('--output-root',type=Path,default=Path('/home/jovyan/seal_revision_20261124/results/degree_balance_v1'))
    parser.add_argument('--export-root',type=Path)
    parser.add_argument('--wait-freeze-hours',type=float,default=8)
    args=parser.parse_args(argv)
    require(math.isfinite(args.wait_freeze_hours) and 0<args.wait_freeze_hours<=8,'freeze wait must be finite, positive and at most8hours')
    require('torch' not in sys.modules,'Torch must not be imported')
    if args.phase=='validation_geometry': result=validation_geometry(args)
    elif args.phase=='wait_freeze_test_geometry': result=wait_freeze_test_geometry(args)
    else:
        require(args.export_root is not None,'--export-root required');result=summarize_exports(args)
    require('torch' not in sys.modules,'Torch must not be imported')
    print(json.dumps({'phase':args.phase,'status':result['status'],'output_root':str(args.output_root)}),flush=True)
    return 0

if __name__=='__main__': main()
