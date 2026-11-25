from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import math
import struct
import sys
import time
import numpy as np
import pytest

SCRIPT=Path(__file__).with_name('degree_balance_audit_v1.py')
SPEC=importlib.util.spec_from_file_location('degree_balance_tested',SCRIPT)
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)
REPO=Path(__file__).resolve().parents[2]
exact,kernels=audit.load_runtime(REPO)


def graph(du,dv,overlap):
    common=list(range(2,2+overlap));a=list(range(2+overlap,2+du));b=list(range(2+du,2+du+dv-overlap))
    edges=[(0,v) for v in common+a]+[(1,v) for v in common+b]
    n=max([1]+[v for pair in edges for v in pair])+2
    rows=[[] for _ in range(n)]
    for u,v in edges:rows[u].append(v);rows[v].append(u)
    starts=np.array([0]+list(np.cumsum([len(r) for r in rows[:-1]])),dtype=np.int64)
    return exact.CSRGraph(starts,np.array([v for row in rows for v in sorted(row)],dtype=np.int32))


def test_hypergeometric_product_matches_exact_combinations_and_cold():
    for m in range(13):
        for k in range(m+1):
            expected,pzero=audit.uniform_union(m,k,3);draws=min(m,3)
            truth=math.comb(m-k,draws)/math.comb(m,draws) if draws<=m-k else 0
            assert expected==pytest.approx(draws*k/m if m else 0)
            assert pzero==pytest.approx(truth)
    assert audit.uniform_union(1000,1)[1]>.79
    assert audit.uniform_union(0,0)==(0,1)


@pytest.mark.parametrize('du,dv,bin_id',[(0,0,-1),(0,10,-1),(1,1,0),(2,3,0),(1,2,1),(1,4,2),(1,16,3),(1,64,4),(2,128,4)])
def test_predeclared_bins_and_cold(du,dv,bin_id):
    assert audit.imbalance_bin(du,dv)[0]==bin_id
    assert audit.imbalance_bin(dv,du)[0]==bin_id


def test_actual_kernel_is_with_replacement_for_large_rows():
    anatomy=audit.verify_kernel_anatomy(kernels)
    assert anatomy['duplicate_observed_d4_k3_seed'] in range(16)
    assert anatomy['copy_all_probe_passed']
    assert 'torch' not in sys.modules
    assert audit.coverage_bound(151,1)==1 < min(151,50)
    assert audit.unconditional_bound(151)==1
    assert audit.unconditional_bound(150)==50


@pytest.mark.parametrize('du,dv,overlap',[(0,400,0),(1,1000,0),(60,500,0),(150,1000,50),(151,151,151),(200,500,100)])
def test_actual_pv_first_hop_bound_with_disjoint_shared_and_unequal_rows(du,dv,overlap):
    g=graph(du,dv,overlap)
    assert len(g.neighbors(0))==du and len(g.neighbors(1))==dv
    assert audit.count_overlap(g.neighbors(0),g.neighbors(1))==overlap
    for seed in range(16):
        cover,bounds,unique=audit.pv_trial(g,0,1,kernels,seed)
        assert all(c>=b for c,b in zip(cover,bounds))
        assert (cover[0]>0)==(du>0) and (cover[1]>0)==(dv>0)
        assert bounds==[min(unique[0],50),min(unique[1],50)]


def test_all_resource_guards_are_predeclared_and_finite(monkeypatch):
    b=audit.Budget()
    with pytest.raises(audit.Guarded,match='20M'):b.check(20_000_001)
    monkeypatch.setattr(audit.resource,'getrusage',lambda *_:SimpleNamespace(ru_maxrss=5*2**30 if sys.platform=='darwin' else 5*2**20))
    with pytest.raises(audit.Guarded,match='4GiB'):b.check()
    monkeypatch.setattr(audit.time,'monotonic',lambda:b.started+1801)
    with pytest.raises(audit.Deadline):b.check()


def test_row_join_uses_original_ids_not_repeated_source_or_position():
    ids=np.array([4,8,11]);run_ids=np.array([11,4,8]);rr=np.array([.1,.2,.3])
    assert np.allclose(audit.align_rr(ids,run_ids,rr),[.2,.3,.1])
    with pytest.raises(ValueError,match='duplicate'):audit.align_rr(ids,np.array([4,4,11]),rr)
    with pytest.raises(ValueError,match='set differs'):audit.align_rr(ids,np.array([4,8,12]),rr)
    with pytest.raises(ValueError,match='invalid reciprocal'):audit.align_rr(ids,ids,np.array([.2,np.nan,.3]))


def test_three_seed_statistics_keep_cold_queries_and_do_not_pool_queries_as_seeds():
    bins=np.array([-1,0,4]);rr={}
    for seed,delta in zip((42,43,44),(.01,.02,.03)):
        rr[('union',seed)]=np.array([.1,.2,.3]);rr[('pv',seed)]=rr[('union',seed)]+delta
    rows=audit.summarize_arrays(bins,rr)
    all_row,cold=rows[:2]
    assert all_row['query_count']==3 and cold['query_count']==1
    assert all_row['pv_minus_union_mrr']['mean']==pytest.approx(.02)
    assert cold['pv_minus_union_mrr']['sample_sd']==pytest.approx(.01)
    assert cold['pv_minus_union_mrr']['seed_count']==3
    assert next(r for r in rows if r['bin_id']==2)['pv_minus_union_mrr']['mean'] is None


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'COUNTS',{'val':3,'test':4})
    base=tmp_path/'base';prepared=base/'data/friendster/prepared';quality=base/'results/quality_friendster';output=base/'results/degree_balance_v1'
    train=prepared/'train_csr';train.mkdir(parents=True)
    rows=[[2],[3,4,5],[0],[1],[1],[1],[],[]];starts=np.array([0]+list(np.cumsum([len(x) for x in rows[:-1]])),dtype='<i8')
    ends=np.array([v for row in rows for v in row],dtype='<i4');starts.tofile(train/'edge_starts.bin')
    (train/'edge_ends.bin').write_bytes(struct.pack('<q',len(ends))+ends.tobytes())
    manifest={'dataset':'friendster','variant':'exact_colleague_80_10_10','num_nodes':8,'train_stored_entries':8,'files':{}}
    subsets={}
    for split,prefix,ids,pairs in [('val','valid',np.array([0,2,4]),np.array([[0,1],[0,6],[7,1],[2,3],[7,1]])),
                                  ('test','test',np.arange(4),np.array([[0,1],[0,6],[7,1],[2,3]]))]:
        originals=prepared/'evaluation'/prefix;originals.mkdir(parents=True)
        warm=np.array([all(len(rows[int(v)])>0 for v in p) for p in pairs])
        paths={};inputs={}
        for name,data in [('source_node',pairs[:,0]),('target_node',pairs[:,1]),('warm_query',warm)]:
            path=originals/(name+'.npy');np.save(path,data)
            inputs[name]={'path':str(path),'sha256':audit.sha(path)}
            manifest['files']['evaluation/'+prefix+'/'+name+'.npy']={'sha256':audit.sha(path)}
        for name,suffix,data in [('query_ids','query_ids',ids),('positive','edge',pairs[ids]),('warm_query','warm_query',warm[ids])]:
            path=quality/'subsets'/(prefix+'_'+suffix+'.npy');path.parent.mkdir(parents=True,exist_ok=True);np.save(path,data)
            paths[name]={'path':str(path),'sha256':audit.sha(path)}
        subsets[split]={'query_count':len(ids),'outputs':paths,'inputs':inputs}
    write(prepared/'manifest.json',manifest);write(prepared/'PREPARED_COMPLETE.json',{'manifest_sha256':audit.sha(prepared/'manifest.json')})
    jobs=[{'name':f'{v}_seed{s}','variant':v,'seed':s,'config_sha256':'a'*64} for v,s in sorted(audit.EXPECTED_JOBS)]
    plan={'protocol':{'dataset':'friendster_colleague_80_10_10','max_nodes_per_hop':200,'per_vertex_oversample':1.5,'num_hops':2,
                      'validation_query_count':3,'test_query_count':4,'max_steps':20000,'implementation_metric_alias':'ogb_mrr'},'graph_root':str(train),'dataset_provenance':{'sha256':audit.sha(prepared/'manifest.json')},
          'graph_input_stats':{str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in train.iterdir()},
          'code':{str(REPO/name):audit.sha(REPO/name) for name in audit.CORE_SOURCES},'subsets':subsets,'jobs':jobs}
    write(quality/'plan.json',plan)
    return SimpleNamespace(base=base,repo=REPO,prepared_root=prepared,quality_root=quality,output_root=output,wait_freeze_hours=.000001,export_root=base)


def freeze_fixture(args):
    plan=audit.read(args.quality_root/'plan.json')
    frozen={'created_unix':time.time(),'plan_sha256':audit.sha(args.quality_root/'plan.json'),'selection_uses_test':False,
            'jobs':[{**j,'checkpoint_sha256':'b'*64,'initial_model_sha256':str(j['seed'])*32,
                     'selection':{'selection_split':'validation','metric':'ogb_mrr'},'training':{'test_scored':False,'steps':20000}} for j in plan['jobs']]}
    # Keep well-formed hashes while paired seeds have equal initial state.
    for j in frozen['jobs']:j['initial_model_sha256']='c'*64
    write(args.quality_root/'FROZEN_CHECKPOINTS.json',frozen)
    return frozen


def test_full_geometry_and_final_rr_join_never_require_model_or_score_files(prepared):
    a=prepared
    val=audit.validation_geometry(a)
    assert val['status']=='complete' and val['queries_completed']==3
    assert val['bound_possible_violations']==val['nonisolated_zero_coverage']==0
    frozen=freeze_fixture(a);test=audit.wait_freeze_test_geometry(a)
    assert test['query_count']==4 and test['bin_counts']['min_degree_zero']==2
    fsha=audit.sha(a.quality_root/'FROZEN_CHECKPOINTS.json');plan=audit.read(a.quality_root/'plan.json')
    ids=np.load(a.quality_root/'subsets/test_query_ids.npy');finals=[]
    for job in frozen['jobs']:
        run=a.quality_root/'runs'/job['name']/'run1';run.mkdir(parents=True)
        rr=np.array([.2,.4,.1,.3]) + ((job['seed']-41)*.01 if job['variant']=='pv' else 0)
        np.save(run/'query_ids.npy',ids);np.save(run/'query_rr.npy',rr)
        row={'name':job['name'],'checkpoint_sha256':job['checkpoint_sha256'],
             'original_query_ids_sha256':audit.sha(run/'query_ids.npy'),'test_metrics':{'custom_sampled_mrr':float(rr.mean())}}
        write(a.quality_root/('test_results_'+job['name']+'.json'),{'frozen_manifest_sha256':fsha,'results':[row]})
        finals.append({**row,'artifacts':{'original_query_ids':{'sha256':audit.sha(run/'query_ids.npy')},
                                         'query_reciprocal_ranks':{'sha256':audit.sha(run/'query_rr.npy')}}})
    write(a.base/'results/FRIENDSTER_QUALITY_COMPLETE.json',{'success':True,'frozen_checkpoints_sha256':fsha,'results':finals})
    result=audit.summarize_exports(a)
    assert result['rows'][0]['pv_minus_union_mrr']['mean']==pytest.approx(.02)
    assert result['rows'][1]['query_count']==2
    assert not list(a.base.rglob('*.pth')) and not list(a.base.rglob('*scores*'))
    rrpath=a.quality_root/'runs/union_seed42/run1/query_rr.npy';rr=np.load(rrpath);rr[:2]=rr[1::-1];np.save(rrpath,rr)
    with pytest.raises(ValueError,match='artifacts differ'):audit.summarize_exports(a)


def test_no_test_geometry_before_joint_freeze_and_initial_predeclare_after_freeze_rejected(prepared,monkeypatch):
    a=prepared
    audit.validation_geometry(a)
    monkeypatch.setattr(audit,'load_subset',lambda *_:pytest.fail('test data accessed before freeze'))
    with pytest.raises(ValueError,match='wait deadline'):audit.wait_freeze_test_geometry(a)
    freeze_fixture(a)
    other=SimpleNamespace(**vars(a));other.output_root=a.output_root/'new_declaration'
    with pytest.raises(ValueError,match='precede joint freeze'):audit.predeclare(other,{'test':'identity'})


@pytest.mark.parametrize("hours", ["inf", "nan", "-inf", "0", "8.01"])
def test_freeze_wait_is_strictly_finite_and_at_most_eight_hours(hours):
    with pytest.raises(ValueError,match="freeze wait must be finite"):
        audit.main(["--phase", "wait_freeze_test_geometry", "--wait-freeze-hours="+hours])
