from pathlib import Path
import hashlib,importlib.util,io,json,math,os,sys,types
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import admission as a
import run_public as run

def npy_bytes(value):
    b=io.BytesIO();np.save(b,value,allow_pickle=False);return b.getvalue()

def test_exact_public_proof_and_unchanged_original_small_freeze_admission(tmp_path):
    before={str(p):a.sha(p)for p in(HERE/'quality_reference').rglob('*')if p.is_file()}
    gate=a.Admission(tmp_path/'not_loaded',HERE/'quality_reference',HERE/'QUALITY_MANIFEST.json',HERE/'scientific_repo')
    assert len(gate.expected)==23 and gate.freeze['selection_uses_test']is False
    assert before=={name:a.sha(name)for name in before}
    assert not(tmp_path/'not_loaded').exists()

def test_whole_numeric_hash_rejects_unselected_corruption_even_with_unchanged_size_and_mtime(tmp_path):
    p=tmp_path/'graph.bin';p.write_bytes(b'selected-prefix-unselected-original')
    row={'path':'graph.bin','bytes':p.stat().st_size,'sha256':a.sha(p)};old=p.stat()
    assert len(a.verify_payloads(tmp_path,[row]))==1
    p.write_bytes(b'selected-prefix-unselected-modified');os.utime(p,ns=(old.st_atime_ns,old.st_mtime_ns))
    assert p.stat().st_size==row['bytes']and p.stat().st_mtime_ns==old.st_mtime_ns
    with pytest.raises(ValueError,match='entire numeric'):a.verify_payloads(tmp_path,[row])

@pytest.mark.parametrize('shape,dtype',[( (3,2),np.int64),((7,),np.bool_)])
def test_exact_NPY_stream_matches_numpy_save_without_creating_a_candidate_file(shape,dtype):
    x=np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    assert a.npy_hash([x[:1],x[1:]],shape,dtype)==hashlib.sha256(npy_bytes(x)).hexdigest()

def test_negative_stream_preserves_repeated_source_query_rows_and_every_candidate():
    source=np.array([5,5,7,9],np.int32);negative=np.array([[1,2,3],[8,2,0],[6,0,2],[1,7,8]],np.int32);ids=np.array([0,1,3])
    blocks=list(a.selected_negative_blocks(source,negative,ids,chunk=2));expected=np.empty((3,3,2),np.int64);expected[:,:,0]=source[ids,None];expected[:,:,1]=negative[ids]
    assert [b.shape[0]for b in blocks]==[6,3]
    assert a.npy_hash(blocks,(9,2),'<i8')==hashlib.sha256(npy_bytes(expected.reshape(-1,2))).hexdigest()
    assert np.array_equal(np.concatenate(blocks),expected.reshape(-1,2))

def test_preflight_reads_no_actual_graph_and_writes_no_output(tmp_path,capsys):
    out=tmp_path/'new';assert run.main(['--experiment','e7','--prepared-root',str(tmp_path/'absent'),'--out',str(out)])==0
    result=json.loads(capsys.readouterr().out);assert result['numeric_files_to_stream_hash']==23 and not result['out_created']and not out.exists()

def test_output_inside_immutable_inputs_is_rejected_before_any_creation(tmp_path):
    prepared=tmp_path/'prepared'
    with pytest.raises(ValueError,match='outside all immutable'):run.main(['--experiment','e12','--prepared-root',str(prepared),'--out',str(prepared/'output')])
    assert not prepared.exists()

def test_real_unchanged_E12_kernel_matches_an_independent_small_structural_oracle(tmp_path,monkeypatch):
    monkeypatch.setenv('NUMBA_CACHE_DIR',str(tmp_path/'cache'));module=run.load_original('e12')
    adjacency=[{1,2},{0,2,3},{0,1,3},{1,2,4},{3},set()]
    starts=np.array([0,*np.cumsum([len(x)for x in adjacency])[:-1]],np.int64);col=np.array([v for row in adjacency for v in sorted(row)],np.int32)
    source=np.array([0,0,5],np.int64);target=np.array([3,4,0],np.int64);negative=np.array([[1,4,5],[1,3,5],[1,2,4]],np.int64)
    def score(u,v):
        common=adjacency[u]&adjacency[v]
        return np.array([len(adjacency[v]),len(adjacency[u])*len(adjacency[v]),len(common),sum(1/math.log(len(adjacency[w]))for w in common),sum(1/len(adjacency[w])for w in common)],float)
    expected=[]
    for q in range(3):
        positive=score(int(source[q]),int(target[q]));scores=np.stack([score(int(source[q]),int(v))for v in negative[q]])
        expected.append(1+(scores>positive).sum(axis=0)+.5*(scores==positive).sum(axis=0))
    actual,operations,bad=module.score_query_rows(starts,col,source,target,negative,np.arange(3,dtype=np.int64))
    assert np.array_equal(actual,np.array(expected))and not bad.any()and np.all(operations>=0)
    assert a.sha(HERE/'original/heuristic_baselines.py')==a.ORIGINAL['e12'][1]

def test_real_unchanged_E7_shared_priority_cap_and_guarded_summary(tmp_path):
    module=run.load_original('e7');nodes=np.arange(300,dtype=np.int64)
    selected=module.choose(nodes,200,20261125,7,1)
    assert len(selected)==200 and len(np.unique(selected))==200 and np.array_equal(selected,module.choose(nodes[::-1],200,20261125,7,1))
    row={'cases':{name:{'status':'guarded'}for name in module.CASES}}
    aggregate=module.aggregate([row]);assert all(r['provided_query_rows']==1 and r['guarded_queries']==1 and r['structural_metric_queries']==0 for r in aggregate.values())
    assert a.sha(HERE/'original/semantic_sampler_audit_v1.py')==a.ORIGINAL['e7'][1]

def test_E12_operational_gate_replacements_restore_original_functions_on_failure(tmp_path):
    sentinel=lambda *x,**k:None
    module=types.SimpleNamespace(verify_test_gate=sentinel,load_inputs=sentinel)
    module.run=lambda *args,**kwargs:(_ for _ in()).throw(RuntimeError('intentional scientific failure'))
    admission=types.SimpleNamespace(prepared=tmp_path,quality=tmp_path,cohorts={'test':{'query_ids_path':tmp_path/'ids.npy'}},stable=lambda:({},{}))
    with pytest.raises(RuntimeError,match='scientific failure'):run.execute_e12(admission,module,tmp_path/'fresh')
    assert module.verify_test_gate is sentinel and module.load_inputs is sentinel
