"""Small tie, permutation, repeated-source, corruption and memory-reader tests."""
import hashlib,io,json
import numpy as np
import pytest
import verify_friendster_scores as v

def encoded(array):
    stream=io.BytesIO();np.save(stream,array,allow_pickle=False);return stream.getvalue()

def test_midpoint_does_not_assume_positive_first_or_tie_order():
    labels=np.array([[0,1,0,0],[0,0,0,1],[1,0,0,0]],dtype=np.int8)
    scores=np.array([[.7,.5,.5,.1],[.1,.7,.5,.5],[.5,.5,.5,.5]],dtype=np.float32)
    ranks,positions,ties=v.midpoint_ranks(labels,scores)
    assert ranks.tolist()==[2.5,2.5,2.5]and positions.tolist()==[1,3,0]and ties.tolist()==[1,1,3]
@pytest.mark.parametrize('change',['nan','none_positive','two_positive','invalid_label','shape'])
def test_bad_score_group_rejected(change):
    y=np.array([[0,1,0]],dtype=np.int8);s=np.array([[.7,.5,.1]],dtype=np.float32)
    if change=='nan':s[0,0]=np.nan
    elif change=='none_positive':y[:]=0
    elif change=='two_positive':y[0,0]=1
    elif change=='invalid_label':y[0,0]=2
    else:s=s[:,:2]
    with pytest.raises(v.InvalidScores):v.midpoint_ranks(y,s)

def test_memory_file_bounds_prevent_whole_vector_copies():
    f=v.MemoryFile(bytearray(b'abcdef'),6);assert f.read(2)==b'ab';f.seek(-2,2);assert f.read()==b'ef'
    with pytest.raises(v.InvalidScores):f.seek(-1)
    big=v.MemoryFile(bytearray(5*v.MIB),5*v.MIB)
    with pytest.raises(v.InvalidScores):big.read()

def test_canonical_npy_hash_matches_actual_numpy_file():
    array=np.arange(18,dtype='<i8').reshape(9,2)
    assert v.canonical_npy_sha(array)==hashlib.sha256(encoded(array)).hexdigest()

@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(v,'Q',2);monkeypatch.setattr(v,'K',3)
    qids=np.array([1717,1740],dtype=np.int64);positive=np.array([[7,8],[7,9]],dtype=np.int64)
    canonical=np.array([[8,10,11,12],[9,13,14,15]],dtype=np.int64)
    orders=np.stack([v.independently_declared_order(q,3,20261127)for q in qids]);scores=np.array([[.5,.7,.5,.1],[1,0,.5,.5]],dtype=np.float32)
    negatives=np.empty((2,3,2),dtype=np.int64);negatives[:,:,0]=7;negatives[:,:,1]=canonical[:,1:]
    subset={'candidate_order_seed':20261127,'outputs':{'negative':{'sha256':hashlib.sha256(encoded(negatives.reshape(-1,2))).hexdigest()}}}
    metrics={'ogb_num_sources':2,'ogb_mrr':.7,'ogb_hits@1':.5,'ogb_hits@10':1.,'ogb_hits@50':1.,'ogb_hits@100':1.,'mrr':.71}
    small={v.BASE+'runs/union_seed42/run1/query_rr.npy':encoded(np.array([.4,1.],dtype=np.float64)),v.BASE+'runs/union_seed42/run1/test_metrics.json':json.dumps(metrics).encode()}
    arrays={'src':np.repeat(qids,4),'dst':np.take_along_axis(canonical,orders,axis=1).ravel(),'label':(orders==0).astype(np.int8).ravel(),'logit':np.take_along_axis(scores,orders,axis=1).ravel()}
    return qids,positive,subset,small,arrays

def execute_tiny(tiny):
    qids,positive,subset,small,arrays=tiny;stream=io.BytesIO();np.savez(stream,**arrays);raw=stream.getvalue()
    return v.rerank_npz(bytearray(raw),len(raw),'union_seed42',small,subset,qids,positive)

def test_complete_tiny_npz_preserves_query_ids_and_repeated_physical_source(tiny):
    result=execute_tiny(tiny)
    assert result['midpoint_mrr']==.7 and result['all_query_RR_bitwise_exact']is True
    assert result['query_IDs_not_physical_sources_verified']is True and result['legacy_mrr_field_not_used']==.71
    assert result['original_negative_NPY_reconstructed_sha256']==tiny[2]['outputs']['negative']['sha256']

@pytest.mark.parametrize('change',['physical_src_instead_queryID','bad_labels','wrong_positive','negative_order','nan','saved_RR','metric','negative_hash'])
def test_tiny_raw_and_sidecar_corruptions_rejected(tiny,change):
    qids,positive,subset,small,arrays=tiny
    if change=='physical_src_instead_queryID':arrays['src'][:]=7
    elif change=='bad_labels':arrays['label'][:]=0
    elif change=='wrong_positive':arrays['dst'][np.flatnonzero(arrays['label']==1)[0]]=30
    elif change=='negative_order':
        positions=np.flatnonzero(arrays['label'][:4]==0);arrays['dst'][positions[:2]]=arrays['dst'][positions[:2][::-1]]
    elif change=='nan':arrays['logit'][0]=np.nan
    elif change=='saved_RR':small[v.BASE+'runs/union_seed42/run1/query_rr.npy']=encoded(np.array([1.,.4],dtype=np.float64))
    elif change=='metric':x=json.loads(small[v.BASE+'runs/union_seed42/run1/test_metrics.json']);x['ogb_mrr']=.8;small[v.BASE+'runs/union_seed42/run1/test_metrics.json']=json.dumps(x).encode()
    else:subset['outputs']['negative']['sha256']='0'*64
    with pytest.raises(v.InvalidScores):execute_tiny(tiny)

def test_permutation_must_be_bijection():
    with pytest.raises(v.InvalidScores):v.restore_candidate_slots(np.array([[1,2,3]]),np.array([[0,0,2]]))

def test_all_six_paired_means_and_sample_sd():
    rows=[{'run':f'{arm}_seed{s}','midpoint_mrr':value}for arm,values in [('union',[.1,.2,.3]),('pv',[.2,.25,.4])]for s,value in zip((42,43,44),values)]
    result=v.aggregate(rows);assert result['union']['mean']==pytest.approx(.2)and result['paired_difference']['mean']==pytest.approx(1/12)
    with pytest.raises(v.InvalidScores):v.aggregate(rows[:-1])
