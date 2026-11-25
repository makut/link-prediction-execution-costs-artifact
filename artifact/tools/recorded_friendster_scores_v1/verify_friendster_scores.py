#!/usr/bin/env python3
"""Bounded independent archival Friendster float-score -> midpoint-rank audit.

No producer imports, graph/model/weights loading, prediction or new experiments.
"""
from pathlib import Path
import argparse,gzip,hashlib,io,json,math,resource,statistics,sys,tarfile,zipfile
import numpy as np
MIB=1<<20
ASSET_BYTES=404147012
ASSET_SHA='e17cdc9678d00a3f372fc8083741a200a63662226f730f47eeda79f89b4ad3b2'
MANIFEST_SHA='9bf816acdce59b0905998adc47437092427cfa8d4c7154da69e5343a8eb15f7d'
PREFIX='lpp_public_quality_evidence_v1/'
BASE='friendster/results/quality_friendster/'
RUNS=tuple(f'{arm}_seed{seed}'for arm in('union','pv')for seed in(42,43,44))
Q,K=4096,1000
NPZ_BYTES=86102998
MAX_RSS=256*MIB
BLOCK_QUERIES=64
class InvalidScores(ValueError):pass
def require(ok,message):
    if not ok:raise InvalidScores(message)
def peak_bytes():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
def memory_gate():require(peak_bytes()<MAX_RSS,'measured process high-water must remain below256MiB')
def digest_bytes(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(MIB),b''):h.update(block)
    return h.hexdigest()
def decode(raw):
    def unique(pairs):
        out={}
        for k,v in pairs:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(InvalidScores('nonfinite JSON')))
def bound_small(raw):
    require(len(raw)<=2*MIB,'small recorded metadata bound');return np.load(io.BytesIO(raw),allow_pickle=False)
def npy_header(dtype,shape):
    out=io.BytesIO();np.lib.format.write_array_header_1_0(out,{'descr':np.dtype(dtype).str,'fortran_order':False,'shape':tuple(shape)});return out.getvalue()
def canonical_npy_sha(array):
    array=np.ascontiguousarray(array);h=hashlib.sha256(npy_header(array.dtype,array.shape));h.update(array.tobytes());return h.hexdigest()
def midpoint_ranks(labels,scores):
    require(labels.ndim==scores.ndim==2 and labels.shape==scores.shape and labels.shape[1]>=2,'complete candidate groups required')
    require(np.isfinite(scores).all()and np.all((labels==0)|(labels==1))and np.all((labels==1).sum(axis=1)==1),'finite scores and exactly one labeled positive per query required')
    positive_position=np.argmax(labels==1,axis=1);positive=np.take_along_axis(scores,positive_position[:,None],axis=1)
    greater=((scores>positive)&(labels==0)).sum(axis=1);equal=((scores==positive)&(labels==0)).sum(axis=1)
    return 1.0+greater+.5*equal,positive_position,equal

def independently_declared_order(query_id,k,seed):
    # Reconstructed from the pinned protocol/source equation, never imported
    # from the producer. The inputs are original QUERY IDs, not physical src IDs.
    return np.random.default_rng(np.random.SeedSequence([int(seed),int(query_id)])).permutation(k+1).astype(np.int32)
def restore_candidate_slots(destinations,orders):
    require(destinations.shape==orders.shape and np.all(np.sort(orders,axis=1)==np.arange(orders.shape[1])),'complete bijective candidate permutation required')
    restored=np.empty_like(destinations);np.put_along_axis(restored,orders,destinations,axis=1);return restored

class MemoryFile(io.RawIOBase):
    def __init__(self,buffer,length):self.view=memoryview(buffer)[:length];self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else len(self.view)+offset
        require(0<=position<=len(self.view),'bounded NPZ seek required');self.position=position;return position
    def read(self,size=-1):
        size=len(self.view)-self.position if size<0 else min(size,len(self.view)-self.position)
        require(0<=size<=4*MIB,'no whole-vector read or NPZ copy permitted');data=self.view[self.position:self.position+size].tobytes();self.position+=size;return data
class HashingReader:
    def __init__(self,source):self.source=source;self.hash=hashlib.sha256();self.count=0
    def read(self,n=-1):
        data=self.source.read(n);self.hash.update(data);self.count+=len(data);return data

def safe_tar_name(member):
    require(member.isfile()and member.name.startswith(PREFIX)and '..'not in Path(member.name).parts and 0<=member.size<=128*MIB,'bounded original regular quality member required')
    return member.name[len(PREFIX):]
def drain(stream,count,collector=None,hasher=None,buffer=None):
    offset=0
    while offset<count:
        block=stream.read(min(MIB,count-offset));require(block,'truncated original member')
        if collector is not None:collector.append(block)
        if hasher is not None:hasher.update(block)
        if buffer is not None:buffer[offset:offset+len(block)]=block
        offset+=len(block);memory_gate()
    require(stream.read(1)==b'','unexpected member trailing bytes')

def gather_small(asset,manifest):
    targets={BASE+'plan.json',BASE+'FROZEN_CHECKPOINTS.json','friendster/results/FRIENDSTER_CHECKPOINTS_JOINTLY_FROZEN.json'}
    targets|={BASE+'subsets/'+name for name in('test_edge.npy','test_query_ids.npy','test_warm_query.npy','test_manifest.json')}
    targets|={BASE+'runs/'+run+'/run1/'+name for run in RUNS for name in('query_rr.npy','query_ids.npy','test_metrics.json','best_selection.json','test_access.json')}
    observed={};used={};total=0;members=0
    with Path(asset).open('rb')as raw:
        wrapper=HashingReader(raw)
        with tarfile.open(fileobj=wrapper,mode='r|gz')as tar:
            for member in tar:
                name=safe_tar_name(member);members+=1;total+=member.size;require(members<=4096 and total<=2*(1<<30),'bounded original quality inventory')
                collect=[]if name in targets else None
                if collect is not None:require(name not in observed and member.size<=2*MIB,'unique bounded small original metadata')
                with tar.extractfile(member)as stream:drain(stream,member.size,collector=collect)
                if collect is not None:
                    data=b''.join(collect);pin=manifest['files'][name];require(member.size==pin['bytes']and digest_bytes(data)==pin['sha256'],'small original metadata identity differs: '+name);observed[name]=data;used[name]={'bytes':member.size,'sha256':pin['sha256']}
        while wrapper.read(MIB):pass
        require(wrapper.count==ASSET_BYTES and wrapper.hash.hexdigest()==ASSET_SHA,'whole compressed public quality archive differs')
    require(set(observed)==targets,'all genuine Friendster metadata/RR/cohort targets required');return observed,used

def provenance(small,manifest):
    plan=decode(small[BASE+'plan.json']);subset=decode(small[BASE+'subsets/test_manifest.json']);frozen=decode(small[BASE+'FROZEN_CHECKPOINTS.json']);joint=decode(small['friendster/results/FRIENDSTER_CHECKPOINTS_JOINTLY_FROZEN.json'])
    qids=bound_small(small[BASE+'subsets/test_query_ids.npy']);positive=bound_small(small[BASE+'subsets/test_edge.npy']);warm=bound_small(small[BASE+'subsets/test_warm_query.npy'])
    require(plan['subsets']['test']==subset and subset['selection_unit']=='original query row, including repeated source IDs'and subset['query_count']==Q and subset['original_query_count']==10000 and subset['negatives_per_query']==K and subset['all_provided_negatives_retained']is True,'exact original query/candidate protocol required')
    require(qids.dtype==np.int64 and qids.shape==(Q,)and np.array_equal(qids,np.sort(np.random.default_rng(subset['subset_seed']).choice(10000,Q,replace=False)).astype(np.int64)),'original sorted4096query-row selection differs')
    require(positive.dtype==np.int64 and positive.shape==(Q,2)and np.unique(positive[:,0]).size==subset['unique_source_count']==4095,'4096query rows/4095physical sources must remain distinct')
    require(warm.shape==(Q,)and np.isin(warm,[0,1]).all()and int((warm==0).sum())==44,'all44 original cold queries must remain included')
    for key,name in(('positive','test_edge.npy'),('query_ids','test_query_ids.npy'),('warm_query','test_warm_query.npy')):require(digest_bytes(small[BASE+'subsets/'+name])==subset['outputs'][key]['sha256'],'original subset payload hash differs')
    require(frozen['plan_sha256']==digest_bytes(small[BASE+'plan.json'])and frozen['selection_uses_test']is False and len(frozen['jobs'])==6 and joint['success']is True and joint['frozen_sha256']==digest_bytes(small[BASE+'FROZEN_CHECKPOINTS.json'])and joint['jobs']==6 and joint['all_validation_selected_before_test']is True,'recorded joint validation freeze differs')
    jobs={row['name']:row for row in frozen['jobs']};require(set(jobs)==set(RUNS),'all six paired frozen jobs required')
    for run,job in jobs.items():
        prefix=BASE+'runs/'+run+'/run1/';selection=decode(small[prefix+'best_selection.json']);access=decode(small[prefix+'test_access.json']);ids=bound_small(small[prefix+'query_ids.npy'])
        require(np.array_equal(ids,qids)and selection==job['selection']and selection['selection_split']=='validation'and job['training']['test_scored']is False and access['mode']=='only_test'and access['started_unix']>=frozen['created_unix'],'recorded validation/test/cohort metadata inconsistent')
        require(manifest['files'][prefix+'best_model.pth']['sha256']==job['checkpoint_sha256'],'recorded weight manifest pin differs; weights are not deserialized')
    for seed in(42,43,44):require(jobs[f'union_seed{seed}']['initial_model_sha256']==jobs[f'pv_seed{seed}']['initial_model_sha256']==joint['paired_initial_hashes'][str(seed)],'recorded paired initial-model hashes differ')
    return plan,subset,qids,positive,warm

def rerank_npz(buffer,size,run,small,subset,qids,positive):
    prefix=BASE+'runs/'+run+'/run1/';expected_rr=bound_small(small[prefix+'query_rr.npy']);require(expected_rr.shape==(Q,)and expected_rr.dtype==np.float64,'original4096float64 RR required')
    rr=np.empty(Q,dtype=np.float64);ranks=np.empty(Q,dtype=np.float64);positions=np.empty(Q,dtype=np.int64);tie_queries=0;schemas={};array_hashes={};negative_hash=hashlib.sha256(npy_header('<i8',(Q*K,2)))
    with zipfile.ZipFile(MemoryFile(buffer,size))as archive:
        info=archive.infolist();require(len(info)==4 and{v.filename for v in info}=={'src.npy','dst.npy','label.npy','logit.npy'},'exact four original arrays required')
        streams={};dtypes={'src':np.dtype('<i8'),'dst':np.dtype('<i8'),'label':np.dtype('|i1'),'logit':np.dtype('<f4')}
        for key,dtype in dtypes.items():
            member=archive.getinfo(key+'.npy');require(member.compress_type==zipfile.ZIP_STORED and not member.flag_bits&1,'original stored unencrypted NPY members required');stream=archive.open(member);version=np.lib.format.read_magic(stream);shape,fortran,actual=np.lib.format.read_array_header_1_0(stream)if version==(1,0)else np.lib.format.read_array_header_2_0(stream)
            require(version in((1,0),(2,0))and shape==(Q*(K+1),)and not fortran and actual==dtype and member.file_size-stream.tell()==Q*(K+1)*dtype.itemsize,'original inspected flat-array schema differs')
            streams[key]=stream;schemas[key]={'shape':list(shape),'dtype':actual.str,'NPY_bytes':member.file_size};array_hashes[key]=hashlib.sha256()
        try:
            for lo in range(0,Q,BLOCK_QUERIES):
                hi=min(Q,lo+BLOCK_QUERIES);count=(hi-lo)*(K+1);arrays={}
                for key,dtype in dtypes.items():
                    raw=streams[key].read(count*dtype.itemsize);require(len(raw)==count*dtype.itemsize,'whole candidate block required');array_hashes[key].update(raw);arrays[key]=np.frombuffer(raw,dtype=dtype).reshape(hi-lo,K+1)
                require(np.array_equal(arrays['src'],np.broadcast_to(qids[lo:hi,None],arrays['src'].shape)),'NPZ.src must be original query IDs, not physical sources or collapsed unique vertices')
                orders=np.stack([independently_declared_order(qid,K,subset['candidate_order_seed'])for qid in qids[lo:hi]])
                require(np.array_equal(arrays['label'],(orders==0).astype(np.int8)),'labels do not follow declared per-query candidate permutation')
                block_ranks,positive_position,ties=midpoint_ranks(arrays['label'],arrays['logit']);ranks[lo:hi]=block_ranks;rr[lo:hi]=1.0/block_ranks;positions[lo:hi]=positive_position;tie_queries+=int((ties>0).sum())
                restored=restore_candidate_slots(arrays['dst'],orders);require(np.array_equal(restored[:,0],positive[lo:hi,1]),'labeled positive destinations differ from frozen4096positive edges')
                require(np.all((restored>=0)&(restored<65608366))and not np.any(restored[:,1:]==positive[lo:hi,0,None])and not np.any(restored[:,1:]==positive[lo:hi,1,None]),'candidate node bounds/self/positive-as-negative inconsistent')
                require(np.all(np.diff(np.sort(restored[:,1:],axis=1),axis=1)>0),'repeated negative destination within a query')
                pairs=np.empty((hi-lo,K,2),dtype='<i8');pairs[:,:,0]=positive[lo:hi,0,None];pairs[:,:,1]=restored[:,1:];negative_hash.update(pairs.tobytes());memory_gate()
            require(all(stream.read(1)==b''for stream in streams.values()),'no trailing vector bytes permitted')
        finally:
            for stream in streams.values():stream.close()
    require(negative_hash.hexdigest()==subset['outputs']['negative']['sha256'],'stream-reconstructed full original65,536,128-byte negative NPY hash differs')
    require(np.array_equal(rr,expected_rr)and canonical_npy_sha(rr)==digest_bytes(small[prefix+'query_rr.npy']),'all4096 independently reranked RR must be bitwise identical to original saved RR file')
    metrics=decode(small[prefix+'test_metrics.json']);mrr=float(np.mean(rr));hits={str(k):float(np.mean(ranks<=k))for k in(1,10,50,100)}
    require(metrics['ogb_num_sources']==Q and math.isclose(mrr,metrics['ogb_mrr'],rel_tol=0,abs_tol=1e-13)and all(v==metrics['ogb_hits@'+k]for k,v in hits.items()),'original midpoint MRR/Hits alias mismatch')
    return {'run':run,'schema':schemas,'queries':Q,'negatives_per_query':K,'candidate_rows':Q*(K+1),'query_IDs_not_physical_sources_verified':True,'all_query_RR_bitwise_exact':True,'RR_NPY_sha256':canonical_npy_sha(rr),'candidate_array_payload_sha256':{k:h.hexdigest()for k,h in array_hashes.items()if k!='logit'},'float32_logit_payload_sha256':array_hashes['logit'].hexdigest(),'original_negative_NPY_reconstructed_bytes':128+Q*K*16,'original_negative_NPY_reconstructed_sha256':negative_hash.hexdigest(),'positive_position_range':[int(positions.min()),int(positions.max())],'distinct_positive_positions':int(np.unique(positions).size),'queries_with_negative_score_ties':tie_queries,'midpoint_mrr':mrr,'midpoint_hits':hits,'legacy_mrr_field_not_used':metrics.get('mrr'),'cold_query_count':44}

def aggregate(rows):
    by={r['run']:r['midpoint_mrr']for r in rows};require(set(by)==set(RUNS),'all six result rows required');result={}
    for arm in('union','pv'):
        values=[by[f'{arm}_seed{s}']for s in(42,43,44)];result[arm]={'seed_MRR':values,'mean':statistics.mean(values),'sample_sd':statistics.stdev(values)}
    differences=[by[f'pv_seed{s}']-by[f'union_seed{s}']for s in(42,43,44)];result['paired_difference']={'seed_differences':differences,'mean':statistics.mean(differences),'sample_sd':statistics.stdev(differences)};return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--quality-asset',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    require(not a.output.exists()and not a.output.is_symlink(),'fresh archival receipt only');require(a.quality_asset.is_file()and not a.quality_asset.is_symlink()and a.quality_asset.stat().st_size==ASSET_BYTES and sha(a.manifest)==MANIFEST_SHA,'exact public asset size/manifest pin required');manifest=decode(a.manifest.read_bytes());memory_gate();small,used=gather_small(a.quality_asset,manifest);plan,subset,qids,positive,warm=provenance(small,manifest)
    buffer=bytearray(NPZ_BYTES);rows=[];npz_pins={};targets={BASE+'runs/'+run+'/run1/test_scores.npz':run for run in RUNS};before=tuple(a.quality_asset.stat()[i]for i in(6,8))
    with a.quality_asset.open('rb')as raw:
        wrapper=HashingReader(raw)
        with tarfile.open(fileobj=wrapper,mode='r|gz')as tar:
            for member in tar:
                name=safe_tar_name(member);run=targets.get(name);h=hashlib.sha256()if run else None
                if run:require(member.size==NPZ_BYTES and name not in npz_pins,'all six unique original NPZ containers required')
                with tar.extractfile(member)as stream:drain(stream,member.size,hasher=h,buffer=buffer if run else None)
                if run:
                    pin=manifest['files'][name];require(pin['bytes']==member.size and pin['sha256']==h.hexdigest(),'whole original score NPZ bytes/hash differ');npz_pins[name]={'bytes':member.size,'sha256':h.hexdigest()};rows.append(rerank_npz(buffer,member.size,run,small,subset,qids,positive));memory_gate()
        while wrapper.read(MIB):pass
        require(wrapper.count==ASSET_BYTES and wrapper.hash.hexdigest()==ASSET_SHA,'second complete original archive readback differs')
    require(set(npz_pins)==set(targets)and tuple(a.quality_asset.stat()[i]for i in(6,8))==before,'archive/cohort changed during reranking')
    signatures=[r['candidate_array_payload_sha256']for r in rows];require(all(s==signatures[0]for s in signatures),'candidate IDs/order/labels differ across six paired runs');result={'kind':'INDEPENDENT_ARCHIVAL_FRIENDSTER_FLOAT_SCORE_RERANK_V1','status':'PASS','quality_archive':{'bytes':ASSET_BYTES,'sha256':ASSET_SHA},'quality_manifest_sha256':MANIFEST_SHA,'source_sha256':sha(Path(__file__)),'queries':Q,'physical_unique_sources':4095,'repeated_physical_source_query_rows_preserved':True,'cold_queries_retained':44,'negative_candidates_per_query':K,'positive_identification':'one stored label==1, independently checked against per-original-query-ID SeedSequence([20261127,query_id]) permutation; not assumed first','rank_formula':'1 + count(negative_score > positive_score) + 0.5*count(negative_score == positive_score), preserving float32 stored comparisons','rows':rows,'aggregates':aggregate(rows),'used_score_NPZ_files':npz_pins,'used_original_small_files':used,'recorded_provenance':{'plan_sha256':digest_bytes(small[BASE+'plan.json']),'freeze_sha256':digest_bytes(small[BASE+'FROZEN_CHECKPOINTS.json']),'original_subset_manifest_sha256':digest_bytes(small[BASE+'subsets/test_manifest.json']),'validation_selected_checkpoint_hashes_and_paired_initial_hashes_consistent':True,'original_negative_subset_reconstructed_without_disk_or_full_array':True},'peak_rss_bytes':peak_bytes(),'peak_rss_limit_bytes_exclusive':MAX_RSS,'one_reused_NPZ_buffer_bytes':NPZ_BYTES,'query_block_size':BLOCK_QUERIES,'score_NPZ_disk_copies':0,'weights_or_models_loaded':0,'graphs_loaded':0,'Torch_imported':'torch'in sys.modules,'new_predictions_or_experiments':0,'remote_actions':0,'limits':['Archival arithmetic/cohort/provenance verification only; no new training, inference, graph loading or quality experiment.','Raw src field encodes original query ID, not physical source vertex. Repeated physical sources remain distinct query rows.','Headline midpoint ogb_* aliases are checked; legacy mrr/Hits fields can differ at ties and are not substituted.','All candidate IDs/order bind to original frozen positive/query-ID files and reconstructed negative NPY whole hash. Graph non-neighbor/leakage status is inherited from recorded preparation audits, not rerun here.','Checkpoint members are streamed/discarded in bounded chunks, never interpreted, deserialized or materialized as model arrays; their pinned manifest hashes agree with genuine recorded validation-freeze metadata. Whole compressed asset is verified twice.','Non-target tar members are discarded in1MiB chunks to avoid tarfile allocating a whole skipped NPZ.']};memory_gate();require(result['Torch_imported']is False,'no Torch/model import allowed');a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'PASS','rows':6,'queries_per_run':Q,'peak_rss_bytes':result['peak_rss_bytes'],'receipt':str(a.output),'receipt_sha256':sha(a.output),'aggregates':result['aggregates']}));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status':'INVALID_ARCHIVAL_SCORE_EVIDENCE','error_type':type(error).__name__,'message':str(error),'new_science':False}),file=sys.stderr);raise SystemExit(2)
