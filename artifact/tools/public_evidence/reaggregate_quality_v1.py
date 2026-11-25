#!/usr/bin/env python3
"""Portable offline quality-table arithmetic, reading no models or graphs.

Use the separate verified quality asset extracted on a suitable compute host.
Friendster uses the six original RR vectors. Citation2 streams six candidate
NPZs through one reused buffer, reading NPY vectors in256query blocks with a
128MiB per-container guard and512MiB measured process peak cap. No NPZ disk
copy, full candidate-array loading, model loading or prediction is performed.
"""
import argparse,gzip,hashlib,io,json,math,resource,statistics,sys,tarfile,time,zipfile
from pathlib import Path,PurePosixPath
import numpy as np

MIB=1<<20
MANIFEST_SHA='9bf816acdce59b0905998adc47437092427cfa8d4c7154da69e5343a8eb15f7d'
QUALITY_BYTES=404147012
QUALITY_SHA='e17cdc9678d00a3f372fc8083741a200a63662226f730f47eeda79f89b4ad3b2'
PREFIX='lpp_public_quality_evidence_v1/'
def require(ok,message):
    if not ok:raise RuntimeError(message)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(MIB),b''):h.update(block)
    return h.hexdigest()
def candidate_ranks(src,dst,label,logit,queries=4096,negatives=1000):
    count=queries*(negatives+1)
    require(all(v.ndim==1 and v.size==count for v in(src,dst,label,logit))and src.dtype.kind in'iu'and dst.dtype.kind in'iu'and label.dtype.kind in'biu'and logit.dtype.kind=='f','complete numeric candidate vectors required')
    require(np.isfinite(logit).all()and np.all((label==0)|(label==1)),'finite logits and binary labels required')
    src,dst,label,logit=(a.reshape(queries,negatives+1)for a in(src,dst,label,logit))
    require(np.all(src==src[:,:1])and len(np.unique(src[:,0]))==queries,'distinct complete source groups required')
    require(np.all((label==1).sum(axis=1)==1)and np.all((label==0).sum(axis=1)==negatives),'one positive and all negatives per group required')
    ranks=np.empty(queries,dtype=np.float64)
    for start in range(0,queries,256):
        stop=min(start+256,queries);y=label[start:stop];scores=logit[start:stop]
        positive=np.sum(np.where(y==1,scores,0),axis=1)
        greater=((scores>positive[:,None])&(y==0)).sum(axis=1)
        equal=((scores==positive[:,None])&(y==0)).sum(axis=1)
        ranks[start:stop]=1+greater+.5*equal
    return ranks
def candidate_rr(*args,**kwargs):return 1.0/candidate_ranks(*args,**kwargs)
def peak_bytes():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin'else 1024)
def score_npz(stream,queries=4096,negatives=1000):
    with zipfile.ZipFile(stream)as archive:
        members=archive.infolist();require(len(members)==4 and{m.filename for m in members}=={'src.npy','dst.npy','label.npy','logit.npy'}and sum(m.file_size for m in members)<=128*MIB,'bounded four-vector candidate NPZ required')
        count=queries*(negatives+1)
        def vector(name,kinds):
            source=archive.open(name+'.npy');version=np.lib.format.read_magic(source)
            require(version in((1,0),(2,0)),'bounded known NPY header required')
            shape,fortran,dtype=(np.lib.format.read_array_header_1_0(source)if version==(1,0)else np.lib.format.read_array_header_2_0(source))
            require(shape==(count,)and not fortran and dtype.kind in kinds and dtype.itemsize<=8 and archive.getinfo(name+'.npy').file_size-source.tell()==count*dtype.itemsize,'exact simple numeric NPY shape/size required')
            return source,dtype
        # Validate complete source groups without materializing an NPY vector.
        sources=[]
        source,dtype=vector('src','iu')
        with source:
            for start in range(0,queries,256):
                rows=min(256,queries-start);raw=source.read(rows*(negatives+1)*dtype.itemsize)
                require(len(raw)==rows*(negatives+1)*dtype.itemsize,'complete source block required')
                block=np.frombuffer(raw,dtype=dtype).reshape(rows,negatives+1)
                require(np.all(block==block[:,:1]),'all candidate sources in a group must agree');sources.extend(block[:,0].tolist())
            require(source.read(1)==b'','no trailing source bytes required')
        require(len(set(sources))==queries,'distinct complete source groups required')
        target,target_dtype=vector('dst','iu')
        with target:
            remaining=count*target_dtype.itemsize
            while remaining:
                raw=target.read(min(MIB,remaining));require(raw,'complete destination vector required');remaining-=len(raw)
            require(target.read(1)==b'','no trailing destination bytes required')
        labels,label_dtype=vector('label','biu');logits,logit_dtype=vector('logit','f');ranks=np.empty(queries,dtype=np.float64)
        with labels,logits:
            for start in range(0,queries,256):
                stop=min(start+256,queries);size=(stop-start)*(negatives+1)
                raw_y=labels.read(size*label_dtype.itemsize);raw_scores=logits.read(size*logit_dtype.itemsize)
                require(len(raw_y)==size*label_dtype.itemsize and len(raw_scores)==size*logit_dtype.itemsize,'complete label/logit blocks required')
                y=np.frombuffer(raw_y,dtype=label_dtype).reshape(stop-start,negatives+1);scores=np.frombuffer(raw_scores,dtype=logit_dtype).reshape(stop-start,negatives+1)
                require(np.isfinite(scores).all()and np.all((y==0)|(y==1))and np.all((y==1).sum(axis=1)==1)and np.all((y==0).sum(axis=1)==negatives),'finite complete original candidate groups required')
                positive=np.sum(np.where(y==1,scores,0),axis=1)
                greater=((scores>positive[:,None])&(y==0)).sum(axis=1);equal=((scores==positive[:,None])&(y==0)).sum(axis=1)
                ranks[start:stop]=1+greater+.5*equal
            require(labels.read(1)==logits.read(1)==b'','no trailing candidate bytes required')
    metrics={'mrr':float(np.mean(1.0/ranks)),'hits':{str(k):float(np.mean(ranks<=k))for k in(1,10,50,100)}}
    require(peak_bytes()<=512*MIB,'512MiB measured process peak exceeded: '+str(peak_bytes()));return metrics
def stream_citation2(asset,manifest):
    files=manifest['files'];targets={f'citation2/results/quality/runs/{arm}_seed{seed}/run1/{name}'for arm in('union','pv')for seed in(42,43,44)for name in('test_scores.npz','test_metrics.json')}
    found={};used={};compressed_hash=hashlib.sha256()
    sizes={files[name]['bytes']for name in targets if name.endswith('.npz')}
    require(len(sizes)==1 and next(iter(sizes))<=128*MIB,'same bounded frozen candidate-container size required')
    # Reuse exactly one allocated NPZ buffer. On macOS repeated large BytesIO
    # allocations can accumulate in allocator high-water RSS despite close().
    candidate_buffer=io.BytesIO(bytes(next(iter(sizes))))
    class HashingInput:
        def __init__(self,stream):self.stream=stream;self.count=0
        def read(self,count):
            require(0<=count<=MIB,'bounded compressed read required');raw=self.stream.read(count);compressed_hash.update(raw);self.count+=len(raw);return raw
    require(asset.is_file()and not asset.is_symlink()and asset.stat().st_size==QUALITY_BYTES,'exact original quality archive size required')
    with asset.open('rb')as source:
        wrapper=HashingInput(source)
        # GzipFile bounds decompression to each requested read. tarfile's direct
        # r|gz mode can inflate a very compressible NPZ member in a huge chunk.
        with gzip.GzipFile(fileobj=wrapper,mode='rb')as uncompressed,tarfile.open(fileobj=uncompressed,mode='r|')as archive:
            for member in archive:
                if not member.name.startswith(PREFIX):continue
                name=member.name[len(PREFIX):]
                if name not in targets:continue
                require(member.isfile()and name not in found and member.size==files[name]['bytes']and member.size<=128*MIB,'exact unique bounded candidate member required')
                h=hashlib.sha256();buffer=candidate_buffer if name.endswith('.npz')else io.BytesIO();buffer.seek(0)
                with archive.extractfile(member)as stream:
                    for block in iter(lambda:stream.read(MIB),b''):h.update(block);buffer.write(block)
                require(h.hexdigest()==files[name]['sha256'],'original candidate/metric memberSHA differs');buffer.seek(0)
                used[name]={'bytes':member.size,'sha256':h.hexdigest()}
                if name.endswith('.npz'):found[name]=score_npz(buffer)
                else:found[name]=json.loads(buffer.getvalue())
                if buffer is not candidate_buffer:buffer.close()
                require(peak_bytes()<=512*MIB,'512MiB measured process peak exceeded')
            while uncompressed.read(MIB):pass
        while wrapper.read(MIB):pass
    candidate_buffer.close()
    require(wrapper.count==QUALITY_BYTES and compressed_hash.hexdigest()==QUALITY_SHA and set(found)==targets,'whole original archiveSHA or six-complete-cohort coverage differs')
    rows=[]
    for arm in('union','pv'):
        for seed in(42,43,44):
            name=f'{arm}_seed{seed}';base=f'citation2/results/quality/runs/{name}/run1/';score=found[base+'test_scores.npz'];metric=found[base+'test_metrics.json']
            require(math.isclose(score['mrr'],metric['ogb_mrr'],rel_tol=1e-11,abs_tol=1e-13)and all(value==metric['ogb_hits@'+key]for key,value in score['hits'].items()),'all original MRR/Hits must match')
            rows.append({'name':name,**score})
    return rows,used
def aggregate(rows):
    require(len(rows)==6 and{r['name']for r in rows}=={f'{arm}_seed{seed}'for arm in('union','pv')for seed in(42,43,44)},'all6paired fits required')
    result={}
    for arm in('union','pv'):
        values=[r['mrr']for r in rows if r['name'].startswith(arm+'_')]
        result[arm]={'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),'seed_mrr':values}
    by={r['name']:r['mrr']for r in rows};differences=[by[f'pv_seed{s}']-by[f'union_seed{s}']for s in(42,43,44)]
    result['paired_difference']={'mean':statistics.mean(differences),'sample_sd':statistics.stdev(differences),'seed_differences':differences};return result
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True);group.add_argument('--quality-root',type=Path);group.add_argument('--quality-asset',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--dataset',choices=('friendster','citation2'),required=True);args=p.parse_args(argv);started=time.monotonic()
    if args.quality_asset:
        require(args.dataset=='citation2','stream mode is bounded actual Citation2 score aggregation')
        manifest_path=args.manifest or Path(str(args.quality_asset)+'.manifest.json');require(manifest_path.is_file()and digest(manifest_path)==MANIFEST_SHA,'pinned original quality sidecar manifest required')
        rows,used=stream_citation2(args.quality_asset,json.loads(manifest_path.read_text()))
        print(json.dumps({'status':'PASS','dataset':'citation2','rows':rows,'aggregates':aggregate(rows),'used_files':used,
          'source_sha256':digest(Path(__file__)),'quality_archive_bytes':QUALITY_BYTES,'quality_archive_sha256':QUALITY_SHA,
          'quality_manifest_sha256':MANIFEST_SHA,'peak_rss_bytes':peak_bytes(),'peak_rss_limit_bytes':512*MIB,
          'elapsed_seconds':time.monotonic()-started,'new_measurements':0,'models_loaded':0,'graphs_loaded':0,
          'quality_NPZ_disk_copies':0,'torch_imported':'torch'in sys.modules},indent=2));return 0
    root=args.quality_root.resolve();manifest=root/'MANIFEST.json';require(manifest.is_file()and not manifest.is_symlink()and digest(manifest)==MANIFEST_SHA,'pinned original quality manifest required')
    files=json.loads(manifest.read_text())['files'];used={}
    def bound(name,limit):
        posix=PurePosixPath(name);require(not posix.is_absolute()and'..'not in posix.parts,'safe quality member required')
        path=root/name;row=files[name]
        require(path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(root)and row['bytes']<=limit and path.stat().st_size==row['bytes']and digest(path)==row['sha256'],'quality member identity/size differs: '+name)
        used[name]={'bytes':row['bytes'],'sha256':row['sha256']};return path
    rows=[]
    for arm in('union','pv'):
        for seed in(42,43,44):
            name=f'{arm}_seed{seed}';quality='quality_friendster'if args.dataset=='friendster'else'quality';base=f'{args.dataset}/results/{quality}/runs/{name}/run1/'
            if args.dataset=='friendster':
                rr=np.load(bound(base+'query_rr.npy',MIB),allow_pickle=False);require(rr.shape==(4096,)and np.isfinite(rr).all()and np.all((rr>0)&(rr<=1)),'complete finite original RR vector required')
            else:
                path=bound(base+'test_scores.npz',128*MIB)
                with path.open('rb')as stream:metrics=score_npz(stream)
                rr=np.array([metrics['mrr']])
            mrr=float(np.mean(rr));metric=json.loads(bound(base+'test_metrics.json',MIB).read_text())
            require(math.isclose(mrr,float(metric['ogb_mrr']),rel_tol=1e-11,abs_tol=1e-13),'original reported MRR differs')
            rows.append({'name':name,'mrr':mrr});del rr
    print(json.dumps({'status':'PASS','dataset':args.dataset,'rows':rows,'aggregates':aggregate(rows),'used_files':used,'new_measurements':0,'models_loaded':0,'graphs_loaded':0},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
