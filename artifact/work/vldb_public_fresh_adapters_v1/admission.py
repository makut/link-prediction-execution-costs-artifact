"""New public admission: exact numerical bytes and frozen cohorts, no old paths.

The original experiment sources and metadata remain unchanged. Model bytes are
not loaded: their validation-selected identities are bound through the exact
public quality manifest and original freeze. No operational timestamp is forged.
"""
from pathlib import Path,PurePosixPath
import copy,hashlib,io,json,os,time

HERE=Path(__file__).resolve().parent
MIB=1<<20
PROOF={'bytes':4032,'sha256':'3a2e2ed2bcab1e7f160d43031f5bf640c354a05a45614e8aa67f60569eb4ad33'}
QUALITY_MANIFEST='9bf816acdce59b0905998adc47437092427cfa8d4c7154da69e5343a8eb15f7d'
PLAN_SHA='9bfb522c4d8472296c2be0cc4064aca75172002b5975fc58ab0ab5c8f467b135'
FREEZE_SHA='471e93af838f6d6a81681b46e4c31b4ce6fdfe5bbd4479196409ac45ea929e6c'
FR_ARCHIVE='ad6690f65f54b6eee0e201108f957d48a47278df6aceb319c8d5850fb00e3d2f'
SOURCES={'scripts/seal_revision/exact_benchmark.py':'3ea7075dd80453a61eb42e8e990d21bf153635e6eff2e26379ba21788ad22544',
 'seal/graph_csr_kernels.py':'19f517b874cfebc9f9388f094c17cf91b559936cd36f81976292c76025cfb2c5',
 'seal/graph_csr_extraction.py':'fd1468b141af9d33ed4045f862342e5f145aa4d0eaee73421754832c9831b1cd'}
ORIGINAL={'e7':('semantic_sampler_audit_v1.py','d19ab63bc217f1ebc5f6856b868d2928972fb763420b6bf95fe6ab7a0d402f78'),
 'e12':('heuristic_baselines.py','dc59aabf813abaf0b34c816199e9fe3edad59e2402856ddd73fd6207c0043104')}
RUNS={f'{arm}_seed{seed}'for arm in('union','pv')for seed in(42,43,44)}

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    path=Path(path);require(path.is_file()and not path.is_symlink(),'regular immutable input required: '+str(path))
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(4*MIB),b''):h.update(block)
    return h.hexdigest()
def snapshot(path):
    info=Path(path).stat();return {'bytes':info.st_size,'mtime_ns':info.st_mtime_ns,'inode':info.st_ino,'device':info.st_dev}
def read(path,limit=4*MIB):
    path=Path(path);require(path.is_file()and not path.is_symlink()and path.stat().st_size<=limit,'bounded regular JSON required')
    def unique(pairs):
        result={}
        for key,value in pairs:require(key not in result,'duplicate JSON key');result[key]=value
        return result
    return json.loads(path.read_bytes(),object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def save(path,value):
    with Path(path).open('x')as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def safe_relative(name):
    p=PurePosixPath(name);require(name and str(p)==name and not p.is_absolute()and '..'not in p.parts and '\\'not in name,'canonical relative payload required');return name
def expected_payloads():
    path=HERE/'PREPARED_23_PAYLOAD_IDENTITY.json';require(path.stat().st_size==PROOF['bytes']and sha(path)==PROOF['sha256'],'exact public23 proof required')
    proof=read(path);require(proof['all_payload_hashes_equal']is True and proof['payload_count']==len(proof['files'])==23,'complete public23 expectation required')
    return proof['files']
def verify_payloads(root,expected):
    """Stream whole public numerical files; no global array or graph copy."""
    root=Path(root).resolve();observed={}
    for row in expected:
        name=safe_relative(row['path']);path=root/name
        require(path.resolve().is_relative_to(root)and not path.is_symlink(),'payload path escape refused')
        before=snapshot(path);require(before['bytes']==row['bytes']and sha(path)==row['sha256'],'entire numeric payload bytes/SHA differ: '+name)
        require(snapshot(path)==before,'numeric payload changed during full hashing')
        observed[str(path)]={**before,'sha256':row['sha256']}
    require(len(observed)==len(expected),'unique whole numerical payloads required');return observed
def npy_hash(blocks,shape,dtype):
    """Hash the exact v1 NPY representation without writing a candidate file."""
    import numpy as np
    h=hashlib.sha256();header=io.BytesIO()
    np.lib.format.write_array_header_1_0(header,{'descr':np.lib.format.dtype_to_descr(np.dtype(dtype)),'fortran_order':False,'shape':tuple(shape)})
    h.update(header.getvalue());count=0
    for block in blocks:
        raw=np.ascontiguousarray(block,dtype=dtype).tobytes();h.update(raw);count+=len(raw)
    require(count==int(np.prod(shape))*np.dtype(dtype).itemsize,'complete NPY byte stream required')
    return h.hexdigest()
def selected_negative_blocks(source,negative,ids,chunk=64):
    import numpy as np
    for at in range(0,len(ids),chunk):
        selected=ids[at:at+chunk];block=np.empty((len(selected),negative.shape[1],2),dtype='<i8')
        block[:,:,0]=source[selected,None];block[:,:,1]=negative[selected];yield block.reshape(-1,2)

class Admission:
    def __init__(self,prepared,quality,quality_manifest,repo):
        self.prepared=Path(prepared).resolve();self.quality=Path(quality).resolve();self.repo=Path(repo).resolve();self.stats={};self.small={}
        self.expected=expected_payloads()
        require(sha(quality_manifest)==QUALITY_MANIFEST,'exact immutable public quality manifest required')
        self.qmanifest=read(quality_manifest)['files'];self.small[str(Path(quality_manifest).resolve())]=QUALITY_MANIFEST
        self.plan=self.quality_json('plan.json');self.freeze=self.quality_json('FROZEN_CHECKPOINTS.json')
        require(sha(self.quality/'plan.json')==PLAN_SHA and sha(self.quality/'FROZEN_CHECKPOINTS.json')==FREEZE_SHA,'unchanged original Friendster plan/freeze required')
        require(self.freeze['selection_uses_test']is False and self.freeze['plan_sha256']==PLAN_SHA and len(self.freeze['jobs'])==6 and{r['name']for r in self.freeze['jobs']}==RUNS,'all6 original validation-selected fits required')
        for job in self.freeze['jobs']:
            require(job['training']['test_scored']is False and job['selection']['selection_split']=='validation','test-independent selection required')
            name=job['name'];config=self.quality_json('configs/'+name+'.json')
            require(sha(self.quality/'configs'/f'{name}.json')==job['config_sha256'],'original fit config identity differs')
            model=self.qmanifest['friendster/results/quality_friendster/runs/'+name+'/run1/best_model.pth']
            require(model['sha256']==job['checkpoint_sha256']and model['original_archive_sha256']==FR_ARCHIVE,'original frozen checkpoint/whole archive binding differs')
            require(config['training']['seed']==job['seed']and config['seal']['graph_csr_use_per_vertex_sampling']==(job['variant']=='pv'),'original config seed/sampler differs')
        for name,pin in SOURCES.items():require(sha(self.repo/name)==pin,'frozen scientific source differs: '+name)
        for role,(name,pin)in ORIGINAL.items():require(sha(HERE/'original'/name)==pin,'original E7/E12 source differs: '+role)
        self.sources=dict(SOURCES);self.cohorts={}
    def quality_raw(self,name):
        path=self.quality/safe_relative(name);key='friendster/results/quality_friendster/'+name;row=self.qmanifest[key]
        require(path.stat().st_size==row['bytes']and sha(path)==row['sha256']and row['original_archive_sha256']==FR_ARCHIVE,'unchanged original quality member differs: '+name)
        self.small[str(path)]=row['sha256'];return path
    def quality_json(self,name):return read(self.quality_raw(name))
    def verify_all(self):
        self.stats=verify_payloads(self.prepared,self.expected)
        manifest=self.prepared/'manifest.json';require(manifest.is_file(),'actual regenerated preparation metadata required')
        self.small[str(manifest)]=sha(manifest)
        for split,key,count in(('valid','val',512),('test','test',4096)):
            self.cohorts[split]=self.verify_cohort(split,key,count)
        self.derived_plan=copy.deepcopy(self.plan)
        self.derived_plan['graph_root']=str(self.prepared/'train_csr')
        self.derived_plan['graph_input_stats']={str(self.prepared/'train_csr'/n):{k:self.stats[str(self.prepared/'train_csr'/n)][k]for k in('bytes','mtime_ns')}for n in('edge_starts.bin','edge_ends.bin')}
        self.gate={'kind':'NEW_PUBLIC_NUMERIC_IDENTITY_ADMISSION','original_plan_sha256':PLAN_SHA,'original_freeze_sha256':FREEZE_SHA,
          'quality_manifest_sha256':QUALITY_MANIFEST,'checkpoint_identity_scope':'Original6 checkpoint SHA declarations bound to immutable full public quality manifest; model bytes not loaded or required for structural scoring.',
          'full_numeric_payloads_verified':23,'fixed_validation_queries':512,'fixed_test_queries':4096,'no_original_metadata_or_mtime_changes':True}
        return self
    def verify_cohort(self,split,key,count):
        import numpy as np
        spec=self.plan['subsets'][key];require(spec['query_count']==count and spec['negatives_per_query']==1000 and spec['all_provided_negatives_retained']is True,'frozen cohort cardinality differs')
        arrays={name:np.load(self.prepared/'evaluation'/split/(name+'.npy'),mmap_mode='r',allow_pickle=False)for name in('source_node','target_node','target_node_neg','warm_query')}
        source,target,negative,warm=[arrays[n]for n in('source_node','target_node','target_node_neg','warm_query')]
        require(source.shape==target.shape==warm.shape==(10000,)and negative.shape==(10000,1000),'full provided query shape differs')
        path=self.quality_raw('subsets/'+split+'_query_ids.npy');ids=np.load(path,mmap_mode='r',allow_pickle=False)
        require(ids.shape==(count,)and ids.dtype==np.dtype('<i8')and np.array_equal(ids,np.sort(np.random.default_rng(spec['subset_seed']).choice(10000,count,replace=False))),'exact original seed-selected query IDs required')
        positive=np.column_stack((source[ids],target[ids])).astype('<i8');selected_warm=np.asarray(warm[ids],dtype=bool)
        hashes={'query_ids':sha(path),'positive':npy_hash([positive],positive.shape,'<i8'),
          'warm_query':npy_hash([selected_warm],selected_warm.shape,np.bool_),
          'negative':npy_hash(selected_negative_blocks(source,negative,ids),(count*1000,2),'<i8')}
        require(all(hashes[name]==row['sha256']for name,row in spec['outputs'].items()),'entire fixed cohort/candidate NPY identity differs')
        for name,suffix in(('positive','edge'),('warm_query','warm_query')):
            require(sha(self.quality_raw('subsets/'+split+'_'+suffix+'.npy'))==hashes[name],'original cohort payload differs')
        return {'arrays':arrays,'ids':ids,'positive':positive,'hashes':hashes,'query_ids_path':path}
    def stable(self):
        for name,row in self.stats.items():require(snapshot(name)=={k:row[k]for k in('bytes','mtime_ns','inode','device')},'whole-admitted numeric payload changed')
        for name,pin in self.small.items():require(sha(name)==pin,'original small evidence changed')
        for name,pin in SOURCES.items():require(sha(self.repo/name)==pin,'scientific helper source changed')
        return self.derived_plan,dict(self.gate)
    def receipt(self):
        return {**self.gate,'prepared_root':str(self.prepared),'quality_root':str(self.quality),'actual_fresh_numeric_file_stats':self.stats,
          'original_small_evidence_sha256':self.small,'cohort_NPY_sha256':{k:v['hashes']for k,v in self.cohorts.items()},
          'original_scientific_source_sha256':{**SOURCES,**{f'original/{name}':pin for name,pin in ORIGINAL.values()}},
          'stat_continuity_scope':'Full byte hashes before execution; actual observed size/mtime/inode/device continuity afterwards. No old timestamp is set or required.'}
