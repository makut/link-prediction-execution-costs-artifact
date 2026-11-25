"""Small bounded contracts for the new clean-consumer controller candidate."""
from pathlib import Path,PurePosixPath
import hashlib,json,math,os,re,socket,time,uuid
HERE=Path(__file__).resolve().parent
BRIDGE=HERE/'candidate/bridge/vldb_pencil_bridge_clean_v1'
GIB=1<<30
TARGET='//home/ucp-research/projects/graph_baselines/seal_revision_20261124/pencil_clean_consumer_v1_20261125.tgz'
ATTRIBUTE='lpp_clean_consumer_persistence_v1'
POD='seal-clean-consumer-20261125-0'
STS='seal-clean-consumer-20261125'
BASE=Path('/home/jovyan/seal_revision_20261124')

def require(ok,message):
    if not ok:raise ValueError(message)
def read(path,limit=32<<20):
    path=Path(path);require(path.is_file()and not path.is_symlink()and path.stat().st_size<=limit,'bounded regular JSON required: '+str(path))
    def unique(pairs):
        result={}
        for key,value in pairs:require(key not in result,'duplicate JSON key');result[key]=value
        return result
    return json.loads(path.read_text(),object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def save(path,value,*,exclusive=False):
    path=Path(path)
    if exclusive:require(not path.exists()and not path.is_symlink(),'preserve existing receipt: '+str(path))
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w')as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)
def sha(path,check=lambda:None):
    path=Path(path);require(path.is_file()and not path.is_symlink(),'regular immutable file required: '+str(path))
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(4<<20),b''):check();h.update(block)
    return h.hexdigest()
def identity(path,check=lambda:None):return {'bytes':Path(path).stat().st_size,'sha256':sha(path,check)}
def copy_bound_json(source,destination,expected,limit=4<<20):
    read(source,limit);raw=Path(source).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected,'whole bound JSON bytes changed before archival copy')
    with Path(destination).open('xb')as stream:stream.write(raw)
    require(sha(destination)==expected,'whole bound JSON archival copy differs')
def snapshot(path):
    st=Path(path).stat();return {'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev}
def clean(exit):
    return type(exit.get('returncode'))is int and exit['returncode']==0 and exit.get('timed_out')is False and exit.get('owned_tree_quiescent')is True and exit.get('orphan_cleanup')is False and exit.get('guard_abort_reason')is None
def valid_sha(value):return isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None
def verify_sources(expected):
    require(valid_sha(expected)and sha(HERE/'SOURCE_MANIFEST.json')==expected,'root-pinned controller source manifest differs')
    rows=read(HERE/'SOURCE_MANIFEST.json')['files'];seen=set()
    for row in rows:
        name=row['path'];path=PurePosixPath(name)
        require(str(path)==name and not path.is_absolute()and '..'not in path.parts and name not in seen,'canonical unique source paths required');seen.add(name)
        target=HERE/name;require(target.resolve().is_relative_to(HERE)and identity(target)=={k:row[k]for k in('bytes','sha256')},'controller/candidate source changed: '+name)
    return {'source_manifest_sha256':expected,'files':len(rows)}
def verify_admission(path,expected,args,*,hostname=None,now=None):
    require(valid_sha(expected)and sha(path)==expected,'independent root admission bytes/SHA differ')
    value=read(path,64<<10)
    require(value['kind']=='PENCIL_CLEAN_CONSUMER_V1_ROOT_ADMISSION'and value['authorized_to_execute']is True
        and value['fresh_owned_statefulset']is True,'fresh explicit root execution admission required')
    require(value['namespace']=='m-utushkin'and value['pod_name']==POD and value['statefulset_name']==STS,'new planned owned pod identity differs')
    for key in('pod_uid','statefulset_uid'):require(str(uuid.UUID(value[key]))==value[key],'literal observed UUID required: '+key)
    host=hostname if hostname is not None else socket.gethostname()
    require(value['hostname']==host==POD,'actual fresh controller hostname differs from root observation')
    require(value['base']==str(args.base.resolve())and value['source_manifest_sha256']==args.source_manifest_sha256,'root source/base admission differs')
    require(value['source_packet_sha256']==args.source_packet_sha256,'independent root source packet pin differs')
    require(value['owned_tree_guard_gib']==args.memory_guard_gib==96 and value['pod_memory_limit_gib']==args.pod_memory_limit_gib==256,'fixed96GiB guard/256GiB pod admission required')
    require(value['gpu_count']==1 and value['gpu_name_contains']=='H100'and value['cpu_request']==27 and value['memory_request_gib']==205,'fixed reviewed oneH10080Single27CPU205/256GiB row required')
    current=time.time()if now is None else now
    require(type(value['observed_unix'])in(int,float)and math.isfinite(value['observed_unix'])and 0<=current-value['observed_unix']<=3600,'root pod observation must be fresh within1hour')
    require(isinstance(value['image'],str)and value['image'],'root-observed image identity required; actual package stack is independently probed')
    return value
def verify_inputs(prepared,validation,check=lambda:None):
    prepared=Path(prepared).resolve();validation=Path(validation).resolve();rows=read(HERE/'metadata/citation2_prepared_expected.json')['files']
    require(len(rows)==19 and len({r['path']for r in rows})==19,'all19 public numeric identities required');observed={}
    for row in rows:
        relative=PurePosixPath(row['path']);require(not relative.is_absolute()and '..'not in relative.parts,'relative public numeric path required')
        path=prepared/row['path'];require(path.resolve().is_relative_to(prepared),'numeric input path escape')
        before=snapshot(path);require(identity(path,check)=={k:row[k]for k in('bytes','sha256')}and snapshot(path)==before,'entire public numeric payload differs: '+row['path'])
        observed[str(path)]={**before,'sha256':row['sha256']}
    contract=read(BRIDGE/'input_contract.json')
    for name,key in(('valid_edge.npy','valid_positive'),('valid_edge_neg.npy','valid_negative')):
        path=validation/name;before=snapshot(path);require(sha(path,check)==contract['payloads'][key]['sha256']and snapshot(path)==before,'exact512-source validation cohort differs')
        observed[str(path)]={**before,'sha256':contract['payloads'][key]['sha256']}
    manifest=prepared/'manifest.json';strict=read(manifest,16<<20);strict=strict.get('dataset_provenance',{}).get('content',strict)
    declared=strict.get('files',[]);lookup={r['path']:{k:r[k]for k in('bytes','sha256')}for r in declared}
    require(len(declared)==len(lookup)==19 and lookup=={r['path']:{k:r[k]for k in('bytes','sha256')}for r in rows},'strict-clean manifest must declare all19 exact public payloads')
    require(strict.get('variant')=='strict_clean'and strict.get('quality_dataset')is True,'strict-clean public context graph required')
    for split in('valid','test'):
        audit=strict.get('clean_heldout_audits',{}).get('train_csr',{}).get(split,{})
        require(audit.get('forward')==audit.get('reverse')==0,'bidirectional held-out cleaning audit required')
    return {'status':'pass','numeric_payloads':19,'validation_sources':512,'negatives_per_source':1000,'actual_files':observed,'strict_manifest_path':str(manifest),'strict_manifest':identity(manifest),'scope':'Full19 numeric hashes plus exact frozen512-source payload hashes; no data fetch, fit or test ranking.'}
def verify_input_continuity(receipt):
    for name,row in receipt['actual_files'].items():require(snapshot(name)=={k:row[k]for k in('bytes','mtime_ns','inode','device')},'admitted numeric input changed')
    require(identity(receipt['strict_manifest_path'])==receipt['strict_manifest'],'strict-clean metadata changed')
    return True
