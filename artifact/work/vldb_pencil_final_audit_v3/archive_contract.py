"""Read-only verification of the existing collector export; no extraction/network."""
from pathlib import Path,PurePosixPath
import hashlib,json,re,tarfile

PREFIX='results/pencil_bridge_v3/'
REMOTE_OUT='results/pencil_bridge_v3_20261125'
BASE='/home/jovyan/seal_revision_20261124'
YT='//home/ucp-research/projects/graph_baselines/seal_revision_20261124/pencil_bridge_v3_20261125.tgz'
MAX_ARCHIVE=128<<20

class InvalidEvidence(ValueError):pass

def require(ok,message):
    if not ok:raise InvalidEvidence(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def regular(path,root=None):
    p=Path(path);require(p.is_file()and not p.is_symlink(),'regular evidence file required: '+str(p))
    if root is not None:require(p.resolve().is_relative_to(Path(root).resolve()),'evidence path escapes local export')
    return p
def read(path,maximum=32<<20):
    p=regular(path);require(p.stat().st_size<=maximum,'bounded JSON required')
    value=json.loads(p.read_text(),parse_constant=lambda value:(_ for _ in()).throw(InvalidEvidence('nonfinite JSON value: '+value)))
    require(isinstance(value,dict),'JSON object required');return value
def identity(path):
    p=regular(path);return {'bytes':p.stat().st_size,'sha256':sha(p)}
def valid_sha(value):return isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None

def verify_export(folder,queue_sha,root_launch=None):
    folder=Path(folder).resolve();export=read(folder/'export.json',2<<20)
    require(export.get('status')=='verified_existing_persisted_archive'and export.get('role')=='pencil'and
        export.get('all_internal_manifest_files_verified')is True,'fully verified final PENCIL collector export required')
    require(export.get('remote_archive_created')is False and all(export.get(k)==0 for k in('backup_calls','YT_writes','pod_lifecycle_mutations')),
        'collector must describe existing persisted data without remote mutations')
    require(type(export.get('scientific_success'))is bool and type(export.get('controller_success'))is bool,'explicit scientific/controller outcomes required')
    pointer=export['pointer'];packed=export['archive_metadata'];download=export['download']
    require(pointer.get('yt_path')==export.get('yt_path')==YT and pointer.get('readback_verified')is True and pointer.get('new_YT_file_nodes')==1,
        'one exact readback-verified final PENCIL YT pointer required')
    require(type(pointer.get('bytes'))is int and 0<pointer['bytes']<=MAX_ARCHIVE and valid_sha(pointer.get('sha256')),'bounded archive identity required')
    require(all(pointer[k]==packed[k]==download[k]for k in('bytes','sha256'))and download['file_count']==packed['files']+1,'archive/pointer/count contradiction')
    snapshot=regular(folder/'final_snapshot.tgz',folder);require(identity(snapshot)=={k:pointer[k]for k in('bytes','sha256')},'immutable final archive hash/bytes differ')
    extraction=read(folder/'extraction_manifest.json');declared={x['path']:{'bytes':x['bytes'],'sha256':x['sha256']}for x in extraction['files']}
    require(len(declared)==len(extraction['files'])==download['file_count'],'complete unique extraction manifest required')
    entries={};total=0
    with tarfile.open(snapshot,'r:gz')as tf:
        for item in tf:
            name=PurePosixPath(item.name)
            require(item.isfile()and item.name not in entries and not name.is_absolute()and'..'not in name.parts and'\\'not in item.name,
                'unique bounded regular archive member required')
            require(item.name=='SHA256_MANIFEST.json'or item.name.startswith(PREFIX),'PENCIL-only result namespace required')
            require(0<=item.size<=32<<20,'single whole evidence member32MiB limit')
            total+=item.size;require(total<=129<<20 and len(entries)<4096,'bounded whole archive payload/count')
            h=hashlib.sha256();count=0
            with tf.extractfile(item)as stream:
                for block in iter(lambda:stream.read(1<<20),b''):count+=len(block);h.update(block)
            require(count==item.size,'truncated archive member');entries[item.name]={'bytes':count,'sha256':h.hexdigest()}
        require(set(entries)==set(declared)and entries==declared,'archive bytes differ from saved extraction manifest')
        mf=tf.getmember('SHA256_MANIFEST.json');require(mf.size<=1<<20,'internal manifest1MiB bound')
        manifest_bytes=tf.extractfile(mf).read();require(hashlib.sha256(manifest_bytes).hexdigest()==packed['manifest_sha256'],'internal SHA manifest changed')
        manifest=json.loads(manifest_bytes)['files'];require(manifest=={k:v for k,v in entries.items()if k!='SHA256_MANIFEST.json'},'internal manifest must name actual tar members exactly')
    extracted=folder/'extracted';require(extracted.is_dir()and not extracted.is_symlink(),'regular extraction root required')
    actual_files={str(p.relative_to(extracted))for p in extracted.rglob('*')if p.is_file()or p.is_symlink()}
    require(actual_files==set(entries),'extra/missing files in local extraction')
    for name,value in entries.items():require(identity(regular(extracted/name,extracted))==value,'local extracted file changed: '+name)
    payload=extracted/PREFIX;remote=folder/'remote_receipts'
    done=read(remote/REMOTE_OUT/'COMPLETE.json');science=read(payload/'SCIENTIFIC_TERMINAL.json');plan=read(payload/'plan.json')
    require(identity(remote/REMOTE_OUT/'SCIENTIFIC_TERMINAL.json')==identity(payload/'SCIENTIFIC_TERMINAL.json')and
        identity(remote/REMOTE_OUT/'plan.json')==identity(payload/'plan.json'),'observed/archived whole scientific marker or plan bytes differ')
    require(type(science.get('success'))is bool and type(done.get('success'))is bool and type(done.get('scientific_complete'))is bool and
        type(science.get('completed_jobs'))is int and 0<=science['completed_jobs']<=48,'scientific/controller outcomes and completed count require literal boolean/integer types')
    require(read(remote/REMOTE_OUT/'archive_pointer.json')==pointer and read(remote/REMOTE_OUT/'archive_local_receipt.json')==packed and done.get('archive_pointer')==pointer,
        'observed final pointer/local receipt/terminal contradiction')
    require(all(done.get(k)is True for k in('controller_finished','owned_tree_quiescent','timing_lock_released','persistence_complete')),
        'clean final process/lock/persistence terminal required')
    require(done.get('scientific_complete')==science.get('success')==export['scientific_success']and done.get('success')==export['controller_success'],
        'scientific/controller/export status contradiction')
    require(done['success']==(science['success']and done.get('failure')is None),'final success must include controller failure state')
    require(science.get('controller_jobs_finished')is True and science.get('owned_tree_quiescent')is True,'scientific jobs must be quiescent')
    controller=done['controller_identity'];require(controller==plan.get('controller_identity')and
        controller.get('source_sha256')==queue_sha and controller.get('entrypoint')==BASE+'/pencil_queue_v3_code/pencil_queue_v3.py'and
        type(controller.get('pid'))is int and type(controller.get('start_ticks'))is int,'frozen final controller identity required')
    launch=read(remote/'pencil_queue_v3_launch.json')
    require(all(launch.get(k)==controller[k]for k in('pid','start_ticks','source_sha256'))and controller['entrypoint']in launch.get('argv',[]),
        'actual launch PID/start/source differs from terminal')
    if root_launch is not None:
        anchor=read(root_launch);anchor=anchor.get('launch_receipt',anchor);require(anchor==launch,'independent root launch receipt changed')
    marker=export['receipt'];require(marker['path']==PREFIX+'SCIENTIFIC_TERMINAL.json'and entries[marker['path']]['sha256']==marker['sha256'],
        'observed scientific terminal hash differs')
    return {'folder':folder,'payload':payload,'export':export,'entries':entries,'science':science,'done':done,'plan':plan,'launch':launch,
        'integrity':{'status':'pass','archive':identity(snapshot),'manifest_files':len(entries)-1,'all_archive_and_extracted_file_hashes_exact':True,
            'root_launch_anchor_checked':root_launch is not None,'transport_success_is_scientific_success':False}}
