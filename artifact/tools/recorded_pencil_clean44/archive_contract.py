"""Independent read-only tar + outside-tar structured attribute + root anchor verifier."""
from pathlib import Path,PurePosixPath
import hashlib,json,math,re,tarfile,uuid
BASE='/home/jovyan/seal_revision_20261124'
REMOTE_CODE=BASE+'/pencil_clean_controller_v1_code'
REMOTE_BRIDGE=REMOTE_CODE+'/candidate/bridge/vldb_pencil_bridge_clean_v1'
REMOTE_OUT=BASE+'/results/pencil_clean_consumer_v1_20261125'
YT='//home/ucp-research/projects/graph_baselines/seal_revision_20261124/pencil_clean_consumer_v1_20261125.tgz'
ATTRIBUTE='lpp_clean_consumer_persistence_v1'
MAX_ARCHIVE=384<<20
class InvalidEvidence(ValueError):pass
def require(ok,message):
    if not ok:raise InvalidEvidence(message)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def regular(path,root=None):
    p=Path(path);require(p.is_file()and not p.is_symlink(),'regular evidence file required: '+str(p))
    if root is not None:require(p.resolve().is_relative_to(Path(root).resolve()),'evidence path escapes local export')
    return p
def decode(raw):
    def unique(pairs):
        out={}
        for k,v in pairs:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(InvalidEvidence('nonfinite JSON')))
def read(path,maximum=32<<20):
    p=regular(path);require(p.stat().st_size<=maximum,'bounded JSON required');value=decode(p.read_text());require(isinstance(value,dict),'JSON object required');return value
def identity(path):
    p=regular(path);return {'bytes':p.stat().st_size,'sha256':sha(p)}
def valid_sha(v):return isinstance(v,str)and re.fullmatch('[0-9a-f]{64}',v)is not None
def path_name(name):
    p=PurePosixPath(name);require(str(p)==name and not p.is_absolute()and '..'not in p.parts and '\\'not in name and len(name)<=512,'canonical relative archive path required');return name

def root_admission(value,launch,contract):
    require(value['kind']=='PENCIL_CLEAN_CONSUMER_V1_ROOT_ADMISSION'and value['authorized_to_execute']is True and value['fresh_owned_statefulset']is True,'fresh root execution admission required')
    require(value['namespace']=='m-utushkin'and value['pod_name']==value['hostname']==launch['hostname']=='seal-clean-consumer-20261125-0'and value['statefulset_name']=='seal-clean-consumer-20261125','fresh actual pod/host identity differs')
    for k in('pod_uid','statefulset_uid'):require(isinstance(value[k],str)and str(uuid.UUID(value[k]))==value[k],'actual literal pod/STS UUID required')
    require(value['base']==BASE and value['source_manifest_sha256']==contract['source_manifest_sha256']and value['source_packet_sha256']==contract['source_packet_sha256'],'fresh root source/base pins differ')
    for k,wanted in {'gpu_count':1,'cpu_request':27,'memory_request_gib':205,'pod_memory_limit_gib':256,'owned_tree_guard_gib':96}.items():require(type(value[k])is int and value[k]==wanted,'root hardware admission differs: '+k)
    require(value['gpu_name_contains']=='H100'and isinstance(value['image'],str)and bool(value['image']),'root actual H100/image observation missing')
    require(type(value['observed_unix'])in(int,float)and math.isfinite(value['observed_unix'])and type(launch['started_unix'])in(int,float)and math.isfinite(launch['started_unix'])and 0<=launch['started_unix']-value['observed_unix']<=3600,'actual launch not within fresh root observation window')
    return value

def verify_export(folder,contract,root_launch,root_admission_path):
    folder=Path(folder).resolve();ex=read(folder/'export.json',2<<20)
    require(ex.get('status')=='verified_existing_persisted_archive'and ex.get('role')=='pencil_clean_consumer'and ex.get('all_internal_manifest_files_verified')is True,'verified existing clean-consumer collection required')
    require(ex.get('remote_archive_created')is False and all(type(ex.get(k))is int and ex[k]==0 for k in('backup_calls','YT_writes','pod_lifecycle_mutations')),'collection must be read-only without lifecycle/writes')
    require(ex.get('yt_path')==YT and ex.get('attribute_name')==ATTRIBUTE,'exact one-file/one-attribute boundary differs')
    snapshot=regular(folder/'final_snapshot.tgz',folder);packed=identity(snapshot)
    require(type(packed['bytes'])is int and 0<packed['bytes']<=MAX_ARCHIVE and ex['archive']==packed,'bounded complete collected archive identity differs')
    attr=read(folder/'persistence_attribute.json',32<<10);attribute_sha=hashlib.sha256(canonical(attr).encode()).hexdigest()
    require(ex['attribute_canonical_json_sha256']==attribute_sha,'independently collected outside-tar attribute SHA differs')
    require(set(attr)=={'schema_version','kind','archive_sha256','archive_bytes','scientific_terminal_sha256','canonical_receipt_json','canonical_receipt_sha256','outer_final_pid_exit'}and type(attr['schema_version'])is int and attr['schema_version']==1 and attr['kind']==ATTRIBUTE,'exact structured persistence attribute required')
    require(type(attr['archive_bytes'])is int and attr['archive_bytes']==packed['bytes']and attr['archive_sha256']==packed['sha256']and attr['outer_final_pid_exit']=='UNKNOWN','attribute/archive/explicit final-PID boundary differs')
    text=attr['canonical_receipt_json'];require(isinstance(text,str)and hashlib.sha256(text.encode()).hexdigest()==attr['canonical_receipt_sha256'],'canonical post-readback receipt SHA differs')
    receipt=decode(text);require(canonical(receipt)==text and receipt['archive']==packed and {k:receipt['readback'][k]for k in('bytes','sha256')}==packed and receipt['target']==YT and receipt['outer_final_pid_exit']=='UNKNOWN','actual canonical whole-file readback binding differs')
    require(type(receipt['readback']['verified_unix'])in(int,float)and math.isfinite(receipt['readback']['verified_unix']),'literal finite persistence observation required')
    declared_rows=read(folder/'extraction_manifest.json')['files'];declared={path_name(r['path']):{k:r[k]for k in('bytes','sha256')}for r in declared_rows};require(len(declared)==len(declared_rows),'unique complete local extraction manifest required')
    entries={};manifest=None;total=0
    with tarfile.open(snapshot,'r|gz')as source:
        for member in source:
            name=path_name(member.name);require(member.isfile()and name not in entries and 0<=member.size<=32<<20,'unique bounded regular archive members required')
            total+=member.size;require(total<=MAX_ARCHIVE and len(entries)<4096,'bounded whole evidence payload/count required')
            h=hashlib.sha256();count=0;raw=[] if name=='SHA256_MANIFEST.json'else None
            with source.extractfile(member)as stream:
                for block in iter(lambda:stream.read(4<<20),b''):
                    count+=len(block);h.update(block)
                    if raw is not None:require(count<=2<<20,'bounded internal manifest required');raw.append(block)
            require(count==member.size,'truncated archive member');entries[name]={'bytes':count,'sha256':h.hexdigest()}
            if raw is not None:manifest=decode(b''.join(raw))
    require(entries==declared and manifest is not None,'whole archive differs from complete extraction manifest')
    require(manifest['kind']=='whole_clean_consumer_evidence_v1','new exact internal manifest schema required')
    rows=manifest['files'];internal={path_name(r['path']):{k:r[k]for k in('bytes','sha256')}for r in rows}
    require(len(internal)==len(rows)==manifest['file_count']and internal=={k:v for k,v in entries.items()if k!='SHA256_MANIFEST.json'}and manifest['payload_bytes']==sum(v['bytes']for v in internal.values()),'internal SHA manifest must bind all whole evidence members exactly')
    extracted=folder/'extracted';require(extracted.is_dir()and not extracted.is_symlink(),'regular extracted root required')
    observed={str(p.relative_to(extracted))for p in extracted.rglob('*')if p.is_file()or p.is_symlink()};require(observed==set(entries),'extra/missing/symlink extracted files')
    for name,value in entries.items():require(identity(regular(extracted/name,extracted))==value,'whole extracted member differs: '+name)
    launch=read(extracted/'LAUNCH.json');science=read(extracted/'SCIENTIFIC_TERMINAL.json');plan=read(extracted/'plan.json');admission=read(extracted/'ROOT_ADMISSION.json')
    require(root_launch is not None and root_admission_path is not None,'independent actual root launch and whole admission anchors are mandatory')
    anchor=read(root_launch);require(anchor.get('launch_receipt',anchor)==launch,'independent actual root launch anchor differs')
    require(identity(root_admission_path)==identity(extracted/'ROOT_ADMISSION.json')and read(root_admission_path)==admission,'independent whole root admission bytes differ')
    require(sha(extracted/'ROOT_ADMISSION.json')==launch['root_admission_sha256']==science['root_admission_sha256']==receipt['root_admission_sha256'],'whole root admission linkage differs')
    root_admission(admission,launch,contract)
    require(launch['kind']=='actual_clean_consumer_controller_launch'and launch['entrypoint']==REMOTE_CODE+'/controller.py'and launch['source_sha256']==contract['controller_source_sha256'],'actual frozen controller source/entrypoint differs')
    owner=launch['identity'];require(type(owner['pid'])is int and owner['pid']>0 and type(owner['start_ticks'])is int and owner['start_ticks']>=0,'actual PID/start identity required')
    require(sha(extracted/'SOURCES.json')==contract['source_manifest_sha256']and read(extracted/'SOURCES.json')['files']==contract['source_files'],'whole archived108-file source manifest differs')
    for k in('source_manifest_sha256','source_packet_sha256'):require(launch[k]==science[k]==receipt[k]==contract[k],'source packet/manifest linkage differs')
    require(sha(extracted/'LAUNCH.json')==science['launch_sha256']==receipt['launch_sha256']and receipt['controller_source_sha256']==contract['controller_source_sha256'],'whole launch/source linkage differs')
    require(sha(extracted/'SCIENTIFIC_TERMINAL.json')==attr['scientific_terminal_sha256']==receipt['scientific_terminal_sha256'],'whole scientific terminal linkage differs')
    require(receipt['pod_uid']==admission['pod_uid']and receipt['statefulset_uid']==admission['statefulset_uid'],'durable actual root pod/STS linkage differs')
    require(science['kind']=='PENCIL_CLEAN_CONSUMER_SCIENTIFIC_TERMINAL_V1'and type(science['scientific_success'])is bool and type(science['completed_jobs'])is int and 0<=science['completed_jobs']<=44 and science['planned_jobs']==receipt['planned_jobs']==44,'literal scientific outcome/planned count required')
    require(type(science['all_planned_science_job_exits_clean'])is bool and type(science['stop_requested'])is bool and science['outer_final_pid_exit']=='UNKNOWN'and science['owned_children_quiescent_before_publication']is True,'quiescent scientific terminal and explicit unknown outer PID required')
    require(receipt['scientific_success']is science['scientific_success']and receipt['all_planned_science_job_exits_clean']is science['all_planned_science_job_exits_clean']and receipt['completed_jobs']==science['completed_jobs'],'durable terminal outcome contradiction')
    if science['scientific_success']:require(science['completed_jobs']==44 and science['all_planned_science_job_exits_clean']is True and science['failure']is None and science['stop_requested']is False,'scientific success must include all44 clean jobs and no failure/stop')
    require(ex['scientific_success']is science['scientific_success'],'collector science outcome differs')
    return {'folder':folder,'payload':extracted,'export':ex,'science':science,'plan':plan,'launch':launch,'admission':admission,
        'integrity':{'status':'pass','archive':packed,'manifest_files':len(internal),'all_archive_and_extracted_file_hashes_exact':True,'root_launch_and_whole_admission_anchors_checked':True,'attribute_canonical_json_sha256':attribute_sha,'attribute_is_inside_tar_hash':False,'outer_final_pid_exit':'UNKNOWN','transport_success_is_scientific_success':False}}
