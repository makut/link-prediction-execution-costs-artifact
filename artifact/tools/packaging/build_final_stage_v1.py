#!/usr/bin/env python3
"""Plan or stage a NEW v0.5 snapshot. No archive freeze, git or publication API.

Preserves the exact471-payload earlier candidate and immutable v0.4. Only
explicit generated metadata is replaced, with original bytes retained. Large
raw evidence/wheels/models remain separate asset references; no graph load.
"""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,os,shutil
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PORTABLE=ROOT/'work/vldb_artifact_v0_5_portable_v1'
BASE=ROOT/'work/vldb_artifact_v0_5_candidate'
BASE_MANIFEST_SHA='b4aae9e737f743e1fefef245bcf7da9dfcf2fa2ee58e8329304cbb114cdb8bec'
CPU_MANIFEST_SHA='fe94c5c9138dfdbb6c4d7dd195ab4aa54531dca7db350fbdb73228c5de3304b4'
PENCIL_MANIFEST_SHA='250f034c2dcd77d4e1bd266f3c01de3ff55f8a6324e1944f141f57c2751352b1'
CLEAN_AUDIT_SHA='4682eef72971ab64f25d64cf2823f3492602497e36580e001751cf799eb9014f'
CLEAN_PEER_SHA='9ef11d0164a22b7d1ab65eb96fe2be3dabd095defe76158ef98a8b02b9c6dbac'
RESERVE=int(1.5*(1<<30));MAX_CODE=384*(1<<20)
ASSETS={
 'public_quality':('outputs/vldb_public_quality_evidence_20261125/lpp_public_quality_evidence_v1.tgz',404147012,'e17cdc9678d00a3f372fc8083741a200a63662226f730f47eeda79f89b4ad3b2','lpp_public_quality_evidence_v1.tgz'),
 'public_systems':('outputs/vldb_public_systems_evidence_20261125/lpp_public_systems_evidence_v1.tgz',83629661,'48232549089f3295c893bb33f6cea0d1aac4a0ee45b91a2669392c90ae1bfe40','lpp_public_systems_evidence_v1.tgz'),
 'cpu_E20_raw':('outputs/seal_after_travel_systems_final_20261125/public_fr_cpu_matched_v2_final_readonly_v1/final_snapshot.tgz',2121570,'546cad6f6b14c9f56844dc3414804e97f0dd303028532910dd74bee9d10f9374','lpp_public_friendster_cpu_matched_v2_raw.tgz'),
 'cpu_pt22_wheels':('work/vldb_cpu_native_wheels_pt22_v1/pt22_native_wheels_v1.tgz',15064339,'5e9edbdc067325cd0b41b5fb292e2cc7343cc535da87ff3bef720cbf85da38cf','pt22_native_wheels_v1.tgz'),
 'pencil_E19_instrumented48_raw':('outputs/seal_after_travel_systems_final_20261125/pencil_bridge_v3_final_readonly_v1/final_snapshot.tgz',9966534,'7ab909e987af900f8ad5f9ea9a559e17e650f78891cb0bc99200a0fdb97fa36c','lpp_public_pencil_instrumented48_raw.tgz'),
 'pencil_E21_clean44_raw':('outputs/seal_after_travel_systems_final_20261125/pencil_clean_consumer_v1_final_readonly_v1/final_snapshot.tgz',10230546,'708b6f32081d187dbe5e092b6782a36002333c3f8b0c5ff7dcbf20ec48c156b0','lpp_public_pencil_clean44_raw.tgz')}
OVERRIDES=('README.md','MANIFEST.json','CITATION.cff','RELEASE_PLAN_v0_5.json','metadata/CANDIDATE_STATUS.json','metadata/V0_5_GATES.json','metadata/SYSTEMS_FROZEN_INPUTS.json','metadata/SYSTEMS_SOURCE_SELECTION.json','LICENSE_PENDING.txt')

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as s:
        for b in iter(lambda:s.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def pin(path):
    p=Path(path);require(p.is_file()and not p.is_symlink(),'regular immutable input required');return{'bytes':p.stat().st_size,'sha256':sha(p)}
def read(path):
    p=Path(path);require(p.stat().st_size<=4<<20,'bounded staging metadata required')
    return json.loads(p.read_bytes(),parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def encoded(value):return(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def safe(name):
    p=PurePosixPath(name);require(name and str(p)==name and not p.is_absolute()and'..'not in p.parts and'\\'not in name and'\0'not in name,'canonical stage path required')
    require('__pycache__'not in p.parts and'.pytest_cache'not in p.parts and not name.endswith(('.pyc','.so','.dylib','.pt','.pth','.ckpt')),'binary/cache/model not admitted into code snapshot')
    if name.startswith(('data/','results/','metadata/evidence/')):
        require(not any(part.lower()=='vk'or part.lower().startswith('vk_')for part in p.parts),'private raw data namespace refused')
    return name

def clean_evidence(audit,peer):
    require((audit is None)==(peer is None),'supply both exact clean audit and numeric peer or neither')
    if audit is None:return{'accepted':False,'reason':'actual clean44 audit/numeric peer not supplied'}
    a=read(audit);p=read(peer);ap=pin(audit)
    require(ap['sha256']==CLEAN_AUDIT_SHA and pin(peer)['sha256']==CLEAN_PEER_SHA,'exact root-accepted clean44 independent audit and numeric peer pins required')
    require(a.get('status')=='scientific_checks_passed'and a.get('scientific_success')is True and a.get('completed_jobs')==a.get('planned_jobs')==44,'exact actual clean44 successful frozen audit required')
    require(p.get('status')=='PASS'and ap['sha256']in json.dumps(p)and ASSETS['pencil_E21_clean44_raw'][2]in json.dumps(p),'independent raw peer must bind exact clean44 actual audit and archive')
    return{'accepted':True,'audit_path':str(Path(audit).resolve()),'audit':ap,'numeric_peer_path':str(Path(peer).resolve()),'numeric_peer':pin(peer)}

def verify_baseline():
    require(sha(BASE/'MANIFEST.json')==BASE_MANIFEST_SHA,'saved original471manifest changed')
    original=read(BASE/'MANIFEST.json');require(len(original['files'])==471,'whole471baseline required')
    for name,row in original['files'].items():require(pin(BASE/name)=={k:row[k]for k in('bytes','sha256')},'original471payload changed: '+name)
    v04=ROOT/'outputs/vldb_submission_2026/lpp_vldb_artifact_v0.4.0.tgz';original_v04=pin(v04)
    require(original_v04=={'bytes':76637178,'sha256':'8657a980e7efe1266f30022223c9dffde75882c53c1e55e6b6dd9ec160470404'},'immutable v0.4 archive changed')
    return{'kind':'ACTUAL_POST_STAGE_BASELINE_PRESERVATION_V1','status':'PASS','all_original471_payload_bytes_rehashed_unchanged':True,'original471manifest_sha256':BASE_MANIFEST_SHA,'v0_4_archive_unchanged':original_v04,
      'write_strategy':'All current replacement metadata written as fresh new files; historical originals linked only at separate prior-snapshot destinations. No write-through to a source hardlink.','remote_actions':0,'YT_writes':0}

def plan(clean_audit=None,clean_peer=None):
    require(sha(BASE/'MANIFEST.json')==BASE_MANIFEST_SHA,'exact prior471candidate manifest required')
    old=read(BASE/'MANIFEST.json');require(old['file_count']==len(old['files'])==471,'complete previous471payload snapshot required')
    files={};generated={};origins={}
    def add(name,path,identity=None,kind='unchanged_explicit_input'):
        safe(name);require(name not in files and name not in generated,'explicit stage destination collision: '+name)
        actual=pin(path);require(identity is None or actual=={k:identity[k]for k in('bytes','sha256')},'whole input bytes differ: '+str(path))
        files[name]={'path':str(Path(path).resolve()),**actual,'lineage':kind}
    def gen(name,value):
        safe(name);require(name not in files and name not in generated,'generated metadata collision: '+name)
        b=value if isinstance(value,bytes)else encoded(value);generated[name]=b
    def source_tree(folder,target):
        manifest=read(folder/'SOURCE_MANIFEST.json');rows=manifest['files']
        rows={row['path']:{k:row[k]for k in('bytes','sha256')}for row in rows}if isinstance(rows,list)else rows
        for name,row in rows.items():add(target+'/'+name,folder/name,row,'unchanged_frozen_offline_audit_dependency')
        add(target+'/SOURCE_MANIFEST.json',folder/'SOURCE_MANIFEST.json')
    def recorded_metadata(role,folder):
        rows={}
        for path in sorted(folder.rglob('*')):
            if not path.is_file()or path.name=='final_snapshot.tgz'or'extracted'in path.relative_to(folder).parts:continue
            name=path.relative_to(folder).as_posix();add('metadata/evidence/'+role+'/recorded_export/'+name,path,kind='unchanged_genuine_recorded_export_sidecar');rows[name]=pin(path)
        gen('metadata/evidence/'+role+'/recorded_export/FILE_INVENTORY.json',{'role':role,'files':rows,'archive_separate':True,'new_observations':0})
    for name,row in old['files'].items():
        destination='metadata/lineage/PRIOR_V0_5_CANDIDATE/'+name if name in OVERRIDES else name
        add(destination,BASE/name,row,'unchanged_prior471candidate_payload')
    add('metadata/lineage/PRIOR_V0_5_CANDIDATE/MANIFEST.json',BASE/'MANIFEST.json',kind='exact_prior471manifest_record')
    for dirname,expected in(('public_friendster_cpu_v1',CPU_MANIFEST_SHA),('public_pencil_clean_v1',PENCIL_MANIFEST_SHA)):
        folder=PORTABLE/dirname;require(sha(folder/'SOURCE_MANIFEST.json')==expected,'reviewed portable source manifest differs')
        manifest=read(folder/'SOURCE_MANIFEST.json');rows=manifest['files']
        if isinstance(rows,list):rows={row['path']:{k:row[k]for k in('bytes','sha256')}for row in rows}
        for name,row in rows.items():add('reproduction/'+dirname+'/'+name,folder/name,row,'reviewed_generic_public_adapter_unchanged')
        add('reproduction/'+dirname+'/SOURCE_MANIFEST.json',folder/'SOURCE_MANIFEST.json',kind='exact_portable_manifest')
    # Exact actual CPU closure and explicit transport correction; no new audit.
    cpu_report=ROOT/'outputs/vldb_public_fr_cpu_audit_20261125/ACTUAL_PUBLIC_FR_CPU_V2_AUDIT_v2_transport_adapter.json'
    cpu_peer=ROOT/'work/vldb_public_fr_cpu_numeric_peer_v1/ACTUAL_RAW_NUMERIC_LIFECYCLE_PEER_v1.json'
    require(sha(cpu_report)=='bd96e6dc5d55af0e8cd7543bc51b1d6a43230e8f43664648acabd1019fd6bb73'and sha(cpu_peer)=='561c170652118da014502ae23d53537d8b267d07d0fc665e220a293bb6107b61','exact independently accepted E20 closure required')
    add('metadata/evidence/E20/ACTUAL_INDEPENDENT_AUDIT.json',cpu_report)
    add('metadata/evidence/E20/ACTUAL_RAW_NUMERIC_PEER.json',cpu_peer)
    add('metadata/evidence/E20/ROOT_LAUNCH.json',ROOT/'work/vldb_gpu_recovery/public_fr_cpu_matched_v2_launch.json')
    add('metadata/evidence/E20/public_fr_cpu_v2_sources.tgz',ROOT/'work/vldb_public_fr_cpu_v2_sources.tgz',{'bytes':1781157,'sha256':'f79cd610e8f24830b896ed4e8b3a37035ab31d0626349699c3b568ebe69c118f'})
    add('metadata/evidence/E20/FIRST_FROZEN_TRANSPORT_SCHEMA_FAIL.json',ROOT/'outputs/vldb_public_fr_cpu_audit_20261125/ACTUAL_PUBLIC_FR_CPU_V2_AUDIT_v1.json')
    add('metadata/evidence/E20/PUBLIC_RAW_ALLOWLIST_REVIEW.json',PORTABLE/'CPU_E20_RAW_PUBLIC_ALLOWLIST_REVIEW_v1.json')
    original=ROOT/'outputs/seal_after_travel_systems_final_20261125/public_fr_cpu_matched_v2_final_readonly_transport_adapter_v1'
    metadata={}
    for p in sorted(original.rglob('*')):
        if not p.is_file()or p.name=='final_snapshot.tgz':continue
        name=p.relative_to(original).as_posix();add('metadata/evidence/E20/recorded_export/'+name,p,kind='unchanged_genuine_or_losslessly_derived_recorded_transport');metadata[name]=pin(p)
    gen('metadata/evidence/E20/recorded_export/FILE_INVENTORY.json',{'files':metadata,'archive_separate':True,'numeric_observations_added':0})
    for dirname,target in(('vldb_public_fr_cpu_audit_v1','tools/recorded_cpu_v2'),('vldb_public_fr_cpu_export_adapter_v1','tools/recorded_cpu_transport')):
        folder=ROOT/'work'/dirname
        for p in sorted(folder.iterdir()):
            if p.is_file()and p.suffix in('.py','.json'):add(target+'/'+p.name,p,kind='unchanged_frozen_E20_audit_or_transport_source')
    add('tools/public_evidence/materialize_cpu_e20_v1.py',HERE/'materialize_cpu_e20_v1.py',kind='new_local_lossless_recorded_asset_materializer')
    add('tools/public_evidence/materialize_pencil_export_v1.py',HERE/'materialize_pencil_export_v1.py',kind='new_offline_lossless_recorded_export_extractor')
    add('tools/public_evidence/extract_friendster_rr_v1.py',HERE/'extract_friendster_rr_v1.py',kind='bounded_small_original_quality_member_extractor_no_numeric_change')
    add('tools/packaging/freeze_local_archive_v1.py',HERE/'freeze_local_archive_v1.py',kind='local_only_regular_member_archive_writer_with_full_readback')
    add('tools/packaging/review_recorded_public_surface_v1.py',HERE/'review_recorded_public_surface_v1.py',kind='local_readonly_all_member_raw_evidence_review')
    add('tools/packaging/materialize_pencil_export_v1.py',HERE/'materialize_pencil_export_v1.py',kind='unchanged_transport_dependency_for_public_surface_review')
    # Packaging provenance is inspectable separately from public science entrypoints.
    for name in('build_final_stage_v1.py','test_final_stage_v1.py','materialize_cpu_e20_v1.py',
                'extract_friendster_rr_v1.py','run_final_offline_roundtrip_v1.py'):
        add('tools/packaging/'+name,HERE/name,kind='exact_local_packaging_or_verification_source_no_remote_execution')
    # E19 remains explicitly the old instrumented48job evidence; never renamed.
    add('metadata/evidence/E19/ACTUAL_DERIVED_AUDIT_V2.json',ROOT/'outputs/seal_after_travel_systems_final_20261125/pencil_bridge_v3_independent_audit_addendum_v2_actual_v1/audit.json',{'bytes':261382,'sha256':'ec3d25768c1c9dd3b444b84ee3420521053472d007c567c383b0abcf612f5ec4'})
    add('metadata/evidence/E19/ACTUAL_NUMBERS_AND_POSTHOC_PEER.json',ROOT/'work/vldb_pencil_replay_peer_v2/ACTUAL_NUMBERS_AND_POSTHOC_PEER_v2.json')
    add('metadata/evidence/E19/PUBLIC_RAW_ALLOWLIST_REVIEW.json',PORTABLE/'PENCIL_E19_RAW_PUBLIC_ALLOWLIST_REVIEW_v1.json')
    recorded_metadata('E19',ROOT/'outputs/seal_after_travel_systems_final_20261125/pencil_bridge_v3_final_readonly_v1')
    for dirname in('vldb_pencil_final_audit_v3','vldb_pencil_replay_addendum_v2','vldb_pencil_queue_v3'):
        source_tree(ROOT/'work'/dirname,'work/'+dirname)
    relocation=HERE/'recorded_pencil_e19_v1'
    require(sha(relocation/'SOURCE_MANIFEST.json')=='01b9530baedd8c48c7486fff18e7f9de33456c4ebe690473b4bac5b64507c4cc','reviewed E19 offline relocation source manifest differs')
    source_tree(relocation,'tools/recorded_pencil_e19_v1')
    for name in('ACTUAL_RELOCATED_COMMAND.json',):
        p=ROOT/'work/vldb_recorded_e19_adapter_roundtrip_v1'/name
        add('metadata/evidence/E19/OFFLINE_PORTABILITY/'+name,p)
    add('metadata/evidence/E15/HARDWARE_CURRENT_PROVENANCE.json',ROOT/'outputs/seal_revision_20261124/hardware_current_provenance_20261125.json',{'bytes':2355,'sha256':'74fff820a646974c0f6535b16558725c2b91edc81d1f60b6073bd2c0dd6d4115'})
    correction=read(ROOT/'work/vldb_pencil_replay_addendum_v2/CORRECTION.json')
    for key,name in(('original_failed_attempt','ORIGINAL_FIRST_FAILED_ATTEMPT.json'),('original_failure_stderr','ORIGINAL_FIRST_FAILURE_STDERR.txt')):
        row=correction[key];path=Path(row['path']);require(path.resolve().is_relative_to(ROOT),'historical failure anchor must be from this reviewed workspace')
        add('metadata/evidence/E19/'+name,path,row,'unchanged_original_first_failed_audit_anchor')
    recovery=ROOT/'outputs/seal_after_travel_systems_final_20261125/pencil_bridge_v3_final_readonly_v1/remote_receipts/RECOVERY_ADMISSION.json'
    require(sha(recovery)=='3589dda0795ca201057016fbd771776ea2838c70a902536d29d76e899f228f73','exact original E19 recovery anchor')
    add('metadata/evidence/E19/RECOVERY_ADMISSION.json',recovery)
    launch=ROOT/'outputs/seal_after_travel_systems_final_20261125/pencil_bridge_v3_final_readonly_v1/remote_receipts/pencil_queue_v3_launch.json'
    require(sha(launch)=='46e942a9e2bfa6f441869f15536cb6e2d2fe06d38c1acddacd3f297ecd5e6f62','exact original E19 root launch anchor')
    add('metadata/evidence/E19/ROOT_LAUNCH.json',launch)
    accepted=clean_evidence(clean_audit,clean_peer)
    if accepted['accepted']:
        add('metadata/evidence/E21/ACTUAL_INDEPENDENT_AUDIT.json',Path(accepted['audit_path']),accepted['audit'])
        add('metadata/evidence/E21/ACTUAL_RAW_NUMERIC_PEER.json',Path(accepted['numeric_peer_path']),accepted['numeric_peer'])
        add('metadata/evidence/E21/PUBLIC_RAW_ALLOWLIST_REVIEW.json',PORTABLE/'PENCIL_E21_RAW_PUBLIC_ALLOWLIST_REVIEW_v1.json')
        source_tree(ROOT/'work/vldb_pencil_clean_final_audit_v1','tools/recorded_pencil_clean44')
        source_tree(ROOT/'work/vldb_pencil_clean_controller_v1','tools/vldb_pencil_clean_controller_v1')
        add('tools/vldb_pencil_clean_controller_v1_sources.tgz',ROOT/'work/vldb_pencil_clean_controller_v1_sources.tgz',{'bytes':304148,'sha256':'8cacc3d5c5c61964d8d7b86c4aa21fef648476ef1d8ab415e91678beac13f75c'})
        recorded_metadata('E21',ROOT/'outputs/seal_after_travel_systems_final_20261125/pencil_clean_consumer_v1_final_readonly_v1')
        add('metadata/evidence/E21/ROOT_LAUNCH.json',ROOT/'work/vldb_clean_consumer_deploy_v1/ACTUAL_CONTROLLER_LAUNCH_v1.json')
        add('metadata/evidence/E21/ROOT_ADMISSION.json',ROOT/'work/vldb_clean_consumer_deploy_v1/ROOT_ADMISSION_v1.json')
    # Actual Linux portability receipts are separate from full reproduction.
    smoke=ROOT/'work/vldb_portable_target_smoke_v1/pencil_actual_linux_v1'
    for p in sorted(smoke.iterdir()):
        if p.is_file()and p.suffix in('.json','.log','.txt'):add('metadata/portability/PENCIL_LINUX_TINY/'+p.name,p)
    for dirname in('cpu_actual_linux_v1','cpu_actual_linux_v2','cpu_actual_linux_v3'):
        folder=ROOT/'work/vldb_portable_target_smoke_v1'/dirname
        if folder.is_dir():
            for p in sorted(folder.rglob('*')):
                if p.is_file()and p.suffix in('.json','.log','.txt'):add('metadata/portability/CPU_LINUX_TINY/'+dirname+'/'+p.relative_to(folder).as_posix(),p)
    for p in(PORTABLE/'CPU_PORTABLE_PACKET_READY_v1.json',PORTABLE/'ROOT_PORTABLE_CLEAN_SOURCE_PEER_v1.json'):
        if p.is_file():add('metadata/portability/'+p.name,p)
    catalog={}
    for key,(path,size,expected,filename)in ASSETS.items():
        actual=pin(ROOT/path);require(actual=={'bytes':size,'sha256':expected},'exact separate external asset differs: '+key)
        catalog[key]={'filename':filename,**actual,'local_source_path':path,'included_in_code_snapshot':False,'publication_state_when_created':'not_published','download_url_at_creation':None}
    gen('metadata/EXTERNAL_ASSETS.json',{'kind':'SEPARATE_IMMUTABLE_PUBLIC_EVIDENCE_AND_CPU_WHEEL_ASSETS','assets':catalog,'graphs_included':False,'publication_performed':False})
    add('metadata/external_assets/CPU_E20_RAW_NOTICE.txt',PORTABLE/'assets/CPU_E20_RAW_NOTICE.txt')
    add('metadata/external_assets/PENCIL_RECORDED_RAW_NOTICE.txt',PORTABLE/'assets/PENCIL_RECORDED_RAW_NOTICE.txt')
    score_checker=ROOT/'work/vldb_friendster_raw_score_check_v1'
    require(sha(score_checker/'SOURCE_MANIFEST.json')=='586e726e048d2449c7ed7865ac4f2d717305cd03ff1baa3d164c92a7e2b78a5b',
            'final independently accepted bounded Friendster full-score source/receipt required')
    source_tree(score_checker,'tools/recorded_friendster_scores_v1')
    for name in('FRIENDSTER_RR_REAGGREGATION_v1.json','CITATION2_FULL_SCORE_REAGGREGATION_v1.json'):
        add('metadata/evidence/QUALITY/OFFLINE_CLOSURE/'+name,ROOT/'work/vldb_final_public_package_peer_v1'/name,kind='actual_independent_offline_quality_reporting_command_receipt')
    add('PUBLIC_COMMANDS.txt',PORTABLE/'PUBLIC_COMMANDS.txt')
    add('PUBLIC_OBJECT_COMMANDS.json',PORTABLE/'PUBLIC_OBJECT_COMMANDS.json')
    gen('README.md',README.encode())
    citation=(BASE/'CITATION.cff').read_text().replace('version: 0.4.0-local-staged','version: 0.5.0').replace('Local staged research artifact; own code MIT, public release pending.','Research artifact; own code MIT. See the external asset catalog for exact evidence files.')
    gen('CITATION.cff',citation.encode())
    gates={'status':'ARTIFACT_V0_5_0_CREATION_RECORD','old471snapshot_preserved':True,'v0_4_unchanged':True,'E20_actual_independent_audit':'PASS','E20_raw_numeric_peer':'PASS','E19':'OLD_INSTRUMENTED48_ONLY','E21_actual_clean44':accepted,
      'portable_CPU_generic_source_manifest_sha256':CPU_MANIFEST_SHA,'portable_PENCIL_clean44_source_manifest_sha256':PENCIL_MANIFEST_SHA,'fresh_full_7arm_or44job_public_adapter_reruns_claimed':False,'fresh_public_independent_numeric_auditor_exists':False,'publication_and_durable_URLs_not_yet_created_at_packaging':True,'code_package_sha_recorded_in_external_archive_manifest':True,'git_actions':0,'remote_actions':0,'YT_writes':0}
    gen('metadata/V0_5_GATES.json',gates)
    gen('metadata/CANDIDATE_STATUS.json',{'version':'0.5.0','creation_role':'reviewed_local_release_package','publication_state_at_creation':'not_published','publisher_action_is_separate':True,'E19_actual_independently_closed':True,'E20_actual_independently_closed':True,'E21_actual_independently_closed':accepted['accepted'],'parent471snapshot_unchanged':True,'creation_record':'metadata/V0_5_GATES.json'})
    pins=read(BASE/'metadata/SYSTEMS_FROZEN_INPUTS.json');selection=read(BASE/'metadata/SYSTEMS_SOURCE_SELECTION.json')
    merged=dict(pins['files']);registry=dict(selection['sources'])
    for name,row in files.items():
        if row['lineage']=='unchanged_prior471candidate_payload':continue
        require(name not in merged and name not in registry,'new source registry entry cannot overwrite old pin')
        merged[name]=row['sha256'];registry[name]={'bytes':row['bytes'],'packaged_sha256':row['sha256'],'lineage':{'kind':row['lineage']}}
    gen('metadata/SYSTEMS_FROZEN_INPUTS.json',{**pins,'files':merged,'prior363pins_unchanged':True,'new_stage_does_not_establish_new_scientific_results':True})
    gen('metadata/SYSTEMS_SOURCE_SELECTION.json',{**selection,'sources':registry,'source_count':len(registry),'immutable_frozen_input_count':len(merged),'previous_entries_unchanged':True})
    gen('metadata/FINAL_STAGE_LINEAGE.json',{'base471manifest_sha256':BASE_MANIFEST_SHA,'base471_payloads_preserved_unchanged':471,'preserved_overrides_prefix':'metadata/lineage/PRIOR_V0_5_CANDIDATE/','immutable_v0_4_sha256':'8657a980e7efe1266f30022223c9dffde75882c53c1e55e6b6dd9ec160470404','source_adapters':{'cpu':CPU_MANIFEST_SHA,'pencil_clean44':PENCIL_MANIFEST_SHA},'code_archive_freeze_or_publication':False})
    total=sum(row['bytes']for row in files.values())+sum(len(b)for b in generated.values());require(total<=MAX_CODE,'bounded code snapshot384MiB required')
    return{'status':'PLAN_NEW_FINAL_STAGE_NOT_FROZEN','files':files,'generated':generated,'catalog':catalog,'gates':gates,'payload_bytes':total,'preserved_prior_payloads':471}

def stage(out,definition):
    require(definition['gates']['E21_actual_clean44']['accepted']is True,'final local snapshot requires exact closed E21 audit and numeric peer')
    out=Path(out).resolve();require(not out.exists()and not out.is_symlink(),'fresh final-stage snapshot only')
    require(not out.is_relative_to(BASE)and not BASE.is_relative_to(out)and not out.is_relative_to(PORTABLE),'new snapshot outside protected input trees')
    out.parent.mkdir(parents=True,exist_ok=True);require(shutil.disk_usage(out.parent).free>=RESERVE+8*(1<<20),'1.5GiB reserve plus metadata allowance required')
    out.mkdir();rows={}
    for name,row in definition['files'].items():
        path=Path(row['path']);require(pin(path)=={k:row[k]for k in('bytes','sha256')},'immutable stage input changed')
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(path,target)
        rows[name]={k:row[k]for k in('bytes','sha256')};rows[name]['lineage']=row['lineage']
    for name,payload in definition['generated'].items():
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb')as stream:stream.write(payload)
        rows[name]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'lineage':'new_explicit_stage_metadata_not_scientific_measurement'}
    baseline=verify_baseline();name='metadata/FINAL_BASELINE_PRESERVATION.json';payload=encoded(baseline)
    with(out/name).open('xb')as stream:stream.write(payload)
    rows[name]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'lineage':'actual_post_stage_rehash_of_all_original471_and_v0_4'}
    require(all(pin(out/n)=={k:r[k]for k in('bytes','sha256')}for n,r in rows.items()),'every new stage member must retain exact bytes')
    manifest={'schema_version':1,'version':'0.5.0','status':'ARTIFACT_V0_5_0_LOCAL_RELEASE_PACKAGE','publication_state_at_creation':'not_published','manifest_excludes_itself':True,'file_count':len(rows),'files':rows,'parent471manifest_sha256':BASE_MANIFEST_SHA,'gates':definition['gates']}
    with(out/'MANIFEST.json').open('xb')as stream:stream.write(encoded(manifest))
    require(sha(BASE/'MANIFEST.json')==BASE_MANIFEST_SHA,'previous staged471manifest must remain unchanged')
    return{'status':manifest['status'],'output':str(out),'payloads':len(rows),'manifest':pin(out/'MANIFEST.json'),'external_assets_copied':False,'remote_actions':0,'YT_writes':0}

README='''# Link prediction artifact v0.5.0

This package contains the completed independently verified E19, E20 and E21
studies, exact public evidence references and portable reproduction commands.
The previous471-payload v0.5 snapshot and immutable v0.4 are preserved. Original
metadata replaced here is retained under metadata/lineage/PRIOR_V0_5_CANDIDATE/.
External assets are identified by filename and whole SHA256 in the catalog.
Publishing/uploading this package is a separate action from creating its bytes.

PUBLIC_OBJECT_COMMANDS.json maps all13 current manuscript objects to exact
inputs, aggregation rules and CLI commands. PUBLIC_COMMANDS.txt maps public
data preparation, recorded-byte verification,
and fresh generic execution to their respective objects. Reproduction sources
under reproduction/ accept ordinary local paths and supplied environments;
CPU and clean PENCIL coordinate through one shared timing lock/active marker.
Their scientific sources remain unchanged. Actual Linux tiny receipts establish
only the declared small portability gates, not full public-run performance.
Fresh public runs have producer validation; the frozen historical independent
auditors require recorded root provenance and do not accept arbitrary local tars.
The separate recorded Friendster score checker independently reranks all6
original float32 NPZs, binds their original query/candidate cohort and reproduces
all saved RR bytes under a256MiB peak limit. It performs archival arithmetic;
checkpoint TAR bytes are discarded without model deserialization/materialization.

E19 is the completed independently verified OLD instrumented48-job consumer
diagnostic. E20 is the completed independently verified seven-arm matched public
Friendster CPU study. E21 is the completed independently verified NEW44-job,
32ONLINE clean-consumer study with disabled per-query memory. Their original
raw archives, true root anchors, frozen independent auditors and numeric peers
are retained separately. These studies/assets are never relabelled or pooled. Query,
setup/conversion, output format, thread/affinity and memory-observation limits
remain necessary when interpreting the tables.

Historical README_v0_3_BASE.md, SYSTEMS_REPRODUCTION.md and
PUBLIC_PENCIL_V2_CANDIDATE.md retain their original bytes and earlier statuses.
Those earlier pending/matched-not-included statements describe past snapshots;
they are superseded by this README, PUBLIC_COMMANDS.txt and the current catalog.
The exact prior471-payload snapshot inventory and replaced metadata are kept
under metadata/lineage/PRIOR_V0_5_CANDIDATE/; the original work candidate is untouched.

Large raw quality/systems/CPU/old48/new44 evidence and the exact CPU two-wheel
packet are separate immutable assets described in metadata/EXTERNAL_ASSETS.json.
No graph is bundled or loaded by this stage; original public data/code notices
are preserved. CPU raw attribution is provided as an external sidecar without
changing the original121-member archive. The artifact's global license applies
to own code; upstream Apache-2.0/MIT and dataset CC-BY-4.0/ODC-BY terms remain.

metadata/V0_5_GATES.json records the exact accepted science/source bindings and
creation-time publication state. An external archive manifest records this
package's whole SHA256. Neither archived verification nor tiny portability
checks count as full fresh seven-arm/44-job performance reproductions.
'''

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--stage',action='store_true');p.add_argument('--clean-audit',type=Path);p.add_argument('--clean-peer',type=Path)
    a=p.parse_args(argv);definition=plan(a.clean_audit,a.clean_peer)
    if a.stage:require(a.out is not None,'--stage needs explicit fresh --out');print(json.dumps(stage(a.out,definition)))
    else:print(json.dumps({'status':definition['status'],'payload_files':len(definition['files'])+len(definition['generated']),'payload_bytes':definition['payload_bytes'],'gates':definition['gates'],'external_assets':definition['catalog'],'output_created':False},indent=2))
if __name__=='__main__':main()
