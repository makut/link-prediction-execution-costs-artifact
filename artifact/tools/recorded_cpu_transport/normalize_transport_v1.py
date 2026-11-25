#!/usr/bin/env python3
"""Lossless local transport mapping of genuine CPUv2 collector sidecars.

No scientific classifier, numeric aggregation, source edit, remote operation or
new observation. All pre-existing evidence bytes remain untouched. The first
frozen auditor failure is retained and bound in the derivation manifest.
"""
from pathlib import Path,PurePosixPath
import argparse,base64,hashlib,json,os

MIB=1<<20
NAME='public_fr_cpu_matched_v2_20261125'
SOURCE_SHA='95322468767da771d7368fa21a4d9ff02091f19d8be6c3d3cf2e9bfc5adab470'
MANIFEST_SHA='48e6dffd615ca7555826846d63dfa2aa2caf72d46d337736de10ac3c10891e1b'
FIRST_FAILURE_SHA='0ea9697cc0531adb531d6520aee720f2b0aedbe60ba08ec9462a0be7197ea44a'
RECEIPTS={'launch':'public_fr_cpu_matched_v2_launch.json','complete':'results/'+NAME+'/COMPLETE.json',
 'pointer':'results/'+NAME+'/ARCHIVE_READY.json','source':'public_fr_cpu_v2_code/cpu_coordinator_v1.py',
 'manifest':'public_fr_cpu_v2_code/SOURCE_MANIFEST.json','build':'public_fr_cpu_matched_v2_build/build_receipt_v1.json'}
LOCKS=('GPU_SUITE_RUNNING.lock','PIPELINE_SUITE_RUNNING.lock','ADDITIONAL_TIMING_RUNNING.lock')

def require(value,message):
    if not value:raise RuntimeError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def pin(raw):return{'bytes':len(raw),'sha256':digest(raw)}
def file_pin(path):
    path=Path(path);require(path.is_file()and not path.is_symlink(),'regular nonsymlink original required');h=hashlib.sha256()
    with path.open('rb')as source:
        for block in iter(lambda:source.read(MIB),b''):h.update(block)
    return{'bytes':path.stat().st_size,'sha256':h.hexdigest()}
def raw(path,limit):
    path=Path(path);require(path.is_file()and not path.is_symlink()and path.stat().st_size<=limit,'bounded regular original input required');return path.read_bytes()
def decode(raw):
    def unique(pairs):
        out={}
        for k,v in pairs:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def encode(value):return(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def safe(name):
    path=PurePosixPath(name);require(name and str(path)==name and not path.is_absolute()and'..'not in path.parts and'\\'not in name and'\0'not in name,'canonical relative receipt path required');return name

def normalize_view(observation,receipt_bytes):
    require(observation['ready']is True,'genuine sidecar not ready')
    require(set(observation['files'])==set(receipt_bytes),'sidecar must contain all6 exact genuine raw receipt names')
    actual={name:base64.b64decode(value,validate=True)for name,value in observation['files'].items()}
    require(actual==receipt_bytes,'decoded sidecar bytes differ from genuine remote receipts')
    # Copy only actually observed fields. Optional current pod/node fields are
    # copied if present in that sidecar; never recovered from cached metadata.
    view={key:observation[key]for key in('owner','timing_locks','standalone_library_observed')}
    view['hostname']=observation['host']
    view['receipt_sha256']={name:digest(value)for name,value in actual.items()}
    view['source_observed_sha256']=digest(actual[RECEIPTS['source']])
    view['source_manifest_observed_sha256']=digest(actual[RECEIPTS['manifest']])
    for key in('actual_cpu_pod_uid','actual_node'):
        if key in observation:view[key]=observation[key]
    return view

def adapt(original_dir,root_launch_file,preserved_failure,output_dir,expected):
    """Validate all inputs, then create a fresh derivative. No audit is run."""
    original_dir=Path(original_dir);output_dir=Path(output_dir);root_launch_file=Path(root_launch_file)
    require(not output_dir.exists()and not output_dir.is_symlink(),'fresh derivative directory required')
    require(not output_dir.resolve().is_relative_to(original_dir.resolve()),'derivative must be outside original evidence')
    input_bytes={name:raw(original_dir/name,256*1024)for name in('export.json','remote_before.json','remote_after.json')}
    require(set(expected)==set(input_bytes),'explicit whole export/both-sidecar pins required')
    require(all(digest(value)==expected[name]for name,value in input_bytes.items()),'whole original export/sidecar pin differs')
    export=decode(input_bytes['export.json']);require(export['kind']=='public_fr_cpu_matched_v2_final_readonly_export','original versioned export kind differs')
    require('before_view'not in export and'after_view'not in export,'adapter handles only original sidecar transport schema')
    failure_raw=raw(preserved_failure,65536);failure=decode(failure_raw)
    require(digest(failure_raw)==FIRST_FAILURE_SHA and failure['status']=='FAIL_NO_ELIGIBLE_PERFORMANCE_RESULTS'and failure['scientific_evidence_eligible']is False,'preserve exact original frozen auditor failure')
    require(set(export['remote_receipts'])==set(RECEIPTS.values()),'exact6 genuine remote receipt paths required')
    receipts={};paths={};total=0
    for name,identity in export['remote_receipts'].items():
        safe(name);path=original_dir/'remote_receipts'/name;value=raw(path,65536);total+=len(value)
        require(total<=131072 and pin(value)==identity,'whole genuine remote receipt bytes differ');receipts[name]=value;paths[name]=path
    require(digest(receipts[RECEIPTS['source']])==export['source_observed_sha256']==SOURCE_SHA and digest(receipts[RECEIPTS['manifest']])==export['source_manifest_observed_sha256']==MANIFEST_SHA,'whole actual coordinator/source manifest pins differ')
    launch_raw=raw(root_launch_file,65536);require(launch_raw==receipts[RECEIPTS['launch']]and digest(launch_raw)==export['root_launch_sha256'],'whole original root launch and genuine remote bytes differ')
    launch=decode(launch_raw);require(launch['source_sha256']==SOURCE_SHA and launch['source_manifest_sha256']==MANIFEST_SHA,'root launch fixed source pins differ')
    views={key:normalize_view(decode(input_bytes[name]),receipts)for key,name in(('before_view','remote_before.json'),('after_view','remote_after.json'))}
    pointer=decode(receipts[RECEIPTS['pointer']]);complete=decode(receipts[RECEIPTS['complete']])
    require(pointer==export['archive']==complete['local_archive'],'genuine archive pointer differs')
    archive=original_dir/'final_snapshot.tgz';archive_identity={k:pointer[k]for k in('bytes','sha256')}
    require(archive.stat().st_size<=64*MIB and file_pin(archive)==archive_identity,'whole unchanged archive differs')
    # Do not infer quiescence or science from this transport mapping. The frozen
    # independent auditor will validate all lifecycle/scientific fields later.
    derived=dict(export);derived.update(views)
    derived['transport_schema_adaptation']={'kind':'LOSSLESS_GENUINE_SIDECARS_TO_EMBEDDED_VIEWS_V1',
      'original_inputs':{name:pin(value)for name,value in input_bytes.items()},'new_observations':0,'scientific_or_timer_changes':0}
    derivative_raw=encode(derived)
    output_dir.mkdir(parents=True)
    os.link(archive,output_dir/'final_snapshot.tgz')
    for name,path in paths.items():
        destination=output_dir/'remote_receipts'/name;destination.parent.mkdir(parents=True,exist_ok=True);os.link(path,destination)
    originals=output_dir/'original_transport';originals.mkdir()
    for name in input_bytes:os.link(original_dir/name,originals/name)
    os.link(preserved_failure,originals/'FIRST_FROZEN_AUDIT_FAIL.json')
    os.link(root_launch_file,originals/'root_launch.json')
    require(file_pin(output_dir/'final_snapshot.tgz')==archive_identity and all(file_pin(output_dir/'remote_receipts'/name)==pin(value)for name,value in receipts.items()),'hardlinked genuine evidence bytes changed')
    require(all(file_pin(originals/name)==pin(value)for name,value in input_bytes.items())and file_pin(originals/'root_launch.json')==pin(launch_raw)and file_pin(originals/'FIRST_FROZEN_AUDIT_FAIL.json')==pin(failure_raw),'original transport/launch/failure changed during derivation')
    with(output_dir/'export.json').open('xb')as stream:stream.write(derivative_raw)
    proof={'kind':'ROOT_AUTHORIZED_LOSSLESS_TRANSPORT_DERIVATION_V1','meaning':'Transport schema only; no new observations, science, timers, gates or result eligibility inferred.',
      'original_directory':str(original_dir.resolve()),'original_inputs':{name:pin(value)for name,value in input_bytes.items()},
      'root_launch':pin(launch_raw),'preserved_first_failure':pin(failure_raw),'genuine_receipts_unchanged':{name:pin(value)for name,value in receipts.items()},
      'whole_archive_unchanged':archive_identity,'derivative_export':pin(derivative_raw),
      'mapping':{'hostname':'raw observed host','receipt_sha256':'SHA256 of exactly matched decoded six genuine raw receipt bytes',
        'source_observed_sha256':'SHA256 of actual decoded coordinator bytes','source_manifest_observed_sha256':'SHA256 of actual decoded source manifest bytes',
        'owner/timing_locks/standalone_library_observed':'copied observations only','actual_cpu_pod_uid/actual_node':'copied only if present in each genuine sidecar'},
      'storage':'hardlinks for byte-identical original archive/receipts/raw-sidecars/launch/failure; new derivative export+manifest only',
      'local_adapter':file_pin(__file__),'scientific_or_timer_changes':0,'new_observations':0,'audits_executed':0,'remote_actions':0,'YT_writes':0}
    proof_raw=encode(proof)
    with(output_dir/'TRANSPORT_DERIVATION_MANIFEST.json').open('xb')as stream:stream.write(proof_raw)
    require(file_pin(output_dir/'final_snapshot.tgz')==proof['whole_archive_unchanged']and all(file_pin(output_dir/'remote_receipts'/n)==p for n,p in proof['genuine_receipts_unchanged'].items()),'hardlinked genuine evidence bytes changed')
    return{'status':'LOSSLESS_TRANSPORT_DERIVATIVE_CREATED_NO_AUDIT','output_dir':str(output_dir),'export':pin(derivative_raw),'derivation_manifest':pin(proof_raw)}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-dir',type=Path,required=True);parser.add_argument('--root-launch',type=Path,required=True);parser.add_argument('--preserved-failure',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--export-sha256',required=True);parser.add_argument('--before-sha256',required=True);parser.add_argument('--after-sha256',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(adapt(args.original_dir,args.root_launch,args.preserved_failure,args.output_dir,
      {'export.json':args.export_sha256,'remote_before.json':args.before_sha256,'remote_after.json':args.after_sha256})))

if __name__=='__main__':main()
