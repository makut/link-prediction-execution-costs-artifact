#!/usr/bin/env python3
"""Relocate and audit the exact RECORDED E19 archive with unmodified frozen checks.

Offline Python3.12/NumPy only. No new scientific execution or recorded-data edits.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
DEPENDENCIES_SHA='e7b401a6af5ba5a8dbc52b17e5af357d53ad9700ae5c64225625b8c4bd57ff2a'
class InvalidRelocation(ValueError):pass
def require(ok,message):
    if not ok:raise InvalidRelocation(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def read(path):
    def unique(pairs):
        out={}
        for k,v in pairs:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    p=Path(path);require(p.is_file()and not p.is_symlink()and p.stat().st_size<=2<<20,'bounded regular adapter metadata required')
    return json.loads(p.read_text(),object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(InvalidRelocation('nonfinite JSON')))
def verify_file(path,pin,root=None):
    path=Path(path);require(path.is_file()and not path.is_symlink(),'whole regular pinned file missing: '+str(path))
    if root is not None:require(path.resolve().is_relative_to(Path(root).resolve()),'pinned path escapes supplied root')
    require(path.stat().st_size==pin['bytes']and sha(path)==pin['sha256'],'whole bytes/SHA differ: '+str(path));return path

def dependency_contract():
    require(sha(HERE/'DEPENDENCIES.json')==DEPENDENCIES_SHA,'frozen adapter dependency pins changed');value=read(HERE/'DEPENDENCIES.json')
    require(value['audit_python_minor']==[3,12]and len(value['source_dependencies'])==115 and len(value['historical_path_aliases'])==3,'exact recorded source/alias/Python contract required');return value

def rooted_path(root,name):
    path=Path(name);require(not path.is_absolute()and '..'not in path.parts and str(path)==name,'canonical relative dependency required');target=Path(root)/path;require(target.resolve().is_relative_to(Path(root).resolve()),'dependency escapes source root');return target

def verify_dependencies(source_root,evidence_root,export,root_launch,contract):
    for row in contract['source_dependencies']:verify_file(rooted_path(source_root,row['path']),row,source_root)
    aliases={}
    for row in contract['historical_path_aliases']:
        target=rooted_path(evidence_root,row['evidence_name']);verify_file(target,row,evidence_root);require(row['historical_absolute_path']not in aliases,'duplicate historical alias');aliases[row['historical_absolute_path']]=target
    verify_file(Path(export)/'final_snapshot.tgz',contract['recorded_archive'],export)
    for row in contract['recorded_export_sidecars']:verify_file(rooted_path(export,row['path']),row,export)
    verify_file(root_launch,contract['root_launch'])
    return aliases

class ExactPathAliases:
    """A module-local Path constructor view, not a pathlib/filesystem monkeypatch."""
    def __init__(self,aliases,author_root,allowed_roots):
        self.aliases=dict(aliases);self.author_root=Path(author_root);self.allowed_roots=[Path(p).resolve()for p in allowed_roots];self.hits={name:0 for name in aliases}
    def __call__(self,*args,**kwargs):
        path=Path(*args,**kwargs);key=str(path)
        if key in self.aliases:self.hits[key]+=1;return self.aliases[key]
        if path.is_absolute()and path.is_relative_to(self.author_root):
            require(any(path.resolve().is_relative_to(root)for root in self.allowed_roots),'unlisted historical author path refused: '+key)
        return path

def load_frozen(source_root,mapper):
    source_root=Path(source_root);folder=source_root/'work/vldb_pencil_replay_addendum_v2'
    spec=importlib.util.spec_from_file_location('recorded_e19_byte_unchanged_addendum',folder/'audit_addendum_v2.py');addendum=importlib.util.module_from_spec(spec);sys.modules[spec.name]=addendum;spec.loader.exec_module(addendum)
    addendum.ROOT=source_root;addendum.HERE=folder;addendum.CORE=source_root/'work/vldb_pencil_final_audit_v3';addendum.BRIDGE=source_root/'work/vldb_pencil_queue_v3/bridge/vldb_pencil_bridge_v2';addendum.Path=mapper
    for name in('archive_contract','trace_audit'):
        if name in sys.modules:require(Path(sys.modules[name].__file__).resolve().parent==addendum.CORE,'unrelated already imported core module refused')
    core=addendum.load_core();core.ROOT=source_root;core.GOLDEN_QUEUE=source_root/'work/vldb_pencil_queue_v3';core.Path=mapper
    # Existing frozen addendum is solely responsible for its already-reviewed
    # one-predicate REPLAY correction. No adapter scientific predicate exists.
    return addendum,addendum.install(core)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root','--artifact-root',dest='source_root',type=Path,required=True);p.add_argument('--evidence-root',type=Path);p.add_argument('--export',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--root-launch',type=Path,required=True);a=p.parse_args(argv)
    require(sys.version_info[:2]==(3,12),'archival E19 AST pins require offline audit Python3.12; GPU science Python3.10 is a separate environment')
    a.source_root=a.source_root.resolve();a.export=a.export.resolve();a.output=a.output.resolve();a.root_launch=a.root_launch.resolve();evidence=(a.evidence_root or a.source_root/'metadata/evidence/E19').resolve()
    require(not a.output.exists()and not a.output.is_symlink(),'fresh archival audit output required')
    for protected in(a.source_root,a.export,HERE,evidence):require(not a.output.is_relative_to(protected)and not protected.is_relative_to(a.output),'output overlaps immutable sources/export/evidence')
    contract=dependency_contract();aliases=verify_dependencies(a.source_root,evidence,a.export,a.root_launch,contract);mapper=ExactPathAliases(aliases,contract['historical_author_root'],(a.source_root,a.export,a.output,evidence))
    addendum,core=load_frozen(a.source_root,mapper);addendum.correction_receipt()
    code=core.main(['--export',str(a.export),'--output',str(a.output),'--root-launch',str(a.root_launch)])
    require(code==0 and(a.output/'audit.json').is_file(),'frozen archival auditor did not produce successful report')
    report=json.loads((a.output/'audit.json').read_text());require(report['status']=='scientific_checks_passed'and report['completed_jobs']==48,'recorded E19 archival result must preserve all48 jobs')
    require(all(n>0 for n in mapper.hits.values()),'every exact historical filesystem alias must be exercised')
    receipt={'kind':'RECORDED_E19_EXACT_PATH_RELOCATION_NOT_NEW_SCIENCE','source_root':str(a.source_root),'export':str(a.export),'output':str(a.output),'audit_python':sys.version,'dependency_contract_sha256':DEPENDENCIES_SHA,'recorded_archive':contract['recorded_archive'],'frozen_addendum_source_sha256':sha(a.source_root/'work/vldb_pencil_replay_addendum_v2/audit_addendum_v2.py'),'correction_receipt_sha256':contract['recorded_correction_sha256'],'audit_json_sha256':sha(a.output/'audit.json'),'exact_aliases':[{'historical_absolute_path':old,'relocated_path':str(target),'uses':mapper.hits[old]}for old,target in aliases.items()],'all_core_addendum_science_source_bytes_unchanged':True,'recorded_json_or_metric_edits':False,'new_observations_or_scientific_jobs':0,'Torch_imported':'torch'in sys.modules,'remote_actions':0,'scope':'Only source-root globals and three exact whole-hash-verified historical local-file aliases are relocated. Historical metadata paths, failures, observations and numeric evidence stay unchanged.'}
    require(receipt['Torch_imported']is False,'archival audit must not import Torch')
    with(a.output/'ARCHIVAL_RELOCATION.json').open('x')as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'RECORDED_E19_ARCHIVAL_AUDIT_PASSED','completed_jobs':48,'audit_json_sha256':receipt['audit_json_sha256'],'new_scientific_jobs':0}));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status':'invalid_recorded_evidence_or_relocation','error_type':type(error).__name__,'message':str(error),'new_science_performed':False}),file=sys.stderr);raise SystemExit(2)
