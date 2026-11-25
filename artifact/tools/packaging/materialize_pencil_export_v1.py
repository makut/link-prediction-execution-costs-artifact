#!/usr/bin/env python3
"""Offline byte-preserving export materializer for recorded E19 or E21.
No YT, pod, producer import, model, fresh observation or numeric audit occurs.
"""
from pathlib import Path,PurePosixPath
import argparse,errno,gzip,hashlib,json,os,shutil,tarfile
MIB=1<<20
ASSETS={'E19':{'bytes':9966534,'sha256':'7ab909e987af900f8ad5f9ea9a559e17e650f78891cb0bc99200a0fdb97fa36c','members':248},
 'E21':{'bytes':10230546,'sha256':'708b6f32081d187dbe5e092b6782a36002333c3f8b0c5ff7dcbf20ec48c156b0','members':221}}
def require(ok,message):
    if not ok:raise RuntimeError(message)
def pin(path):
    p=Path(path);require(p.is_file()and not p.is_symlink(),'regular nonsymlink evidence required');h=hashlib.sha256()
    with p.open('rb')as s:
        for b in iter(lambda:s.read(MIB),b''):h.update(b)
    return{'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def safe(name):
    p=PurePosixPath(name);require(name and len(name)<=512 and str(p)==name and not p.is_absolute()and'..'not in p.parts and'\\'not in name and'\0'not in name,'safe canonical relative evidence path required');return name
def decode(raw):
    def unique(pairs):
        out={}
        for k,v in pairs:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def declared_extraction(raw):
    value=decode(raw)['files']
    if isinstance(value,dict):return{safe(n):{k:r[k]for k in('bytes','sha256')}for n,r in value.items()}
    out={}
    for r in value:
        name=safe(r['path']);require(name not in out,'unique original extraction declaration');out[name]={k:r[k]for k in('bytes','sha256')}
    return out
def materialize(role,metadata,archive,out):
    require(role in ASSETS,'recorded asset role required');expected=ASSETS[role]
    metadata=Path(metadata).resolve();archive=Path(archive).resolve();out=Path(out).resolve()
    require(not out.exists()and not out.is_symlink(),'fresh offline export required')
    require(not out.is_relative_to(metadata)and not metadata.is_relative_to(out)and not archive.is_relative_to(out),'output must be disjoint from input evidence')
    require(archive.stat().st_size==expected['bytes']and pin(archive)=={k:expected[k]for k in('bytes','sha256')},'exact unchanged recorded raw asset required')
    inventory=metadata/'FILE_INVENTORY.json';require(inventory.stat().st_size<=2*MIB,'bounded metadata inventory')
    spec=decode(inventory.read_bytes());require(spec['role']==role,'metadata/asset role mismatch');files=spec['files'];require(0<len(files)<=128,'bounded genuine sidecar inventory')
    total=0
    for name,identity in files.items():
        safe(name);require(name!='final_snapshot.tgz'and not name.startswith('extracted/')and pin(metadata/name)==identity,'exact original sidecar/anchor bytes required')
        total+=identity['bytes'];require(total<=8*MIB,'whole sidecar8MiB bound')
    require('export.json'in files and'extraction_manifest.json'in files,'genuine collection/export manifest required')
    declared=declared_extraction((metadata/'extraction_manifest.json').read_bytes())
    require(len(declared)==expected['members'],'original complete extraction inventory count differs')
    # First stream validates every raw member and decompressed bound BEFORE any
    # destination is created. The second stream only copies the same pinned tar.
    def stream_members(destination=None):
        observed={};payload=0
        class Bounded:
            def __init__(self,s):self.s=s;self.n=0
            def read(self,n=-1):
                limit=386*MIB
                if n<0:n=limit-self.n+1
                b=self.s.read(min(n,limit-self.n+1));self.n+=len(b);require(self.n<=limit,'whole decompressed TAR bound');return b
        with gzip.open(archive,'rb')as gz:
            bounded=Bounded(gz)
            with tarfile.open(fileobj=bounded,mode='r|')as tar:
                for m in tar:
                    name=safe(m.name);require(m.isfile()and not m.issparse()and name not in observed and len(observed)<512 and 0<=m.size<=32*MIB,'bounded unique regular tar payload required')
                    payload+=m.size;require(payload<=384*MIB and name in declared,'whole payload/declared allowlist bound')
                    h=hashlib.sha256();output=None
                    if destination is not None:
                        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);output=target.open('xb')
                    try:
                        with tar.extractfile(m)as source:
                            for b in iter(lambda:source.read(MIB),b''):
                                h.update(b)
                                if output is not None:output.write(b)
                    finally:
                        if output is not None:output.close()
                    observed[name]={'bytes':m.size,'sha256':h.hexdigest()}
                    require(observed[name]==declared[name],'recorded tar member differs from original genuine extraction declaration')
            while bounded.read(MIB):pass
        require(observed==declared,'all exact original raw members required');return observed
    stream_members()
    out.parent.mkdir(parents=True,exist_ok=True);require(shutil.disk_usage(out.parent).free>=int(1.5*(1<<30))+384*MIB,'1.5GiB reserve plus bounded extraction allowance')
    out.mkdir()
    def copy_or_link(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(source,target)
        except OSError as exc:
            if exc.errno!=errno.EXDEV:raise
            with source.open('rb')as i,target.open('xb')as o:shutil.copyfileobj(i,o,MIB)
    for name in files:copy_or_link(metadata/name,out/name)
    copy_or_link(archive,out/'final_snapshot.tgz');extracted=out/'extracted';extracted.mkdir();observed=stream_members(extracted)
    require(pin(out/'final_snapshot.tgz')=={k:expected[k]for k in('bytes','sha256')}and all(pin(out/n)==v for n,v in files.items()),'unchanged archive/sidecar recheck required')
    receipt={'kind':'OFFLINE_RECORDED_PENCIL_EXPORT_MATERIALIZATION','role':role,'raw_asset':{k:expected[k]for k in('bytes','sha256')},'members':len(observed),
      'original_sidecar_inventory':pin(inventory),'all_member_and_sidecar_bytes_unchanged':True,'new_observations':0,'new_numeric_audits':0,'science_jobs':0,'remote_actions':0,'YT_writes':0}
    with(out/'OFFLINE_MATERIALIZATION.json').open('x')as s:json.dump(receipt,s,indent=2);s.write('\n')
    return receipt
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--role',choices=tuple(ASSETS),required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);print(json.dumps(materialize(a.role,a.metadata,a.archive,a.out)))
if __name__=='__main__':main()
