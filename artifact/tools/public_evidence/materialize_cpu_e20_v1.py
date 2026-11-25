#!/usr/bin/env python3
"""Materialize unchanged recorded E20 metadata plus its separate exact raw asset.
No observation, experiment, numeric audit, network call or source edit occurs.
"""
from pathlib import Path,PurePosixPath
import argparse,errno,hashlib,json,os,shutil
ARCHIVE={'bytes':2121570,'sha256':'546cad6f6b14c9f56844dc3414804e97f0dd303028532910dd74bee9d10f9374'}
def require(ok,message):
    if not ok:raise RuntimeError(message)
def pin(path):
    p=Path(path);require(p.is_file()and not p.is_symlink(),'regular nonsymlink evidence required');h=hashlib.sha256()
    with p.open('rb')as s:
        for b in iter(lambda:s.read(1<<20),b''):h.update(b)
    return{'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def safe(name):
    p=PurePosixPath(name);require(name and str(p)==name and not p.is_absolute()and'..'not in p.parts and'\\'not in name,'canonical relative metadata path required');return name
def materialize(metadata,archive,out):
    metadata=Path(metadata).resolve();archive=Path(archive).resolve();out=Path(out).resolve()
    require(not out.exists()and not out.is_symlink(),'fresh materialized directory required')
    require(not out.is_relative_to(metadata)and not metadata.is_relative_to(out)and not archive.is_relative_to(out),'materialization must be outside source/evidence inputs')
    require(pin(archive)==ARCHIVE,'exact separately published E20 archive required')
    inventory=metadata/'FILE_INVENTORY.json';require(inventory.stat().st_size<=2<<20,'bounded metadata inventory required')
    files=json.loads(inventory.read_text())['files'];require(0<len(files)<=32,'bounded exact recorded metadata inventory required')
    total=0
    for name,identity in files.items():
        safe(name);require(name!='final_snapshot.tgz'and pin(metadata/name)==identity,'recorded metadata bytes differ')
        total+=identity['bytes'];require(total<=2<<20,'whole metadata2MiB limit')
    require('export.json'in files and any(n.startswith('remote_receipts/')for n in files),'genuine export/remote receipts required')
    out.mkdir(parents=True)
    def copy_or_link(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(source,target)
        except OSError as error:
            if error.errno!=errno.EXDEV:raise
            with source.open('rb')as i,target.open('xb')as o:shutil.copyfileobj(i,o,1<<20)
    for name in files:copy_or_link(metadata/name,out/name)
    copy_or_link(archive,out/'final_snapshot.tgz')
    require(pin(out/'final_snapshot.tgz')==ARCHIVE and all(pin(out/n)==v for n,v in files.items()),'copied/linked evidence changed')
    receipt={'kind':'LOSSLESS_RECORDED_E20_FILE_MATERIALIZATION_ONLY','archive':ARCHIVE,'metadata_inventory':pin(inventory),
      'all_metadata_files_unchanged':True,'new_observations':0,'scientific_jobs':0,'numeric_audits':0,'remote_actions':0,'YT_writes':0}
    with(out/'MATERIALIZATION.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
    return receipt
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);print(json.dumps(materialize(a.metadata,a.archive,a.out)))
if __name__=='__main__':main()
