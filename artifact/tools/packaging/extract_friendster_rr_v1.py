#!/usr/bin/env python3
"""Extract only18 small Friendster RR/ID/metric files +true quality manifest.
The exact full quality asset is streamed; models/full scores are never copied.
"""
from pathlib import Path,PurePosixPath
import argparse,gzip,hashlib,json,tarfile
MIB=1<<20
ASSET={'bytes':404147012,'sha256':'e17cdc9678d00a3f372fc8083741a200a63662226f730f47eeda79f89b4ad3b2'}
MANIFEST='9bf816acdce59b0905998adc47437092427cfa8d4c7154da69e5343a8eb15f7d'
PREFIX='lpp_public_quality_evidence_v1/'
def require(ok,message):
    if not ok:raise RuntimeError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def pin(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as s:
        for b in iter(lambda:s.read(MIB),b''):h.update(b)
    return{'bytes':Path(path).stat().st_size,'sha256':h.hexdigest()}
def extract(asset,out):
    asset=Path(asset).resolve();out=Path(out).resolve();require(not out.exists()and not out.is_symlink(),'fresh small quality output required')
    require(not asset.is_symlink()and pin(asset)==ASSET,'whole exact public quality asset required')
    wanted={f'friendster/results/quality_friendster/runs/{arm}_seed{seed}/run1/{name}'for arm in('union','pv')for seed in(42,43,44)for name in('query_rr.npy','query_ids.npy','test_metrics.json')}|{'MANIFEST.json'}
    retained={}
    with gzip.open(asset,'rb')as uncompressed,tarfile.open(fileobj=uncompressed,mode='r|')as tar:
        for m in tar:
            require(m.name.startswith(PREFIX),'exact quality namespace required');name=m.name[len(PREFIX):]
            if name not in wanted:continue
            require(m.isfile()and not m.issparse()and name not in retained and 0<=m.size<=2*MIB,'bounded regular selected quality payload required')
            raw=tar.extractfile(m).read(m.size+1);require(len(raw)==m.size,'whole selected member required');retained[name]=raw
        while uncompressed.read(MIB):pass
    require(set(retained)==wanted and digest(retained['MANIFEST.json'])==MANIFEST,'all18 small files +original quality manifest required')
    files=json.loads(retained['MANIFEST.json'])['files']
    for name,raw in retained.items():
        if name!='MANIFEST.json':require({'bytes':len(raw),'sha256':digest(raw)}=={k:files[name][k]for k in('bytes','sha256')},'selected original member hash differs')
    require(sum(map(len,retained.values()))<=8*MIB,'small quality subset8MiB bound')
    out.mkdir(parents=True)
    for name,raw in retained.items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as s:s.write(raw)
    return{'kind':'SMALL_ORIGINAL_FRIENDSTER_QUALITY_MEMBERS','files':19,'bytes':sum(map(len,retained.values())),'full_quality_asset':ASSET,'models_loaded':0,'models_or_full_scores_copied':0,'numeric_recomputations':0,'remote_actions':0,'YT_writes':0}
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--quality-asset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv);print(json.dumps(extract(a.quality_asset,a.out)))
if __name__=='__main__':main()
