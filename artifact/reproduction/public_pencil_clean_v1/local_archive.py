"""Whole local evidence archive; no remote storage or publication."""
from pathlib import Path
import hashlib,tarfile,uuid
from common import require,identity,save,snapshot
def archive_payload(out,archive,check):
    """Whole static evidence, excluding only declared cache directories."""
    out=Path(out);archive=Path(archive);require(not archive.exists(),'preserve existing coarse archive')
    files=[];total=0
    for path in sorted(out.rglob('*')):
        require(not path.is_symlink(),'output symlink refused during finalization')
        if not path.is_file()or path.relative_to(out).parts[0]=='cache':continue
        check();row={'path':str(path.relative_to(out)),**identity(path,check)};files.append(row);total+=row['bytes']
    require(total<=384<<20,'whole evidence exceeds384MiB; no truncation or partial success')
    manifest={'kind':'whole_clean_consumer_evidence_v1','files':files,'file_count':len(files),'payload_bytes':total,
        'excluded':'only declared cache/ directories; no raw,trace,guard,exit or failed-job evidence omitted'}
    save(out/'SHA256_MANIFEST.json',manifest,exclusive=True)
    temporary=archive.with_name(archive.name+'.partial-'+uuid.uuid4().hex[:8])
    with tarfile.open(temporary,'x:gz',compresslevel=1)as target:
        for row in files:
            check();path=out/row['path'];before=snapshot(path)
            target.add(path,arcname=row['path'],recursive=False)
            require(identity(path,check)=={k:row[k]for k in('bytes','sha256')}and snapshot(path)==before,'evidence changed during whole archive creation')
        target.add(out/'SHA256_MANIFEST.json',arcname='SHA256_MANIFEST.json',recursive=False)
    require(temporary.stat().st_size<=384<<20,'bounded coarse archive required')
    expected={row['path']:{k:row[k]for k in('bytes','sha256')}for row in files};expected['SHA256_MANIFEST.json']=identity(out/'SHA256_MANIFEST.json')
    observed={}
    with tarfile.open(temporary,'r|gz')as source:
        for member in source:
            check();require(member.isfile()and member.name in expected and member.name not in observed,'exact regular archive members required')
            h=hashlib.sha256();count=0;stream=source.extractfile(member)
            for block in iter(lambda:stream.read(4<<20),b''):check();h.update(block);count+=len(block)
            observed[member.name]={'bytes':count,'sha256':h.hexdigest()}
    require(observed==expected,'whole local coarse archive readback differs')
    temporary.replace(archive);return {**identity(archive,check),'file_count':len(expected),'whole_readback_verified':True}
