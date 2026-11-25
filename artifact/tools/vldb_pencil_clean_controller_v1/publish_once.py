"""One fixed coarse YT file plus one bounded attribute; no overwrite or retry.

Run only as a deadline-bound owned child of the reviewed controller. This
module performs no operation merely by import; tests inject an in-memory client.
"""
from pathlib import Path
import argparse,hashlib,json,time
from contracts import TARGET,ATTRIBUTE,require,read,identity,sha,save

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def utf8_metadata(value):
    """Normalize only YSON UTF8 byte strings; no numerical or content coercion."""
    if isinstance(value,bytes):return value.decode('utf-8')
    if isinstance(value,dict):
        result={utf8_metadata(k):utf8_metadata(v)for k,v in value.items()}
        require(len(result)==len(value),'duplicate decoded metadata keys');return result
    if isinstance(value,list):return [utf8_metadata(v)for v in value]
    return value
def publish(client,archive,context):
    archive=Path(archive);observed=identity(archive)
    require(observed==context['archive'],'whole local archive differs from finalized context')
    require(context['target']==TARGET and context['outer_final_pid_exit']=='UNKNOWN','fixed final target/explicit outer-exit boundary required')
    require(len(canonical(context).encode())<=24<<10,'bounded publication context required before any remote write')
    require(not client.exists(TARGET),'coarse target already exists; no overwrite or resume')
    # Atomic exclusive creation prevents a check/write race. Never delete on
    # failure; partial remote state and the local archive remain explicit.
    client.create('file',TARGET,recursive=False,ignore_existing=False)
    with archive.open('rb')as stream:client.write_file(TARGET,stream)
    h=hashlib.sha256();count=0
    stream=client.read_file(TARGET)
    try:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block);count+=len(block)
    finally:
        if hasattr(stream,'close'):stream.close()
    require({'bytes':count,'sha256':h.hexdigest()}==observed,'whole coarse archive readback SHA/bytes differ')
    payload={**context,'readback':{'bytes':count,'sha256':h.hexdigest(),'verified_unix':time.time()},
        'scope':'All pre-publication scientific/setup phase exits are in the tar; the publisher child exit is recorded later locally. This post-readback attribute is outside the tar hash; outer final controller PID exit remains UNKNOWN.'}
    text=canonical(payload)
    attribute={'schema_version':1,'kind':'lpp_clean_consumer_persistence_v1',
        'archive_sha256':observed['sha256'],'archive_bytes':observed['bytes'],
        'scientific_terminal_sha256':context['scientific_terminal_sha256'],
        'canonical_receipt_json':text,'canonical_receipt_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'outer_final_pid_exit':'UNKNOWN'}
    require(len(canonical(attribute).encode())<=32<<10,'bounded structured attribute must be at most32KiB')
    client.set(TARGET+'/@'+ATTRIBUTE,attribute)
    actual=utf8_metadata(client.get(TARGET+'/@'+ATTRIBUTE))
    require(set(actual)==set(attribute)and all(actual[k]==v for k,v in attribute.items()),'exact structured attribute readback differs')
    require(hashlib.sha256(actual['canonical_receipt_json'].encode()).hexdigest()==attribute['canonical_receipt_sha256'],
        'canonical persistence JSON readback SHA differs')
    return {'status':'ONE_COARSE_FILE_AND_ATTRIBUTE_READBACK_VERIFIED','archive':observed,
        'target':TARGET,'attribute_name':ATTRIBUTE,'attribute':attribute,
        'attribute_canonical_json_sha256':hashlib.sha256(canonical(attribute).encode()).hexdigest(),
        'remote_file_nodes_created':1,'file_writes':1,'attribute_writes':1,
        'outer_final_pid_exit':'UNKNOWN','attribute_is_inside_archive_hash':False}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--context',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    require(not a.output.exists(),'preserve existing persistence receipt')
    import yt.wrapper as yt
    client=yt.YtClient(proxy='jupiter.yt.vk.team',config={'proxy':{'request_timeout':15000,'retries':{'count':1}},
        'read_retries':{'enable':False},'write_retries':{'enable':False}})
    result=publish(client,a.archive,read(a.context,64<<10));save(a.output,result,exclusive=True)
    print(json.dumps({'status':result['status'],'target':TARGET,'archive':result['archive'],
        'attribute_canonical_json_sha256':result['attribute_canonical_json_sha256']}));return 0
if __name__=='__main__':raise SystemExit(main())
