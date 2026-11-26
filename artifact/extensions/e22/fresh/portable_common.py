"""Portable path binding around unchanged frozen CPU v2 scientific functions."""
from pathlib import Path,PurePosixPath
import hashlib,importlib,json,math,os,sys
HERE=Path(__file__).resolve().parent
FROZEN=HERE/'frozen_e22'
RECORDED_COHORT=HERE/'recorded_cohort'
MIB,GIB=1<<20,1<<30
ORIGINAL_MANIFEST_SHA='d30698edff9d9d6b1d959e78bedc7ea226359fb667941e70f43a87945443012b'
SOURCE_PACKET={'bytes':100500,'sha256':'eacd8d86d6849d8fa5d263272d66026d613f302a8efd558750cbd5964529922c'}
WHEEL_PACKET={'bytes':15064339,'sha256':'5e9edbdc067325cd0b41b5fb292e2cc7343cc535da87ff3bef720cbf85da38cf'}

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(4*MIB),b''):h.update(block)
    return h.hexdigest()
def pin(path):
    path=Path(path);require(path.is_file()and not path.is_symlink(),'regular immutable source required');return{'bytes':path.stat().st_size,'sha256':sha(path)}
def read(path,limit=16*MIB):
    path=Path(path);require(path.is_file()and not path.is_symlink()and path.stat().st_size<=limit,'bounded regular JSON required')
    def unique(pairs):
        out={}
        for k,v in pairs:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(path.read_bytes(),object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def save(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    temp.replace(path)
def verify_portable_sources():
    manifest=read(HERE/'SOURCE_MANIFEST.json');bindings={}
    for name,value in manifest['files'].items():
        rel=PurePosixPath(name);require(not rel.is_absolute()and'..'not in rel.parts and str(rel)==name,'safe public source path required')
        path=HERE/name;require(path.resolve().is_relative_to(HERE)and pin(path)==value,'public portable source changed: '+name);bindings[str(path)]=value
    require(sha(FROZEN/'SOURCE_MANIFEST.json')==ORIGINAL_MANIFEST_SHA,'unchanged original CPU v2 source manifest required')
    bindings[str(HERE/'SOURCE_MANIFEST.json')]=pin(HERE/'SOURCE_MANIFEST.json');return bindings

def load_originals():
    sys.path.insert(0,str(FROZEN))
    module=importlib.import_module('cpu_common_v1')
    require(Path(module.__file__).resolve()==FROZEN/'cpu_common_v1.py','original common module origin differs')
    module.verify_sources()
    return module

def paths(out,full):
    out=Path(out).resolve();full=Path(full).resolve();runtime=out/'runtime'
    return{'root':out,'out':out/'results','full':full,'base':runtime,'native':runtime/'native_i64','build':runtime/'cpp_build',
      'venv':runtime/'citation2_cpu_matched_v1_env','wheels':runtime/'citation2_cpu_matched_v1_wheels','cache':runtime/'numba_cache',
      'cohort':out/'results/cohort','cohort_archive':out/'results/cohort/citation2_exact_int32.json',
      'cohort_plan':out/'results/cohort/FROZEN_128_SELECTION.json'}

def configure_worker(config):
    """Only path/host admission changes; scientific module bytes stay exact."""
    verify_portable_sources();config=read(config)
    expected=paths(config['root'],config['full'])
    require(config['paths']=={k:str(v)for k,v in expected.items()},'portable worker path binding differs')
    require(config['source_manifest_sha256']==sha(HERE/'SOURCE_MANIFEST.json'),'portable worker source manifest changed')
    original=load_originals()
    original.BASE=expected['base']
    original.layout=lambda base:dict(expected)
    return original,expected

def independent_ordered_selection():
    # Replay the actual recorded cohort. The original loader revalidates its
    # node/oracle hashes; the sampler/cohort-generation entrypoint is not run.
    original=load_originals();selection=original.read(RECORDED_COHORT/'FROZEN_128_SELECTION.json')
    require(selection['count']==128 and selection['graph']['nodes']==2927963 and selection['graph']['stored_entries']==60703502,'public strict-clean Citation2 128-case cohort differs')
    return selection
