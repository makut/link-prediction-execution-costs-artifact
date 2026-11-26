#!/usr/bin/env python3
"""Offline isolated CPU-native environment helpers. No subprocess/network API.

The coordinator owns every returned creation/install/probe command. The only
filesystem writes here are an exact fresh3-file wheel unpack, one plain .pth
inside the fresh owned venv, and a fresh CLI probe receipt. Base packages are
never installed, patched, removed or copied.
"""
from pathlib import Path,PurePosixPath
import argparse,gzip,hashlib,importlib,importlib.metadata as md,importlib.util,json,os,platform,re,shutil,sys,sysconfig,tarfile

MIB=1<<20
RESERVE=int(1.5*(1<<30))
PACKET={'bytes':15064339,'sha256':'5e9edbdc067325cd0b41b5fb292e2cc7343cc535da87ff3bef720cbf85da38cf'}
MANIFEST_SHA='0ad0063fea1bbae0ee7ddff749766450d75e791e13d7316ec199208194e2ac16'
WHEELS={
 'torch_sparse-0.6.18+pt22cu121-cp310-cp310-linux_x86_64.whl':{'bytes':5049112,'sha256':'67877b8d085a54b228e86ab52bcaeb0eceed11df19d686d3e25c46f4e7417837','version':'0.6.18+pt22cu121'},
 'torch_scatter-2.1.2+pt22cu121-cp310-cp310-linux_x86_64.whl':{'bytes':10864848,'sha256':'caab7dd4a8783d49b2a983331e93ebf968c9c76ff4daaca158e2f6fb9c8028bf','version':'2.1.2+pt22cu121'}}
VERSIONS={'torch':'2.2.2+cu121','numpy':'1.24.4','scipy':'1.10.1','numba':'0.58.0'}
NATIVE_VERSIONS={'torch_sparse':'0.6.18+pt22cu121','torch_scatter':'2.1.2+pt22cu121'}
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')
BASE_SITE_ROOTS=(Path('/opt/python'),Path('/usr'))
VENV_NAME='citation2_cpu_matched_v1_env'
PTH_NAME='lpp_public_fr_cpu_base_sites_v1.pth'
SAINT_BLOB='440455853e0309b00376cbe9ad25d7226cfa5e0e'
WHEEL_MEMBERS={
 'torch_scatter/__init__.py':{'bytes': 4399, 'sha256': '18d3f1bf4431ecab75c81c38b6fb9d63a4119c24c96b8bd05facca18a68463c8'},
 'torch_scatter/_scatter_cpu.so':{'bytes': 418912, 'sha256': '62f3f8a9405edd9611091ec81723ff3eb2ecf5d79d9de4e9f26603c44094c7eb'},
 'torch_scatter/_scatter_cuda.so':{'bytes': 3753392, 'sha256': 'acb04f740e586e59082bf3faf88765b8522b1a798c13f9146b9ac0cfeb854eff'},
 'torch_scatter/_segment_coo_cpu.so':{'bytes': 480352, 'sha256': 'f59e7ca768a0d6fcb0bf00b43e753ffd59ad049303d1e6958dbfc6737b0f4112'},
 'torch_scatter/_segment_coo_cuda.so':{'bytes': 58787280, 'sha256': 'd1629bb3253bcdbe8078094fbd0fee92a7e18bcf3a0871e110034808e4bfcea8'},
 'torch_scatter/_segment_csr_cpu.so':{'bytes': 422984, 'sha256': '3715224a3b932dbe19b064a46b368ca72d43d285b94094691b336fde1cf9eb8a'},
 'torch_scatter/_segment_csr_cuda.so':{'bytes': 10233248, 'sha256': '9194cd32769d4f36c5656204ad85488d4d37f8119593dd3baf4a6ee6981aafec'},
 'torch_scatter/_version_cpu.so':{'bytes': 134896, 'sha256': 'c0c6fa0c67ad794e7046548dd1d1640930d37503b57995dd831b1a6b59838052'},
 'torch_scatter/_version_cuda.so':{'bytes': 134896, 'sha256': '56248553c6b0ae53ee46b69630eab749716151ce118b151b3d71eccbd4fcb4ba'},
 'torch_sparse/__init__.py':{'bytes': 3462, 'sha256': '7449626978acc0ed05ba8efc88c7bdefb58c16d3dc481fc6a427487b9ad2408a'},
 'torch_sparse/_convert_cpu.so':{'bytes': 163976, 'sha256': '65d8bc76cc5611ba1d22dffa90ffeeb03081e2e76d067a31b258341436194be9'},
 'torch_sparse/_convert_cuda.so':{'bytes': 242144, 'sha256': '836bbcf0442a02f4a3277a5849024f7e0ad40792e74ece89d3cc2dae01e68b4b'},
 'torch_sparse/_diag_cpu.so':{'bytes': 159728, 'sha256': '1f39cd5a7eaa40760cfa47b8bae9c016a8cabb3f1a46eaed650cb4b0b58264d0'},
 'torch_sparse/_diag_cuda.so':{'bytes': 196912, 'sha256': '4af768ec5860613fb02437ab2ae7e86cd9321c811be9c7e84b0d8e05c4f9e936'},
 'torch_sparse/_ego_sample_cpu.so':{'bytes': 201016, 'sha256': 'f1068a0190634881426c1369786c111e136f2bce75d938d17fed5262cb10838c'},
 'torch_sparse/_ego_sample_cuda.so':{'bytes': 201016, 'sha256': '25c4e627c0017d61eb81ea2c4222105bccb7a603899378c5519745f98687a02a'},
 'torch_sparse/_hgt_sample_cpu.so':{'bytes': 258640, 'sha256': 'dbaea2181aeeca028da828c7dde98bbfc0d77399ffc8e1152f1da79721091dab'},
 'torch_sparse/_hgt_sample_cuda.so':{'bytes': 258640, 'sha256': 'dbaea2181aeeca028da828c7dde98bbfc0d77399ffc8e1152f1da79721091dab'},
 'torch_sparse/_metis_cpu.so':{'bytes': 444776, 'sha256': '9a974dfc5d254297fb95959fe689146e6ae093a49516372ea30a3fc336df1087'},
 'torch_sparse/_metis_cuda.so':{'bytes': 444776, 'sha256': '0660331e23c97c9f6df05384c9f77cc6f0e94818d294768bdb48f28c8994d4e7'},
 'torch_sparse/_neighbor_sample_cpu.so':{'bytes': 394016, 'sha256': 'f820e4adf31821caf9db49be5efaf22f69d10383224d78c4387f7e04ccdbb287'},
 'torch_sparse/_neighbor_sample_cuda.so':{'bytes': 394016, 'sha256': 'f820e4adf31821caf9db49be5efaf22f69d10383224d78c4387f7e04ccdbb287'},
 'torch_sparse/_relabel_cpu.so':{'bytes': 200856, 'sha256': '5b7b2f0bea32a166c1b8b5cec84536b634301d1df950ac74ae1731da4aef95a9'},
 'torch_sparse/_relabel_cuda.so':{'bytes': 200856, 'sha256': 'a0afef89e5fc90eb94844a82499e53e836d19e3b9f2cb16d8186e5e6f0533bea'},
 'torch_sparse/_rw_cpu.so':{'bytes': 163776, 'sha256': 'b09df4f6d41544158035b0c95dbf6183c0d6f7af455995a098b2d4faa5254f09'},
 'torch_sparse/_rw_cuda.so':{'bytes': 221488, 'sha256': 'c7f96d44f0bce5ea29789ccebe0dc0f52f97791935df6adb58ef76a653d9ca54'},
 'torch_sparse/_saint_cpu.so':{'bytes': 172000, 'sha256': 'a04b9fdd3f862dc8a4b7fea22a6abba0db2790407ed4074b92ea24431e326ffa'},
 'torch_sparse/_saint_cuda.so':{'bytes': 172000, 'sha256': '159af4474c2c858f19afa433b1cb8db9be88bdc0399ccbdff7964e7dc3ef17d7'},
 'torch_sparse/_sample_cpu.so':{'bytes': 184416, 'sha256': '3fcade9978088c79fd0a0b5baf3bdbd25401a8eef070595da4a80c7851c91807'},
 'torch_sparse/_sample_cuda.so':{'bytes': 184416, 'sha256': 'c8ce444b406a70e2d7498e93abbdf8a760f3bf89ddf7c7e1fa7e6c392d09d946'},
 'torch_sparse/_spmm_cpu.so':{'bytes': 775280, 'sha256': 'c7c501eb6c87c9a05c5472cecd2f5c031393b9e2bd6338063cf4e026c7921c86'},
 'torch_sparse/_spmm_cuda.so':{'bytes': 14566824, 'sha256': '70dc8b50806ad3113ec227279b55a56aa6702bea50772c1adc6698945c3b7850'},
 'torch_sparse/_version_cpu.so':{'bytes': 47688, 'sha256': 'a64d705bd188809666fdad426705621a3c829328a86015dca4840e0bcc9b4b96'},
 'torch_sparse/_version_cuda.so':{'bytes': 47688, 'sha256': '4642fc8d119363324de1524bb0b4e98f31dcf5fb5472d70c53469c4edf5c6551'},
 'torch_sparse/saint.py':{'bytes': 699, 'sha256': 'f0928fe9c8cba36021a7b6c470c7ce1b99719b876e1c254cba4cf9d2a717ca34'},
}

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(MIB),b''):h.update(block)
    return h.hexdigest()
def info(path):
    path=Path(path);require(path.is_file()and not path.is_symlink(),'regular bound environment file required: '+str(path))
    return {'bytes':path.stat().st_size,'sha256':sha(path)}
def decode(raw):
    def unique(pairs):
        result={}
        for key,value in pairs:require(key not in result,'duplicate JSON key');result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def safe(name):
    value=PurePosixPath(name);require(name and str(value)==name and not value.is_absolute()and '..'not in value.parts and '\\'not in name and '\x00'not in name,'canonical wheel member required');return name
def fresh(path):
    path=Path(path);require(not path.exists()and not path.is_symlink(),'fresh owned environment output required');return path
def layout(base):
    base=Path(base).resolve();venv=base/VENV_NAME;wheels=base/'citation2_cpu_matched_v1_wheels';python=venv/'bin/python';base_python=Path(sys.executable)
    return {'venv':venv,'python':python,'base_python':base_python,'wheels':wheels,'wheel_archive':base/'cpu_pt22_native_wheels_v1.tgz',
      'create_argv':[str(base_python),'-m','venv','--without-pip',str(venv)],
      'install_argv':[str(python),'-m','pip','--isolated','install','--no-index','--no-deps','--no-compile','--no-cache-dir','--disable-pip-version-check','--ignore-installed','--prefix',str(venv),
        *[str(wheels/name)for name in sorted(WHEELS)]]}

def unpack_wheels(archive,destination):
    archive=Path(archive);require(info(archive)==PACKET,'whole official2-wheel packet bytes/SHA differs')
    destination=fresh(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(destination.parent).free>=RESERVE+32*MIB,'reserve plus bounded32MiB unpack allowance required');destination.mkdir()
    expected={'DOWNLOAD_MANIFEST.json':{'bytes':1105,'sha256':MANIFEST_SHA},**{name:{k:row[k]for k in('bytes','sha256')}for name,row in WHEELS.items()}}
    observed={};total=0
    with gzip.open(archive,'rb')as source,tarfile.open(fileobj=source,mode='r|')as tf:
        for member in tf:
            name=safe(member.name);require(member.isfile()and not member.issparse()and name in expected and name not in observed and member.size==expected[name]['bytes'],'exact3regular official packet members required')
            total+=member.size;require(total<=32*MIB,'bounded whole wheel packet raw bytes required');target=destination/name;h=hashlib.sha256();count=0
            with tf.extractfile(member)as stream,target.open('xb')as output:
                for block in iter(lambda:stream.read(MIB),b''):
                    require(shutil.disk_usage(destination).free>=RESERVE+len(block),'1.5GiB live reserve required');output.write(block);h.update(block);count+=len(block)
            observed[name]={'bytes':count,'sha256':h.hexdigest()};require(observed[name]==expected[name],'unpacked official wheel/member SHA differs')
    require(observed==expected,'all3 official packet members required');manifest=decode((destination/'DOWNLOAD_MANIFEST.json').read_bytes())
    require({row['name']:{k:row[k]for k in('bytes','sha256')}for row in manifest['wheels']}=={name:{k:row[k]for k in('bytes','sha256')}for name,row in WHEELS.items()},'original public wheel manifest differs')
    return {'status':'PASS','packet':PACKET,'download_manifest_sha256':MANIFEST_SHA,'members':observed,'destination':str(destination.resolve()),
      'package_installations':0,'base_packages_changed':False,'network_calls':0,'subprocesses_started':0}

def validate_base_sites(paths):
    result=[]
    for name in paths:
        require(isinstance(name,str)and '\n'not in name and '\r'not in name,'plain base-site path required')
        path=Path(name);require(path.is_absolute()and path.is_dir(),'existing absolute base site required');path=path.resolve()
        require(path.name in('site-packages','dist-packages')and any(path.is_relative_to(root.resolve())for root in BASE_SITE_ROOTS),'base package fallback only under/opt/python or/usr')
        if str(path)not in result:result.append(str(path))
    require(result,'nonempty existing inherited base sites required');return result
def add_base_site(venv,before):
    venv=Path(venv);require(not venv.is_symlink()and venv.name==VENV_NAME and venv.is_dir()and str(venv.resolve())!=before['prefix'],'fresh owned isolated venv required; base install refused');venv=venv.resolve()
    cfg=venv/'pyvenv.cfg';require(cfg.is_file()and not cfg.is_symlink()and cfg.stat().st_size<=8192,'new regular pyvenv.cfg required')
    text=cfg.read_text();require(re.search(r'^include-system-site-packages\s*=\s*false\s*$',text,re.M|re.I),'plain isolated venv required; system-site-packages disabled')
    require(before['python'].startswith('3.10.'),'same Python3.10 base required');site=venv/'lib/python3.10/site-packages'
    require(site.is_dir()and not site.is_symlink()and site.resolve().is_relative_to(venv),'owned venv site directory required')
    sites=validate_base_sites(before['base_sites']);require(all(not Path(path).is_relative_to(venv)for path in sites),'base fallback cannot reference isolated env')
    path=fresh(site/PTH_NAME);raw=('\n'.join(sites)+'\n').encode()
    with path.open('xb')as stream:stream.write(raw)
    return {'path':str(path),'identity':info(path),'base_sites':sites,'plain_path_lines_only':True,'import_statements':0,'base_packages_modified':False}

def source_record(module):
    path=Path(module.__file__).resolve();return {'origin':str(path),**info(path),'version':str(module.__version__)}
def native_record(name,*,import_package):
    module=importlib.import_module(name)if import_package else None
    spec=importlib.util.find_spec(name)if module is None else None
    if module is None and spec is None:return {'installed':False,'version':None,'origin':None,'binaries':{}}
    path=Path(module.__file__ if module is not None else spec.origin).resolve();folder=path.parent
    version=md.version(name.replace('_','-'));binaries={p.name:{'path':str(p.resolve()),**info(p)}for p in sorted(folder.glob('*.so'))}
    record={'installed':True,'version':version,'origin':str(path),'source':info(path),'binaries':binaries}
    if module is not None:require(str(module.__version__)==version,'loaded native version differs from installed metadata')
    if name=='torch_sparse':
        wrapper=folder/'saint.py';raw=wrapper.read_bytes();record['saint_wrapper']={'path':str(wrapper),**info(wrapper),'git_blob':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}
    return record

def probe(mode,venv=None):
    require(mode in('base','isolated'),'base or isolated mode required')
    require(sys.platform.startswith('linux')and sys.version_info[:2]==(3,10)and platform.machine()=='x86_64','targetLinuxx86_64Python3.10 required before numerical imports')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit empty CUDA_VISIBLE_DEVICES required before imports')
    require(all(os.environ.get(name)=='1'for name in THREADS),'all4 numerical thread counts must be1')
    if mode=='isolated':
        require(venv is not None and Path(venv).name==VENV_NAME and Path(sys.prefix).resolve()==Path(venv).resolve(),'probe must execute inside the exact new isolated venv')
    else:require(venv is None,'base probe must not claim an isolated venv')
    numerical={};modules={}
    for name,version in VERSIONS.items():
        module=importlib.import_module(name);require(md.version(name)==str(module.__version__)==version,'recorded base numerical version differs: '+name)
        numerical[name]=source_record(module);modules[name]=module
    require(modules['torch'].get_num_threads()==modules['numba'].get_num_threads()==1,'actual CPU numerical worker counts must be1')
    native={name:native_record(name,import_package=mode=='isolated')for name in NATIVE_VERSIONS}
    sites=validate_base_sites([name for name in sys.path if name and Path(name).is_dir()and Path(name).resolve().name in('site-packages','dist-packages')and any(Path(name).resolve().is_relative_to(root.resolve())for root in BASE_SITE_ROOTS)])
    executable=Path(sys.executable);resolved=executable.resolve();record={'schema_version':1,'kind':'public_fr_cpu_environment_probe_v1','mode':mode,
      'python':'.'.join(map(str,sys.version_info[:3])),'platform':platform.platform(),'machine':platform.machine(),'executable':str(executable),'resolved_executable':str(resolved),'executable_identity':info(resolved),
      'prefix':str(Path(sys.prefix).resolve()),'base_prefix':str(Path(sys.base_prefix).resolve()),'base_sites':sites,'packages':{name:md.version(name)for name in VERSIONS},
      'numerical_modules':numerical,'native':native,'thread_environment':{name:os.environ[name]for name in THREADS},'CUDA_VISIBLE_DEVICES':'',
      'torch_cpu_threads':modules['torch'].get_num_threads(),'numba_threads':modules['numba'].get_num_threads(),'torch_cuda_runtime':str(modules['torch'].version.cuda),
      'probe_source':{'path':str(Path(__file__).resolve()),**info(Path(__file__).resolve())},'GPU_API_calls_by_probe':0,'models_loaded':0,'package_installations':0}
    if mode=='isolated':
        venv=Path(venv).resolve();site=venv/'lib/python3.10/site-packages';pth=site/PTH_NAME;record['isolated_venv']=str(venv)
        order=[str(Path(p).resolve())for p in sys.path if p and Path(p).is_dir()and Path(p).resolve().name in('site-packages','dist-packages')]
        require(str(site)in order and all(order.index(str(site))<order.index(p)for p in sites),'isolated site must precede every inherited base site')
        record['package_site_path_order']=order;record['isolated_site_precedes_base_fallback']=True
        record['pyvenv_cfg']={'path':str(venv/'pyvenv.cfg'),**info(venv/'pyvenv.cfg')};record['base_site_pth']={'path':str(pth),**info(pth),'base_sites':pth.read_text().splitlines()}
        validate_isolated(record,venv)
    return record

def validate_isolated(isolated,venv):
    venv=Path(venv).resolve();require(venv.name==VENV_NAME and isolated['mode']=='isolated'and isolated['prefix']==str(venv)and isolated['isolated_venv']==str(venv),'exact isolated environment identity required')
    require(isolated['packages']==VERSIONS and isolated['CUDA_VISIBLE_DEVICES']==''and all(isolated['thread_environment'].get(n)=='1'for n in THREADS)and isolated['torch_cpu_threads']==isolated['numba_threads']==1 and isolated['torch_cuda_runtime']=='12.1','CPU/thread/base numerical compatibility differs')
    own_site=str(venv/'lib/python3.10/site-packages');order=isolated['package_site_path_order']
    require(isolated['isolated_site_precedes_base_fallback']is True and own_site in order and all(p in order and order.index(own_site)<order.index(p)for p in isolated['base_sites']),'isolated site precedence differs from declared fallback')
    actual={}
    for name,version in NATIVE_VERSIONS.items():
        row=isolated['native'][name];origin=Path(row['origin']);require(row['installed']is True and row['version']==version and origin.is_absolute()and origin.resolve().is_relative_to(venv),'native package must come only from isolated venv')
        actual[name+'/__init__.py']=row['source']
        for filename,entry in row['binaries'].items():
            require(Path(entry['path']).is_absolute()and Path(entry['path']).resolve().is_relative_to(venv),'native binary escaped new venv');actual[name+'/'+filename]={k:entry[k]for k in('bytes','sha256')}
        if name=='torch_sparse':
            wrapper=row['saint_wrapper'];require(wrapper['git_blob']==SAINT_BLOB and Path(wrapper['path']).is_absolute()and Path(wrapper['path']).resolve().is_relative_to(venv),'exact frozen saint wrapper required');actual['torch_sparse/saint.py']={k:wrapper[k]for k in('bytes','sha256')}
    require(actual==WHEEL_MEMBERS,'all installed native/source bytes must match the exact official wheel members')
    require(isolated['base_site_pth']['base_sites']==isolated['base_sites']and Path(isolated['base_site_pth']['path']).resolve().is_relative_to(venv)and Path(isolated['pyvenv_cfg']['path']).resolve().is_relative_to(venv),'exact owned plain fallback/config required')
    return True
def verify(before,isolated,after):
    require(before['mode']==after['mode']=='base','two genuine base probes required')
    require(before['python'].startswith('3.10.')and isolated['python']==before['python']and before['machine']==isolated['machine']=='x86_64','recorded CPU Python/architecture target differs')
    stable=('python','platform','machine','executable','resolved_executable','executable_identity','prefix','base_prefix','base_sites','packages','numerical_modules','native','thread_environment','CUDA_VISIBLE_DEVICES','torch_cpu_threads','numba_threads','torch_cuda_runtime','probe_source')
    require(all(before.get(k)==after.get(k)for k in stable),'base package/source/binary/interpreter identity changed')
    require(before['packages']==VERSIONS and before['torch_cuda_runtime']=='12.1'and before['CUDA_VISIBLE_DEVICES']==''and before['torch_cpu_threads']==before['numba_threads']==1 and all(before['thread_environment'].get(n)=='1'for n in THREADS),'recorded base numerical/thread contract differs')
    require(isolated['numerical_modules']==before['numerical_modules']and isolated['packages']==before['packages']and isolated['resolved_executable']==before['resolved_executable']and isolated['executable_identity']==before['executable_identity']and isolated['base_prefix']==before['base_prefix']and isolated['base_sites']==before['base_sites']and isolated['probe_source']==before['probe_source'],'isolated environment must inherit exact unchanged numerical origins/interpreter/probe')
    validate_isolated(isolated,isolated['isolated_venv'])
    return {'status':'PASS','base_before_after_exact':True,'inherited_numerical_origins_exact':True,'native_wheel_source_and_binary_hashes_exact':True,
      'saint_wrapper_git_blob':SAINT_BLOB,'two_official_wheels_only':True,'no_base_package_install_or_patch':True,'CUDA_VISIBLE_DEVICES':'','GPU_API_calls_by_probe':0}
def environment_bindings(isolated,venv):
    validate_isolated(isolated,venv);bindings={}
    def add(path,value):
        path=str(Path(path).resolve());identity={k:value[k]for k in('bytes','sha256')};require(path not in bindings or bindings[path]==identity,'environment path has conflicting identities');require(info(path)==identity,'bound environment bytes changed');bindings[path]=identity
    for row in isolated['numerical_modules'].values():add(row['origin'],row)
    for name,row in isolated['native'].items():
        add(row['origin'],row['source'])
        for entry in row['binaries'].values():add(entry['path'],entry)
        if name=='torch_sparse':entry=row['saint_wrapper'];add(entry['path'],entry)
    for name in('pyvenv_cfg','base_site_pth','probe_source'):entry=isolated[name];add(entry['path'],entry)
    add(isolated['resolved_executable'],isolated['executable_identity']);return bindings

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--probe',choices=('base','isolated'),required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--venv',type=Path);args=parser.parse_args(argv)
    output=fresh(args.output);output=output.resolve();require(not args.venv or not output.is_relative_to(args.venv.resolve()),'probe output must be outside isolated environment')
    value=probe(args.probe,args.venv);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x')as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','mode':args.probe,'output':str(output),'receipt':info(output),'GPU_API_calls_by_probe':0}));return 0
if __name__=='__main__':raise SystemExit(main())
