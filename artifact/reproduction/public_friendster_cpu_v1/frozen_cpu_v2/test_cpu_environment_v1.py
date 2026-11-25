"""Offline native-environment gates; no native installation or Torch import."""
from pathlib import Path
import ast,copy,hashlib,io,json,os,sys,tarfile,types,zipfile
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import cpu_environment_v1 as e

def put(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path
def fake_receipts(venv):
    venv=Path(venv).resolve();site=venv/'lib/python3.10/site-packages';origin=lambda n:'/opt/python/lib/python3.10/site-packages/'+n+'/__init__.py'
    modules={n:{'origin':origin(n),'bytes':1,'sha256':'1'*64,'version':v}for n,v in e.VERSIONS.items()}
    before={'mode':'base','python':'3.10.12','platform':'Linux','machine':'x86_64','executable':'/opt/python/bin/python','resolved_executable':'/usr/bin/python3.10','executable_identity':{'bytes':1,'sha256':'1'*64},
      'prefix':'/opt/python','base_prefix':'/usr','base_sites':['/opt/python/lib/python3.10/site-packages','/usr/lib/python3/dist-packages'],'packages':dict(e.VERSIONS),'numerical_modules':modules,
      'native':{n:{'installed':False,'version':None,'origin':None,'binaries':{}}for n in e.NATIVE_VERSIONS},'thread_environment':{n:'1'for n in e.THREADS},'CUDA_VISIBLE_DEVICES':'',
      'torch_cpu_threads':1,'numba_threads':1,'torch_cuda_runtime':'12.1','probe_source':{'path':'/source/cpu_environment_v1.py','bytes':1,'sha256':'1'*64}}
    isolated=copy.deepcopy(before);isolated.update(mode='isolated',prefix=str(venv),isolated_venv=str(venv),executable=str(venv/'bin/python'))
    isolated['pyvenv_cfg']={'path':str(venv/'pyvenv.cfg'),'bytes':1,'sha256':'1'*64};isolated['base_site_pth']={'path':str(site/e.PTH_NAME),'bytes':1,'sha256':'1'*64,'base_sites':list(before['base_sites'])}
    isolated['isolated_site_precedes_base_fallback']=True;isolated['package_site_path_order']=[str(site),*before['base_sites']]
    for name,version in e.NATIVE_VERSIONS.items():
        package=site/name;row={'installed':True,'version':version,'origin':str(package/'__init__.py'),'source':copy.deepcopy(e.WHEEL_MEMBERS[name+'/__init__.py']),'binaries':{}}
        for rel,pin in e.WHEEL_MEMBERS.items():
            if rel.startswith(name+'/')and rel.endswith('.so'):row['binaries'][Path(rel).name]={'path':str(site/rel),**pin}
        if name=='torch_sparse':row['saint_wrapper']={'path':str(package/'saint.py'),**e.WHEEL_MEMBERS['torch_sparse/saint.py'],'git_blob':e.SAINT_BLOB}
        isolated['native'][name]=row
    return before,isolated,copy.deepcopy(before)

def test_returned_commands_create_plain_venv_and_only_offline2wheel_install(tmp_path):
    layout=e.layout(tmp_path);assert layout['venv']==tmp_path/e.VENV_NAME and '--without-pip'in layout['create_argv']and '--system-site-packages'not in layout['create_argv']
    argv=layout['install_argv'];assert argv[:4]==[str(layout['python']),'-m','pip','--isolated']
    assert all(flag in argv for flag in('--no-index','--no-deps','--no-compile','--no-cache-dir','--ignore-installed'))
    assert argv[argv.index('--prefix')+1]==str(layout['venv'])and set(Path(x).name for x in argv if x.endswith('.whl'))==set(e.WHEELS)

def test_actual_official_archive_and_all35_embedded_source_binary_hashes(tmp_path):
    source=HERE.parent/'vldb_cpu_native_wheels_pt22_v1/pt22_native_wheels_v1.tgz'
    assert e.info(source)==e.PACKET
    target=tmp_path/'wheels';receipt=e.unpack_wheels(source,target);assert receipt['status']=='PASS'and receipt['package_installations']==receipt['network_calls']==0
    actual={}
    for name in e.WHEELS:
        with zipfile.ZipFile(target/name)as archive:
            for entry in archive.infolist():
                if entry.filename not in e.WHEEL_MEMBERS:continue
                h=hashlib.sha256()
                with archive.open(entry)as stream:
                    for block in iter(lambda:stream.read(e.MIB),b''):h.update(block)
                actual[entry.filename]={'bytes':entry.file_size,'sha256':h.hexdigest()}
    assert actual==e.WHEEL_MEMBERS and sum(n.endswith('.so')for n in actual)==32
    with pytest.raises(RuntimeError,match='fresh'):e.unpack_wheels(source,target)

def test_bad_packet_fails_before_destination_or_base_modification(tmp_path):
    source=put(tmp_path/'bad.tgz',b'not-the-frozen-packet');out=tmp_path/'wheels'
    with pytest.raises(RuntimeError,match='packet'):e.unpack_wheels(source,out)
    assert not out.exists()

def test_plain_base_fallback_is_fresh_owned_and_supports_nested_sites(tmp_path,monkeypatch):
    root=tmp_path/'base';sites=[root/'opt/python/lib/python3.10/site-packages',root/'usr/lib/python3/dist-packages']
    for p in sites:p.mkdir(parents=True)
    monkeypatch.setattr(e,'BASE_SITE_ROOTS',(root/'opt/python',root/'usr'))
    venv=tmp_path/e.VENV_NAME;site=venv/'lib/python3.10/site-packages';site.mkdir(parents=True);cfg=put(venv/'pyvenv.cfg',b'include-system-site-packages = false\n')
    before={'prefix':str(root/'opt/python'),'python':'3.10.12','base_sites':[str(p)for p in sites]};old=e.info(cfg)
    receipt=e.add_base_site(venv,before);pth=Path(receipt['path']);assert pth.read_text()=='\n'.join(str(p)for p in sites)+'\n'and'import 'not in pth.read_text()
    assert e.info(cfg)==old and receipt['import_statements']==0
    with pytest.raises(RuntimeError,match='fresh'):e.add_base_site(venv,before)
    with pytest.raises(RuntimeError,match='base install'):e.add_base_site(venv,{**before,'prefix':str(venv)})

def test_fallback_refuses_executable_or_external_paths_and_systemsite_venv(tmp_path):
    for paths in(["/usr/lib/python3\nimport secret"],['/tmp/foreign/site-packages']):
        with pytest.raises(RuntimeError):e.validate_base_sites(paths)
    venv=tmp_path/e.VENV_NAME;(venv/'lib/python3.10/site-packages').mkdir(parents=True);put(venv/'pyvenv.cfg',b'include-system-site-packages = true\n')
    with pytest.raises(RuntimeError,match='system-site-packages'):e.add_base_site(venv,{'prefix':'/opt/python','python':'3.10.12','base_sites':[]})
    assert not(venv/'lib/python3.10/site-packages'/e.PTH_NAME).exists()

def test_full_declared_before_isolated_after_contract_matches_exact_native_wheels(tmp_path):
    before,isolated,after=fake_receipts(tmp_path/e.VENV_NAME);result=e.verify(before,isolated,after)
    assert result['status']=='PASS'and result['inherited_numerical_origins_exact']and result['saint_wrapper_git_blob']==e.SAINT_BLOB

@pytest.mark.parametrize('fault',['base_after','inherited_origin','native_origin','native_hash','wrapper_blob','thread','cuda','extra_native','fallback_escape','interpreter','site_precedence'])
def test_numerical_origin_base_mutation_abi_binary_and_isolation_faults_refused(tmp_path,fault):
    before,isolated,after=fake_receipts(tmp_path/e.VENV_NAME)
    if fault=='base_after':after['packages']['torch']='wrong'
    elif fault=='inherited_origin':isolated['numerical_modules']['numpy']['origin']='/tmp/another/numpy.py'
    elif fault=='native_origin':isolated['native']['torch_sparse']['origin']='/opt/python/lib/python3.10/site-packages/torch_sparse/__init__.py'
    elif fault=='native_hash':next(iter(isolated['native']['torch_sparse']['binaries'].values()))['sha256']='0'*64
    elif fault=='wrapper_blob':isolated['native']['torch_sparse']['saint_wrapper']['git_blob']='0'*40
    elif fault=='thread':isolated['thread_environment']['NUMBA_NUM_THREADS']='8'
    elif fault=='cuda':isolated['CUDA_VISIBLE_DEVICES']='0'
    elif fault=='extra_native':isolated['native']['torch_sparse']['binaries']['unapproved.so']={'path':str(tmp_path/e.VENV_NAME/'lib/unapproved.so'),'bytes':1,'sha256':'0'*64}
    elif fault=='fallback_escape':isolated['base_site_pth']['path']=str(tmp_path/e.VENV_NAME/'../outside.pth')
    elif fault=='site_precedence':isolated['package_site_path_order']=isolated['package_site_path_order'][1:]+isolated['package_site_path_order'][:1]
    else:isolated['resolved_executable']='/another/python'
    with pytest.raises(RuntimeError):e.verify(before,isolated,after)

def test_probe_rejects_cpu_target_or_cuda_visibility_before_importing_torch(monkeypatch):
    monkeypatch.setattr(e.sys,'platform','darwin');monkeypatch.setattr(e.importlib,'import_module',lambda name:pytest.fail('no numerical import before target gate'))
    with pytest.raises(RuntimeError,match='targetLinux'):e.probe('base')
    monkeypatch.setattr(e.sys,'platform','linux');monkeypatch.setattr(e.sys,'version_info',(3,10,12));monkeypatch.setattr(e.platform,'machine',lambda:'x86_64');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(RuntimeError,match='empty CUDA'):e.probe('base')

def test_module_has_no_subprocess_network_or_GPU_api_calls():
    tree=ast.parse((HERE/'cpu_environment_v1.py').read_text());imports={alias.name for node in ast.walk(tree)if isinstance(node,ast.Import)for alias in node.names}
    assert not imports&{'subprocess','requests','urllib','socket','yt','torch','numpy','scipy','numba'}
    assert not any(isinstance(node,ast.Attribute)and node.attr in('cuda','is_available','device_count')and isinstance(node.value,ast.Name)and node.value.id=='torch'for node in ast.walk(tree))

def test_environment_bindings_are_flat_hashes_and_detect_file_changes(tmp_path,monkeypatch):
    venv=tmp_path/e.VENV_NAME;site=venv/'lib/python3.10/site-packages';values={}
    for rel in('torch_sparse/__init__.py','torch_sparse/saint.py','torch_sparse/_saint_cpu.so','torch_scatter/__init__.py','torch_scatter/_scatter_cpu.so'):
        path=put(site/rel,rel.encode());values[rel]=e.info(path)
    monkeypatch.setattr(e,'WHEEL_MEMBERS',values);before,isolated,after=fake_receipts(venv)
    for name in e.VERSIONS:
        path=put(tmp_path/'base'/name/'__init__.py',name.encode());isolated['numerical_modules'][name]={'origin':str(path),**e.info(path),'version':e.VERSIONS[name]}
    for key in('pyvenv_cfg','base_site_pth','probe_source'):
        path=Path(isolated[key]['path'])if key!='probe_source'else tmp_path/'probe.py';put(path,key.encode());isolated[key]={'path':str(path),**e.info(path),**({'base_sites':isolated['base_sites']}if key=='base_site_pth'else{})}
    executable=put(tmp_path/'python',b'python');isolated['resolved_executable']=str(executable);isolated['executable_identity']=e.info(executable)
    bindings=e.environment_bindings(isolated,venv);assert all(set(value)=={'bytes','sha256'}and Path(name).is_absolute()for name,value in bindings.items())
    next(iter(isolated['native']['torch_sparse']['binaries'].values()));put(site/'torch_sparse/_saint_cpu.so',b'changed')
    with pytest.raises(RuntimeError,match='bytes changed'):e.environment_bindings(isolated,venv)
