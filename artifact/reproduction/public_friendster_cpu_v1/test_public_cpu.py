"""Tiny local path/admission/state tests, never a full-graph performance run."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,importlib.util,json,os,struct,subprocess,sys
import pytest
import portable_common as c
import run_public_cpu as r
from local_execution import Execution

def parse(tmp_path,*extra):return r.arguments(['--full-csr',str(tmp_path/'data with spaces/full_csr'),'--wheel-packet',str(tmp_path/'assets/native.tgz'),'--out',str(tmp_path/'fresh results'),'--python',str(tmp_path/'supplied env/bin/python'),*extra])

def test_default_plan_is_generic_complete_readonly_and_no_numerical_imports(tmp_path):
    args=parse(tmp_path);before=set(tmp_path.rglob('*'));p=r.plan(args)
    assert set(tmp_path.rglob('*'))==before and not args.out.exists()
    assert p['ordered_sets']==128 and p['warmup']==1 and p['repeats']==3
    assert p['numba_widths']==[32,64]and p['numba_methods']==['scan','pairwise']
    assert len(p['native_modes'])==3 and[j['name']for j in p['jobs']]==list(r.PHASES)
    assert p['default_plan_graph_reads']==p['default_plan_numeric_imports']==p['default_plan_output_creations']==0
    assert not any(n in sys.modules for n in('torch','numpy','numba','scipy'))
    cmd=next(j['argv']for j in p['jobs']if j['name']=='native_friendster')
    assert cmd[cmd.index('--graph-path')+1]==str(args.full_csr)
    assert cmd[cmd.index('--count')+1]=='128'and cmd[cmd.index('--repeats')+1]=='3'
    assert '--include-upstream'in cmd and'--prepare-native'in cmd
    assert '/home/jovyan/seal_revision_20261124'not in json.dumps({k:v for k,v in p.items()if k!='sources'})

def test_lexical_supplied_interpreter_symlink_is_preserved(tmp_path):
    actual=tmp_path/'real executable';actual.write_text('not executed')
    alias=tmp_path/'supplied env/bin/python';alias.parent.mkdir(parents=True);alias.symlink_to(actual)
    args=parse(tmp_path);p=r.plan(args)
    assert p['jobs'][0]['argv'][0]==str(alias)and p['jobs'][0]['argv'][0]!=str(actual)

@pytest.mark.parametrize('bad',['0','-1','nan','inf','181'])
def test_memory_bounds_refuse_invalid_or_increased_caps(tmp_path,bad):
    with pytest.raises(RuntimeError):parse(tmp_path,'--memory-budget-gib',bad)

@pytest.mark.parametrize('bad',['0','-1','nan','inf','6.01'])
def test_total_deadline_finite_and_at_most6hours(tmp_path,bad):
    with pytest.raises(RuntimeError):parse(tmp_path,'--max-hours',bad)

def test_existing_output_and_overlapping_source_input_refused(tmp_path):
    args=parse(tmp_path);args.out.mkdir()
    with pytest.raises(RuntimeError,match='fresh'):parse(tmp_path)
    for out in(c.HERE/'inside_source',tmp_path/'data with spaces/full_csr/output'):
        with pytest.raises(RuntimeError):parse(tmp_path,'--out',str(out))

def test_public_manifest_and_all39_frozen_original_files_match():
    c.verify_portable_sources();original=c.load_originals();bindings=original.verify_sources()
    assert len(bindings)==39
    pins=original.SCIENCE_PINS
    native=[name for name in pins if name.startswith('original/native/')]
    assert len(native)==8
    for name in native:assert c.sha(c.FROZEN/name)==pins[name]
    assert c.sha(c.FROZEN/'scientific_repo/seal/graph_csr_kernels.py')==pins['scientific_repo/seal/graph_csr_kernels.py']

def test_worker_only_rebinds_generic_paths(tmp_path):
    L=c.paths(tmp_path/'run arbitrary',tmp_path/'full arbitrary');config=tmp_path/'config.json'
    c.save(config,{'root':str(L['root']),'full':str(L['full']),'paths':{k:str(v)for k,v in L.items()},'source_manifest_sha256':c.sha(c.HERE/'SOURCE_MANIFEST.json')})
    original,actual=c.configure_worker(config)
    assert actual==L and original.layout(L['base'])==L
    assert c.pin(c.FROZEN/'cpu_worker_v1.py')['sha256']=='aa07786f148b0ade804c41640e5eb2832694c009317888e544192c00a50f4337'
    value=c.read(config);value['paths']['full']='/changed/full';config.write_text(json.dumps(value))
    with pytest.raises(RuntimeError):c.configure_worker(config)

def tiny_full(tmp_path):
    root=tmp_path/'tiny input';root.mkdir()
    (root/'edge_starts.bin').write_bytes(struct.pack('<qqq',0,1,2))
    (root/'edge_ends.bin').write_bytes(struct.pack('<qii',2,1,2));(root/'num_nodes.txt').write_text('3\n')
    return root,{name:c.pin(root/name)for name in('edge_starts.bin','edge_ends.bin','num_nodes.txt')}

def test_whole3_tiny_hash_header_and_stat_admission(tmp_path):
    original=c.load_originals();root,pins=tiny_full(tmp_path)
    proof=original.admit_original(root,pins,(3,2));original.stable_original(proof)
    path=root/'edge_ends.bin';old=path.read_bytes();path.write_bytes(struct.pack('<q',3)+old[8:]);pins['edge_ends.bin']=c.pin(path)
    with pytest.raises(RuntimeError,match='dimensions'):original.admit_original(root,pins,(3,2))
    path.write_bytes(old)
    with pytest.raises(RuntimeError):original.admit_original(root,pins,(3,2))
    with pytest.raises(RuntimeError):original.stable_original(proof)

def test_owned5functions_ast_equal_original_without_remote_entrypoint():
    old=ast.parse((c.FROZEN/'third_party/owned_supervision_v3.py').read_text());new=ast.parse((c.HERE/'owned_supervision.py').read_text())
    import portable_ast_v1 as canonical
    names=('enable_subreaper','process_identity','owned_groups','signal_confirmed','run_owned')
    for name in names:
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name==name)
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name==name)
        assert canonical.canonical_sha256(a)==canonical.canonical_sha256(b)
    assert'yt.wrapper'not in(c.HERE/'owned_supervision.py').read_text()

@pytest.mark.parametrize('quiet,changed',[(False,False),(True,True),(True,False)])
def test_shared_active_marker_preserved_without_exact_quiescence(tmp_path,quiet,changed):
    args=parse(tmp_path);L=c.paths(args.out,args.full_csr);L['out'].mkdir(parents=True)
    ex=Execution(args,L,{})
    ex.active=tmp_path/'PUBLIC_SYSTEMS_ACTIVE.json';ex.token={'role':'cpu7','pid':123,'start_ticks':456,'nonce':'unit fixture'}
    c.save(ex.active,dict(ex.token,pid=999)if changed else ex.token)
    ex.cleanup=lambda:quiet;closed=[];ex.lock=SimpleNamespace(close=lambda:closed.append(True))
    if quiet and not changed:assert ex.__exit__(None,None,None)is False
    else:
        with pytest.raises(RuntimeError):ex.__exit__(None,None,None)
    assert closed==[True]
    done=c.read(L['out']/'COMPLETE.json');assert done['success']==(quiet and not changed)
    assert ex.active.exists()==(not quiet or changed)

def test_execution_requires_reviewed_manifest_before_any_work(tmp_path):
    args=parse(tmp_path,'--execute');p=r.plan(args)
    with pytest.raises(RuntimeError):r.execute(args,p)
    assert not args.out.exists()

def test_supplied_nested_plain_base_site_route_is_admitted(tmp_path):
    import portable_environment as env
    base=tmp_path/'supplied venv';system=tmp_path/'system prefix';fallback=tmp_path/'alternate base/lib/python3.10/site-packages'
    for p in(base,system,fallback):p.mkdir(parents=True)
    module=SimpleNamespace(BASE_SITE_ROOTS=None)
    before={'prefix':str(base),'base_prefix':str(system),'base_sites':[str(fallback)]}
    roots=env.set_roots(module,before)
    assert roots==(base,system,fallback)
    original=c.load_originals();import cpu_environment_v1 as frozen
    previous=frozen.BASE_SITE_ROOTS
    try:
        frozen.BASE_SITE_ROOTS=roots
        assert frozen.validate_base_sites(before['base_sites'])==[str(fallback)]
    finally:frozen.BASE_SITE_ROOTS=previous

def test_wheel_install_is_two_pinned_offline_wheels_only(tmp_path):
    p=r.plan(parse(tmp_path));cmd=next(j['argv']for j in p['jobs']if j['name']=='install_offline_native')
    assert all(flag in cmd for flag in('--no-index','--no-deps','--ignore-installed','--no-cache-dir'))
    assert len([x for x in cmd if x.endswith('.whl')])==2
    assert c.WHEEL_PACKET=={'bytes':15064339,'sha256':'5e9edbdc067325cd0b41b5fb292e2cc7343cc535da87ff3bef720cbf85da38cf'}

def test_plain_subprocess_plan_creates_no_bytecode_or_output(tmp_path):
    import shutil
    clone=tmp_path/'source clone'
    shutil.copytree(c.HERE,clone,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    before={p.relative_to(clone).as_posix()for p in clone.rglob('*')}
    env=dict(os.environ);env.pop('PYTHONDONTWRITEBYTECODE',None)
    out=tmp_path/'fresh no-output'
    command=[sys.executable,str(clone/'run_public_cpu.py'),'--full-csr',str(tmp_path/'not-opened data'),
      '--wheel-packet',str(tmp_path/'not-opened wheels.tgz'),'--out',str(out)]
    ran=subprocess.run(command,capture_output=True,text=True,env=env,timeout=30)
    assert ran.returncode==0,ran.stderr
    assert not out.exists()
    assert before=={p.relative_to(clone).as_posix()for p in clone.rglob('*')}
    assert not list(clone.rglob('*.pyc'))
