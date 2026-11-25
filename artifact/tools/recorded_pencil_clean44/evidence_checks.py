"""Fresh supplied platform, full public input identity and native build checks."""
from pathlib import PurePosixPath
from archive_contract import BASE,REMOTE_BRIDGE,REMOTE_OUT,require,read,identity,valid_sha,sha
from science_core import clean_exit,guards,optional_vision_policy,finite_number
EXPECTED={'torch':'2.5.0+cu121','numpy':'1.24.4','scipy':'1.10.1','numba':'0.58.0','torch-geometric':'2.7.0','transformers':'4.46.2','wandb':'0.18.7','rich':'14.2.0','tqdm':'4.67.1'}
def phase(payload,job):
    ex=read(payload/(job['name']+'.exit.json'));clean_exit(ex,job['gpu']);require(ex['timeout_seconds']<=job['timeout_seconds'],'phase exit timeout exceeds declared bound')
    require(ex.get('memory_budget_gib')==96 and ex.get('guard_policy')=='unchanged stable_pss_individual_rss_cgroup90_v3 sampled2s','unchanged sampled2s96GiB guard required')
    report=guards(payload/(job['name']+'.guard.jsonl'));log=payload/(job['name']+'.log');require(log.is_file()and not log.is_symlink()and log.stat().st_size<=4<<20,'complete bounded phase stdout required')
    return ex,report

def platform(payload,contract,plan,config):
    before=read(payload/'platform_before.json');after=read(payload/'platform_after.json');require(before==after,'fresh platform/native/numeric/HF sources changed across clean run')
    require(before['schema_version']==1 and before['mode']=='base'and before['python'].startswith('3.10.')and before['platform'].startswith('Linux-')and before['executable']==config['python'],'actual supplied Linux/Python3.10 lexical interpreter differs')
    require(all(before['packages'].get(k)==v for k,v in EXPECTED.items()),'exact target numerical/HF package versions differ')
    versions={k:EXPECTED[k]for k in('torch','numpy','scipy','numba','torch-geometric')};versions['torch-sparse']=before['torch_sparse_version']
    require(before['critical_versions']==versions and before['torch']=='2.5.0+cu121'and before['torch_cuda']=='12.1'and before['numpy']=='1.24.4'and before['pyg']=='2.7.0'and before['torch_sparse_version'].split('+')[0]=='0.6.18','target CUDA/native/numerical admission differs')
    modules=before['critical_modules'];require(set(modules)=={'torch','numpy','scipy','numba','torch_sparse','torch_geometric'},'all6 actual numerical/native origins required')
    for row in modules.values():require(PurePosixPath(row['origin']).is_absolute()and valid_sha(row['source_sha256']),'literal whole numerical origin source hashes required')
    require(before['saint_wrapper']=={'sha256':contract['native']['saint_wrapper_sha256'],'git_blob':contract['native']['saint_wrapper_git_blob']},'fixed original native saint wrapper differs')
    binary=before['torch_sparse_binaries'];require('_saint_cpu.so'in binary and binary['_saint_cpu.so']['sha256']==contract['native']['saint_cpu_sha256'],'fixed original native saint binary differs')
    for name,row in binary.items():require(PurePosixPath(name).name==name and name.endswith('.so')and set(row)=={'bytes','sha256'}and type(row['bytes'])is int and 0<row['bytes']<=256<<20 and valid_sha(row['sha256']),'whole native extension binary identities required')
    extras=before['extra_modules'];require(set(extras)=={'transformers','wandb','rich','tqdm'},'all4 supplied HF/extra origins required')
    for name,row in extras.items():require(row['version']==EXPECTED[name]and PurePosixPath(row['origin']).is_absolute()and valid_sha(row['sha256']),'actual pinned extra source/version differs')
    optional_vision_policy(before['runtime_compatibility'],before,contract)
    gpu=before['visible_gpu'];require(type(gpu['count'])is int and gpu['count']==1 and 'H100'in gpu['name']and gpu['uuid'].startswith('GPU-')and type(gpu['total_memory_bytes'])is int and 79*(1<<30)<=gpu['total_memory_bytes']<=82*(1<<30),'one actual H10080 visible GPU required')
    cpu=before['cpu_observation'];require(isinstance(cpu['models_from_bounded_cpuinfo'],list)and bool(cpu['models_from_bounded_cpuinfo'])and all(isinstance(v,str)and bool(v)for v in cpu['models_from_bounded_cpuinfo'])and type(cpu['cpuinfo_prefix_truncated'])is bool,'actual CPU platform observation required')
    affinity=cpu['cpu_affinity'];require(isinstance(affinity,list)and bool(affinity)and len(set(affinity))==len(affinity)and all(type(v)is int and v>=0 for v in affinity),'actual CPU affinity observation required')
    require(cpu['cgroup_cpu_max']is None or isinstance(cpu['cgroup_cpu_max'],str),'actual cgroup CPU quota observation required')
    cgroup=read(payload/'initial_cgroup.json');require(type(cgroup['max_bytes'])is int and cgroup['max_bytes']==256*(1<<30),'actual finite256GiB cgroup required')
    for job in [*plan['setup_jobs'],*plan['after_jobs']]:phase(payload,job)
    return before,{'status':'pass','pre_post_exact':True,'visible_gpu':gpu,'cpu_observation':cpu,'all_numeric_native_HF_source_hashes_and_policy_checked':True,'actual_cgroup_limit_bytes':cgroup['max_bytes'],'no_legacy_venv_recovery_or_old_PID_gate':True}

def cpp_build(payload,contract):
    receipt=read(payload/'cpp_build/build_receipt_v1.json');library=identity(payload/'cpp_build/native_assoc_v1.so');base=REMOTE_BRIDGE+'/corrected_native';source=base+'/native_assoc_v1.cpp';wanted=REMOTE_OUT+'/cpp_build/native_assoc_v1.so'
    require(receipt['schema_version']==1 and receipt['kind']=='corrected_native_build_v1'and receipt['source']==source and receipt['source_sha256']==contract['cpp']['source_sha256']and receipt['library']==wanted and receipt['library_sha256']==library['sha256'],'actual fresh native C++ source/build/library differs')
    require(0<library['bytes']<=4<<20 and isinstance(receipt['compiler'],str)and PurePosixPath(receipt['compiler']).is_absolute()and isinstance(receipt['compiler_version'],str)and 0<len(receipt['compiler_version'])<=4096 and receipt['platform'].startswith('Linux-'),'fresh bounded Linux compiler/library receipt required')
    require(receipt['argv']==[receipt['compiler'],'-std=c++17','-O3','-DNDEBUG','-fPIC','-pthread','-shared',source,'-o',wanted+'.building'],'actual frozen C++ build flags/source/output differ')
    return {**contract['cpp'],'library_sha256':library['sha256'],'build_receipt_sha256':sha(payload/'cpp_build/build_receipt_v1.json')}

def public_inputs(payload,contract,config):
    value=read(payload/'INPUT_IDENTITY.json');require(value['status']=='pass'and type(value['numeric_payloads'])is int and value['numeric_payloads']==19 and value['validation_sources']==512 and value['negatives_per_source']==1000,'all19 full public identities and frozen512/1000 cohort required')
    expected={config['prepared_root']+'/'+r['path']:{k:r[k]for k in('bytes','sha256')}for r in contract['public_inputs']}
    for name,key in(('valid_edge.npy','valid_positive'),('valid_edge_neg.npy','valid_negative')):expected[config['validation_root']+'/'+name]={'sha256':contract['input_contract']['payloads'][key]['sha256']}
    actual=value['actual_files'];require(set(actual)==set(expected),'complete19+2 actual whole input identity inventory required')
    for name,wanted in expected.items():
        row=actual[name];require(all(row.get(k)==v for k,v in wanted.items())and all(type(row[k])is int and row[k]>=0 for k in('bytes','mtime_ns','inode','device')),'full numeric identity or actual continuity observation differs')
    require(sum(r['bytes']for r in contract['public_inputs'])==5593015516 and len(contract['public_inputs'])==19,'frozen complete public numeric bytes differ')
    require(value['strict_manifest_path']==config['prepared_root']+'/manifest.json'and type(value['strict_manifest']['bytes'])is int and 0<value['strict_manifest']['bytes']<=16<<20 and valid_sha(value['strict_manifest']['sha256']),'whole strict manifest identity required')
    return value

def strict_manifest(raw,contract,input_identity):
    meta=raw['input_metadata'];manifest=meta['strict_manifest'];require(manifest['sha256']==input_identity['strict_manifest']['sha256'],'actual strict manifest SHA differs from full input admission')
    strict=manifest['content'];strict=strict.get('dataset_provenance',{}).get('content',strict)
    rows=strict['files'];actual={r['path']:{k:r[k]for k in('bytes','sha256')}for r in rows};wanted={r['path']:{k:r[k]for k in('bytes','sha256')}for r in contract['public_inputs']}
    require(len(rows)==len(actual)==19 and actual==wanted,'actual strict manifest omits/changes one of19 numeric payloads')
