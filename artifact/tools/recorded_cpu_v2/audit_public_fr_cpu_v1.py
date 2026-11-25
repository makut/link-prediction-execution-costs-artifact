#!/usr/bin/env python3
"""Independent bounded LOCAL audit of new public Friendster-only CPU controls.

No graph/model/native-library loading, producer classifiers, remote calls or
scientific reruns. All medians are independently recomputed from raw repeats.
V1 failed before science; it is accepted only by the explicitly synthetic
Python test API. There is no CLI option permitting synthetic evidence.
"""
from pathlib import Path,PurePosixPath
import argparse,ast,collections,gzip,hashlib,io,json,math,random,statistics,struct,tarfile,zipfile

ROOT=Path(__file__).resolve().parents[2]
MIB=1<<20
GIB=1<<30
BASE='/home/jovyan/seal_revision_20261124'
MAX_ARCHIVE,MAX_RAW,MAX_MEMBER,MAX_MEMBERS=64*MIB,96*MIB,16*MIB,257
SOURCE_VERSIONS={
 'bb7014a6cdc202a735373d5537c678b07521bdaeaf6d2521195f695405734990':{'version':1,'bytes':1775086,'manifest_sha256':'c3c54792e86a08cea32699dc615bbe9c6f6360fce86b2376771ec5ae0a8393c4','actual_science_eligible':False},
 'f79cd610e8f24830b896ed4e8b3a37035ab31d0626349699c3b568ebe69c118f':{'version':2,'bytes':1781157,'manifest_sha256':'48e6dffd615ca7555826846d63dfa2aa2caf72d46d337736de10ac3c10891e1b','actual_science_eligible':True}}
HISTORICAL={'bytes':83629661,'sha256':'48232549089f3295c893bb33f6cea0d1aac4a0ee45b91a2669392c90ae1bfe40','manifest_sha256':'2800804cfc2278d399b95a7c1112f05b79e98fe004679466876c7f9185980b2c'}
WHEEL_PACKET={'bytes':15064339,'sha256':'5e9edbdc067325cd0b41b5fb292e2cc7343cc535da87ff3bef720cbf85da38cf'}
WHEEL_MANIFEST='0ad0063fea1bbae0ee7ddff749766450d75e791e13d7316ec199208194e2ac16'
WHEELS={'torch_sparse-0.6.18+pt22cu121-cp310-cp310-linux_x86_64.whl':{'bytes':5049112,'sha256':'67877b8d085a54b228e86ab52bcaeb0eceed11df19d686d3e25c46f4e7417837'},
 'torch_scatter-2.1.2+pt22cu121-cp310-cp310-linux_x86_64.whl':{'bytes':10864848,'sha256':'caab7dd4a8783d49b2a983331e93ebf968c9c76ff4daaca158e2f6fb9c8028bf'}}
NUMERIC={'torch':'2.2.2+cu121','numpy':'1.24.4','scipy':'1.10.1','numba':'0.58.0'}
NATIVE_VERSIONS={'torch_sparse':'0.6.18+pt22cu121','torch_scatter':'2.1.2+pt22cu121'}
N,M=65608366,3612134270
RECORD_SHA='bed119047100f89080081fe638ed60816d0256bfeea9750418668078cdb2d975'
FROZEN_SHA='d2cac5cc0ef136aa0a59466cb9251f2e191e267a09904e373ece78479b55dfb4'
CPP_SHA='67a9e651b7bad7889a73751bff3b33b4309fb2ca8ddfe2660acf851141045d20'
NATIVE_HASHES={'rowptr_sha256':'28060771441d71ba71877e55c0a48230c54d1b2715dd4edd08042019cf4a7a81','col_sha256':'012e1d0a907d2eb6b46d72f31be8b3ca722d88432b2c0509162c4ff487f25a5d'}
FULL_FILES={'edge_starts.bin':{'bytes':524866928,'sha256':'b6ebf026fa54498f241af834b96b5496b2a1bcbf8be3edea2f4b3a1cf0a674e9'},'edge_ends.bin':{'bytes':14448537088,'sha256':'0f74897efa3172252a8371bb0d2c8e0129602475ebb84a863ec1210f9c5e5973'},'num_nodes.txt':{'bytes':9,'sha256':'8551541f6f2ef955481af83ef5e6f92c5f206c968d5d9db6a4437efecf767753'}}
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')
MODES=('reference_full_clear','persistent_touched_reset','upstream_torch_sparse_direct')
PHASES=('base_environment_before','create_isolated_env','install_offline_native','isolated_environment','base_environment_after_install','build_standalone_native','full_input_before','native_friendster','numba32','numba64','full_input_after','isolated_environment_after','base_environment_after')
KERNELS=('_build_induced_edges_pairwise','_fill_neighbors_with_owner_numba','_build_induced_edges_neighbor_scan')
LOCKS=('GPU_SUITE_RUNNING.lock','PIPELINE_SUITE_RUNNING.lock','ADDITIONAL_TIMING_RUNNING.lock')

def require(ok,message):
    if not ok:raise RuntimeError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(MIB),b''):h.update(block)
    return h.hexdigest()
def identity(path):
    path=Path(path);require(path.is_file()and not path.is_symlink(),'regular nonsymlink evidence required');return{'bytes':path.stat().st_size,'sha256':sha(path)}
def record(raw):return{'bytes':len(raw),'sha256':digest(raw)}
def decode(raw):
    def unique(pairs):
        out={}
        for k,v in pairs:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def encoded(value):return(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def safe(name):
    p=PurePosixPath(name);require(name and len(name)<=4096 and str(p)==name and not p.is_absolute()and'..'not in p.parts and'\\'not in name and'\0'not in name,'safe canonical relative evidence path required');return name
def read(path,limit=16*MIB):
    path=Path(path);require(path.is_file()and not path.is_symlink()and path.stat().st_size<=limit,'bounded regular evidence JSON required');return decode(path.read_bytes())
def literal_constants(raw):
    out={}
    for node in ast.parse(raw.decode()).body:
        if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try:out[node.targets[0].id]=ast.literal_eval(node.value)
            except(ValueError,TypeError):pass
    return out
def bounded_tar(path,max_raw,max_member,max_count,prefix='',keep=None):
    files={};identities={};total=0
    class CountingReader:
        # Includes TAR headers, PAX records, padding and trailing decompressed
        # data, which are not covered by the sum of member payload lengths.
        def __init__(self,source):self.source=source;self.total=0;self.limit=max_raw+4096*max_count+MIB
        def read(self,size=-1):
            if size<0:size=self.limit-self.total+1
            raw=self.source.read(min(size,self.limit-self.total+1));self.total+=len(raw)
            require(self.total<=self.limit,'whole decompressed TAR stream bound exceeded');return raw
    with gzip.open(path,'rb')as gzip_source:
      source=CountingReader(gzip_source)
      with tarfile.open(fileobj=source,mode='r|')as archive:
        for m in archive:
            require(m.name.startswith(prefix),'fixed archive prefix differs');name=safe(m.name[len(prefix):])
            require(m.isfile()and not m.issparse()and name not in identities and 0<=m.size<=max_member and len(identities)<max_count,'unique bounded regular archive member required')
            total+=m.size;require(total<=max_raw,'whole raw archive bound exceeded');h=hashlib.sha256();parts=[]
            retain=keep is None or name in keep
            with archive.extractfile(m)as stream:
                for block in iter(lambda:stream.read(MIB),b''):
                    h.update(block)
                    if retain:parts.append(block)
            identities[name]={'bytes':m.size,'sha256':h.hexdigest()}
            if retain:files[name]=b''.join(parts)
      while source.read(MIB):pass
    return files,identities,total

def independent_selection(e1,frozen):
    rows=e1['records'];require(len(rows)==768 and e1['provenance']['graph']['nodes']==N and e1['provenance']['graph']['stored_entries']==M,'whole original768fullFriendster source required')
    groups=collections.defaultdict(list)
    for i,row in enumerate(rows):
        nodes=row['nodes'];require(2<=len(nodes)<=10000 and all(type(x)is int and 0<=x<N for x in nodes)and len(set(nodes))==len(nodes),'frozen ordered S invalid')
        require(nodes[:2]==row['pair']and row['virtual_graph_nodes']==N and row['added_isolates']==0,'original unpadded endpoint-first S required')
        require(digest(struct.pack('<'+'q'*len(nodes),*nodes))==row['node_set_sha256'],'ordered S hash differs')
        groups[(row['sampler'],row['cap'],row['workload'])].append(i)
    ranked=[sorted(groups[key],key=lambda i:hashlib.sha256((RECORD_SHA+':20261125:'+str(i)).encode()).hexdigest())for key in sorted(groups,key=repr)]
    chosen=[];round_index=0
    while len(chosen)<128:
        for group in ranked:
            if round_index<len(group):chosen.append(group[round_index])
            if len(chosen)==128:break
        round_index+=1
    chosen.sort();require(chosen==[r['archive_record_index']for r in frozen['selected']],'independently reselected128cohort differs')
    for item in frozen['selected']:
        row=rows[item['archive_record_index']];require(all(row[k]==v for k,v in item.items()if k!='archive_record_index'),'frozen case metadata/oracle differs')
    return chosen

def load_anchors(source_packet,historical_systems_asset,wheel_packet):
    source_packet=Path(source_packet);source_info=identity(source_packet);require(source_info['sha256']in SOURCE_VERSIONS,'source packet is not independently approved')
    contract=dict(SOURCE_VERSIONS[source_info['sha256']]);require(source_info['bytes']==contract['bytes'],'whole source packet length differs')
    sources,source_rows,_=bounded_tar(source_packet,32*MIB,16*MIB,64)
    manifest_raw=sources['SOURCE_MANIFEST.json'];require(digest(manifest_raw)==contract['manifest_sha256'],'frozen source manifest differs')
    manifest=decode(manifest_raw);expected={r['path']:{k:r[k]for k in('bytes','sha256')}for r in manifest['files']}
    require(expected=={n:r for n,r in source_rows.items()if n!='SOURCE_MANIFEST.json'},'every exact source payload required')
    require(digest(sources['FROZEN_128_SELECTION.json'])==FROZEN_SHA and digest(sources['records/friendster_exact_int32.json'])==RECORD_SHA and digest(sources['original/native/native_assoc_v1.cpp'])==CPP_SHA,'independent scientific source/cohort anchors differ')
    frozen=decode(sources['FROZEN_128_SELECTION.json']);e1=decode(sources['records/friendster_exact_int32.json']);independent_selection(e1,frozen)
    version=contract['version'];contract.update(name=f'public_fr_cpu_matched_v{version}_20261125',source_dir=f'{BASE}/public_fr_cpu_v{version}_code',
      venv=f'{BASE}/public_fr_cpu_matched_v{version}_env',native=f'{BASE}/public_fr_cpu_matched_v{version}_data/native_i64',build=f'{BASE}/public_fr_cpu_matched_v{version}_build',wheels=f'{BASE}/public_fr_cpu_matched_v{version}_wheels',packet_sha256=source_info['sha256'])
    require(identity(historical_systems_asset)=={k:HISTORICAL[k]for k in('bytes','sha256')},'immutable historical public systems asset differs')
    names={'MANIFEST.json','E18/friendster/upstream_native_admission.json',*{'E18/friendster/'+mode+'.json'for mode in MODES}}
    old,_,_=bounded_tar(historical_systems_asset,160*MIB,128*MIB,512,'lpp_public_systems_evidence_v1/',names)
    require(set(old)==names and digest(old['MANIFEST.json'])==HISTORICAL['manifest_sha256'],'exact historical public manifest/native anchors required')
    old_manifest=decode(old['MANIFEST.json']);ordered=None
    for mode in MODES:
        name='E18/friendster/'+mode+'.json';require(record(old[name])=={k:old_manifest['files'][name][k]for k in('bytes','sha256')},'old native original member differs')
        report=decode(old[name]);mapping={r['archive_record_index']:r['ordered_output_sha256']for r in report['records']}
        require(len(mapping)==128 and(ordered is None or mapping==ordered),'all3 historical native ordered-array hashes must agree');ordered=mapping
    require(identity(wheel_packet)==WHEEL_PACKET,'whole officialpt22wheel packet differs');wheel_sources,_,_=bounded_tar(wheel_packet,32*MIB,16*MIB,3)
    require(set(wheel_sources)==set(WHEELS)|{'DOWNLOAD_MANIFEST.json'}and digest(wheel_sources['DOWNLOAD_MANIFEST.json'])==WHEEL_MANIFEST,'exact official download manifest required')
    members={}
    for name,pin in WHEELS.items():
        require(record(wheel_sources[name])==pin,'whole official native wheel SHA differs')
        with zipfile.ZipFile(io.BytesIO(wheel_sources[name]))as archive:
            require(sum(x.file_size for x in archive.infolist())<=128*MIB,'bounded official wheel ZIP required')
            for entry in archive.infolist():
                name=entry.filename
                if not(name.endswith('.so')or name in('torch_sparse/__init__.py','torch_scatter/__init__.py','torch_sparse/saint.py')):continue
                h=hashlib.sha256()
                with archive.open(entry)as stream:
                    for block in iter(lambda:stream.read(MIB),b''):h.update(block)
                members[name]={'bytes':entry.file_size,'sha256':h.hexdigest()}
    require(len(members)==35 and sum(n.endswith('.so')for n in members)==32,'all35 official native/source member anchors required')
    return{'contract':contract,'source_info':source_info,'source_files':sources,'source_manifest':manifest,'frozen_plan':frozen,'e1':e1,'ordered_hashes':ordered,'wheel_members':members,
      'upstream_fixture':decode(old['E18/friendster/upstream_native_admission.json'])}

class Data:
    def __init__(self,files):self.files=files
    def raw(self,name):require(name in self.files,'missing raw scientific receipt: '+name);return self.files[name]
    def obj(self,name):return decode(self.raw(name))
    def proof(self,name):return record(self.raw(name))
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-12)
def finite_positive(value):return type(value)in(int,float)and math.isfinite(value)and value>0
def times(row,key):
    values=row[key];require(isinstance(values,list)and len(values)==3 and all(finite_positive(x)for x in values),'all3 strictly finite positive raw times required');return values
def clean(exit,owned=True,max_timeout=21600):
    require(type(exit.get('pid'))is int and exit['pid']>0 and type(exit.get('returncode'))is int and exit['returncode']==0 and exit.get('timed_out')is False,'clean actual subprocess PID/exit required')
    require(finite_positive(exit.get('timeout_seconds'))and exit['timeout_seconds']<=max_timeout,'finite positive bounded actual subprocess timeout required')
    if owned:require(exit.get('owned_tree_quiescent')is True and exit.get('orphan_cleanup')is False and exit.get('guard_abort_reason')is None,'owned subtree/guard failure cannot count as science')
def same_plan(a,b):
    a=json.loads(json.dumps(a));b=json.loads(json.dumps(b))
    for value in(a,b):value['archive'].pop('original_local_path',None)
    require(a==b,'exact unchanged128selection/seed/order/strata/oracles required')
def full_input(value):
    require(value['kind']=='whole_public_fullCSR_SHA_v1'and value['nodes']==N and value['stored_entries']==M,'fullFriendster dimensions required; trainCSR rejected')
    prefix=BASE+'/public_friendster_repro_v1/new_data/full_csr/'
    require(set(value['files'])=={prefix+n for n in FULL_FILES},'all3 exact public fullCSR inputs required')
    for name,pin in FULL_FILES.items():
        row=value['files'][prefix+name];require({k:row[k]for k in('bytes','sha256')}==pin and all(type(row[k])is int and row[k]>=0 for k in('mtime_ns','device','inode')),'whole public input hash/stat binding differs')
    return value
def host(value):
    require(value.get('hostname')=='lpp-cpu-0'and isinstance(value.get('boot_id'),str)and value['boot_id']and isinstance(value.get('cpu_models'),list)and value['cpu_models']and isinstance(value.get('cpu_affinity'),list)and value['cpu_affinity']and all(type(v)is int and v>=0 for v in value['cpu_affinity']),'documented actual CPU host/boot/model/affinity required')
    return value

def environment(data,anchors):
    c=anchors['contract'];before=data.obj('base_environment_before.json');after_install=data.obj('base_environment_after_install.json');after=data.obj('base_environment_after.json');iso=data.obj('isolated_environment.json');iso_after=data.obj('isolated_environment_after.json')
    require(before==after_install==after and iso==iso_after,'actual before/after-install/final base or isolated environment changed')
    require(before['mode']=='base'and iso['mode']=='isolated'and before['packages']==iso['packages']==NUMERIC,'recorded numerical stack differs')
    require(before['python'].startswith('3.10.')and before['machine']==iso['machine']=='x86_64'and iso['prefix']==iso['isolated_venv']==c['venv'],'Linuxcp310isolated interpreter identity differs')
    require(iso['python']==before['python']and before['prefix']=='/opt/python'and before['base_prefix']=='/usr','actual base/isolated Python and plain /opt/python to /usr interpreter route differ')
    sites=before['base_sites'];require(isinstance(sites,list)and sites and len(sites)==len(set(sites)),'nonempty unique base package sites required')
    for name in sites:
        require(isinstance(name,str)and'\n'not in name and'\r'not in name,'plain inherited base site required')
        path=PurePosixPath(name);require(path.is_absolute()and str(path)==name and'..'not in path.parts and path.name in('site-packages','dist-packages')and any(path.is_relative_to(root)for root in(PurePosixPath('/opt/python'),PurePosixPath('/usr'))),'base package fallback must be canonical under /opt/python or /usr')
    for value in(before,iso):
        require(value['CUDA_VISIBLE_DEVICES']==''and value['torch_cuda_runtime']=='12.1'and value['torch_cpu_threads']==value['numba_threads']==1 and value['thread_environment']=={k:'1'for k in THREADS},'CPU-only one-thread numerical environment required')
        require(value['GPU_API_calls_by_probe']==value['models_loaded']==value['package_installations']==0,'environment probe scope differs')
        require(value['probe_source']=={'path':c['source_dir']+'/cpu_environment_v1.py',**record(anchors['source_files']['cpu_environment_v1.py'])},'actual environment probe source differs')
    for key in('numerical_modules','resolved_executable','executable_identity','base_prefix','base_sites','probe_source'):
        require(iso[key]==before[key],'isolated numerical origins/interpreter changed: '+key)
    site=c['venv']+'/lib/python3.10/site-packages';order=iso['package_site_path_order']
    require(iso['isolated_site_precedes_base_fallback']is True and site in order and all(p in order and order.index(site)<order.index(p)for p in before['base_sites']),'new venv site must precede every inherited base package site')
    actual={}
    for name,version in NATIVE_VERSIONS.items():
        row=iso['native'][name];require(row['installed']is True and row['version']==version and row['origin']==site+'/'+name+'/__init__.py','native package loaded outside new isolated venv')
        actual[name+'/__init__.py']=row['source']
        for filename,entry in row['binaries'].items():
            require(entry['path']==site+'/'+name+'/'+safe(filename),'native binary origin differs');actual[name+'/'+filename]={k:entry[k]for k in('bytes','sha256')}
        if name=='torch_sparse':
            wrapper=row['saint_wrapper'];require(wrapper['path']==site+'/torch_sparse/saint.py'and wrapper['git_blob']=='440455853e0309b00376cbe9ad25d7226cfa5e0e','pinned saint wrapper differs');actual['torch_sparse/saint.py']={k:wrapper[k]for k in('bytes','sha256')}
    require(actual==anchors['wheel_members'],'installed35native/source hashes differ from independently verified official wheels')
    require(iso['pyvenv_cfg']['path']==c['venv']+'/pyvenv.cfg'and iso['base_site_pth']['path']==site+'/lpp_public_fr_cpu_base_sites_v1.pth'and iso['base_site_pth']['base_sites']==before['base_sites'],'owned plain fallback/config differs')
    route=data.obj('base_site_binding.json');require(route['path']==iso['base_site_pth']['path']and route['identity']=={k:iso['base_site_pth'][k]for k in('bytes','sha256')}and route['base_sites']==before['base_sites']and route['plain_path_lines_only']is True and route['import_statements']==0 and route['base_packages_modified']is False,'plain .pth gate differs')
    wheel=data.obj('wheel_extraction.json');require(wheel['status']=='PASS'and wheel['packet']==WHEEL_PACKET and wheel['download_manifest_sha256']==WHEEL_MANIFEST and wheel['members']=={'DOWNLOAD_MANIFEST.json':{'bytes':1105,'sha256':WHEEL_MANIFEST},**WHEELS}and wheel['destination']==c['wheels']and wheel['package_installations']==wheel['network_calls']==wheel['subprocesses_started']==0 and wheel['base_packages_changed']is False,'exact two-wheel offline setup differs')
    for name in('environment_admission.json','environment_verification.json'):
        receipt=data.obj(name);require(receipt['status']=='PASS'and all(receipt[k]is True for k in('base_before_after_exact','inherited_numerical_origins_exact','native_wheel_source_and_binary_hashes_exact','two_official_wheels_only','no_base_package_install_or_patch')),'producer environment receipt contradicts independent check')
    return before,iso

def guards(data):
    stages=data.obj('stages.json')['stages'];require([r['name']for r in stages]==list(PHASES),'all13 genuine owned setup/science/final phases in order required')
    counts={}
    for stage in stages:
        name=stage['name'];clean(stage['exit']);require(stage['exit']==data.obj(name+'.exit.json'),'stage/actual owned exit receipt differs')
        maximum={'base_environment_before':120,'create_isolated_env':120,'install_offline_native':600,'isolated_environment':120,'base_environment_after_install':120,'build_standalone_native':180,'full_input_before':600,'native_friendster':13000,'numba32':1800,'numba64':1800,'full_input_after':600,'isolated_environment_after':120,'base_environment_after':120}[name]
        require(stage['exit']['timeout_seconds']<=maximum,'actual phase timeout exceeds original finite bound')
        require(len(data.raw(name+'.log'))<=4*MIB,'bounded whole stdout log required');raw=data.raw(name+'.guard.jsonl');require(len(raw)<=8*MIB,'bounded whole guard log required')
        errors=0;observations=0
        for line in raw.splitlines():
            require(len(line)<=MIB,'bounded guard row required');row=decode(line)
            if 'sampling_error'in row:
                errors+=1;require(errors<3,'three guard failures cannot count as clean science');continue
            errors=0;decision=row['decision'];require(decision['abort_reason']is None and decision.get('decision','continue')=='continue'and decision['memory_budget_bytes']==180*GIB,'memory guard abort or budget changed')
            stable=row.get('pss_complete')is True and row.get('membership_stable')is True and type(row.get('tree_pss_bytes'))is int and row['tree_pss_bytes']>=0
            require(not stable or row['tree_pss_bytes']<=180*GIB,'stable PSS exceeds180GiB')
            require(row.get('max_process_rss_bytes')is None or(type(row['max_process_rss_bytes'])is int and 0<=row['max_process_rss_bytes']<=180*GIB),'individual RSS exceeds180GiB')
            charged=decision.get('cgroup',{});cur,maximum=charged.get('current_bytes'),charged.get('max_bytes')
            require(stable or not(type(cur)is int and type(maximum)is int and maximum>0 and cur>=maximum*9//10),'unknown own PSS with cgroup90% cannot be clean')
            require(decision['own_pss_estimate_bytes']==(row['tree_pss_bytes']if stable else None),'guard PSS basis differs');observations+=1
        require(observations>0,'at least one actual valid guard decision required per phase')
        counts[name]={'valid_decision_rows':observations,'final_sampling_errors':errors}
    return counts

def runtime_native(value,conversion,c):
    require(value['original_conversion_manifest']==conversion and value['conversion_performed_by_this_worker']is False,'native worker must reuse exact existing conversion')
    require(type(value['content_verification_seconds'])in(int,float)and math.isfinite(value['content_verification_seconds'])and value['content_verification_seconds']>=0,'separately recorded native content-verification duration required')
    expected={c['native']+'/rowptr.i8':{'bytes':8*(N+1),'sha256':NATIVE_HASHES['rowptr_sha256']},c['native']+'/col.i8':{'bytes':8*M,'sha256':NATIVE_HASHES['col_sha256']}}
    actual={r['path']:{k:r[k]for k in('bytes','sha256')}for r in value['runtime_files']};require(actual==expected and len(value['runtime_files'])==2,'both globally hashed native arrays required')
def literal_fixture(value):
    require(value['status']=='pass'and value['literal_cases']==4 and value['ordered_raw_edges_and_eids']is True and value['arbitrary_root_packaging_and_independent_BFS_labels']is True,'all4 actual ordered/EID/arbitrary-root BFS fixtures required')
def native_software(value,iso,anchors):
    require(value['torch']=='2.2.2+cu121'and value['torch_sparse']=='0.6.18+pt22cu121'and value['torch_num_threads']==1,'actual CPU Torch/native compatibility differs')
    require(value['installed_wrapper_git_blob']=='440455853e0309b00376cbe9ad25d7226cfa5e0e'and value['installed_wrapper_sha256']==anchors['wheel_members']['torch_sparse/saint.py']['sha256'],'actual installed native wrapper differs')
    binaries={Path(r['path']).name:r['sha256']for r in value['extension_binaries']};expected={n:row['sha256']for n,row in iso['native']['torch_sparse']['binaries'].items()if n.startswith('_saint')}
    require(binaries==expected and all(r['path']==anchors['contract']['venv']+'/lib/python3.10/site-packages/torch_sparse/'+Path(r['path']).name for r in value['extension_binaries']),'actually admitted upstream native binaries differ')
def conversion_and_build(data,full,iso,anchors):
    c=anchors['contract'];build=decode(data.raw('build_standalone_native.log').strip())
    source=c['source_dir']+'/original/native/native_assoc_v1.cpp';library=c['build']+'/native_assoc_v1.so'
    require(build['source']==source and build['source_sha256']==CPP_SHA and build['library']==library and isinstance(build['compiler_version'],str)and build['compiler_version'],'actual standalone source/compiler/library build proof required')
    require(build['argv']==[build['compiler'],'-std=c++17','-O3','-DNDEBUG','-fPIC','-pthread','-shared',source,'-o',library+'.building'],'unchanged standalone C++17/O3 flags required')
    require(isinstance(build['library_sha256'],str)and len(build['library_sha256'])==64,'actual standalone binarySHA required')
    value=data.obj('native/native_conversion_receipt.json');require(value['status']=='complete'and value['external_helper_sha256']==digest(anchors['source_files']['original/external_operator_benchmark.py']),'frozen actual converter proof required');literal_fixture(value['actual_native_fixture']);clean(data.obj('native/native_conversion_exit.json'),False,1800)
    meta=value['conversion'];graph=meta['graph'];source_files=[{'path':BASE+'/public_friendster_repro_v1/new_data/full_csr/'+n,'bytes':FULL_FILES[n]['bytes'],'mtime_ns':full['files'][BASE+'/public_friendster_repro_v1/new_data/full_csr/'+n]['mtime_ns']}for n in('edge_starts.bin','edge_ends.bin')]
    require(graph==value['graph']and graph['n']==N and graph['m']==M and graph['endian']=='little'and graph['source_files']==source_files,'conversion bound to different source graph')
    require(graph['source_storage_bytes']==8*N+8+4*M and graph['native_int64_storage_bytes']==8*(N+1+M),'source/native storage arithmetic differs')
    require(all(meta[k]==v for k,v in NATIVE_HASHES.items())and meta['native_dtype']=='little-endian int64'and meta['sentinel_added']is True,'historical global native content identity differs')
    require(finite_positive(meta['conversion_seconds'])and finite_positive(value['setup_wall_seconds']),'conversion/setup reported separately required')
    estimate=value['estimate'];expected={'source_storage_bytes':8*N+8+4*M,'native_int64_storage_bytes':8*(N+1+M),'per_query_assoc_bytes':8*N,'public_api_global_row_bytes':0,'conversion_chunk_upper_bytes':24000000,'conservative_peak_bytes':8*N+8+4*M+8*(N+1+M)+8*N+24000000+GIB}
    require(all(estimate[k]==v for k,v in expected.items()),'converter storage/scratch/peak arithmetic differs')
    verified=value['runtime_native_content_verification'];require(verified['status']=='pass','converter actual content readback must pass')
    runtime_native({'original_conversion_manifest':meta,'conversion_performed_by_this_worker':False,'content_verification_seconds':verified['seconds'],'runtime_files':verified['files']},meta,c)
    fixture=data.obj('native/upstream_native_admission.json');require(fixture['status']=='complete'and fixture['fixture']is True and fixture['all_executed_arms_exact']is True and fixture['native_admission_fixture']['status']=='pass'and fixture['native_admission_fixture']['cases_per_arm']==4 and fixture['native_admission_fixture']['arms']==['direct'],'actual installed native4case fixture required before conversion')
    native_software(fixture['provenance'],iso,anchors);clean(data.obj('native/upstream_native_admission_exit.json'),False,120)
    expected_fixture=anchors['upstream_fixture'];require(len(fixture['records'])==4,'all4 original extension fixture cases required')
    for actual,old in zip(fixture['records'],expected_fixture['records']):
        require(all(actual[k]==old[k]for k in('nodes','pair','node_set_sha256','canonical_edges_sha256','drnl_sha256','sum_out_degrees')),'independent frozen tiny fixture oracle differs')
        measured=actual['methods']['direct'];require(measured['status']=='ok'and measured['exact_edges']is True and measured['exact_drnl']is True,'native tiny fixture not exact')
        for key in('raw_operator_seconds','edge_seconds','drnl_seconds'):times(measured,key)
    return meta,build

def native_results(data,full,iso,anchors,meta,build):
    c=anchors['contract'];plan=data.obj('native/replay_plan.json');same_plan(plan,anchors['frozen_plan']);done=data.obj('native/COMPLETE.json')
    require(done['complete']is True and done['scientific_terminal']is True and done['success']is True and done['replay_plan_sha256']==data.proof('native/replay_plan.json')['sha256'],'native parent scientifically incomplete')
    require(len(done['arms'])==3 and{r['mode']for r in done['arms']}==set(MODES),'all3native arms required')
    parent=data.obj('native/parent_plan.json');expected_sources={Path(name).name:digest(raw)for name,raw in anchors['source_files'].items()if name.startswith('original/native/')}
    require(parent['source_hashes']==expected_sources and parent['replay_plan_sha256']==data.proof('native/replay_plan.json')['sha256']and parent['mode_order_seed']==20261125,'native original source/order/replay bindings differ')
    mode_order=list(MODES[:2])+[MODES[2]];random.Random(20261125).shuffle(mode_order);require(parent['modes']==mode_order and[r['mode']for r in done['arms']]==mode_order,'frozen seeded sequential native arm order differs')
    pc=parent['configuration'];require(pc['count']==128 and pc['prepare_native']is True and pc['include_upstream']is True and pc['warmup']==1 and pc['repeats']==3 and pc['max_query_neighbors']==20000000 and pc['max_assoc_gib']==.5 and pc['max_peak_gib']==48 and pc['conversion_max_native_gib']==28 and pc['native_dir']==c['native']and pc['library']==build['library'],'frozen native parent settings differ')
    native_folder=BASE+'/results/'+c['name']+'/native';helper=c['source_dir']+'/original/external_operator_benchmark.py';helper_sha=digest(anchors['source_files']['original/external_operator_benchmark.py'])
    extra={'archive':c['source_dir']+'/records/friendster_exact_int32.json','output_dir':native_folder,'execute':True,'build_dir':None,'admission_marker':BASE+'/results/'+c['name']+'/admission.json','external_helper':helper,'external_helper_sha256':helper_sha,'graph_path':BASE+'/public_friendster_repro_v1/new_data/full_csr','conversion_timeout_seconds':1800,'timeout_seconds':3600,'order_seed':20261125}
    require(all(pc[k]==v for k,v in extra.items()),'native parent actual original source/admission/timeout arguments differ')
    bindings=data.obj('native/runtime_bindings.json');require(done['runtime_bindings_sha256']==data.proof('native/runtime_bindings.json')['sha256'],'native runtime binding receipt differs')
    expected_bindings={c['source_dir']+'/original/native/'+filename:pin for filename,pin in expected_sources.items()}
    expected_bindings.update({build['library']:build['library_sha256'],c['source_dir']+'/records/friendster_exact_int32.json':RECORD_SHA,native_folder+'/replay_plan.json':data.proof('native/replay_plan.json')['sha256'],helper:helper_sha})
    require(bindings==expected_bindings,'all exact native actual source/library/input/helper/replay bindings required')
    summaries={};rows_by_mode={}
    for arm in done['arms']:
        mode=arm['mode'];name='native/'+mode+'.json';report=data.obj(name);clean(arm['exit'],False,3600);require(arm['exit']==data.obj('native/'+mode+'_exit.json')and arm['eligible']is True and arm['report_sha256']==data.proof(name)['sha256'],'native worker exit/report identity differs')
        require(report['status']=='complete'and report['record_count']==len(report['records'])==128 and report['all_executed_outputs_exact']is True and report['archive_sha256']==RECORD_SHA and report['replay_plan_sha256']==data.proof('native/replay_plan.json')['sha256'],'all128 native records and source binding required')
        provenance=report['provenance'];require(provenance['worker_sha256']==digest(anchors['source_files']['original/native/replay_worker_v1.py'])and provenance['adapter_sha256']==digest(anchors['source_files']['original/native/native_adapter_v1.py'])and provenance['library_sha256']==build['library_sha256'],'actual native worker/adapter/library source differs')
        cfg=report['configuration'];require(cfg['mode']==mode and cfg['warmup']==1 and cfg['repeats']==3 and cfg['max_query_neighbors']==20000000 and cfg['max_assoc_gib']==.5 and cfg['max_peak_gib']==48 and cfg['native_dir']==c['native']and cfg['library']==build['library'],'actual native per-worker settings changed')
        require(cfg['archive']==extra['archive']and cfg['replay_plan']==native_folder+'/replay_plan.json'and cfg['output']==native_folder+'/'+mode+'.json'and cfg['external_helper']==(helper if mode==MODES[2]else None)and cfg['external_helper_sha256']==(helper_sha if mode==MODES[2]else None),'native actual per-worker original paths/helper arguments differ')
        literal_fixture(report['actual_native_admission']);runtime_native(report['native_setup'],meta,c)
        budget=report['budget'];local=160*max(anchors['e1']['records'][r['archive_record_index']]['induced_directed_entries']for r in plan['selected'])+256*max(len(anchors['e1']['records'][r['archive_record_index']]['nodes'])for r in plan['selected'])
        require(budget['native_int64_storage_and_potential_file_working_set_bytes']==8*(N+1+M)and budget['association_bytes_per_worker']==8*N and budget['local_output_upper_estimate_bytes']==local and budget['conservative_peak_estimate_bytes']==8*(N+1+M)+8*N+local+GIB,'native work/storage budget arithmetic differs')
        state=report['state_initialization'];require(state['persistent_assoc_bytes_per_worker']==(8*N if mode=='persistent_touched_reset'else 0)and state['assoc_bytes_during_each_call']==8*N,'per-worker8N association-state accounting differs')
        if mode!='upstream_torch_sparse_direct':
            require(finite_positive(state['initialization_seconds'])and state['library_sha256']==build['library_sha256']and state['library']==build['library'],'standalone init must be separately recorded')
            require(state['version']=='persistent_assoc_touched_reset_v1'and state['mode']==mode and state['nodes']==N and state['stored_entries']==M and state['owner_pid']==arm['exit']['pid']and state['initialization_policy']==('once per worker'if mode=='persistent_touched_reset'else'full-N allocation/init inside every native call'),'standalone state metadata must bind actual fresh worker/mode/dimensions/policy')
        else:
            require(state['policy']=='unmodified upstream full-N association allocation/init inside each call','upstream full-N association policy differs')
            native_software(report['upstream_software'],iso,anchors)
        sums=collections.defaultdict(list)
        for actual,frozen in zip(report['records'],plan['selected']):
            index=frozen['archive_record_index'];require(actual['archive_record_index']==index and actual['node_set_sha256']==frozen['node_set_sha256']and actual['status']=='ok'and all(actual[k]is True for k in('exact_edges','exact_drnl','exact_ordered_rows_cols_and_original_eids'))and actual['root_positions']==[0,1],'native case/order/semantics differ')
            require(actual['ordered_output_sha256']==anchors['ordered_hashes'][index],'native ordered rows/cols/original-eid hashes differ from independent historical oracles')
            raw,edge,drnl=[times(actual,key)for key in('raw_operator_seconds','edge_seconds','drnl_seconds')];require(all(a<=b+1e-9 for a,b in zip(raw,edge)),'raw native timer must be nested within packaged edge timer')
            for key,values in(('raw_operator_seconds',raw),('edge_seconds',edge),('drnl_seconds',drnl),('edge_plus_drnl_seconds',[a+b for a,b in zip(edge,drnl)])):sums[key].append(statistics.median(values))
        summaries['native_'+mode]={'records':128,'sum_per_case_medians_seconds':{k:math.fsum(v)for k,v in sums.items()},'state_initialization_outside_query_timer':state,'native_content_verification_outside_query_timer_seconds':report['native_setup']['content_verification_seconds']};rows_by_mode[mode]=report['records']
    return summaries,plan

def numba_results(data,full,iso,anchors,meta,plan,expected_host):
    summaries={};configuration={'selection_count':128,'selection_seed':20261125,'warmup':1,'repeats':3,'methods':['scan','pairwise'],'worker_count':1,'cpu_threads':1,'max_query_neighbors':20000000}
    for width in(32,64):
        report=data.obj('friendster_neighbors'+str(width)+'.json');require(report['status']=='complete'and report['record_count']==len(report['records'])==128 and report['all_executed_outputs_exact']is True and report['configuration']==configuration and report['neighbor_width']==width and report['numba_threads']==1 and report['torch_imported']is False,'both fixed compiled128-case physical-width workers required')
        require(host(report['host'])==expected_host and report['original_full_payload_admission']==full and report['admission_sha256']==data.proof('admission.json')['sha256'],'Numba actual host/input/admission binding differs');same_plan(report['replay_plan'],plan)
        require(report['source_bindings']=={anchors['contract']['source_dir']+'/'+name:record(raw)for name,raw in anchors['source_files'].items()},'Numba actual source bindings differ')
        require(report['original_input_stat_header_proof']=={p:{k:r[k]for k in('bytes','mtime_ns')}for p,r in full['files'].items()if Path(p).name!='num_nodes.txt'},'Numba original stat/header continuity differs')
        physical=report['physical_arrays'];require(physical=={'offset_dtype':'int64','neighbor_dtype':'int'+str(width),'offset_count':N,'neighbor_count':M,'full_graph_copy_or_conversion_performed':False,'native_rowptr_view_excludes_existing_final_sentinel':width==64},'actual original32/reused64 physical representation differs')
        require(set(report['compiled_kernels'])==set(KERNELS)and all(isinstance(v,list)and v and all(isinstance(s,str)and s for s in v)for v in report['compiled_kernels'].values()),'all3 actual nopython kernel signatures required')
        for name in('numpy','scipy','numba'):require(report['numerical_environment'][name]=={'version':NUMERIC[name],'origin':iso['numerical_modules'][name]['origin']},'actual NumBa numerical origins differ')
        require(type(report['setup_seconds_excluded'])in(int,float)and math.isfinite(report['setup_seconds_excluded'])and report['setup_seconds_excluded']>=0,'setup outside query times must be recorded')
        if width==64:runtime_native(report['native_setup'],meta,anchors['contract'])
        sums={name:collections.defaultdict(list)for name in('scan','pairwise')}
        for actual,frozen in zip(report['records'],plan['selected']):
            require(all(actual[k]==v for k,v in frozen.items()if k in('archive_record_index','node_set_sha256','canonical_edges_sha256','drnl_sha256','sum_out_degrees','induced_directed_entries'))and actual['exact_archived_oracles']is True and set(actual['methods'])=={'scan','pairwise'},'NumBa exact case/order/oracles differ')
            require(actual['method_order']==list(actual['methods'])and sorted(actual['method_order'])==['pairwise','scan'],'Numba recorded method order differs from measured insertion order')
            for method,row in actual['methods'].items():
                require(row['status']=='ok'and row['exact_edges']is True and row['exact_drnl']is True,'every compiled method-case must be exact')
                edge,drnl,total=[times(row,key)for key in('edge_seconds','drnl_seconds','edge_plus_drnl_seconds')];require(all(close(a+b,c)for a,b,c in zip(edge,drnl,total)),'edge+DRNL raw repetition arithmetic differs')
                for key,values in(('edge_seconds',edge),('drnl_seconds',drnl),('edge_plus_drnl_seconds',total)):sums[method][key].append(statistics.median(values))
        for method,values in sums.items():summaries['numba'+str(width)+'_'+method]={'records':128,'sum_per_case_medians_seconds':{k:math.fsum(v)for k,v in values.items()},'setup_seconds_excluded':report['setup_seconds_excluded']}
    return summaries

def export_outer(export_dir,root_launch_file,anchors,synthetic):
    folder=Path(export_dir);c=anchors['contract'];version=c['version'];name=c['name']
    export=read(folder/'export.json');flag=export.get('synthetic_fixture')
    require((synthetic and flag=='NOT_SCIENTIFIC_EVIDENCE')or(not synthetic and flag is None),'synthetic fixture is not scientific evidence')
    require(synthetic or c['actual_science_eligible'],'failed v1 has no actual science eligibility')
    require(export['kind']==f'public_fr_cpu_matched_v{version}_final_readonly_export','versioned read-only export kind differs')
    paths={'launch':f'public_fr_cpu_matched_v{version}_launch.json','complete':'results/'+name+'/COMPLETE.json','pointer':'results/'+name+'/ARCHIVE_READY.json','manifest':f'public_fr_cpu_v{version}_code/SOURCE_MANIFEST.json','coordinator':f'public_fr_cpu_v{version}_code/cpu_coordinator_v1.py','build':f'public_fr_cpu_matched_v{version}_build/build_receipt_v1.json'}
    require(set(export['remote_receipts'])==set(paths.values()),'all6 exact genuine raw receipt paths required')
    receipts={};total=0
    for rel,pin in export['remote_receipts'].items():
        safe(rel);p=folder/'remote_receipts'/rel;require(identity(p)==pin and pin['bytes']<=65536,'exact bounded genuine remote receipt differs')
        total+=pin['bytes'];require(total<=131072,'whole remote receipts bound exceeded');receipts[rel]=p.read_bytes()
    require(identity(root_launch_file)['bytes']<=65536 and receipts[paths['launch']]==Path(root_launch_file).read_bytes(),'root launch must match genuine remote launch bytes')
    launch=decode(receipts[paths['launch']]);source_sha=digest(anchors['source_files']['cpu_coordinator_v1.py']);manifest_sha=c['manifest_sha256'];entry=c['source_dir']+'/cpu_coordinator_v1.py'
    require(launch['source_packet_sha256']==c['packet_sha256']and launch['source_manifest_sha256']==manifest_sha and launch['source_sha256']==source_sha and launch['entrypoint']==entry,'root launch/source packet binding differs')
    require(type(launch['pid'])is int and launch['pid']>0 and type(launch['start_ticks'])is int and launch['start_ticks']>0 and launch['stdin_dependency']is False and launch['host']=='lpp-cpu-0','actual detached CPU launch identity required')
    require(launch['argv']==['/opt/python/bin/python3','-B','-u',entry,'--execute','--source-manifest-sha256',manifest_sha,'--max-hours','6'],'actual fixed6hour CPU invocation differs')
    for key in('YT_writes','GPU_allocations','CPU_lifecycle_actions'):require(launch[key]==0,'launch scope contains new resource actions')
    require(receipts[paths['manifest']]==anchors['source_files']['SOURCE_MANIFEST.json']and receipts[paths['coordinator']]==anchors['source_files']['cpu_coordinator_v1.py'],'observed actual source bytes differ from frozen sources')
    require(export['source_observed_sha256']==source_sha and export['source_manifest_observed_sha256']==manifest_sha,'outer observed source hashes differ')
    token={k:launch[k]for k in('pid','start_ticks','entrypoint','source_sha256','source_manifest_sha256')};token['kind']=name
    complete=decode(receipts[paths['complete']]);pointer=decode(receipts[paths['pointer']]);require(complete['controller_identity']==token,'genuine final controller PID/start/source identity differs')
    require(all(complete[k]is True for k in('controller_finished','owned_tree_quiescent','timing_lock_released'))and type(complete['scientific_complete'])is bool and type(complete['success'])is bool,'genuine final controller/quiescence/lock release required')
    require(complete['local_archive']==pointer and export['archive']==pointer and pointer['path']==BASE+'/'+name+'.tgz','genuine final archive pointer differs')
    require(type(pointer['bytes'])is int and 0<pointer['bytes']<=MAX_ARCHIVE and type(pointer['files'])is int and 0<pointer['files']<MAX_MEMBERS,'bounded complete archive pointer required')
    for key in('YT_writes','base_package_installs','GPU_or_pod_lifecycle_actions'):require(complete[key]==0,'controller scope contains new resource actions')
    require(pointer['YT_writes']==0,'archive contains YT mutation')
    build=decode(receipts[paths['build']]);observed=[]
    for view in(export['before_view'],export['after_view']):
        owner=view['owner'];require(owner['pid']==launch['pid']and owner['expected_start_ticks']==launch['start_ticks']and owner['quiescent']is True and type(owner['exists'])is bool,'actual same-owner quiescence required')
        if owner['exists']:require(owner['start_ticks']==launch['start_ticks']and owner['state']=='Z','live or reused PID is not quiescent')
        require(view['timing_locks']=={n:False for n in LOCKS}and view['hostname']=='lpp-cpu-0','actual host/three absent timing locks required')
        require(view['receipt_sha256']=={n:digest(raw)for n,raw in receipts.items()}and view['source_observed_sha256']==source_sha and view['source_manifest_observed_sha256']==manifest_sha,'stable before/after genuine receipt/source hashes required')
        library=view['standalone_library_observed'];require(library['path']==c['build']+'/native_assoc_v1.so'and type(library['bytes'])is int and library['bytes']>0 and library['sha256']==build['library_sha256'],'observed standalone library hash differs from genuine build')
        observed.append(library)
    require(observed[0]==observed[1],'standalone library changed during collection')
    for key in('actual_cpu_pod_uid','actual_node'):
        a,b=export['before_view'].get(key),export['after_view'].get(key)
        require(a==b and(a is None or isinstance(a,str)and a),'optional current pod/node receipt changed')
    archive=folder/'final_snapshot.tgz';require(archive.stat().st_size<=MAX_ARCHIVE and identity(archive)=={k:pointer[k]for k in('bytes','sha256')},'whole raw final archive differs')
    return export,complete,token,receipts,paths

def archive_data(export_dir,anchors,pointer):
    files,rows,total=bounded_tar(Path(export_dir)/'final_snapshot.tgz',MAX_RAW,MAX_MEMBER,MAX_MEMBERS,anchors['contract']['name']+'/')
    raw=files.pop('MANIFEST.json');rows.pop('MANIFEST.json');require(digest(raw)==pointer['manifest_sha256'],'whole raw archive manifest differs')
    manifest=decode(raw);require(manifest['kind']=='public_full_friendster_matched_cpu_v1_raw'and manifest['YT_writes']==0 and manifest['files']==rows and len(rows)==pointer['files'],'every raw archive payload hash/count required')
    sources={n[8:]:v for n,v in files.items()if n.startswith('sources/')}
    require(sources==anchors['source_files'],'every exact archived frozen source payload+manifest required')
    require(all(n.startswith(('sources/','results/'))for n in files),'raw archive namespace differs')
    results={n[8:]:v for n,v in files.items()if n.startswith('results/')};require('COMPLETE.json'not in results and'ARCHIVE_READY.json'not in results,'raw packet predates genuine outer final receipts')
    return Data(results),{'payload_files':len(rows),'raw_bytes_including_manifest':total,'manifest':record(raw),'result_files':len(results),'source_files':len(sources)}

def scientific_lifecycle(data,complete,token,anchors):
    c=anchors['contract'];plan=data.obj('plan.json');expected_sources={c['source_dir']+'/'+n:record(raw)for n,raw in anchors['source_files'].items()}
    require(plan['kind']==c['name']and plan['controller_identity']==token and plan['source_bindings']==expected_sources and plan['max_hours']==6 and plan['numba_widths']==[32,64]and plan['YT_writes']==0,'actual source/finite plan binding differs')
    terminal=data.obj('SCIENTIFIC_TERMINAL.json');jobs=data.obj('scientific_jobs.json')['jobs'];require(terminal['jobs']==jobs,'scientific jobs/terminal receipt differs')
    require(complete['scientific_complete']is True and complete['success']is True and complete['failure']is None and terminal['success']is True and terminal['failure']is None,'failed/incomplete scientific run has no eligible performance ratios')
    require(all(terminal[k]is True for k in('controller_jobs_finished','owned_tree_quiescent','timing_lock_retained'))and finite_positive(terminal['runtime_seconds'])and terminal['runtime_seconds']<=21600,'finite six-hour scientifically terminal owned run required')
    require(all(terminal[k]==0 for k in('private_or_GPU_jobs','new_original_input_copies','YT_writes')),'terminal scope differs')
    expected=[{'name':'native_friendster','eligible':True,'record_count':128,'modes':list(MODES),'global_native_hashes':NATIVE_HASHES},*({'name':'numba'+str(w),'eligible':True,'record_count':128,'width':w,'methods':['scan','pairwise']}for w in(32,64))]
    require(jobs==expected,'all3 exact scientific job denominators required')
    actual_host=host(data.obj('host.json'));admission=data.obj('admission.json')
    require(admission['kind']==f'public_fr_cpu_fresh_admission_v{c["version"]}'and admission['complete']is True and admission['controller_identity']==token and admission['host']==actual_host and admission['source_manifest_sha256']==c['manifest_sha256']and admission['environment_probe_sha256']==data.proof('isolated_environment.json')['sha256']and admission['offline_wheels']==data.obj('wheel_extraction.json'),'genuine CPU admission/source/environment/host binding differs')
    original=full_input(data.obj('full_input_before.json'));require(original==full_input(data.obj('full_input_after.json'))==admission['full_original'],'all3 whole source payload hashes/stat continuity before/after required')
    return original,actual_host

def audit_export(export_dir,root_launch_file,anchors,synthetic=False):
    """Audit only immutable local bytes. Synthetic testing is Python-only."""
    export,complete,token,receipts,paths=export_outer(export_dir,root_launch_file,anchors,synthetic)
    data,inventory=archive_data(export_dir,anchors,export['archive'])
    full,actual_host=scientific_lifecycle(data,complete,token,anchors)
    before,iso=environment(data,anchors);guard=guards(data)
    conversion,build=conversion_and_build(data,full,iso,anchors)
    require(decode(receipts[paths['build']])==build,'original whole build JSON stdout and genuine external build receipt differ')
    admission=data.obj('native/admission.json');require(admission['path']==BASE+'/results/'+anchors['contract']['name']+'/admission.json'and admission['sha256']==data.proof('admission.json')['sha256']and admission['content']==data.obj('admission.json'),'native parent local admission binding differs')
    summaries,plan=native_results(data,full,iso,anchors,conversion,build)
    summaries.update(numba_results(data,full,iso,anchors,conversion,plan,actual_host))
    ratios={}
    for comparator in('native_reference_full_clear','native_upstream_torch_sparse_direct','numba32_scan','numba32_pairwise','numba64_scan','numba64_pairwise'):
        a=summaries[comparator]['sum_per_case_medians_seconds'];b=summaries['native_persistent_touched_reset']['sum_per_case_medians_seconds']
        ratios[comparator+'_over_native_persistent_touched_reset']={k:a[k]/b[k]for k in('edge_seconds','edge_plus_drnl_seconds')}
    report={'kind':'INDEPENDENT_PUBLIC_FULL_FRIENDSTER_MATCHED_CPU_AUDIT_V1','status':'PASS_SYNTHETIC_FIXTURE'if synthetic else'PASS_ACTUAL_PUBLIC_CPU_MATCHED_RESULTS','scientific_evidence_eligible':not synthetic,'source_packet':anchors['source_info'],'source_manifest_sha256':anchors['contract']['manifest_sha256'],'archive':export['archive'],'inventory':inventory,'host':actual_host,'denominator':{'existing_ordered_sets':128,'warmup':1,'repetitions_per_case':3,'native_arms':3,'physical_numba_widths':[32,64],'fixed_numba_methods':['scan','pairwise'],'selection_independently_recomputed':True,'ordered_rows_cols_original_eids_oracles':'independent immutable historical E18; three component hashes per case, not repetitions'},'raw_recomputed_summaries':summaries,'ratios_of_sums_per_case_medians':ratios,'guard_rows':guard,'direct_host_evidence':'Coordinator+both NumBa workers; standalone/upstream native subprocess workers inherit the owned unchanged local parent execution and have no direct per-worker boot receipt.','scope':['Public fullFriendster only; no two-graph, quality, sampler or training claim.','Query edge and edge+DRNL boundaries only; setup/conversion/JIT/warmup/persistent state initialization/content hashes outside query timers.','Controller OS exit code is not inferred from completion; genuine final receipt and observed same-owner quiescence are checked.','Numba method order records are checked against insertion order; its pinned original NumPy PRNG source is retained, without importing numeric libraries in this auditor.'],'local_auditor':identity(__file__),'remote_actions':0,'graph_model_or_native_library_loads':0,'YT_writes':0}
    if synthetic:report['synthetic_fixture']='NOT_SCIENTIFIC_EVIDENCE'
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir',type=Path,required=True);parser.add_argument('--root-launch',type=Path,required=True)
    parser.add_argument('--source-packet',type=Path,required=True);parser.add_argument('--historical-systems-asset',type=Path,required=True);parser.add_argument('--wheel-packet',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);require(not args.output.exists()and not args.output.is_symlink(),'fresh auditor output required')
    destination=args.output.resolve()
    protected_dirs=[args.export_dir.resolve(),Path(__file__).resolve().parent,args.source_packet.resolve().parent]
    # A report may live in a fresh outputs subdirectory, but never mutate the
    # selected evidence export or source packet directory, nor this auditor.
    require(not any(destination.is_relative_to(p)for p in protected_dirs),'auditor output must be outside evidence and source trees')
    require(destination not in{p.resolve()for p in(args.root_launch,args.source_packet,args.historical_systems_asset,args.wheel_packet)},'auditor output cannot replace evidence')
    try:
        anchors=load_anchors(args.source_packet,args.historical_systems_asset,args.wheel_packet)
        report=audit_export(args.export_dir,args.root_launch,anchors)
    except(Exception)as error:
        report={'kind':'INDEPENDENT_PUBLIC_FULL_FRIENDSTER_MATCHED_CPU_AUDIT_V1','status':'FAIL_NO_ELIGIBLE_PERFORMANCE_RESULTS','scientific_evidence_eligible':False,'error':{'type':type(error).__name__,'message':str(error)[:2000]},'local_auditor':identity(__file__),'remote_actions':0,'graph_model_or_native_library_loads':0,'YT_writes':0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb')as stream:stream.write(encoded(report))
    print(json.dumps({'status':report['status'],'output':str(args.output),'output_identity':identity(args.output)}))
    return 0 if report['scientific_evidence_eligible']else 1

if __name__=='__main__':raise SystemExit(main())
