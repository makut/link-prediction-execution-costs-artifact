#!/usr/bin/env python3
"""Fresh portable seven-arm public strict-clean Citation2 CPU study; plan by default."""
from pathlib import Path
import argparse,importlib,io,math,os,shutil,sys,tarfile,time
sys.dont_write_bytecode=True
from portable_common import *
from local_execution import Execution

PHASES=('base_environment_before','create_isolated_env','install_offline_native','isolated_environment','base_environment_after_install',
 'build_standalone_native','full_input_before','native_citation2','numba32','numba64','full_input_after','isolated_environment_after','base_environment_after')

def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-csr',dest='full_csr',type=Path,required=True);p.add_argument('--wheel-packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--python',type=Path,default=Path(sys.executable),help='Supplied Linux Python3.10 with exact numerical stack; lexical venv path retained')
    p.add_argument('--coordination-dir',type=Path);p.add_argument('--memory-budget-gib',type=float,default=8);p.add_argument('--max-hours',type=float,default=6)
    p.add_argument('--source-manifest-sha256');p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv);a.out=a.out.resolve();a.full_csr=a.full_csr.resolve();a.wheel_packet=a.wheel_packet.resolve()
    a.python=Path(os.path.abspath(a.python));a.coordination_dir=(a.coordination_dir or a.out.parent).resolve()
    require(all(math.isfinite(v)and v>0 for v in(a.memory_budget_gib,a.max_hours))and a.memory_budget_gib<=180 and a.max_hours<=6,'finite positive<=180GiB and<=6hour bounds required')
    require(not os.path.lexists(a.out),'fresh output; preserve previous/failure experiment')
    for path in(HERE,a.full_csr,a.wheel_packet):require(not a.out.is_relative_to(path)and not path.is_relative_to(a.out),'source/fullinput/wheel and output must be disjoint')
    require(not a.coordination_dir.is_relative_to(HERE)and not a.coordination_dir.is_relative_to(a.full_csr)and not a.coordination_dir.is_relative_to(a.out),'coordination must be outside source/input/output trees')
    return a

def native_command(L,python):
    return[str(python),'-B','-u',str(FROZEN/'original/native/run_parent_replay_v1.py'),'--archive',str(L['cohort_archive']),
      '--count','128','--output-dir',str(L['out']/'native'),'--execute','--library',str(L['build']/'native_assoc_v1.so'),
      '--native-dir',str(L['native']),'--prepare-native','--graph-path',str(L['full']),'--admission-marker',str(L['out']/'admission.json'),
      '--include-upstream','--external-helper',str(FROZEN/'original/external_operator_benchmark.py'),
      '--external-helper-sha256','525e88c904fd03e10fc617853c8d733e88af90487cbbdd9122ced97962df112c',
      '--conversion-max-native-gib','1','--max-assoc-gib','0.05','--max-peak-gib','8','--max-query-neighbors','20000000',
      '--warmup','1','--repeats','3','--timeout-seconds','3600','--conversion-timeout-seconds','1800','--order-seed','20261125']

def plan(a):
    bindings=verify_portable_sources();original=load_originals();selection=independent_ordered_selection();L=paths(a.out,a.full_csr)
    python=L['venv']/'bin/python';envprobe=HERE/'portable_environment.py'
    jobs=[{'name':'base_environment_before','argv':[str(a.python),'-B',str(envprobe),'--probe','base','--output',str(L['out']/'base_environment_before.json')],'timeout_seconds':120},
      {'name':'create_isolated_env','argv':[str(a.python),'-m','venv','--without-pip',str(L['venv'])],'timeout_seconds':120},
      {'name':'install_offline_native','argv':[str(python),'-m','pip','--isolated','install','--no-index','--no-deps','--no-compile','--no-cache-dir','--disable-pip-version-check','--ignore-installed','--prefix',str(L['venv']),
         *[str(L['wheels']/name)for name in sorted(importlib.import_module('cpu_environment_v1').WHEELS)]],'timeout_seconds':600}]
    for name,mode,interpreter in(('isolated_environment','isolated',python),('base_environment_after_install','base',a.python)):
        cmd=[str(interpreter),'-B',str(envprobe),'--probe',mode,'--before',str(L['out']/'base_environment_before.json'),'--output',str(L['out']/(name+'.json'))]
        if mode=='isolated':cmd+=['--venv',str(L['venv'])]
        jobs.append({'name':name,'argv':cmd,'timeout_seconds':120})
    jobs+=[{'name':'build_standalone_native','argv':[str(python),'-B',str(FROZEN/'original/native/build_native_v1.py'),'--output-dir',str(L['build'])],'timeout_seconds':180},
      {'name':'full_input_before','operation':'stream_both_strict_clean_CSR_payload_SHA_and_actual_header_stat','timeout_seconds':600},
      {'name':'native_citation2','argv':native_command(L,python),'timeout_seconds':13000}]
    for width in(32,64):jobs.append({'name':'numba'+str(width),'argv':[str(python),'-B','-u',str(HERE/'portable_worker.py'),'--config',str(L['out']/'portable_config.json'),'--width',str(width)],'timeout_seconds':1800})
    jobs.append({'name':'full_input_after','operation':'stream_both_strict_clean_CSR_payload_SHA_and_actual_header_stat','timeout_seconds':600})
    for name,mode,interpreter in(('isolated_environment_after','isolated',python),('base_environment_after','base',a.python)):
        cmd=[str(interpreter),'-B',str(envprobe),'--probe',mode,'--before',str(L['out']/'base_environment_before.json'),'--output',str(L['out']/(name+'.json'))]
        if mode=='isolated':cmd+=['--venv',str(L['venv'])]
        jobs.append({'name':name,'argv':cmd,'timeout_seconds':120})
    require([j['name']for j in jobs]==list(PHASES),'exact13phases required')
    return{'kind':'PUBLIC_PORTABLE_STRICT_CLEAN_CITATION2_CPU7_V1','execute':a.execute,'source_manifest_sha256':sha(HERE/'SOURCE_MANIFEST.json'),
      'original_E22_source_packet':SOURCE_PACKET,'original_E22_manifest_sha256':ORIGINAL_MANIFEST_SHA,'sources':bindings,
      'train_csr_dimensions':{'nodes':original.N,'stored_entries':original.M},'both_expected_payloads':original.FULL_FILES,
      'frozen_selection_sha256':sha(RECORDED_COHORT/'FROZEN_128_SELECTION.json'),'ordered_sets':128,'warmup':1,'repeats':3,
      'native_modes':['reference_full_clear','persistent_touched_reset','upstream_torch_sparse_direct'],'numba_widths':[32,64],'numba_methods':['scan','pairwise'],
      'wheel_packet':WHEEL_PACKET,'paths':{k:str(v)for k,v in L.items()},'jobs':jobs,'memory_budget_gib':a.memory_budget_gib,'max_hours':a.max_hours,
      'coordination_dir':str(a.coordination_dir),'default_plan_graph_reads':0,'default_plan_numeric_imports':0,'default_plan_output_creations':0,
      'scope':'Fresh public local rerun of the exact recorded cohort around unchanged scientific functions; no node sets are resampled. Different host timings are new evidence, not byte-identical reproduction.',
      'runtime_storage':'native 509,051,728 bytes plus isolated environment/caches; excluded from 96 MiB raw evidence budget; no source graph copy',
      'YT_writes':0,'remote_actions':0}

def source_bindings_after_environment(before,isolated,L):
    from portable_environment import set_roots
    env=importlib.import_module('cpu_environment_v1');set_roots(env,before)
    return env.environment_bindings(isolated,L['venv'])

def pack_local(L):
    archive=L['root']/'public_cpu7_raw.tgz';require(not archive.exists(),'fresh local archive required')
    entries={}
    for prefix,folder,names in(('results',L['out'],[p for p in L['out'].rglob('*')if p.is_file()]),
      ('sources',HERE,[HERE/name for name in read(HERE/'SOURCE_MANIFEST.json')['files']]+[HERE/'SOURCE_MANIFEST.json'])):
        for p in names:
            require(not p.is_symlink()and p.stat().st_size<=16*MIB,'bounded regular raw/source archive member required');entries[prefix+'/'+p.relative_to(folder).as_posix()]=p
    require(len(entries)<=256 and sum(p.stat().st_size for p in entries.values())<=96*MIB,'bounded raw evidence archive inventory required')
    manifest={'kind':'NEW_PUBLIC_CPU7_RAW_LOCAL','files':{name:pin(p)for name,p in entries.items()},'YT_writes':0,'remote_actions':0}
    encoded=(json.dumps(manifest,indent=2)+'\n').encode()
    with tarfile.open(archive,'w:gz')as tar:
        for name,p in entries.items():tar.add(p,arcname='public_citation2_cpu7/'+name,recursive=False)
        m=tarfile.TarInfo('public_citation2_cpu7/MANIFEST.json');m.size=len(encoded);tar.addfile(m,io.BytesIO(encoded))
    require(archive.stat().st_size<=64*MIB and all(pin(p)==manifest['files'][name]for name,p in entries.items()),'whole immutable archive bound/source check')
    return{'path':str(archive),**pin(archive),'manifest_sha256':hashlib.sha256(encoded).hexdigest(),'files':len(entries)}

def execute(a,definition):
    require(sys.platform.startswith('linux'),'actual execution requires Linux')
    require(a.source_manifest_sha256==sha(HERE/'SOURCE_MANIFEST.json'),'reviewed public source manifest pin required')
    require(a.python.is_file()and pin(a.wheel_packet)==WHEEL_PACKET,'supplied interpreter and exact separate offline wheel asset required')
    L=paths(a.out,a.full_csr);original=load_originals();original.BASE=L['base'];original.layout=lambda base:dict(L)
    coordinator=importlib.import_module('cpu_coordinator_v1');env=importlib.import_module('cpu_environment_v1');bindings=dict(definition['sources']);started=time.monotonic();scientific=[]
    with Execution(a,L,bindings)as execution:
        save(L['out']/'plan.json',definition)
        save(L['out']/'portable_config.json',{'root':str(L['root']),'full':str(L['full']),'paths':{k:str(v)for k,v in L.items()},'source_manifest_sha256':definition['source_manifest_sha256']})
        for path in(L['out']/'plan.json',L['out']/'portable_config.json'):bindings[str(path)]=pin(path)
        for job in definition['jobs']:
            name=job['name']
            if name.startswith('full_input_'):
                # The whole hash admission itself runs in the owned process
                # supervisor, with the same finite guard as numerical phases.
                command=[str(L['venv']/'bin/python'),'-B',str(HERE/'portable_input.py'),'--config',str(L['out']/'portable_config.json'),'--phase',name.rsplit('_',1)[1]]
                outcome=execution.run(name,command,job['timeout_seconds'])
            else:outcome=execution.run(name,job['argv'],job['timeout_seconds'])
            if name=='base_environment_before':
                from portable_environment import set_roots
                before=read(L['out']/'base_environment_before.json');set_roots(env,before)
                save(L['out']/'wheel_extraction.json',env.unpack_wheels(a.wheel_packet,L['wheels']))
            elif name=='create_isolated_env':save(L['out']/'base_site_binding.json',env.add_base_site(L['venv'],before))
            elif name=='base_environment_after_install':
                isolated=read(L['out']/'isolated_environment.json');save(L['out']/'environment_admission.json',env.verify(before,isolated,read(L['out']/'base_environment_after_install.json')))
                bindings.update(source_bindings_after_environment(before,isolated,L))
            elif name=='build_standalone_native':
                build=read(L['build']/'build_receipt_v1.json');require(build['source_sha256']==original.SCIENCE_PINS['original/native/native_assoc_v1.cpp']and build['library_sha256']==sha(L['build']/'native_assoc_v1.so'),'actual fresh standalone source/library differs')
                for p in(L['build']/'native_assoc_v1.so',L['build']/'build_receipt_v1.json'):bindings[str(p)]=pin(p)
            elif name=='full_input_before':
                full=read(L['out']/'full_input_before.json');host=original.host_receipt();original.verify_host(host,host);save(L['out']/'host.json',host)
                L['cohort'].mkdir()
                for source in sorted(RECORDED_COHORT.iterdir()):
                    require(source.is_file() and not source.is_symlink(), 'regular fixed cohort payload required')
                    target=L['cohort']/source.name;shutil.copyfile(source,target);require(pin(source)==pin(target),'fixed cohort copy differs')
                cohort=read(L['cohort']/'COHORT_READY.json')
                require(cohort['archive']==pin(L['cohort_archive']) and cohort['plan']==pin(L['cohort_plan']) and cohort['protocol']==pin(FROZEN/'COHORT_PROTOCOL.json'),'original fixed cohort bindings differ')
                for source in L['cohort'].iterdir():bindings[str(source)]=pin(source)
                save(L['out']/'admission.json',{'kind':'citation2_cpu_fresh_admission_v1','complete':True,'controller_identity':execution.token,
                  'host':host,'full_original':full,'source_manifest_sha256':ORIGINAL_MANIFEST_SHA,
                  'portable_source_manifest_sha256':definition['source_manifest_sha256'],'environment_probe_sha256':sha(L['out']/'isolated_environment.json'),
                  'offline_wheels':read(L['out']/'wheel_extraction.json'),'cohort':cohort,'meaning':'Genuine fresh generic local CPU admission; no internal hostname/pod/storage gate.'})
                for path in(L['out']/'admission.json',L['out']/'host.json',L['out']/'full_input_before.json'):bindings[str(path)]=pin(path)
            elif name=='native_citation2':scientific.append({'name':name,**coordinator.classify_native(L['out']/'native',outcome,build['library_sha256'],full,L['native'])})
            elif name in('numba32','numba64'):
                width=int(name[5:]);scientific.append({'name':name,**coordinator.classify_numba(L['out']/('citation2_neighbors'+str(width)+'.json'),outcome,width,L['out']/'native',host)})
            elif name=='full_input_after':require(read(L['out']/'full_input_after.json')==full,'both whole original Citation2 inputs/stat changed')
            elif name=='base_environment_after':
                require(read(L['out']/'isolated_environment_after.json')==isolated,'isolated environment changed')
                save(L['out']/'environment_verification.json',env.verify(before,isolated,read(L['out']/'base_environment_after.json')))
                original.verify_host(host,original.host_receipt())
            save(L['out']/'scientific_jobs.json',{'jobs':scientific})
        require(len(scientific)==3 and all(row['eligible']is True for row in scientific),'all3 science groups required')
        execution.check_bindings()
        save(L['out']/'runtime_bindings_final.json',bindings)
        save(L['out']/'SCIENTIFIC_TERMINAL.json',{'success':True,'jobs':scientific,'source_manifest_sha256':definition['source_manifest_sha256'],
          'all7arms_all128sets_all3repetitions':True,'runtime_seconds':time.monotonic()-started,'timing_excludes':'setup/conversion/JIT/warmup/content validation/once-per-worker initialization',
          'new_graph_source_copies':0,'YT_writes':0,'remote_actions':0})
        save(L['out']/'ARCHIVE_READY.json',pack_local(L))
    return 0

def main(argv=None):
    a=arguments(argv);definition=plan(a);print(json.dumps(definition,indent=2),flush=True)
    return execute(a,definition)if a.execute else 0
if __name__=='__main__':raise SystemExit(main())
