#!/usr/bin/env python3
"""Finite independent16-consumer V3 supplement; preserve all V2 history."""
from __future__ import annotations
import argparse
import ast
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tarfile
import time

BASE=Path('/home/jovyan/seal_revision_20261124')
REPO=BASE/'cpu_code'
SOURCE=BASE/'memory_benchmark_v3.py'
OUT=BASE/'results/memory_guard_recheck_v3'
ARCHIVE=BASE/'memory_guard_recheck_v3.tgz'
YT_PATH='//home/ucp-research/projects/graph_baselines/seal_revision_20261124/memory_guard_recheck_v3.tgz'
ORIGINAL_DEADLINE=1795626589.0
MAX_JSON=32 << 20
GIB=2**30
REPRESENTATIONS=('mixed-mmap','mixed-resident','scipy-int64','conversion-peak')

def require(value, message):
    if not value:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def file_info(path):
    p=Path(path)
    require(p.is_file() and not p.is_symlink(), 'regular source missing: '+str(p))
    return {'bytes':p.stat().st_size,'sha256':sha(p)}

def load(path):
    require(Path(path).stat().st_size <= MAX_JSON, 'JSON size guard')
    value=json.loads(Path(path).read_text())
    require(isinstance(value,dict), 'JSON object required')
    return value

def write(path, data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def owned_cpu_processes(base=BASE, proc=Path('/proc')):
    """Conservative read-only ownership probe, preserving zombie ancestry."""
    rows={}
    for p in proc.iterdir():
        if not p.name.isdigit(): continue
        try:
            stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            argv=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
            entry=''
            for i in range(1,len(argv)):
                if argv[i]=='-m': entry=argv[i+1] if i+1<len(argv) else '';break
                if not argv[i].startswith('-'): entry=argv[i];break
            owned=str(base) in ' '.join(argv) and (entry in ('scripts.seal_revision.exact_benchmark','scripts.seal_revision.memory_benchmark')
                or Path(entry).name in ('exact_benchmark.py','memory_benchmark.py','memory_benchmark_v2.py','memory_benchmark_v3.py','dispatch_benchmark_v1.py'))
            rows[int(p.name)]={'pid':int(p.name),'ppid':int(stat[1]),'state':stat[0],'owned':owned,'entrypoint':entry}
        except (OSError,ValueError,IndexError): pass
    descendants={8767,11283}
    while True:
        expanded=descendants|{pid for pid,r in rows.items() if r['ppid'] in descendants}
        if expanded==descendants: break
        descendants=expanded
    result=[r for pid,r in sorted(rows.items()) if pid!=os.getpid() and r['state']!='Z' and (r['owned'] or pid in descendants)]
    for pid in (8767,11283):
        if (proc/str(pid)).exists() and pid not in rows: result.append({'pid':pid,'entrypoint':'unreadable_owned_identity'})
    return result

def assert_bindings(bindings):
    require(all(file_info(path)==expected for path,expected in bindings.items()), 'bound input/source/admission changed')

def enable_subreaper():
    """Orphaned row sessions remain descendants of this private queue process."""
    import ctypes
    require(sys.platform.startswith('linux'),'Linux owned-tree supervision required')
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(36,1,0,0,0)==0,'cannot enable private child subreaper')


def process_identity(pid,proc=Path('/proc')):
    try:
        fields=(proc/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'ppid':int(fields[1]),'pgid':int(fields[2]),'start_ticks':int(fields[19]),'state':fields[0]}
    except(OSError,ValueError,IndexError):return None


def owned_groups(owner_pid,proc=Path('/proc')):
    """Only observed descendants, including sessions adopted by our subreaper."""
    owner=process_identity(owner_pid,proc);require(owner is not None,'owned supervisor identity unavailable')
    pending=[owner_pid];seen=set();groups={}
    while pending:
        parent=pending.pop()
        if parent in seen:continue
        seen.add(parent)
        try:children=[int(x)for x in(proc/str(parent)/'task'/str(parent)/'children').read_text().split()]
        except(OSError,ValueError):continue
        for pid in children:
            current=process_identity(pid,proc)
            if current is None or current['ppid']!=parent:continue
            pending.append(pid)
            if current['state']!='Z'and current['pgid']>=2 and current['pgid']!=owner['pgid']:
                groups.setdefault(current['pgid'],[]).append(current)
    return groups


def signal_confirmed(groups,sig,*,reader=process_identity,killpg=os.killpg):
    signaled=[]
    for pgid,members in groups.items():
        for saved in members:
            now=reader(saved['pid'])
            if now is not None and now['state']!='Z'and now['start_ticks']==saved['start_ticks']and now['pgid']==pgid:
                try:killpg(pgid,sig);signaled.append(pgid)
                except ProcessLookupError:pass
                break
    return signaled


def run_owned(argv,log,*,cwd=REPO,timeout=7200,clock=time.monotonic,sleep=time.sleep,
              popen=subprocess.Popen,groups_reader=owned_groups,signaler=signal_confirmed):
    require(timeout>0,'no time remains for owned child')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    grace=min(10,timeout/4);soft_deadline=clock()+timeout-grace;hard_deadline=soft_deadline+grace
    with log.open('x')as stream:
        child=popen(argv,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    close_fds=True,start_new_session=True)
        tracked={};timed_out=False;orphan_cleanup=False;signals=[]
        def observe():
            latest=groups_reader(os.getpid())
            for pgid,members in latest.items():
                saved=tracked.setdefault(pgid,[]);known={(m['pid'],m['start_ticks'])for m in saved}
                saved.extend(m for m in members if(m['pid'],m['start_ticks'])not in known)
            return latest
        while child.poll()is None:
            observe()
            if clock()>=soft_deadline:timed_out=True;break
            sleep(min(.5,max(0,soft_deadline-clock())))
        remaining=observe()
        orphan_cleanup=bool(child.poll()is not None and remaining)
        if timed_out or orphan_cleanup:
            signals+=signaler(tracked,signal.SIGTERM)
            cleanup_deadline=min(hard_deadline,clock()+grace)
            while clock()<cleanup_deadline and observe():sleep(min(.2,max(0,cleanup_deadline-clock())))
            observe();signals+=signaler(tracked,signal.SIGKILL)
        code=child.wait(timeout=10)
        quiescence_deadline=clock()+5
        while clock()<quiescence_deadline and groups_reader(os.getpid()):
            sleep(min(.1,max(0,quiescence_deadline-clock())))
        # Reap only our adopted direct children; never discover/signals by name.
        for directory in Path('/proc').iterdir()if Path('/proc').is_dir()else[]:
            if directory.name.isdigit()and int(directory.name)!=child.pid:
                value=process_identity(int(directory.name))
                if value and value['ppid']==os.getpid()and value['state']=='Z':
                    try:os.waitpid(value['pid'],os.WNOHANG)
                    except ChildProcessError:pass
        return {'pid':child.pid,'returncode':code,'timed_out':timed_out,'timeout_seconds':timeout,
                'orphan_cleanup':orphan_cleanup,'owned_pgids_signaled':sorted(set(signals)),
                'owned_tree_quiescent':not groups_reader(os.getpid())}


def dataset_graph(base,dataset):
    return base/('data/friendster/full_csr' if dataset=='friendster' else 'data/vk/train_csr')


def graph_metadata(path):
    result={}
    for name in ('edge_starts.bin','edge_ends.bin','timestamps.bin'):
        p=path/name
        if not p.exists():continue
        require(p.is_file() and not p.is_symlink(),'invalid graph file')
        st=p.stat();result[str(p)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    starts=path/'edge_starts.bin';ends=path/'edge_ends.bin'
    with ends.open('rb') as stream:count=struct.unpack('<q',stream.read(8))[0]
    require(starts.stat().st_size%8==0 and count>=0 and ends.stat().st_size==8+count*4,'GraphCSR header/size mismatch')
    return {'files':result,'nodes':starts.stat().st_size//8,'adjacency_entries':count,
            'scope':'Read-only header/stat identity and archived download provenance; full graph bytes are not rehashed'}


def admission(base=BASE,scanner=owned_cpu_processes):
    cpu=base/'results/cpu/COMPLETE.json'
    native=base/'results/native_int32_postqueue_20261124_172000'
    dispatch=base/'results/dispatch_queue_v1'
    if not cpu.exists() or not (native/'status.json').exists() or not (dispatch/'status.json').exists():return None
    c=load(cpu)
    require(c.get('complete') is True and c.get('downstream_queue_eligible') is True
            and c.get('final_sources_verified') is True and c.get('controller_finished') is True,'CPU terminal is not verified/eligible')
    require(not (cpu.parent/'ATTENTION_REQUIRED.json').exists(),'CPU attention marker exists')
    n=load(native/'status.json');d=load(dispatch/'status.json')
    if n.get('stage') not in ('complete','failed') or d.get('stage') not in ('complete','failed'):return None
    bindings={str(cpu):file_info(cpu),str(native/'status.json'):file_info(native/'status.json'),str(dispatch/'status.json'):file_info(dispatch/'status.json')}
    rows=n.get('records',[])
    require(len(rows)==2 and {r.get('name')for r in rows}=={'scipy110','scipy115'},'native scientific two-job terminal missing')
    for row in rows:
        p=native/(row['name']+'_exit.json')
        require(load(p)==row and row.get('returncode')==0 and row.get('main_suite_overlap') is False,'native scientific exit invalid')
        bindings[str(p)]=file_info(p)
    observed=n.get('cpu_complete')
    if observed is None:
        p=native/'cpu_complete_observed.json';observed=load(p);bindings[str(p)]=file_info(p)
    require(observed['sha256']==bindings[str(cpu)]['sha256'],'native CPU marker differs')
    exit_path=dispatch/'dispatch_exit.json'
    if not exit_path.exists():return None
    child=load(exit_path)
    require(type(child.get('returncode')) is int,'dispatch child has not finished')
    bindings[str(exit_path)]=file_info(exit_path)
    if d['stage']=='complete':
        p=dispatch/'COMPLETE.json'
        if not p.exists():return None
        terminal=load(p)
        require(terminal.get('scientific_complete') is True and terminal.get('success') is True and child['returncode']==0,
                'dispatch scientific completion invalid')
        bindings[str(p)]=file_info(p)
        mode='complete'
    else:
        mode='failed_after_owned_child_finished'
    if scanner(base):return None
    return {'bindings':bindings,'native_terminal_stage':n['stage'],'dispatch_terminal_mode':mode,'active_owned_cpu_processes':[]}


def build_origin(base,v3_source):
    recovery=load(base/'results/cpu/recovery_plan_v2.json')
    old=Path(recovery['memory_v2']['path']).resolve()
    require(old.is_relative_to(base.resolve()) and old.name=='memory_benchmark_v2.py','V2 source outside experiment')
    old_info=file_info(old)
    require(old_info['sha256']==recovery['memory_v2']['sha256'],'V2 source differs from original recovery')
    tree=ast.parse(v3_source.read_text())
    previous=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name)and t.id=='PREVIOUS_V2_SHA256'for t in n.targets))
    require(previous==old_info['sha256'],'V3 previous-V2 source chain mismatch')
    datasets={};bindings={str(old):old_info,str(base/'results/cpu/recovery_plan_v2.json'):file_info(base/'results/cpu/recovery_plan_v2.json')}
    for dataset in ('friendster','vk'):
        folder=base/'results/cpu'/(dataset+'_memory');report=load(folder/'report.json')
        require(report['implementation']['source_sha256']==old_info['sha256'],'original raw is not the bound V2 implementation')
        files={}
        for path in sorted(folder.rglob('*')):
            require(not path.is_symlink(),'symlink in V2 raw history')
            if path.is_file():
                value=file_info(path);files[str(path.relative_to(folder))]=value;bindings[str(path)]=value
            else:require(path.is_dir(),'nonregular V2 history')
        graph=graph_metadata(dataset_graph(base,dataset))
        require(graph['nodes']==report['graph']['nodes'] and graph['adjacency_entries']==report['graph']['adjacency_entries'],'V2 graph shape/header differs')
        datasets[dataset]={'report':{'path':str(folder/'report.json'),**file_info(folder/'report.json')},'files':files,'graph':report['graph'],'current_graph_metadata':graph}
    downloads={}
    for name in ('input_download_manifest.json','friendster_DOWNLOAD_COMPLETE.json','vk_DOWNLOAD_COMPLETE.json'):
        path=base/'data'/name
        if path.is_file():
            require(path.stat().st_size<=2 << 20,'small download provenance guard')
            value=file_info(path);downloads[name]={'file':{'path':str(path),**value},'content':load(path)};bindings[str(path)]=value
    return {'schema_version':1,'source':{'path':str(old),**old_info},'source_sha256':old_info['sha256'],
            'recovery_plan':{'path':str(base/'results/cpu/recovery_plan_v2.json'),**file_info(base/'results/cpu/recovery_plan_v2.json')},'datasets':datasets,'download_provenance':downloads,
            'policy':'Original V2 reports, samples, failures and CPU export are not rewritten, reclassified or merged with V3.'},bindings


def assert_graphs(origin):
    for item in origin['datasets'].values():
        for path,expected in item['current_graph_metadata']['files'].items():
            st=Path(path).stat();require({'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}==expected,'read-only input graph metadata changed')


def job_argv(dataset):
    return [sys.executable,'-u',str(SOURCE),'--graph-path',str(dataset_graph(BASE,dataset)),'--file-endian','little',
            '--representations',*REPRESENTATIONS,'--workers','16','--hold-seconds','60','--memory-budget-gib','180',
            '--timeout-seconds','3600','--out',str(OUT/dataset)]


def summarize_dataset(dataset,exit_data,source_sha,origin_v2_sha,folder):
    report_path=folder/'report.json';rows=[];valid=True
    if report_path.exists():
        report=load(report_path)
        require(report['implementation']['source_sha256']==source_sha and report['implementation']['previous_v2_sha256']==origin_v2_sha,
                'V3 report/source chain mismatch')
        require(report['budget_guard_policy']['version']=='stable_pss_individual_rss_cgroup90_v3','unknown V3 guard policy')
        runs=report['runs'];require(len({(r['representation'],r['consumers'])for r in runs})==len(runs),'duplicate V3 case')
        require(all(r['representation']in REPRESENTATIONS and r['consumers']==16 for r in runs),'unexpected V3 case')
        by_name={r['representation']:r for r in runs}
    else:by_name={}
    for name in REPRESENTATIONS:
        run=by_name.get(name);raw=run.get('status')if run else'not_produced';steady=run.get('steady_shared',{})if run else{}
        measured=bool(run and raw=='complete' and run.get('returncode')==0 and run.get('abort_reason') is None
            and steady.get('steady_status')=='complete' and steady.get('valid_samples',0)>=2 and steady.get('expected_consumers')==16
            and all(type(steady.get(k))in(int,float)and math.isfinite(steady[k])and steady[k]>0
                    for k in('median_tree_pss_bytes','median_tree_pss_gib')))
        skipped=bool(run and dataset=='vk' and name=='conversion-peak' and raw=='skipped_preflight')
        outcome='complete'if measured else'expected_preflight'if skipped else'failed'
        valid &= outcome!='failed'
        rows.append({'dataset':dataset,'representation':name,'consumers':16,'raw_status':raw,'measured':measured,'outcome_class':outcome,
            'steady_pss_bytes':steady.get('median_tree_pss_bytes')if measured else None,
            'steady_pss_gib':steady.get('median_tree_pss_gib')if measured else None,
            'valid_samples':steady.get('valid_samples',0),'abort_reason':run.get('abort_reason')if run else'missing raw row',
            'preflight_estimated_peak_bytes':run.get('preflight',{}).get('estimated_peak_bytes')if run else None})
    return {'dataset':dataset,'exit':exit_data,'report':{'path':str(report_path),**file_info(report_path)}if report_path.exists()else None,
            'rows':rows,'all_rows_valid_or_expected_skip':bool(valid and exit_data['returncode']in(0,1)
                and not exit_data['timed_out']and not exit_data.get('orphan_cleanup'))}


def pack_bundle(base,out,source,origin,extra_sources=()):
    target=base/'memory_guard_recheck_v3.tgz';require(not target.exists(),'existing recheck archive; no overwrite')
    members={}
    for path in sorted(out.rglob('*')):
        require(not path.is_symlink(),'symlink in new V3 results')
        if path.is_file():
            require(not path.name.endswith('.tmp'),'unfinished V3 artifact')
            require(path.suffix in('.json','.jsonl','.log'),'unexpected array/binary in V3 raw results')
            members['results/memory_guard_recheck_v3/'+str(path.relative_to(out))]=path
        else:require(path.is_dir(),'nonregular V3 result')
    members.update({'sources/memory_benchmark_v3.py':source,'sources/queue_memory_guard_recheck_v3.py':Path(__file__).resolve(),
                    'sources/memory_benchmark_v2_origin.py':Path(origin['source']['path'])})
    require(len({p.name for p in extra_sources})==len(extra_sources),'duplicate provenance source basename')
    for path in extra_sources:
        require(path.resolve().is_relative_to(base.resolve())and path.suffix=='.py'and path.stat().st_size<=1 << 20,
                'extra provenance source must be a small Python file inside BASE')
        members['provenance_sources/'+path.name]=path
    files={name:file_info(path)for name,path in members.items()}
    require(sum(v['bytes']for v in files.values())<=64 << 20,'bounded raw archive guard64MiB')
    manifest=(json.dumps({'schema_version':1,'files':files,'scope':'Separate V3 supplement, all new raw reports/samples, actual sources, plan/admission and V2 origin hash bindings; no old rows relabeled or graph payload copied.'},indent=2)+'\n').encode()
    with target.open('xb')as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0)as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT)as tar:
                for name,path in sorted(members.items()):
                    require(file_info(path)==files[name],'archive input changed')
                    item=tarfile.TarInfo(name);item.mode=0o644;item.size=files[name]['bytes']
                    with path.open('rb')as stream:tar.addfile(item,stream)
                    require(file_info(path)==files[name],'archive input changed during read')
                item=tarfile.TarInfo('SHA256_MANIFEST.json');item.mode=0o644;item.size=len(manifest);tar.addfile(item,io.BytesIO(manifest))
    require(target.stat().st_size+sum(v['bytes']for v in files.values())+len(manifest)<=64 << 20,
            'combined compressed/unpacked supplement exceeds64MiB')
    return target


def upload_once():
    require(time.time()<ORIGINAL_DEADLINE,'original deadline reached')
    expected=file_info(ARCHIVE)
    import yt.wrapper as yt
    client=yt.YtClient(proxy='jupiter.yt.vk.team',config={'proxy':{'request_timeout':45000}})
    require(not client.exists(YT_PATH),'existing YT recheck file; no overwrite')
    with client.Transaction(timeout=120000):
        client.create('file',YT_PATH)
        with ARCHIVE.open('rb')as stream:client.write_file(YT_PATH,stream)
        stream=client.read_file(YT_PATH);h=hashlib.sha256();size=0
        try:
            for part in iter(lambda:stream.read(1 << 20),b''):h.update(part);size+=len(part)
        finally:stream.close()
        require((size,h.hexdigest())==(expected['bytes'],expected['sha256']),'archive readback mismatch')
        require(time.time()<ORIGINAL_DEADLINE,'deadline during persistence')
    write(OUT/'upload_receipt.json',{'yt_path':YT_PATH,**expected,'readback_verified':True,'saved_unix':time.time()})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admission-deadline-unix',type=float)
    p.add_argument('--extra-source',type=Path,action='append',default=[],help='Hash-bound small .py inside BASE, archived in the same runtime object')
    p.add_argument('--_upload',action='store_true',help=argparse.SUPPRESS);args=p.parse_args(argv)
    if args._upload:upload_once();return 0
    now=time.time();deadline=min(ORIGINAL_DEADLINE,now+12*3600)
    if args.admission_deadline_unix is not None:
        require(now<args.admission_deadline_unix<=deadline,'admission deadline must remain within12h/original');deadline=args.admission_deadline_unix
    require(now<deadline and not OUT.exists()and not ARCHIVE.exists(),'expired or existing V3 supplement; no resume/overwrite')
    OUT.mkdir(parents=True);sources={str(SOURCE):file_info(SOURCE),str(Path(__file__).resolve()):file_info(Path(__file__).resolve())}
    require(all(not path.is_symlink()for path in args.extra_source),'symlink provenance source forbidden')
    extra_sources=[path.resolve()for path in args.extra_source]
    require(len({path.name for path in extra_sources})==len(extra_sources),'duplicate extra source basename')
    for path in extra_sources:
        require(path.is_relative_to(BASE.resolve())and path.suffix=='.py'and path.stat().st_size<=1 << 20,'invalid extra provenance source')
        sources[str(path)]=file_info(path)
    plan={'schema_version':1,'kind':'memory_guard_recheck_v3','created_unix':now,'admission_deadline_unix':deadline,
        'original_deadline_unix':ORIGINAL_DEADLINE,'jobs':{d:job_argv(d)for d in('friendster','vk')},
        'owned_job_timeout_seconds':7200,'yt_file':YT_PATH,'separate_V3_only':True,'source_bindings':sources,
        'provenance_sources':[str(path)for path in extra_sources]}
    write(OUT/'plan.json',plan);sources[str(OUT/'plan.json')]=file_info(OUT/'plan.json')
    def status(stage,**fields):write(OUT/'status.json',{'stage':stage,'updated_unix':time.time(),'admission_deadline_unix':deadline,**fields})
    try:
        admitted=None
        while time.time()<deadline:
            assert_bindings(sources);admitted=admission()
            if admitted is not None:break
            status('waiting_for_cpu_native_dispatch');time.sleep(min(30,max(0,deadline-time.time())))
        require(admitted is not None and time.time()<deadline,'admission deadline; no measurement launched')
        origin,old_bindings=build_origin(BASE,SOURCE);write(OUT/'origin_v2.json',origin)
        bindings={**sources,**admitted['bindings'],**old_bindings,str(OUT/'origin_v2.json'):file_info(OUT/'origin_v2.json')}
        write(OUT/'admission.json',{**admitted,'read_only_graph_metadata':{d:v['current_graph_metadata']for d,v in origin['datasets'].items()}})
        enable_subreaper()
        datasets=[]
        for dataset in('friendster','vk'):
            assert_bindings(bindings);assert_graphs(origin);require(not owned_cpu_processes(),'active CPU benchmark at V3 launch')
            require(not(OUT/dataset).exists(),'existing dataset supplement output')
            status('running',dataset=dataset)
            exit_data=run_owned(plan['jobs'][dataset],OUT/(dataset+'.log'),timeout=min(7200,max(0,ORIGINAL_DEADLINE-time.time())))
            write(OUT/(dataset+'_exit.json'),exit_data)
            assert_bindings(bindings);assert_graphs(origin);require(not owned_cpu_processes(),'owned child remains after V3 job')
            datasets.append(summarize_dataset(dataset,exit_data,sources[str(SOURCE)]['sha256'],origin['source_sha256'],OUT/dataset))
        summary={'schema_version':1,'kind':'memory_guard_recheck_v3','datasets':datasets,'source_v3':sources[str(SOURCE)],
            'origin_v2':{'source_sha256':origin['source_sha256'],'origin_manifest_sha256':sha(OUT/'origin_v2.json')},
            'scope':'Separate16-consumer V3 supplement. Raw V2 rows remain unchanged. Nonunique RSS sums are diagnostic only; preflight estimates are not measurements.'}
        summary['success']=all(d['all_rows_valid_or_expected_skip']for d in datasets)
        write(OUT/'summary.json',summary);status('archiving')
        archive=pack_bundle(BASE,OUT,SOURCE,origin,extra_sources)
        assert_bindings(bindings);assert_graphs(origin);status('uploading_one_archive')
        upload=run_owned([sys.executable,'-u',str(Path(__file__).resolve()),'--_upload'],OUT/'upload.log',cwd=BASE,
                         timeout=min(900,max(0,ORIGINAL_DEADLINE-time.time())))
        require(upload['returncode']==0 and not upload['timed_out'],'bounded upload failed')
        pointer=load(OUT/'upload_receipt.json');require(pointer['readback_verified'] is True and pointer['yt_path']==YT_PATH
            and pointer['sha256']==sha(archive)and pointer['bytes']==archive.stat().st_size,'upload receipt mismatch')
        assert_bindings(bindings);assert_graphs(origin);require(not owned_cpu_processes(),'active CPU benchmark before publication')
        write(OUT/'archive_pointer.json',pointer)
        terminal={'schema_version':1,'kind':'memory_guard_recheck_v3','stage':'complete','complete':True,'controller_finished':True,
            'scientific_terminal':True,'persisted':True,'success':summary['success'],
            'terminal_status':'complete'if summary['success']else'completed_with_failures','datasets_attempted':2,
            'summary_sha256':sha(OUT/'summary.json'),'archive_pointer_sha256':sha(OUT/'archive_pointer.json'),
            'archive_sha256':pointer['sha256'],'yt_path':YT_PATH,'finished_unix':time.time()}
        status('complete',success=summary['success']);write(OUT/'COMPLETE.json',terminal)
    except BaseException as error:
        status('failed',error_type=type(error).__name__,error=str(error)[:240]);raise
    print(json.dumps({'stage':'complete','success':summary['success'],'out':str(OUT)}));return 0


if __name__=='__main__':raise SystemExit(main())
