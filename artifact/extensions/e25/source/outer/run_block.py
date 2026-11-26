#!/usr/bin/env python3
"""One complete eight-app allocation block; local PVC raw archive only, no YT/API/release."""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib.util,io,json,math,os,signal,socket,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent
SOURCE_STATE='FROZEN_BOUND'
PACKET_MANIFEST_SHA='3fd8783173e4a850018d860459881aafd1dd49e8d4cda101eb273641e47af086'
PROTOCOL_SHA='e89b371089bd01f71579fdc00ec06004373bb549542658cc223914759f392e18'
ENV_POLICY='fixed27_env_v5'
MAX_SECONDS=4200
PRODUCER_SECONDS=3900
MIB=1<<20
GIB=1<<30
MAX_JSON=32*MIB
MAX_MEMBER=32*MIB
MAX_RAW=384*MIB
MAX_ARCHIVE=128*MIB
MAX_FILES=4096
ARMS=('reference_union_compiled','reference_union_native_persistent')
HORIZONS={'short320':dict(steps=10,measured=320,consumed=576,checkpoints=[32,256,576]),'long6400':dict(steps=200,measured=6400,consumed=6656,checkpoints=[32,256,1280,3456,6656])}
WORKLOAD=dict(bytes=192128,sha256='340e61cdd1eb8d329d52be164f919480e6a41feb4c620852df0393c2e9a192f0')
MODEL_SHA='f88254079fe6c152b4c0153c857a63516348353d578f918622e7f6719559618e'
IMAGE_SHA='2c0d776d47eb7d3b763f5a42c70d51bff2ba6b1f9cb1f01b4adbad76b658cbb1'
FR3={'edge_starts.bin':dict(bytes=524866928,sha256='b6ebf026fa54498f241af834b96b5496b2a1bcbf8be3edea2f4b3a1cf0a674e9'),'edge_ends.bin':dict(bytes=14448537088,sha256='0f74897efa3172252a8371bb0d2c8e0129602475ebb84a863ec1210f9c5e5973'),'num_nodes.txt':dict(bytes=9,sha256='8551541f6f2ef955481af83ef5e6f92c5f206c968d5d9db6a4437efecf767753')}
PRIOR_FAILED_REFERENCE=dict(archive='E25_ENV_V4_FAILED_DIAGNOSTIC_v1.tgz',bytes=2697876,sha256='5a470ea09dcfef2c754273dcb9d63288739b7c779799952e5f24a853b25c2365',completed_jobs=9,status='FAILED_EXCLUDED_FROM_SCIENCE',scope='Reference only; preserve this capsule once in the final aggregate, with older diagnostics nested. No old app enters new measurements.')

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON,
            'bounded regular JSON required: ' + str(path))
    return json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            digest.update(block)
    return digest.hexdigest()

def info(path, maximum=MAX_MEMBER):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum,
            'bounded regular payload required: ' + str(path))
    return dict(bytes=path.stat().st_size, sha256=sha(path))

def save_new(path, value):
    path = Path(path)
    require(not os.path.lexists(path), 'preserve existing evidence; no overwrite: ' + str(path))
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    require(len(data) <= MAX_JSON, 'bounded new JSON required')
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def safe_relative(name):
    rel = PurePosixPath(name)
    return bool(name) and not rel.is_absolute() and '..' not in rel.parts and '\\' not in name and str(rel) == name

def source_packet(root, manifest_name, expected_sha, *, inventory=True):
    root = Path(root)
    require(root.resolve() == root and root.is_dir() and not root.is_symlink(), 'canonical source packet directory required')
    require(isinstance(expected_sha, str) and len(expected_sha) == 64, 'final root-bound source manifest SHA required')
    manifest = root / manifest_name
    require(info(manifest)['sha256'] == expected_sha, 'source manifest SHA differs: ' + str(manifest))
    data = read(manifest)
    rows = data['files']
    if isinstance(rows, list):
        require(len({r['path'] for r in rows}) == len(rows), 'unique manifest source names required')
        rows = {r['path']: r for r in rows}
    require(isinstance(rows, dict) and 0 < len(rows) < MAX_FILES, 'source manifest count exceeded')
    total = 0
    for name, pin in rows.items():
        require(safe_relative(name), 'safe relative source path required')
        rel = PurePosixPath(name)
        require('__pycache__' not in rel.parts and not any('cache' in p.lower() for p in rel.parts)
                and rel.suffix not in ('.so', '.dylib', '.dll', '.o', '.a', '.pyc', '.nbc', '.nbi', '.bin', '.pt', '.pth', '.pkl')
                and rel.name not in ('.env', 'token', 'credentials') and rel.suffix not in ('.key', '.pem'),
                'cache, native binary, graph, checkpoint or credential refused in source')
        if rel.suffix == '.npy':
            require(name == 'workloads/friendster_pairs.npy' and pin == WORKLOAD, 'only exact small source workload binary permitted')
        require((root / name).resolve() == root / name
                and info(root / name) == {k: pin[k] for k in ('bytes', 'sha256')}, 'source differs: ' + name)
        total += pin['bytes']
    require(total <= 32 * MIB, 'small source packet cap exceeded')
    if inventory:
        observed = set()
        for path in root.rglob('*'):
            require(not path.is_symlink(), 'source symlink refused')
            if path.is_file() and path != manifest:
                observed.add(str(path.relative_to(root)))
        require(observed == set(rows), 'source inventory differs')
    return data

def lock_available(coordination):
    coordination = Path(coordination)
    require(not os.path.lexists(coordination / 'PUBLIC_SYSTEMS_ACTIVE.json'), 'foreign or stale timing active marker preserved')
    path = coordination / 'PUBLIC_SYSTEMS_RUNNING.lock'
    require(not path.is_symlink(), 'timing lock symlink refused')
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream, fcntl.LOCK_UN)
    return dict(path=str(path), active_marker_absent=True, exclusive_flock_acquired_and_released=True,
                checked_unix=time.time(), scope='Persistent lock file retained; live flock availability observed.')

def successful_exit(receipt):
    require(type(receipt.get('returncode')) is int and receipt['returncode'] == 0
            and receipt.get('timed_out') is False and receipt.get('orphan_cleanup') is False
            and receipt.get('owned_tree_quiescent') is True and receipt.get('actual_wait_returned') is True
            and receipt.get('owned_pgids_signaled') == [] and receipt.get('guard_abort_reason') is None,
            'actual wait0, no timeout/orphan/signals/guard, and owned-tree quiescence required')

def archive_results(archive, files):
    archive = Path(archive)
    require(not os.path.lexists(archive) and 0 < len(files) <= MAX_FILES, 'fresh bounded archive required')
    pinned = {name: info(path) for name, path in files.items()}
    require(sum(row['bytes'] for row in pinned.values()) <= MAX_RAW - MIB, 'small result/source raw archive cap exceeded')
    for name in files:
        rel = PurePosixPath(name)
        require(name.startswith('e25/') and not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,
                'safe regular archive namespace required')
    manifest = (json.dumps(dict(files=pinned, graphs_included=False, numba_cache_included=False,
                  scope='Frozen E25 source and bounded textual scientific outputs with actual post-exit lifecycle provenance; no graph payloads, native scratch, compiled libraries, caches, credentials or checkpoints.'),
                  indent=2, sort_keys=True) + '\n').encode()
    require(len(manifest) <= MIB, 'manifest1MiB cap exceeded')
    with tarfile.open(archive, 'x:gz') as tar:
        for name, path in sorted(files.items()):
            require(info(path) == pinned[name], 'result changed before archive')
            member = tarfile.TarInfo(name)
            member.size, member.mode = pinned[name]['bytes'], 0o644
            with path.open('rb') as stream:
                tar.addfile(member, stream)
            require(info(path) == pinned[name], 'result changed during archive')
        member = tarfile.TarInfo('SHA256_MANIFEST.json')
        member.size, member.mode = len(manifest), 0o644
        tar.addfile(member, io.BytesIO(manifest))
    expected = info(archive, MAX_ARCHIVE)
    observed = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            require(member.isfile() and member.name not in observed and member.size <= MAX_MEMBER,
                    'regular bounded unique archived members required')
            digest, count = hashlib.sha256(), 0
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(MIB), b''):
                    count += len(block)
                    digest.update(block)
            require(count == member.size, 'truncated archived member')
            observed[member.name] = dict(bytes=count, sha256=digest.hexdigest())
    require(observed == {**pinned, 'SHA256_MANIFEST.json': dict(bytes=len(manifest), sha256=hashlib.sha256(manifest).hexdigest())}
            and info(archive, MAX_ARCHIVE) == expected, 'all whole archive members and final SHA must match')
    return dict(**expected, archive=str(archive), files=len(pinned), manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                all_whole_member_hashes_verified=True, raw_payload_bytes=sum(r['bytes'] for r in observed.values()),
                compressed_cap_bytes=MAX_ARCHIVE, raw_cap_bytes=MAX_RAW, no_graph_payloads=True)

def cleanup(support):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        support.signal_confirmed(support.owned_groups(os.getpid()), sig)
        until = time.monotonic() + 5
        while time.monotonic() < until and support.owned_groups(os.getpid()):
            time.sleep(.1)
    return not support.owned_groups(os.getpid())

def pss_row(sample):
 """Derive eligibility from original memberships/process records, not flags."""
 if 'sampling_error'in sample:return None,'sampling_error'
 before,after=sample['membership_before'],sample['membership_after'];processes=sample['processes']
 def members(rows):
  require(len({r['pid']for r in rows})==len(rows),'duplicate PSS PID')
  require(all(type(r['pid'])is int and r['pid']>0 and type(r['start_ticks'])is int and r['start_ticks']>0 for r in rows),'invalid PSS membership identity')
  return {r['pid']:r['start_ticks']for r in rows}
 a,b=members(before),members(after)
 stable=sample['membership_before_complete']is True and sample['membership_after_complete']is True and a==b
 complete=bool(processes)and len(processes)==len(a)and all(type(r['pss_bytes'])is int and r['pss_bytes']>=0 for r in processes)
 require(len({r['pid']for r in processes})==len(processes),'duplicate PSS process record')
 if complete:require({r['pid']for r in processes}==set(a),'complete PSS records/membership differ')
 require(sample['membership_stable']is stable and sample['pss_complete']is complete
  and sample['publication_eligible']is(stable and complete),'raw PSS membership/completeness flags contradict records')
 require(sample['decision']['abort_reason']is None,'observed resource guard abort')
 same_phase=sample['phase_before']==sample['phase_after'];eligible=stable and complete and same_phase
 require(sample['phase_stable']is same_phase and sample['phase_publication_eligible']is eligible,'phase eligibility flag contradicts recorded phases')
 if complete:require(sample['tree_pss_bytes']==sum(r['pss_bytes']for r in processes),'PSS tree sum differs from original process PSS')
 if eligible:
  require(sample['backend']=='linux-proc-smaps_rollup','actual Linux PSS backend required')
  return (sample['phase_before'],sum(r['pss_bytes']for r in processes)),'eligible'
 return None,'ineligible'


def small(path):
    with Path(path).open()as stream:raw=stream.read(65537)
    require(len(raw)<=65536,'bounded local platform metadata');return raw.strip()

def current_mount(base):
    rows=[]
    for line in small('/proc/self/mountinfo').splitlines():
        left,right=line.split(' - ',1);a,z=left.split(),right.split()
        if a[4]==str(base):rows.append(dict(root=a[3],mount_point=a[4],major_minor=a[2],filesystem=z[0],source=z[1],options=a[5].split(','),super_options=z[2].split(',')))
    require(len(rows)==1,'unique explicit BASE mount required');return rows[0]

def platform_admission(args):
    require(info(args.platform_admission_json)['sha256']==args.platform_admission_sha256,'whole root platform proof SHA differs')
    p=read(args.platform_admission_json);require(p['kind']=='E25_V6_ACTUAL_BLOCK_PLATFORM_ADMISSION'and p['block']==args.repetition_block and p['actual_scientific_jobs_started']==0,'genuine pre-block actual platform proof required')
    base=Path(p['base']);require(base.resolve()==base and base.is_dir()and not base.is_symlink(),'canonical actual PVC BASE required')
    require(socket.gethostname()==p['hostname']and os.environ.get('POD_UID')==p['pod_uid']and p['sts_uid']and small('/proc/sys/kernel/random/boot_id')==p['boot_id'],'actual Pod/host/boot differs from root admission')
    require(isinstance(p['image_digest'],str)and p['image_digest'].endswith(IMAGE_SHA),'same original image digest required; digest is root API-attested')
    chosen=p['cpu_affinity'];require(isinstance(chosen,list)and len(chosen)==27 and chosen==sorted(set(chosen))and all(type(x)is int for x in chosen)and sorted(os.sched_getaffinity(0))==chosen,'actual inherited exact27 pool differs')
    require(small('/sys/fs/cgroup/cpu.max')==p['cpu_max']=='2700000 100000'and small('/sys/fs/cgroup/memory.max')==p['memory_max']=='274877906944','actual CPU quota/memory differs')
    topology=[]
    for cpu in chosen:
        root=Path('/sys/devices/system/cpu')/('cpu'+str(cpu));top=root/'topology'
        topology.append(dict(cpu=cpu,physical_package_id=small(top/'physical_package_id'),core_id=small(top/'core_id'),thread_siblings_list=small(top/'thread_siblings_list'),numa_nodes=sorted(x.name for x in root.glob('node[0-9]*'))))
    require(topology==p['topology'],'actual CPU topology differs')
    storage=p['storage'];mount=current_mount(base);capacity=os.statvfs(base)
    require(storage['PVC_name']=='claim-vk-b'and storage['PVC_uid']=='cfd8708a-cd0b-4cd9-914c-a860abf63ef5'and storage['volume_name']=='pvc-'+storage['PVC_uid']and storage['subPath']=='lpp_seal_e25_20261126_pvc_01','actual root PVC identity differs')
    require(storage['mount']==mount and mount['root']=='/'+storage['subPath']and 'rw'in mount['options']and 'ro'not in mount['options']and 'ro'not in mount['super_options'],'current PVC mount differs')
    require(base.stat().st_dev==storage['device']and mount['major_minor']==str(os.major(storage['device']))+':'+str(os.minor(storage['device']))and not(capacity.f_flag&os.ST_RDONLY)and capacity.f_bavail*capacity.f_frsize>=40*GIB and(capacity.f_files==0 or capacity.f_favail>0),'current pre-block40GiB/RW/inode/device admission failed')
    require(storage['mount_readonly']is False and storage['bytes_available']>=40*GIB and storage['write_fsync_readback_unlink']is True and(storage['inodes_total']==0 or storage['inodes_available']>0),'root owned PVC write proof absent')
    rootfs=os.statvfs('/home/jovyan');require(p['root_layer']['mount_readonly']is False and p['root_layer']['write_fsync_readback_unlink']is True and not(rootfs.f_flag&os.ST_RDONLY)and rootfs.f_bavail*rootfs.f_frsize>=4*GIB,'root layer admission failed')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20).stdout.strip().splitlines()
    require(len(gpu)==1,'one whole H100 required');uuid,name,memory,driver=[x.strip()for x in gpu[0].split(',')]
    require(dict(uuid=uuid,name=name,memory_total_mib=int(memory),driver=driver)==p['gpu']and'H100'in name and int(memory)>=79000,'actual GPU/image-type binding differs')
    busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20).stdout.strip();require(not busy,'GPU busy; no overlap permitted')
    for path in(args.packet,args.friendster_root,args.platform_admission_json,args.coordination,args.out.parent):
        require(path.resolve()==path and path.is_relative_to(base),'all run/input/source paths must be canonical within admitted PVC')
        parent=path
        while not parent.exists():parent=parent.parent
        require(parent.stat().st_dev==storage['device'],'run path backing device differs')
    require(not args.out.exists()and not os.path.lexists(args.out),'fresh block output required')
    require(not args.out.is_relative_to(args.packet)and not args.packet.is_relative_to(args.out)and not args.out.is_relative_to(args.friendster_root)and not args.friendster_root.is_relative_to(args.out),'source/input/output must be disjoint')
    require(not args.coordination.is_relative_to(args.out)and not args.coordination.is_relative_to(args.packet)and not args.coordination.is_relative_to(args.friendster_root),'coordination outside source/input/output required')
    require(args.friendster_root==base/'data/friendster/full_csr','existing byte-pinned compact graph required; no re-ingress')
    lock_available(args.coordination)
    return p

def boundary(value,chosen):
    require(value['status']=='PASS'and value['policy']==ENV_POLICY and value['failed_predicates']==[]and all(x is True for x in value['predicates'].values()),'own boundary hard guard failed')
    require(value['predicates']['boundary_own_leader_mask_exact']is True and len(value['processes'])==1 and value['processes'][0]['leader_mask']==chosen,'own boundary must retain exact27 pool')
    r=value['boundary_rescan'];require(r['terminal_kind']=='fully_stable_pass'and 1<=r['attempt_count']<=8 and len(r['attempts'])==r['attempt_count']and r['elapsed_seconds']<=0.5,'complete bounded own boundary receipt required')

def disk_guard(root,p):
    path=root/'disk_observations.jsonl';require(0<info(path,8*MIB)['bytes']<=8*MIB,'nonempty bounded disk journal required');last=0;count=0
    with path.open('rb')as stream:
        while True:
            line=stream.readline(4097)
            if not line:break
            require(len(line)<=4096 and line.endswith(b'\n'),'complete bounded disk line required');v=json.loads(line)
            require(type(v['observed_monotonic_ns'])is int and v['observed_monotonic_ns']>last and v['path']==str(root)and v['device']==p['storage']['device']and v['mount_readonly']is False and 4*GIB<=v['bytes_available']<=v['bytes_total']and v['inodes_reported']is(v['inodes_total']>0)and(not v['inodes_reported']or 0<v['inodes_available']<=v['inodes_total']),'recorded runtime4GiB/PVC/inode/clock guard failed')
            last=v['observed_monotonic_ns'];count+=1
    latest_path=root/'output_inventory_observation.json';require(info(latest_path,4096)['bytes']<=4096,'latest sampled inventory4KiB cap');latest=read(latest_path)
    require(latest['disk_observed_monotonic_ns']==last<=latest['observed_monotonic_ns']and latest['observed_non_runtime_bytes']<=2*GIB and latest['complete_entry_observations']is(latest['vanished_entries']==0),'sampled non-atomic output cap proof differs')
    return dict(disk_history=info(path),observations=count,latest_inventory=info(latest_path),scope='Recorded sampled thresholds only, not reservation or full statvfs block-arithmetic reconstruction.')

def pss_journal(path):
    require(info(path)['bytes']>0,'raw PSS journal required');counts=dict(eligible=0,ineligible=0,sampling_error=0);maximum=None
    with path.open('rb')as stream:
        while True:
            line=stream.readline(MIB+1)
            if not line:break
            require(len(line)<=MIB and line.endswith(b'\n'),'complete bounded raw PSS line required');sample=json.loads(line);row,status=pss_row(sample);counts[status]+=1
            if 'environment'in sample:require(sample['environment']['status']!='FAIL','known external CPU guard failure')
            if row:maximum=row[1]if maximum is None else max(maximum,row[1])
    return dict(raw=info(path),observation_counts=counts,max_all_eligible_phases_bytes=maximum,scope='Unknown CPU/disk does not filter PSS. Missing eligible PSS remains unobserved, not zero; sampled non-atomic app-tree, partial residency.')

def validate_block(args,p,launch,result):
    successful_exit(result);root=args.out/'producer';done=read(root/'COMPLETE.json');plan=read(root/'plan.json');terminal=read(root/'SCIENTIFIC_TERMINAL.json');records=read(root/'job_records.json')
    require(done['success']is True and done['controller_finished']is True and done['owned_tree_quiescent']is True and done['failure']is None and done['controller_guard_failure']is None and done['environment_guard_success']is True and done['environment_policy']==ENV_POLICY and done['source_manifest_sha256']==PACKET_MANIFEST_SHA,'complete successful block producer required')
    require(done['repetition_block']==terminal['repetition_block']==plan['repetition_block']==args.repetition_block and done['global_protocol_jobs']==24 and len(records)==8 and done['records']==records==terminal['records'],'exact global repetition block8 required')
    require(plan['execute']is True and plan['source_manifest_sha256']==PACKET_MANIFEST_SHA and plan['protocol_sha256']==PROTOCOL_SHA and hashlib.sha256(json.dumps(plan['protocol'],sort_keys=True).encode()).hexdigest()==PROTOCOL_SHA and len(plan['jobs'])==8 and 0<plan['max_hours']<=PRODUCER_SECONDS/3600,'exact frozen block source/protocol/budget required')
    require(info(root/'PLATFORM_ADMISSION.json')['sha256']==done['platform_admission_sha256']==terminal['platform_admission_sha256']==args.platform_admission_sha256 and read(root/'PLATFORM_ADMISSION.json')==p and plan['platform_admission']['proof']==p and plan['platform_admission']['sha256']==args.platform_admission_sha256,'original dynamic platform proof must bind all original outputs')
    require(done['controller_identity']['pid']==launch['pid']and done['controller_identity']['start_ticks']==launch['start_ticks'],'actual producer identity differs')
    require(not any(x.name.endswith(('ENVIRONMENT_FAILURE.json','CONTROLLER_GUARD_FAILURE.json'))for x in root.rglob('*')),'any known resource failure excludes the whole block')
    expected={(h,w,args.repetition_block,a)for h in HORIZONS for w in(1,8)for a in ARMS};jobs={j['name']:j for j in plan['jobs']}
    require(len(jobs)==8 and len({r['name']for r in records})==8 and {(r['horizon'],r['workers'],r['repetition'],r['arm'])for r in records}==expected and set(jobs)=={r['name']for r in records},'exact unique eight-app grid required')
    streams={h:set()for h in HORIZONS};prefixes=set();pss={};digests={}
    for record in records:
        j=jobs[record['name']];require(all(record[k]==j[k]for k in('name','horizon','workers','repetition','arm'))and record['name']==f"{j['horizon']}_w{j['workers']}_r{args.repetition_block}_{j['arm']}",'global original job name differs')
        successful_exit(record['exit']);folder=root/'jobs'/record['name'];require(info(folder/'report.json')['sha256']==record['report_sha256'],'original app report SHA differs');r=read(folder/'report.json');h=HORIZONS[j['horizon']];cfg=j['config']
        require(r['status']=='complete'and r['config']==cfg and cfg['arm']==j['arm']and cfg['num_workers']==j['workers']and cfg['batch_size']==32 and cfg['warmup_pairs']==256 and cfg['steps']==h['steps']and cfg['checkpoints']==h['checkpoints']and r['consumed_graphs']==h['consumed']and r['measured_graphs']==h['measured']and[p['total_trained_graphs']for p in r['checkpoints']]==h['checkpoints']and r['model']['initial_state_sha256']==MODEL_SHA,'original source/model/checkpoint/consumption contract differs')
        require(len(r['worker_receipts'])==j['workers']and all(w['status']=='closed'and w['native_closed']is True for w in r['worker_receipts']),'normal complete worker close required')
        for owner in [r,*r['worker_receipts']]:
            for key in('environment_start','environment_end'):boundary(owner[key],p['cpu_affinity'])
        streams[j['horizon']].add(r['input_stream_sha256']);prefixes.add(r['input_stream_sha256']if j['horizon']=='short320'else r['semantic_input_prefixes']['576'])
        require(h['consumed']<=r['completed_extractions_including_unconsumed_prefetch']<=h['consumed']+32*j['workers'],'prefetch count bound differs')
        if j['arm']==ARMS[1]:
            require(read(folder/'native_post_exit_integrity.json')['outside_application_lifetime']is True and read(folder/'native_scratch_cleanup.json')['after_actual_exit_and_hash']is True and not(folder/'runtime/native').exists(),'actual native post-exit integrity and scratch cleanup required')
        pss[record['name']]=pss_journal(root/(record['name']+'.pss.jsonl'));digests[record['name']]=record['report_sha256']
    require(all(len(x)==1 for x in streams.values())and len(prefixes)==1 and terminal['success']is True and terminal['jobs']==8 and len(terminal['stream_comparisons'])==4 and all(x['all_consumed_inputs_exact']is True for x in terminal['stream_comparisons'])and terminal['cross_horizon_input_prefix576']['all_arms_workers_in_block_exact']is True,'full within-block consumed-input and cross576 equality required')
    for name in('build_native','actual_cuda_qualification'):successful_exit(read(root/(name+'.exit.json')))
    q=read(root/'qualification/QUALIFIED.json');require(q['success']is True and q['actual_consumers']==4 and q['device']=='cuda'and q['workers']==[1,8]and q['environment_policy']==ENV_POLICY and q['source_manifest_sha256']==PACKET_MANIFEST_SHA and q['initial_model_sha256']==MODEL_SHA and all(x['actual_wait_returned']is True and x['returncode']==0 for x in q['actual_child_exits']),'genuine source-bound actual H1001/8 qualification required')
    summary=read(root/'scientific_summary.json');require(summary['status']=='complete'and summary['jobs']==8 and len(summary['rows'])==4 and len(summary['paired_compiled_over_native'])==4 and len(summary['sampled_PSS'])==8,'raw block summary only; no three-repetition statistics')
    identity=read(root/'input_identity.json');require(identity['whole_hashes']is True and set(identity['files'])=={str(args.friendster_root/n)for n in FR3}and all(row['bytes']==FR3[Path(name).name]['bytes']and row['sha256']==FR3[Path(name).name]['sha256']for name,row in identity['files'].items()),'whole compact input hashes differ')
    require(read(root/'host.json')['hostname']==p['hostname']and read(root/'host.json')['boot_id']==p['boot_id']and read(root/'hardware.json')['target']['uuid']==p['gpu']['uuid']and read(root/'status.json')['phase']=='complete','actual block host/GPU/final status differs')
    disk=disk_guard(root,p);lock=lock_available(args.coordination);capacity=os.statvfs(Path(p['base']));require(not(capacity.f_flag&os.ST_RDONLY)and capacity.f_bavail*capacity.f_frsize>=4*GIB,'runtime PVC reserve failed after block')
    source_packet(args.packet,'SOURCE_MANIFEST.json',PACKET_MANIFEST_SHA)
    return dict(kind='E25_V6_COMPLETE_EIGHT_APP_BLOCK',success=True,repetition_block=args.repetition_block,block_jobs=8,global_cohort_jobs=24,all_three_blocks_required=True,packet_manifest_sha256=PACKET_MANIFEST_SHA,protocol_sha256=PROTOCOL_SHA,platform_admission_sha256=args.platform_admission_sha256,report_sha256=digests,consumed_input_streams={h:next(iter(v))for h,v in streams.items()},cross_horizon_prefix576=next(iter(prefixes)),PSS=pss,disk_guard=disk,timing_lock=lock,prior_failed_reference=PRIOR_FAILED_REFERENCE,YT_writes=0,scope='Complete block only; not a complete24 cohort. Root independently audits/collects and observes wrapper OS exit0 before releasing its owned allocation.')

def collect_files(args):
    files={};packet=args.packet;root=args.out/'producer'
    for path in packet.rglob('*'):
        require(not path.is_symlink(),'source symlink refused')
        if path.is_file():files['e25/sources/packet/'+str(path.relative_to(packet))]=path
    for directory,subdirs,names in os.walk(root,topdown=True,followlinks=False):
        directory=Path(directory);subdirs[:]=[x for x in subdirs if x not in('runtime','shared_numba_cache','__pycache__')]
        for name in names:
            path=directory/name;require(not path.is_symlink(),'result symlink refused')
            if path.suffix in('.json','.jsonl','.log','.txt','.csv','.tsv'):files['e25/results/'+str(path.relative_to(root))]=path
            elif path==root/'friendster_pairs.npy':require(info(path)==WORKLOAD,'exact public workload only');files['e25/results/friendster_pairs.npy']=path
    for name in('platform_admission.json','preflight.json','child_launch.json','child_exit.json','SCIENTIFIC_TERMINAL.json','producer.log'):
        files['e25/envelope/'+name]=args.out/name
    files['e25/sources/envelope/run_block.py']=Path(__file__)
    files['e25/sources/envelope/SOURCE_READY.json']=HERE/'SOURCE_READY.json'
    return files

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repetition-block',type=int,choices=(1,2,3),required=True);parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--friendster-root',type=Path,required=True);parser.add_argument('--platform-admission-json',type=Path,required=True);parser.add_argument('--platform-admission-sha256',required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--coordination',type=Path,required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv);started=time.monotonic()
    require(args.execute and SOURCE_STATE=='FROZEN_BOUND'and all(isinstance(x,str)and len(x)==64 for x in(PACKET_MANIFEST_SHA,PROTOCOL_SHA)),'UNBOUND source template; root exact source/actual admission binding required before execution')
    for key in('packet','friendster_root','platform_admission_json','out','coordination'):setattr(args,key,getattr(args,key).resolve())
    ready=read(HERE/'SOURCE_READY.json');require(ready['status']=='SOURCE_READY_FOR_ROOT_PLATFORM_ADMISSION'and ready['ready_for_allocation']is True and ready['files']['run_block.py']['sha256']==sha(Path(__file__))and ready['producer_manifest_sha256']==PACKET_MANIFEST_SHA and ready['protocol_sha256']==PROTOCOL_SHA,'exact final source readiness required')
    source_packet(args.packet,'SOURCE_MANIFEST.json',PACKET_MANIFEST_SHA);p=platform_admission(args);args.out.mkdir();save_new(args.out/'platform_admission.json',p);save_new(args.out/'preflight.json',dict(block=args.repetition_block,platform=info(args.platform_admission_json),wrapper=info(Path(__file__)),max_seconds=MAX_SECONDS,producer_max_seconds=PRODUCER_SECONDS,YT_writes=0))
    spec=importlib.util.spec_from_file_location('e25_owned_block_support',args.packet/'support/owned_supervision.py');support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support);support.enable_subreaper();launch={};success=False;failure=None;archive=None;deadline=started+MAX_SECONDS
    def interrupted(signum,frame):raise RuntimeError('owned block wrapper interrupted')
    previous={sig:signal.signal(sig,interrupted)for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGALRM)};signal.setitimer(signal.ITIMER_REAL,max(1,deadline-time.monotonic()-20))
    try:
        command=[sys.executable,'-B','-u',str(args.packet/'run_lifecycle.py'),'--friendster-root',str(args.friendster_root),'--out',str(args.out/'producer'),'--coordination-dir',str(args.coordination),'--max-hours',str(PRODUCER_SECONDS/3600),'--repetition-block',str(args.repetition_block),'--platform-admission-json',str(args.platform_admission_json),'--platform-admission-sha256',args.platform_admission_sha256,'--source-manifest-sha256',PACKET_MANIFEST_SHA,'--execute']
        def popen(*values,**kwargs):
            kwargs['env']=dict(kwargs['env'],PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',E25_EXPECTED_CPU_AFFINITY=json.dumps(p['cpu_affinity'],separators=(',',':')));began=time.monotonic_ns();child=subprocess.Popen(*values,**kwargs);actual=support.process_identity(child.pid);require(actual is not None,'actual child PID/start absent');launch.update(pid=child.pid,start_ticks=actual['start_ticks'],entrypoint=str(args.packet/'run_lifecycle.py'),launch_monotonic_ns=began,argv=command);save_new(args.out/'child_launch.json',launch);return child
        result=support.run_owned(command,args.out/'producer.log',cwd=args.packet,timeout=min(PRODUCER_SECONDS+60,deadline-time.monotonic()-180),popen=popen);result.update(actual_wait_returned=True,start_ticks=launch['start_ticks'],wait_returned_monotonic_ns=time.monotonic_ns(),guard_abort_reason=None);save_new(args.out/'child_exit.json',result)
        require(not support.owned_groups(os.getpid()),'owned descendants remain after actual wait');terminal=validate_block(args,p,launch,result);save_new(args.out/'SCIENTIFIC_TERMINAL.json',terminal);archive=archive_results(args.out/'block_raw.tgz',collect_files(args))
        with Path(archive['archive']).open('rb')as stream:os.fsync(stream.fileno())
        fd=os.open(args.out,os.O_RDONLY);os.fsync(fd);os.close(fd);save_new(args.out/'archive_local_receipt.json',archive);success=True
    except BaseException as exc:failure=dict(type=type(exc).__name__,message=str(exc)[:2000],partial_block_excluded=True,automatic_retry=False)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);quiet=cleanup(support);save_new(args.out/'COMPLETE.json',dict(kind='E25_V6_LOCAL_BLOCK_WRAPPER_TERMINAL',success=success and quiet,repetition_block=args.repetition_block,block_jobs=8,global_cohort_jobs=24,all_three_blocks_required=True,owned_tree_quiescent=quiet,failure=failure,archive=archive,YT_reads=0,YT_writes=0,statefulset_actions=0,automatic_retry=False,scope='Root must separately observe actual wrapper OS exit0, independently audit and durably collect this block, then release only its owned allocation. No complete24 or final persistence claim.'))
        for sig,old in previous.items():signal.signal(sig,old)
    print(json.dumps(dict(success=success and quiet,block=args.repetition_block,out=str(args.out),archive=archive,YT_writes=0),sort_keys=True));return 0 if success and quiet else 1

if __name__=='__main__':raise SystemExit(main())
