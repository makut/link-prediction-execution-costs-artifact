#!/usr/bin/env python3
"""One detached, bounded E23 envelope; no scheduler or input writes.

The root operator verifies live Kubernetes UIDs immediately before launch. The
container independently verifies hostname/Unix UID and POD_UID when available.
Only one successful, verified, bounded final archive may be written to YT.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
MIB = 1 << 20
MAX_JSON = 32 * MIB
MAX_MEMBER = 32 * MIB
MAX_RAW = 384 * MIB
MAX_ARCHIVE = 128 * MIB
MAX_FILES = 4096
YT_ROOT = '//home/ucp-research/projects/graph_baselines/seal_revision_20261124'
YT_FINAL = YT_ROOT + '/pipeline_repeats_e23_20261126_final_v1.tgz'


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
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def process(pid, proc=Path('/proc')):
    folder = proc / str(pid)
    try:
        before = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        with (folder / 'cmdline').open('rb') as stream:
            raw = stream.read((64 << 10) + 1)
        after = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        require(before[19] == after[19] and len(raw) <= 64 << 10,
                'bounded stable PID/start observation required')
        return dict(pid=pid, start_ticks=int(after[19]), state=after[0],
                    argv=[x.decode(errors='replace') for x in raw.split(b'\0') if x])
    except FileNotFoundError:
        if not folder.exists():
            return None
        raise


def quiet(identity, reader=process):
    require(type(identity.get('pid')) is int and identity['pid'] > 0
            and type(identity.get('start_ticks')) is int and identity['start_ticks'] > 0,
            'actual positive PID/start_ticks required')
    now = reader(identity['pid'])
    if now is None:
        return dict(quiescent=True, observation='absent', pid=identity['pid'], start_ticks=identity['start_ticks'])
    require(now['start_ticks'] == identity['start_ticks'], 'PID reused; no inference or signals permitted')
    if now['state'] == 'Z':
        return dict(quiescent=True, observation='zombie', pid=identity['pid'], start_ticks=identity['start_ticks'])
    require(identity['entrypoint'] in now['argv'], 'live bootstrap/controller entrypoint differs')
    return None


def source_packet(root, manifest_name, expected_sha, *, inventory=False):
    root = Path(root)
    require(root.resolve() == root and not root.is_symlink(), 'canonical source packet directory required')
    manifest = root / manifest_name
    require(sha(manifest) == expected_sha, 'source manifest SHA differs: ' + str(manifest))
    data = read(manifest)
    rows = data['files']
    rows = rows if isinstance(rows, dict) else {r['path']: r for r in rows}
    require(len(rows) < MAX_FILES, 'source manifest count exceeded')
    for name, pin in rows.items():
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,
                'safe relative source path required')
        require((root / name).resolve() == root / name
                and info(root / name) == {k: pin[k] for k in ('bytes', 'sha256')}, 'source differs: ' + name)
    if inventory:
        observed = set()
        for path in root.rglob('*'):
            require(not path.is_symlink(), 'source symlink refused')
            if path.is_file() and path != manifest:
                observed.add(str(path.relative_to(root)))
        require(observed == set(rows), 'source inventory differs')
    return data


def load_support(packet):
    spec = importlib.util.spec_from_file_location('e23_outer_owned_support', packet / 'support/owned_supervision.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def admission(contract, outer_manifest_sha):
    require(sys.platform == 'linux', 'actual envelope execution requires Linux')
    require(socket.gethostname() == contract['hostname'] and os.getuid() == contract['unix_uid'],
            'actual hostname or Unix UID differs')
    require(str(HERE) == contract['outer_source'], 'outer source path differs')
    require(os.environ.get('POD_UID') in (None, '', contract['pod_uid']), 'visible POD_UID differs')
    require(contract['max_total_seconds'] == 21600 and contract['bootstrap_wait_seconds'] == 5400
            and contract['final_reserve_seconds'] == 900 and contract['yt_final'] == YT_FINAL,
            'frozen finite admission or fixed YT destination differs')
    base = Path(contract['base'])
    require(base.resolve() == base and base.is_dir(), 'canonical existing base required')
    source_packet(HERE, 'SOURCE_MANIFEST.json', outer_manifest_sha, inventory=True)
    packet = Path(contract['packet'])
    source_packet(packet, 'PACKET_MANIFEST.json', contract['packet_manifest_sha256'], inventory=True)
    source_packet(Path(contract['bootstrap_source']), 'SOURCE_MANIFEST.json', contract['bootstrap_source_manifest_sha256'])
    require(sha(Path(contract['bootstrap_source']) / 'input_bootstrap.py') == contract['bootstrap_source_sha256'],
            'original input bootstrap source differs')
    for key in ('science_out', 'outer_out', 'archive'):
        path = Path(contract[key])
        require(path.is_relative_to(base) and path.resolve() == path and not os.path.lexists(path),
                'fresh canonical bounded output required: ' + key)
    require(Path(contract['science_out']) != Path(contract['outer_out']), 'separate science and envelope directories required')
    trees = [Path(contract[k]) for k in ('science_out', 'outer_out', 'packet', 'outer_source',
                                        'citation2_root', 'friendster_root', 'bootstrap_source', 'bootstrap_out')]
    require(all(not left.is_relative_to(right) and not right.is_relative_to(left)
                for index, left in enumerate(trees) for right in trees[index + 1:]),
            'all graph, source, bootstrap and result trees must be disjoint')
    return dict(hostname_observed=socket.gethostname(), unix_uid_observed=os.getuid(),
                pod_uid_root_attested=contract['pod_uid'], statefulset_uid_root_attested=contract['statefulset_uid'],
                pod_uid_independently_visible=os.environ.get('POD_UID'),
                binding_scope='Root must verify actual Kubernetes Pod and StatefulSet UIDs immediately before launch; container independently verifies hostname, Unix UID, and POD_UID only when exposed.',
                outer_source_manifest_sha256=outer_manifest_sha, packet_manifest_sha256=contract['packet_manifest_sha256'],
                source_sha256=sha(Path(__file__)), admission_unix=time.time())


def bootstrap_ready(contract, reader=process):
    root = Path(contract['bootstrap_out'])
    require(not (root / 'FAILED.json').exists(), 'input bootstrap FAILED; no scientific launch')
    launch = read(contract['bootstrap_launch'])
    expected = dict(pid=contract['bootstrap_pid'], start_ticks=contract['bootstrap_start_ticks'],
                    entrypoint=str(Path(contract['bootstrap_source']) / 'input_bootstrap.py'),
                    source_sha256=contract['bootstrap_source_sha256'], output=str(root),
                    stdin_dependency=False, detached_session=True, scientific_jobs_started=0, automatic_retry=False)
    require(all(launch.get(k) == v for k, v in expected.items()), 'root-bound actual bootstrap launch differs')
    binding = launch['admission']
    require(binding['hostname'] == contract['hostname'] and binding['uid'] == contract['unix_uid']
            and binding['pod_uid_root_launch_binding'] == contract['pod_uid']
            and binding['statefulset_uid_root_launch_binding'] == contract['statefulset_uid']
            and binding['base'] == contract['base'] and binding['source_path'] == contract['bootstrap_source'],
            'actual bootstrap Pod/STS/Unix/source binding differs')
    quiescence = quiet(launch, reader)
    if not (root / 'COMPLETE.json').exists():
        require(quiescence is None, 'bootstrap exited without genuine COMPLETE; no scientific launch')
        return None
    if quiescence is None:
        return None
    complete = read(root / 'COMPLETE.json')
    for key in ('input_preparation_complete', 'all19_C2_numeric_payloads_exact', 'all3_Friendster_numeric_payloads_exact',
                'base_environment_unchanged', 'prepared_inputs_ready_for_separately_owned_scientific_controller',
                'preparation_child_reaped'):
        require(complete.get(key) is True, 'input preparation invariant absent: ' + key)
    for key in ('package_installs', 'YT_writes', 'new_YT_nodes', 'new_YT_chunks'):
        require(type(complete.get(key)) is int and complete[key] == 0, 'bootstrap no-install/no-write invariant absent')
    require(complete['citation2_prepared_root'] == contract['citation2_root']
            and complete['friendster_root'] == contract['friendster_root'], 'prepared graph roots differ')
    plan = read(root / 'plan.json')
    require(plan['binding'] == binding and plan['source_sha256'] == contract['bootstrap_source_sha256']
            and plan['source_manifest_sha256'] == contract['bootstrap_source_manifest_sha256']
            and plan['training_or_model_scoring'] is False and plan['automatic_retry'] is False
            and type(plan['package_installs']) is int and plan['package_installs'] == 0
            and 0 < plan['max_hours'] <= 2 and plan['max_new_disk_bytes'] <= 90 * (1 << 30)
            and plan['disk_reserve_bytes'] >= 4 * (1 << 30), 'bootstrap plan/source/resource invariants differ')
    source = Path(contract['bootstrap_source'])
    source_packet(source, 'SOURCE_MANIFEST.json', contract['bootstrap_source_manifest_sha256'])
    expected_c2 = read(source / 'EXPECTED_C2_19.json')
    expected_fr = read(source / 'EXPECTED_FRIENDSTER_FULL3.json')
    c2 = read(root / 'ACTUAL_C2_19_IDENTITY.json')
    fr = read(root / 'ACTUAL_FRIENDSTER_FULL3_IDENTITY.json')
    require(c2['all19_numeric_payloads_exact'] is True and len(c2['files']) == 19
            and c2['files'] == expected_c2['files'] and c2['prepared_root'] == contract['citation2_root'],
            'all19 actual prepared payload bindings required')
    require(fr['all3_numeric_payloads_exact'] is True and len(fr['files']) == 3
            and fr['files'] == expected_fr['files'] and fr['root'] == contract['friendster_root'],
            'all3 Friendster payload bindings required')
    require(plan['unchanged_preparer_sources'] == expected_c2['preparer_source_sha256'], 'unchanged preparer hashes required')
    require(read(root / 'base_before.json') == read(root / 'base_after.json'), 'base environment changed')
    require(read(root / 'status.json')['phase'] == 'complete', 'bootstrap final complete status required')
    names = ('plan.json', 'COMPLETE.json', 'ACTUAL_C2_19_IDENTITY.json', 'ACTUAL_FRIENDSTER_FULL3_IDENTITY.json',
             'base_before.json', 'base_after.json', 'status.json', 'input_download_manifest.json')
    return dict(actual_launch=launch, actual_launch_identity=info(contract['bootstrap_launch']), process=quiescence,
                receipt_files={name: info(root / name) for name in names},
                bootstrap_exit_code_scope='Not a child of this envelope; genuine COMPLETE, reaped preparation child and root-bound bootstrap process quiescence are checked. No bootstrap wait returncode is invented.')


def lock_available(coordination):
    path = Path(coordination) / 'PUBLIC_SYSTEMS_RUNNING.lock'
    require(not path.is_symlink(), 'timing lock symlink refused')
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream, fcntl.LOCK_UN)
    return dict(path=str(path), exclusive_flock_acquired_and_released=True, checked_unix=time.time(),
                scope='Persistent lock file is retained; lock availability is observed after actual child exit.')


def scientific_terminal(contract, exit_receipt, child_launch):
    require(type(exit_receipt.get('returncode')) is int and exit_receipt['returncode'] == 0
            and exit_receipt['timed_out'] is False and exit_receipt['orphan_cleanup'] is False
            and exit_receipt['owned_tree_quiescent'] is True,
            'actual wait(0), no timeout/orphan cleanup, and owned-tree quiescence required')
    require(exit_receipt['pid'] == child_launch['pid'], 'actual child PID/exit differs')
    root = Path(contract['science_out'])
    complete = read(root / 'COMPLETE.json')
    require(all(complete.get(k) is True for k in ('controller_finished', 'success', 'owned_tree_quiescent'))
            and complete.get('failure') is None and type(complete.get('YT_writes')) is int and complete['YT_writes'] == 0,
            'successful original scientific terminal required')
    records = read(root / 'job_records.json')
    require(isinstance(records, list) and len(records) == 40 and complete['records'] == records
            and sum(r['kind'] == 'pipeline' for r in records) == 36
            and all(r['validation_error'] is None for r in records), 'all40 jobs/36 timing reports required')
    summary = read(root / 'scientific_summary.json')
    require(summary['status'] == 'complete' and summary['timing_jobs'] == 36
            and len(summary['reference_prefix_comparisons']) == 36 and len(summary['rows']) == 12
            and len(summary['paired_time_ratios']) == 12, 'complete finite-window scientific summary required')
    plan = read(root / 'plan.json')
    require(plan['execute'] is True and len(plan['jobs']) == 40
            and plan['protocol_sha256'] == contract['protocol_sha256'], 'predeclared full protocol differs')
    require(read(root / 'status.json')['phase'] == 'complete', 'final runner complete status required')
    require(read(root / 'hardware.json')['target']['uuid'] == contract['gpu_uuid'], 'actual target GPU UUID differs')
    quiet_after = quiet(child_launch)
    require(quiet_after is not None, 'actual child must be quiescent after wait')
    return dict(scientific_success=True, timing_jobs=36, declared_jobs=40, complete_identity=info(root / 'COMPLETE.json'),
                summary_identity=info(root / 'scientific_summary.json'), process=quiet_after,
                timing_lock=lock_available(contract['coordination']))


def supervise_child(contract, support, timeout, evidence, *, reader=process, argv=None):
    packet = Path(contract['packet'])
    command = argv or [contract['python'], '-B', '-u', str(packet / 'run_pipeline_repeats.py'),
        '--citation2-root', contract['citation2_root'], '--friendster-root', contract['friendster_root'],
        '--out', contract['science_out'], '--coordination-dir', contract['coordination'],
        '--max-hours', str(max(.001, (timeout - 30) / 3600)), '--execute']
    source = packet / 'run_pipeline_repeats.py'
    launch = {}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
               WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    def popen(*args, **kwargs):
        kwargs['env'] = env
        child = subprocess.Popen(*args, **kwargs)
        identity = reader(child.pid)
        require(identity is not None, 'actual launched child process identity unavailable')
        launch.update(pid=child.pid, start_ticks=identity['start_ticks'], entrypoint=str(source), argv=command,
                      source_sha256=sha(source), packet_manifest_sha256=contract['packet_manifest_sha256'],
                      started_unix=time.time(), stdin_dependency=False, start_new_session=True)
        save_new(evidence / 'child_launch.json', launch)
        return child
    result = support.run_owned(command, evidence / 'scientific.log', cwd=packet, timeout=timeout, popen=popen)
    result.update(start_ticks=launch['start_ticks'], finished_unix=time.time(), actual_wait_returned=True)
    save_new(evidence / 'child_exit.json', result)
    return launch, result


def fixture_pair_identity(path):
    """Only the two declared tiny preflight workloads need binary packaging."""
    path = Path(path)
    identity = info(path, 4096)
    report = read(path.parent / 'report.json')
    manifest = report['manifest']
    require(report.get('benchmark_version') == 4 and report['graph']['path'] == 'fixture'
            and manifest['path'] == str(path.resolve()) and manifest['sha256'] == identity['sha256']
            and manifest['rows'] == 128 and manifest['positive_occurrences'] == 64
            and manifest['columns'] == ['src', 'dst', 'label'], 'tiny fixture workload/report identity differs')
    return identity


def collect_files(contract, evidence):
    files = {}
    packet = Path(contract['packet'])
    for path in packet.rglob('*'):
        require(not path.is_symlink(), 'source symlink refused')
        if path.is_file():
            files['e23/sources/packet/' + str(path.relative_to(packet))] = path
    for path in HERE.iterdir():
        require(not path.is_symlink() and path.is_file(), 'outer source inventory must remain flat regular files')
        files['e23/sources/outer/' + path.name] = path
    bootstrap_source = Path(contract['bootstrap_source'])
    for row in read(bootstrap_source / 'SOURCE_MANIFEST.json')['files']:
        files['e23/sources/bootstrap/' + row['path']] = bootstrap_source / row['path']
    files['e23/sources/bootstrap/SOURCE_MANIFEST.json'] = bootstrap_source / 'SOURCE_MANIFEST.json'
    science = Path(contract['science_out'])
    for path in science.rglob('*'):
        require(not path.is_symlink(), 'scientific result symlink refused')
        if not path.is_file() or 'numba_cache' in path.relative_to(science).parts:
            continue
        relative = str(path.relative_to(science))
        permitted_workload = relative in ('manifests/citation2_pairs.npy', 'manifests/friendster_pairs.npy')
        permitted_fixture = relative in ('jobs/guard_refusal/pairs.npy', 'jobs/gpu_fixture/pairs.npy')
        if permitted_fixture:
            fixture_pair_identity(path)
        require(path.suffix in ('.json', '.jsonl', '.log', '.txt', '.csv', '.tsv')
                or permitted_workload or permitted_fixture,
                'unexpected binary/source under result tree: ' + relative)
        files['e23/results/' + relative] = path
    for path in evidence.iterdir():
        require(not path.is_symlink(), 'envelope symlink refused')
        if path.is_file():
            require(path.suffix in ('.json', '.log'), 'bounded textual outer evidence only')
            files['e23/envelope/' + path.name] = path
    bootstrap = Path(contract['bootstrap_out'])
    for name in ('plan.json', 'COMPLETE.json', 'ACTUAL_C2_19_IDENTITY.json', 'ACTUAL_FRIENDSTER_FULL3_IDENTITY.json',
                 'base_before.json', 'base_after.json', 'status.json', 'input_download_manifest.json'):
        files['e23/bootstrap/' + name] = bootstrap / name
    files['e23/bootstrap/actual_launch.json'] = Path(contract['bootstrap_launch'])
    files['e23/envelope/actual_outer_launch.json'] = Path(contract['outer_launch'])
    return files


def archive_results(archive, files):
    archive = Path(archive)
    require(not os.path.lexists(archive) and 0 < len(files) <= MAX_FILES, 'fresh bounded archive required')
    pinned = {name: info(path) for name, path in files.items()}
    require(sum(row['bytes'] for row in pinned.values()) <= MAX_RAW - MIB, 'small result/source raw archive cap exceeded')
    for name in files:
        rel = PurePosixPath(name)
        require(name.startswith('e23/') and not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,
                'safe regular archive namespace required')
    manifest = (json.dumps(dict(files=pinned, graphs_included=False, numba_cache_included=False,
                  scope='Frozen E23 source and small outputs with actual post-exit lifecycle provenance; no graph payloads, environment trees, credentials or checkpoints.'),
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


def upload_once(archive, expected, target, *, client=None):
    require(target == YT_FINAL and info(archive, MAX_ARCHIVE) == {k: expected[k] for k in ('bytes', 'sha256')},
            'only pinned successful bounded final archive may be uploaded')
    if client is None:
        import yt.wrapper as yt
        client = yt.YtClient(proxy='jupiter.yt.vk.team', config={
            'proxy': {'request_timeout': 120000, 'heavy_request_timeout': 120000,
                      'enable_proxy_discovery': False}, 'read_retries': {'enable': False},
            'write_retries': {'enable': False}, 'read_parallel': {'enable': False}, 'write_parallel': {'enable': False}})
    require(client.exists(YT_ROOT) and not client.exists(target), 'existing parent and absent exact final node required; no overwrite')
    with client.Transaction(timeout=600000):
        require(not client.exists(target), 'final node already exists in transaction; no overwrite/retry')
        client.create('file', target, recursive=False)
        with Path(archive).open('rb') as stream:
            client.write_file(target, stream, file_writer={'desired_chunk_size': 512 * MIB, 'block_size': MIB})
        chunks = int(client.get(target + '/@chunk_count'))
        require(chunks == 1 and int(client.get(target + '/@uncompressed_data_size')) == expected['bytes'],
                'single file/single chunk byte bound required before commit')
        digest, count = hashlib.sha256(), 0
        stream = client.read_file(target)
        try:
            for block in iter(lambda: stream.read(MIB), b''):
                count += len(block)
                require(count <= MAX_ARCHIVE, 'final YT whole readback byte cap')
                digest.update(block)
        finally:
            stream.close()
        require(dict(bytes=count, sha256=digest.hexdigest()) == {k: expected[k] for k in ('bytes', 'sha256')}
                and info(archive, MAX_ARCHIVE) == {k: expected[k] for k in ('bytes', 'sha256')},
                'whole final YT SHA/size readback differs')
    return dict(yt_path=target, bytes=count, sha256=digest.hexdigest(), readback_verified=True,
                new_YT_file_nodes=1, chunk_count=chunks, write_file_calls=1, automatic_retry=False,
                transaction_committed=True, completed_unix=time.time())


def cleanup(support):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        support.signal_confirmed(support.owned_groups(os.getpid()), sig)
        until = time.monotonic() + 5
        while time.monotonic() < until and support.owned_groups(os.getpid()):
            time.sleep(.1)
    return not support.owned_groups(os.getpid())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest-sha256', required=True)
    args = parser.parse_args(argv)
    contract = read(HERE / 'CONTRACT.json')
    binding = admission(contract, args.source_manifest_sha256)
    out = Path(contract['outer_out'])
    out.mkdir()
    start = time.monotonic()
    deadline = start + contract['max_total_seconds']
    support = load_support(Path(contract['packet']))
    support.enable_subreaper()
    save_new(out / 'admission.json', binding)
    identity = process(os.getpid())
    require(identity is not None, 'actual envelope process identity unavailable')
    save_new(out / 'controller_identity.json', dict(**identity, entrypoint=str(Path(__file__)),
        source_sha256=sha(Path(__file__)), source_manifest_sha256=args.source_manifest_sha256))
    success, scientific_success, failure, pointer = False, False, None, None
    phase = 'waiting_for_verified_bootstrap'
    def interrupted(signum, frame):
        raise RuntimeError('bounded envelope interrupted by signal ' + str(signum))
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, contract['max_total_seconds'] - 30)
    try:
        wait_until = start + contract['bootstrap_wait_seconds']
        bootstrap = None
        while bootstrap is None:
            require(time.monotonic() < wait_until, 'bootstrap admission wait deadline reached')
            bootstrap = bootstrap_ready(contract)
            if bootstrap is None:
                time.sleep(min(10, max(.1, wait_until - time.monotonic())))
        save_new(out / 'bootstrap_admitted.json', bootstrap)
        phase = 'running_complete_scientific_grid'
        source_packet(Path(contract['packet']), 'PACKET_MANIFEST.json', contract['packet_manifest_sha256'], inventory=True)
        lock_available(contract['coordination'])
        require(not os.path.lexists(contract['science_out']), 'fresh science output required immediately before launch')
        timeout = deadline - time.monotonic() - contract['final_reserve_seconds']
        require(timeout >= 1800, 'insufficient bounded time for complete scientific controller; no partial-grid launch')
        launch, result = supervise_child(contract, support, timeout, out)
        terminal = scientific_terminal(contract, result, launch)
        save_new(out / 'SCIENTIFIC_TERMINAL.json', terminal)
        scientific_success = True
        phase = 'packing_post_exit_small_results'
        require(not support.owned_groups(os.getpid()), 'owned descendants remain before final archive')
        root_launch = read(contract['outer_launch'])
        require(root_launch['pid'] == identity['pid'] and root_launch['start_ticks'] == identity['start_ticks']
                and root_launch['entrypoint'] == str(Path(__file__))
                and root_launch['source_manifest_sha256'] == args.source_manifest_sha256
                and root_launch['source_sha256'] == sha(Path(__file__)), 'actual detached outer launch binding differs')
        source_packet(Path(contract['packet']), 'PACKET_MANIFEST.json', contract['packet_manifest_sha256'], inventory=True)
        source_packet(HERE, 'SOURCE_MANIFEST.json', args.source_manifest_sha256, inventory=True)
        source_packet(Path(contract['bootstrap_source']), 'SOURCE_MANIFEST.json', contract['bootstrap_source_manifest_sha256'])
        require(info(contract['bootstrap_launch']) == bootstrap['actual_launch_identity']
                and all(info(Path(contract['bootstrap_out']) / name) == pin
                        for name, pin in bootstrap['receipt_files'].items()),
                'admitted bootstrap evidence changed before final packaging')
        receipt = archive_results(contract['archive'], collect_files(contract, out))
        save_new(out / 'archive_local_receipt.json', receipt)
        require(time.monotonic() < deadline - 660, 'final persistence reserve exhausted; preserve local archive')
        phase = 'single_final_YT_transaction'
        pointer = upload_once(contract['archive'], receipt, contract['yt_final'])
        save_new(out / 'archive_pointer.json', pointer)
        success = True
    except BaseException as exc:
        # Do not serialize arbitrary client exception text/configuration.
        failure = dict(type=type(exc).__name__, phase=phase,
                       scope='Envelope failed; inspect bounded local evidence. Scientific success is reported separately; no automatic retries.')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        quiescent = cleanup(support)
        save_new(out / 'COMPLETE.json', dict(controller_finished=True, success=success and quiescent,
            scientific_success=scientific_success, owned_tree_quiescent=quiescent, failure=failure,
            archive_pointer=pointer, started_unix=binding['admission_unix'], completed_unix=time.time(),
            elapsed_seconds=time.monotonic() - start, final_YT_write_file_calls=1 if pointer else '0 or 1; transaction may have failed; no retry',
            statefulset_lifecycle_actions=0, package_installs=0, automatic_retry=False))
        for sig, old in previous.items():
            signal.signal(sig, old)
    return 0 if success and quiescent else 1


if __name__ == '__main__':
    raise SystemExit(main())
