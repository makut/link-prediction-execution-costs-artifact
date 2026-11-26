#!/usr/bin/env python3
"""One bounded v4 E25 envelope; root binds persistent storage and ENVv3 CPU admission.

No graph ingestion, package installation, cluster action or retry is performed.
Only a successful post-exit bounded archive may be persisted to one fixed YT file.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import struct
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
YT_FINAL = YT_ROOT + '/native_lifecycle_e25_20261126_final_v1.tgz'
ARMS = ('reference_union_compiled', 'reference_union_native_persistent')
HORIZONS = {
    'short320': dict(steps=10, measured=320, consumed=576, checkpoints=[32, 256, 576]),
    'long6400': dict(steps=200, measured=6400, consumed=6656, checkpoints=[32, 256, 1280, 3456, 6656]),
}
FR3 = {
    'edge_starts.bin': dict(bytes=524866928, sha256='b6ebf026fa54498f241af834b96b5496b2a1bcbf8be3edea2f4b3a1cf0a674e9'),
    'edge_ends.bin': dict(bytes=14448537088, sha256='0f74897efa3172252a8371bb0d2c8e0129602475ebb84a863ec1210f9c5e5973'),
    'num_nodes.txt': dict(bytes=9, sha256='8551541f6f2ef955481af83ef5e6f92c5f206c968d5d9db6a4437efecf767753'),
}
WORKLOAD = dict(bytes=192128, sha256='340e61cdd1eb8d329d52be164f919480e6a41feb4c620852df0393c2e9a192f0')
MODEL_SHA = 'f88254079fe6c152b4c0153c857a63516348353d578f918622e7f6719559618e'
TEXT_SUFFIXES = ('.json', '.jsonl', '.log', '.txt', '.csv', '.tsv')
PRIOR_ABORTED = dict(bytes=1168934, sha256='34d1dbc0aba48f3ce72b766a655f48ee42b857edf70a7d19415d12aa90490d39')


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


def process(pid, proc=Path('/proc')):
    folder = proc / str(pid)
    try:
        before = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        with (folder / 'cmdline').open('rb') as stream:
            raw = stream.read((64 << 10) + 1)
        after = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        require(len(before) >= 50 and len(after) >= 50 and before[19] == after[19]
                and len(raw) <= 64 << 10, 'bounded stable PID/start observation required')
        # Linux /proc/PID/stat field52 is exit_code; index49 after the comm field.
        return dict(pid=pid, start_ticks=int(after[19]), state=after[0],
                    kernel_exit_code=int(after[49]),
                    argv=[x.decode(errors='replace') for x in raw.split(b'\0') if x])
    except FileNotFoundError:
        if not folder.exists():
            return None
        raise


def quiet(identity, reader=process, *, require_zombie_zero=True):
    require(type(identity.get('pid')) is int and identity['pid'] > 0
            and type(identity.get('start_ticks')) is int and identity['start_ticks'] > 0,
            'actual positive PID/start_ticks required')
    now = reader(identity['pid'])
    if now is None:
        return dict(quiescent=True, observation='absent', pid=identity['pid'], start_ticks=identity['start_ticks'],
                    exit_code_observed=False, scope='Absent original process; no wait returncode is invented.')
    require(now['start_ticks'] == identity['start_ticks'], 'PID reused; no inference or signals permitted')
    if now['state'] == 'Z':
        code = now.get('kernel_exit_code')
        require(not require_zombie_zero or type(code) is int and code == 0,
                'observed zombie must have actual kernel exit_code0')
        return dict(quiescent=True, observation='zombie', pid=identity['pid'], start_ticks=identity['start_ticks'],
                    exit_code_observed=True, kernel_exit_code=code,
                    scope='Actual Linux stat field52; original ingress is not this envelope child.')
    require(identity['entrypoint'] in now['argv'], 'live ingress/controller entrypoint differs')
    return None


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


def load_support(packet):
    spec = importlib.util.spec_from_file_location('e25_outer_owned_support', packet / 'support/owned_supervision.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_text(path):
    with Path(path).open('r') as stream:
        result = stream.read((64 << 10) + 1)
    require(len(result) <= 64 << 10, 'bounded environment metadata required')
    return result.strip()


def cpu_list(value):
    result = set()
    require(isinstance(value, str) and 0 < len(value) <= 64 << 10, 'bounded CPU-list text required')
    for part in value.split(','):
        bounds = part.split('-')
        require(len(bounds) in (1, 2) and all(x.isdigit() for x in bounds), 'canonical numeric CPU-list required')
        first, last = int(bounds[0]), int(bounds[-1])
        require(0 <= first <= last < 4096, 'finite CPU IDs required')
        result.update(range(first, last + 1))
    return sorted(result)


def cpu_binding(contract, phase):
    selected = contract.get('expected_cpu_affinity')
    require(isinstance(selected, list) and len(selected) == 27 and selected == sorted(set(selected))
            and all(type(x) is int and 0 <= x < 4096 for x in selected), 'root must bind exact27 unique sorted CPU IDs')
    actual = sorted(os.sched_getaffinity(0))
    effective = cpu_list(small_text('/sys/fs/cgroup/cpuset.cpus.effective'))
    quota = ' '.join(small_text('/sys/fs/cgroup/cpu.max').split())
    memory = small_text('/sys/fs/cgroup/memory.max')
    boot = small_text('/proc/sys/kernel/random/boot_id')
    require(actual == selected, 'actual outer/launcher CPU mask differs; no re-pin or relaxation')
    require(set(selected).issubset(effective), 'selected CPU removed from current effective cpuset')
    require(quota == contract['expected_cpu_max'] == '2700000 100000', 'exact27CPU quota/period changed')
    require(memory == contract['expected_memory_max'] == '274877906944', 'bound memory limit changed')
    require(boot == contract['expected_boot_id'], 'root-bound boot ID changed')
    thread_masks = []
    for folder in sorted(Path('/proc/self/task').iterdir(), key=lambda p: int(p.name)):
        tid = int(folder.name)
        try:
            mask = sorted(os.sched_getaffinity(tid))
        except ProcessLookupError:
            continue
        require(bool(mask) and set(mask).issubset(selected), 'outer thread can execute outside chosen CPU mask')
        thread_masks.append(dict(tid=tid, cpu_affinity=mask))
    visible = os.environ.get('E25_EXPECTED_CPU_AFFINITY')
    if visible is not None:
        require(json.loads(visible) == selected, 'visible expected CPU-mask contract differs')
    return dict(status='PASS', phase=phase, expected_cpu_affinity=selected, actual_cpu_affinity=actual,
                effective_ambient_cpuset=effective, cpu_max=quota, memory_max=memory, boot_id=boot,
                outer_thread_masks=thread_masks, monotonic_ns=time.monotonic_ns(),
                scope='Exact inherited fixed mask; ambient pool may change only outside selected CPUs. This envelope never chooses/re-pins a CPU mask. Producer owns bounded in-app guards.')


def prior_aborted_binding(contract):
    require(contract['prior_aborted_archive_bytes'] == PRIOR_ABORTED['bytes']
            and contract['prior_aborted_archive_sha256'] == PRIOR_ABORTED['sha256'], 'only exact latest available ENVv3 local-evidence capsule admitted')
    archive = Path(contract['prior_aborted_archive'])
    require(archive == Path(contract['base']) / 'results/E25_ENV_V3_EXTERNAL_STOP_DIAGNOSTIC_v1.tgz'
            and info(archive, PRIOR_ABORTED['bytes']) == PRIOR_ABORTED, 'preserved available ENVv3 local-evidence capsule identity differs')
    receipt = HERE / 'PRIOR_ABORTED_PRESERVATION_RECEIPT.json'
    require(info(receipt)['sha256'] == contract['prior_aborted_receipt_sha256'], 'exact root preservation receipt changed')
    data = read(receipt)
    require(data['bytes'] == PRIOR_ABORTED['bytes'] and data['sha256'] == PRIOR_ABORTED['sha256']
            and data['archive'] == archive.name and data['kind'] == 'ENV_V3_AVAILABLE_LOCAL_EVIDENCE_PRESERVATION_RECEIPT'
            and data['full24_success'] is False and data['raw_partial_job_metrics_preserved'] is False
            and data['actual_OS_exit0_proven'] is False and data['observed_completed_jobs_only'] == 2
            and data['prior_failed_env_v2_capsule_nested_once'] is True and data['prior_failed_v1_nested_inside_v2'] is True
            and data['all_available_selected_member_hashes_verified'] is True and data['local_fsync'] is True
            and data['members'] == 81 and data['raw_payload_bytes'] == 1762608
            and data['no_graph_payloads'] is True and data['YT_writes'] == 0,
            'root available81-member local-evidence proof required; lost raw metrics or OS exit must not be invented')
    return dict(archive=str(archive), **PRIOR_ABORTED, preservation_receipt=info(receipt),
                full24_success=False, raw_partial_job_metrics_preserved=False, actual_OS_exit0_proven=False,
                observed_completed_jobs_only=2,
                scope='Available local observations/source only; no raw partial ENVv3 metrics, OS exit or full24 success. ENVv2 stays nested once, with ENVv1 inside it; no old job enters new24 metrics.')


def persistent_storage_binding(contract):
    """Bind the exact actual platform amendment; recheck current mounted BASE."""
    require(contract.get('storage_policy') == 'persistent_project_mount_v4'
            and isinstance(contract.get('root_storage_amendment_sha256'), str)
            and len(contract['root_storage_amendment_sha256']) == 64
            and type(contract.get('expected_storage_device')) is int and contract['expected_storage_device'] >= 0,
            'root must bind actual persistent storage amendment and device before execution')
    require(all(type(contract[key]) is int for key in ('storage_initial_minimum_bytes_available',
                'storage_initial_minimum_inodes_when_exposed', 'storage_runtime_minimum_bytes_available',
                'storage_runtime_minimum_inodes_when_exposed'))
            and contract['storage_initial_minimum_bytes_available'] == 80 * (1 << 30)
            and contract['storage_initial_minimum_inodes_when_exposed'] == 65536
            and contract['storage_runtime_minimum_bytes_available'] == 4 * (1 << 30)
            and contract['storage_runtime_minimum_inodes_when_exposed'] == 1,
            'initial admission and runtime thresholds must remain distinct and predeclared')
    path = HERE / 'ROOT_STORAGE_AMENDMENT.json'
    require(info(path)['sha256'] == contract['root_storage_amendment_sha256'], 'actual platform amendment changed')
    proof = read(path)
    require(proof['kind'] == 'E25_ENV_V4_PREDECLARED_STORAGE_PLATFORM_AMENDMENT'
            and proof['status'] == 'FROZEN_ACTUAL_PLATFORM_BEFORE_QUALIFICATION_AND_FULL_SCIENCE'
            and proof['base'] == contract['base'] and proof['pod_uid'] == contract['pod_uid']
            and proof['statefulset_uid'] == contract['statefulset_uid'] and proof['pod_name'] == contract['hostname']
            and proof['boot_id'] == contract['expected_boot_id'] and proof['node'] == contract['expected_node']
            and proof['cpu_affinity'] == contract['expected_cpu_affinity'] and proof['gpu_uuid'] == contract['gpu_uuid']
            and proof['expected_storage_device'] == contract['expected_storage_device']
            and proof['scientific_jobs_started'] == 0 and proof['YT_writes'] == 0,
            'actual source-predeclared Pod/node/CPU/GPU/storage platform binding required')
    actual_path = HERE / 'ROOT_STORAGE_ACTUAL_ADMISSION.json'
    require(info(actual_path)['sha256'] == proof['actual_admission_sha256'] == contract['root_actual_storage_admission_sha256'],
            'actual original storage-admission bytes must match the amendment and root pin')
    actual = read(actual_path)
    require(actual == proof['actual_admission'] and actual['kind'] == 'ROOT_ACTUAL_PVC_STORAGE_ADMISSION_V4'
            and actual['status'] == 'PASS_ACTUAL_MOUNT_AND_OWN_DIRECTORY_ONLY'
            and actual['base'] == contract['base'] and actual['expected_storage_device'] == contract['expected_storage_device']
            and actual['boot_id'] == contract['expected_boot_id'] and actual['cpu_affinity'] == contract['expected_cpu_affinity']
            and actual['large_payloads_on_PVC'] is True and actual['actual_scientific_jobs_started'] == 0,
            'actual nested admission must be preserved byte-identifiably and scoped to the owned mount')
    kube = actual['kubernetes']
    require(kube['namespace'] == contract['namespace'] and kube['pod_uid'] == contract['pod_uid']
            and kube['statefulset_uid'] == contract['statefulset_uid'] and kube['pod'] == contract['hostname']
            and kube['node'] == contract['expected_node'] and kube['qos'] == 'Guaranteed'
            and kube['PVC_claim'] == proof['PVC_claim_name'] == contract['intended_existing_claim'] == 'claim-vk-b'
            and kube['PVC_uid'] == proof['PVC_claim_uid'] == contract['intended_existing_claim_uid']
            and kube['PV_volume_name'] == proof['PV_volume_name'] and kube['storage_class'] == proof['storage_class']
            and isinstance(kube['PV_binding_scope'], str) and kube['PV_binding_scope'],
            'root must bind actual PVC UID/volumeName/Pod mount; device number alone is not PVC identity')
    project = proof['project_subdirectory']
    require(safe_relative(project) and project == contract['storage_project_subdirectory']
            and actual['project_owner']['claim'] == contract['intended_existing_claim']
            and actual['project_owner']['claim_uid'] == contract['intended_existing_claim_uid']
            and proof['all_source_graph_results_native_and_temp_cache_paths_under_BASE'] is True,
            'only the actual new project subPath and declared large-payload paths are admitted')
    for probe in ('PVC_write_probe', 'root_layer_write_probe'):
        require(actual[probe]['write_fsync_readback_and_unlink'] is True and actual[probe]['directory_fsync'] is True,
                'actual owned-file and root-layer write/fsync/read/unlink proofs required')
    require(actual['root_layer_filesystem']['mount_readonly'] is False, 'root layer was observed readonly at admission')
    initial = actual['after']
    require(proof['initial_free_threshold_bytes'] == 80 * (1 << 30) and proof['initial_inode_threshold'] == 65536
            and proof['live_free_threshold_bytes'] == 4 * (1 << 30) and proof['max_nonruntime_science_outputs_bytes'] == 2 * (1 << 30)
            and type(initial['device']) is int and initial['device'] == contract['expected_storage_device']
            and type(initial['bytes_available']) is int and proof['initial_free_bytes'] == initial['bytes_available'] >= 80 * (1 << 30)
            and initial['mount_readonly'] is False and type(initial['inodes_total']) is int and initial['inodes_total'] >= 0
            and type(initial['inodes_available']) is int and proof['initial_inodes_available'] == initial['inodes_available'] >= 0
            and (initial['inodes_total'] == 0 or initial['inodes_available'] >= 65536),
            'initial80GiB/64K exposed inode admission must not be confused with runtime4GiB/positive inode guards')
    expected_mount = proof['mount']
    require(actual['mount'] == expected_mount and expected_mount['mount_point'] == contract['base']
            and expected_mount['root'] == '/' + project, 'actual mount must expose the isolated project subPath at BASE')
    base = Path(contract['base'])
    require(base.resolve() == base and base.is_dir() and not base.is_symlink() and str(base) == contract['storage_mount_path'],
            'canonical persistent project BASE required')
    observed_mounts = []
    for line in small_text('/proc/self/mountinfo').splitlines():
        left, right = line.split(' - ', 1); fields, tail = left.split(), right.split()
        point = fields[4].replace('\\040', ' ').replace('\\011', '\t').replace('\\134', '\\')
        if point == str(base):
            observed_mounts.append(dict(mount_point=point, mount_root=fields[3], major_minor=fields[2],
                mount_options=fields[5].split(','), filesystem_type=tail[0], source=tail[1], super_options=tail[2].split(',')))
    require(len(observed_mounts) == 1, 'project BASE must be an explicit unique container mount')
    mount = observed_mounts[0]
    require('rw' in mount['mount_options'] and 'ro' not in mount['mount_options'] and 'ro' not in mount['super_options']
            and mount['mount_root'] == expected_mount['root'] and mount['filesystem_type'] == expected_mount['filesystem']
            and mount['major_minor'] == expected_mount['major_minor'] and mount['source'] == expected_mount['source'],
            'current project mount differs from root-pinned actual PVC mount')
    stat, capacity = base.stat(), os.statvfs(base)
    require(stat.st_dev == contract['expected_storage_device']
            and mount['major_minor'] == str(os.major(stat.st_dev)) + ':' + str(os.minor(stat.st_dev))
            and not bool(capacity.f_flag & os.ST_RDONLY) and capacity.f_bavail * capacity.f_frsize >= 4 * (1 << 30)
            and (capacity.f_files == 0 or capacity.f_favail > 0), 'current BASE device/readwrite/runtime reserve differs')
    for key in ('packet', 'outer_source', 'friendster_root', 'ingress_source', 'ingress_out', 'coordination'):
        candidate = Path(contract[key])
        require(candidate.is_relative_to(base) and candidate.resolve() == candidate, 'all run trees must be inside persistent BASE')
        while not candidate.exists():
            require(candidate != base, 'persistent BASE disappeared'); candidate = candidate.parent
        require(candidate.stat().st_dev == stat.st_dev, 'run tree uses another backing device')
    for key, value in actual['temporary_and_cache_environment'].items():
        temporary = Path(value)
        require(key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME')
                and temporary.is_relative_to(base) and temporary.resolve() == temporary
                and os.environ.get(key) == value, 'current temp/cache environment differs from actual PVC admission')
    return dict(policy=contract['storage_policy'], amendment=info(path), original_actual_admission=info(actual_path),
                device=stat.st_dev, mount=mount, root_kubernetes_binding=kube,
                current_bytes_available=capacity.f_bavail * capacity.f_frsize, current_inodes_available=capacity.f_favail,
                current_inodes_reported=capacity.f_files > 0, observed_monotonic_ns=time.monotonic_ns(),
                scope='Actual Bound PVC UID/volumeName plus Pod mount/subPath/owned-directory proof and current local mount/statvfs; direct PV claimRef was not visible. Historical node807 readonly status is not a current global-health assertion. No capacity reservation or continuous health/bandwidth guarantee.')


def admission(contract, outer_manifest_sha):
    require(contract.get('source_only_preparation') is False
            and contract.get('execution_admission_state') == 'ROOT_BOUND_V4_PERSISTENT_SOURCE_AND_LIVE_GATES_PASS',
            'unbound v4 source-only template cannot execute')
    require(sys.platform == 'linux', 'actual envelope execution requires Linux')
    require(socket.gethostname() == contract['hostname'] and os.getuid() == contract['unix_uid'],
            'actual hostname or Unix UID differs')
    require(str(HERE) == contract['outer_source'], 'outer source path differs')
    require(os.environ.get('POD_UID') == contract['pod_uid'], 'mandatory exposed POD_UID differs or is absent')
    require(contract.get('environment_policy') == 'fixed27_env_v3', 'explicit new fixed27ENVv3 policy required')
    require(isinstance(contract.get('root_cpu_predeclaration_sha256'), str)
            and len(contract['root_cpu_predeclaration_sha256']) == 64, 'root must bind final actual CPU predeclaration receipt')
    require(info(HERE / 'ROOT_CPU_PREDECLARATION.json')['sha256'] == contract['root_cpu_predeclaration_sha256'],
            'root pre-outcome CPU-mask predeclaration receipt changed')
    environment = cpu_binding(contract, 'outer_admission')
    require(contract['max_total_seconds'] == 21600 and contract['ingress_wait_seconds'] == 5400
            and contract['producer_max_seconds'] == 14400 and contract['final_reserve_seconds'] == 900
            and contract['yt_final'] == YT_FINAL, 'frozen finite admission or fixed YT destination differs')
    for key in ('packet_manifest_sha256', 'protocol_sha256'):
        require(isinstance(contract.get(key), str) and len(contract[key]) == 64, 'root must fill final pin: ' + key)
    attestation = contract['root_prelaunch_attestation']
    require(all(attestation.get(k) is True for k in ('actual_cpu_qualification_pass', 'actual_h100_tiny_qualification_pass',
            'peer_source_review_pass', 'live_pod_and_statefulset_uid_match', 'no_overlapping_timing_jobs', 'fixed27_cpu_mask_admission_pass',
            'persistent_project_storage_admission_pass', 'controller_guard_fix_source_peer_pass'))
            and attestation['packet_manifest_sha256'] == contract['packet_manifest_sha256']
            and attestation['protocol_sha256'] == contract['protocol_sha256']
            and attestation['expected_cpu_affinity'] == contract['expected_cpu_affinity'],
            'root must bind actual qualifications, peer PASS, live UID admission and source/protocol before launch')
    base = Path(contract['base'])
    require(base.resolve() == base and base.is_dir() and contract['coordination'] == str(base / 'results'),
            'canonical existing base and shared timing coordination required')
    storage = persistent_storage_binding(contract)
    source_packet(HERE, 'SOURCE_MANIFEST.json', outer_manifest_sha)
    source_packet(Path(contract['packet']), 'SOURCE_MANIFEST.json', contract['packet_manifest_sha256'])
    source_packet(Path(contract['ingress_source']), 'SOURCE_MANIFEST.json', contract['ingress_source_manifest_sha256'])
    prior = prior_aborted_binding(contract)
    require(sha(Path(contract['ingress_source']) / 'ingress_friendster.py') == contract['ingress_source_sha256'],
            'original ingress source differs')
    for key in ('science_out', 'outer_out', 'archive'):
        path = Path(contract[key])
        require(path.is_relative_to(base) and path.resolve() == path and not os.path.lexists(path),
                'fresh canonical bounded output required: ' + key)
    trees = [Path(contract[k]) for k in ('science_out', 'outer_out', 'packet', 'outer_source',
                                        'friendster_root', 'ingress_source', 'ingress_out')]
    require(all(not left.is_relative_to(right) and not right.is_relative_to(left)
                for index, left in enumerate(trees) for right in trees[index + 1:]),
            'all graph, source, ingress and result trees must be disjoint')
    return dict(hostname_observed=socket.gethostname(), unix_uid_observed=os.getuid(),
                pod_uid_independently_visible=os.environ.get('POD_UID'), statefulset_uid_root_attested=contract['statefulset_uid'],
                binding_scope='Root verifies live Kubernetes Pod/STS UIDs; container independently requires hostname, Unix UID and exposed POD_UID.',
                root_prelaunch_attestation=attestation, outer_source_manifest_sha256=outer_manifest_sha,
                environment=environment, storage=storage, prior_aborted_diagnostic=prior,
                packet_manifest_sha256=contract['packet_manifest_sha256'], protocol_sha256=contract['protocol_sha256'],
                source_sha256=sha(Path(__file__)), admission_unix=time.time())


def ingress_ready(contract, reader=process):
    root, source = Path(contract['ingress_out']), Path(contract['ingress_source'])
    require(not (root / 'FAILED.json').exists(), 'ingress FAILED; no scientific launch')
    identity = dict(pid=contract['ingress_pid'], start_ticks=contract['ingress_start_ticks'],
                    entrypoint=str(source / 'ingress_friendster.py'))
    quiescence = quiet(identity, reader)
    if not (root / 'COMPLETE.json').exists():
        require(quiescence is None, 'ingress exited without genuine COMPLETE; no scientific launch')
        return None
    if quiescence is None:
        return None
    complete, plan = read(root / 'COMPLETE.json'), read(root / 'plan.json')
    require(info(root / 'COMPLETE.json')['sha256'] == contract['ingress_complete_sha256'],
            'root-observed genuine ingress COMPLETE SHA differs')
    require(all(complete.get(k) is True for k in ('success', 'all3_numeric_payloads_exact', 'base_environment_unchanged'))
            and complete['files'] == FR3 and complete['data_root'] == contract['friendster_root']
            and complete['version_before'] == complete['version_after'] == plan['version_before'],
            'exact all3 ingress completion and unchanged runtime required')
    for key in ('scientific_jobs_started', 'package_installs', 'YT_writes', 'new_YT_nodes', 'new_YT_chunks'):
        require(type(complete.get(key)) is int and complete[key] == 0, 'ingress no-science/no-install/no-write invariant absent')
    expected = dict(pod_uid=contract['pod_uid'], sts_uid=contract['statefulset_uid'], unix_uid=contract['unix_uid'],
                    hostname=contract['hostname'], source_manifest_sha256=contract['ingress_source_manifest_sha256'],
                    data_root=contract['friendster_root'], expected_files=FR3, max_seconds=5400,
                    max_disk_growth_bytes=40 * (1 << 30), reserve_bytes=4 * (1 << 30),
                    package_installs=0, YT_writes=0, scientific_jobs_started=0, automatic_whole_retry=False)
    require(all(type(plan.get(k)) is type(v) and plan[k] == v for k, v in expected.items()),
            'ingress source/UID/finite resource/no-write bindings differ')
    source_packet(source, 'SOURCE_MANIFEST.json', contract['ingress_source_manifest_sha256'])
    pins = read(source / 'EXPECTED_FRIENDSTER_FULL3.json')
    require(pins['files'] == FR3 and pins['num_nodes'] == 65608366 and pins['stored_entries'] == 3612134270,
            'frozen full Friendster identities differ')
    require(read(root / 'status.json')['phase'] == 'complete', 'genuine ingress final complete status required')
    require(read(root / 'downloads.json') == complete['downloads'] and set(complete['downloads']) == set(FR3)
            and all(all(row[k] == FR3[name][k] for k in ('bytes', 'sha256')) and row['YT_writes'] == 0
                    and row['path'] == str(Path(contract['friendster_root']) / name)
                    for name, row in complete['downloads'].items()), 'original whole-download receipt bindings differ')
    snapshots = {}
    for name, pin in FR3.items():
        path = Path(contract['friendster_root']) / name
        require(path.is_file() and not path.is_symlink() and path.resolve() == path and path.stat().st_size == pin['bytes'],
                'ingress graph stat binding differs')
        st = path.stat()
        snapshots[name] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino, device=st.st_dev)
    names = ('plan.json', 'COMPLETE.json', 'status.json', 'downloads.json')
    return dict(root_bound_original_identity=identity, process=quiescence,
                ingress_source_manifest_sha256=contract['ingress_source_manifest_sha256'],
                ingress_source_sha256=contract['ingress_source_sha256'], files=FR3, stat_snapshots=snapshots,
                receipt_files={name: info(root / name) for name in names},
                exit_code_scope='Ingress is not our child. Genuine COMPLETE plus original PID/start quiescence; an observed zombie must have actual Linux stat field52 exit_code0. Absent does not imply an observed wait0.',
                hashing_scope='Exact ingress whole-hash receipts and graph stat bindings admitted; producer independently rehashes full inputs. No duplicate15GiB hash here.')


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


def finite_numbers(value):
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_numbers(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def pid_start_matches(value, expected):
    # Process state R/S can legitimately change during a readonly scan.
    return isinstance(value, dict) and value.get('pid') == expected['pid'] and value.get('start_ticks') == expected['start_ticks']


def environment_process(value, expected, selected, baseline_cgroup):
    """Reclassify one complete recorded task bracket without opening its paths."""
    bracket = value['brackets']
    require(type(expected['pid']) is int and expected['pid'] > 0
            and type(expected['start_ticks']) is int and expected['start_ticks'] > 0
            and value['pid'] == expected['pid'] and value['start_ticks'] == expected['start_ticks']
            and value['expected_start_ticks'] == expected['start_ticks']
            and pid_start_matches(bracket['process_before'], expected)
            and pid_start_matches(bracket['process_after'], expected), 'observed own PID/start mismatch is a hard failure, never retried')
    require(type(bracket['begin_monotonic_ns']) is int and type(bracket['end_monotonic_ns']) is int
            and 0 < bracket['begin_monotonic_ns'] <= bracket['end_monotonic_ns'], 'actual ordered full task-bracket clocks required')
    require(bracket['cgroup_before'] == bracket['cgroup_after'] == value['cgroup_membership'] == baseline_cgroup,
            'observed own cgroup mismatch/unobserved is a hard failure')
    before, after = bracket['task_ids_before'], bracket['task_ids_after']
    for ids in (before, after):
        require(isinstance(ids, list) and 0 < len(ids) <= 256
                and all(type(tid) is int and tid > 0 for tid in ids) and ids == sorted(set(ids))
                and expected['pid'] in ids, 'bounded unique task membership must contain the own leader')
    require([r['tid'] for r in bracket['tasks_before']] == before
            and [r['tid'] for r in bracket['tasks_after']] == after
            and [r['tid'] for r in bracket['task_affinity_observations']] == before,
            'full before/affinity/after task rows must be retained, never dropped')
    starts, unobserved = {}, set()
    def note(tid, identity):
        if identity is None:
            unobserved.add(tid)
            return
        require(type(identity.get('pid')) is int and identity['pid'] == tid
                and type(identity.get('start_ticks')) is int and identity['start_ticks'] > 0,
                'actual task PID/start identity required')
        starts.setdefault(tid, set()).add(identity['start_ticks'])
    for row in bracket['tasks_before'] + bracket['tasks_after']:
        note(row['tid'], row['identity'])
    masks, tasks, vanished, leader = [], [], [], None
    for row in bracket['task_affinity_observations']:
        tid, first, last, mask = row['tid'], row['before'], row['after'], row['cpu_affinity']
        note(tid, first)
        note(tid, last)
        if mask is None:
            unobserved.add(tid)
        else:
            require(isinstance(mask, list) and mask and all(type(cpu) is int for cpu in mask)
                    and mask == sorted(set(mask)) and set(mask).issubset(selected),
                    'any observed empty/widened CPU mask is an immediate hard failure')
        if tid == expected['pid']:
            leader = mask
        if first is None or last is None or mask is None or first['start_ticks'] != last['start_ticks']:
            vanished.append(tid)
            continue
        if mask not in masks:
            masks.append(mask)
        tasks.append(dict(tid=tid, start_ticks=first['start_ticks'], mask_index=masks.index(mask)))
    require(all(len(identities) == 1 for identities in starts.values()), 'any observed TID reuse/start mismatch is an immediate hard failure')
    require(leader == value['leader_mask'] == selected and expected['pid'] not in unobserved,
            'own leader mask/start must be observed exact on every attempt')
    require(value['task_masks'] == masks and value['tasks'] == tasks and value['vanished_tids'] == vanished
            and value['process_identity_exact'] is True and value['task_identities_exact'] is True,
            'legacy task-mask/index/start summary must match full retained brackets')
    reasons = []
    if before != after:
        reasons.append('task_membership_changed')
    if unobserved:
        reasons.append('ancillary_task_unobserved')
    stable = not reasons and len(tasks) == len(before)
    require(value['stable'] is stable and set(value['instability_reasons']) == set(reasons),
            'task stability/ancillary-only classification must be independently consistent with full brackets')
    return stable, bracket['begin_monotonic_ns'], bracket['end_monotonic_ns']


def environment_attempt(value, contract, environment_contract, expected):
    required = ('resource_observation_readable', 'same_hostname', 'same_boot_id', 'same_cgroup_identity',
                'same_visible_quota_hierarchy', 'same_exact_cpu_quota', 'same_memory_limit', 'same_cpu_weight',
                'same_effective_mems', 'chosen_cpus_still_allowed', 'chosen_cpus_still_online',
                'stable_process_cgroups_exact', 'stable_process_leader_masks_exact', 'observed_thread_masks_within_chosen',
                'observed_process_PID_starts_exact', 'observed_TID_starts_exact', 'observed_process_cgroups_exact',
                'observed_process_leader_masks_exact', 'boundary_own_PID_start_observed_consistent',
                'boundary_own_leader_mask_exact', 'boundary_own_cgroup_observed_exact', 'boundary_instability_is_ancillary_task_only')
    require(value['status'] == 'PASS' and value['policy'] == 'fixed27_env_v3' and value['failed_predicates'] == []
            and all(value['predicates'].get(k) is True for k in required)
            and all(flag is True for flag in value['predicates'].values()),
            'any hard-negative resource/mask/PID/cgroup predicate prevents science success; no retry can hide it')
    selected, resources, base = contract['expected_cpu_affinity'], value['resources'], environment_contract['resources']
    require(resources['hostname'] == contract['hostname'] and resources['boot_id'] == contract['expected_boot_id']
            and resources['cpu_max'] == contract['expected_cpu_max'] and resources['memory_max'] == contract['expected_memory_max']
            and resources['cgroup'] == base['cgroup'] and resources['visible_quota_hierarchy'] == base['visible_quota_hierarchy']
            and resources['cpu_weight'] == base['cpu_weight'] and resources['ambient_mems'] == base['ambient_mems']
            and set(selected).issubset(resources['ambient_cpu_pool']) and set(selected).issubset(resources['online_cpus']),
            'unchanged fixed27 resources/visible limits/weight/mems and chosen availability required on every attempt')
    require(len(value['processes']) == 1, 'own boundary rescans must observe one identified process')
    stable, begin, end = environment_process(value['processes'][0], expected, selected, '0::' + base['cgroup']['membership'])
    require(value['process_scan_complete_stable'] is stable and type(value['observed_monotonic_ns']) is int
            and 0 < value['observed_monotonic_ns'] <= begin <= end, 'observed boundary clock/stability binding differs')
    return stable, value['observed_monotonic_ns'], end


def environment_observation(value, contract, environment_contract):
    """Admit successful bounded boundary rescan, rechecking every retained attempt."""
    rescan = value['boundary_rescan']
    require(rescan['max_attempts'] == 8 and rescan['max_wall_seconds'] == 0.5 and rescan['pause_seconds'] == 0.01
            and type(rescan['attempt_count']) is int and 1 <= rescan['attempt_count'] <= 8
            and len(rescan['attempts']) == rescan['attempt_count']
            and type(rescan['elapsed_seconds']) in (int, float) and math.isfinite(rescan['elapsed_seconds'])
            and 0 <= rescan['elapsed_seconds'] <= 0.5 and rescan['terminal_kind'] == 'fully_stable_pass'
            and len(rescan['expected_owners']) == 1, 'exact bounded<=8/0.5s successful readonly rescan receipt required')
    owner = rescan['expected_owners'][0]
    first, previous_end = None, None
    for index, attempt in enumerate(rescan['attempts'], 1):
        require(attempt['attempt'] == index, 'all boundary attempts must be retained in original order')
        stable, observed, end = environment_attempt(attempt['observation'], contract, environment_contract, owner)
        require(previous_end is None or observed >= previous_end, 'boundary rescan clocks must be ordered')
        first = observed if first is None else first
        previous_end = end
        if index < rescan['attempt_count']:
            require(not stable and attempt['classification'] == 'TRANSIENT_ANCILLARY_TASK_SCAN',
                    'only observed ancillary task churn may cause a boundary-only rescan')
        else:
            require(stable and attempt['classification'] == 'FULLY_STABLE_PASS', 'last boundary attempt must be fully stable')
    require((previous_end - first) / 1e9 <= rescan['elapsed_seconds'] + 1e-6,
            'all observed bracket work must fit inside recorded bounded rescan time')
    terminal = {key: data for key, data in value.items() if key not in ('label', 'boundary_rescan')}
    require(terminal == rescan['attempts'][-1]['observation'], 'successful top-level boundary receipt must retain the actual final raw observation')


def producer_environment(contract, root, complete, plan):
    require(complete['environment_policy'] == 'fixed27_env_v3' and complete['environment_guard_success'] is True
            and plan['environment_policy'] == 'fixed27_env_v3'
            and plan['selected_cpu_affinity'] == contract['expected_cpu_affinity'], 'complete newENVv3 producer gate required')
    path = root / 'ENVIRONMENT_ADMISSION.json'
    require(info(path)['sha256'] == complete['environment_admission_sha256'], 'original ENVv3 admission receipt SHA differs')
    admission = read(path)
    environment_contract = admission['contract']
    require(admission['status'] == 'PASS' and admission['policy'] == 'fixed27_env_v3'
            and admission['chosen_cpu_affinity'] == contract['expected_cpu_affinity']
            and environment_contract['policy'] == 'fixed27_env_v3'
            and environment_contract['chosen_cpu_affinity'] == contract['expected_cpu_affinity']
            and environment_contract['no_exclusive_or_continuous_placement_guarantee'] is True,
            'root-bound fixed27 ENVv3 admission and stated limitations required')
    environment_observation(admission['after'], contract, environment_contract)
    require(not any(p.name.endswith('ENVIRONMENT_FAILURE.json') for p in root.rglob('*')),
            'any new-cohort environment failure prevents science success')
    log = root / 'environment_events.jsonl'
    require(log.is_file() and not log.is_symlink() and 0 < log.stat().st_size <= MAX_MEMBER,
            'bounded nonempty ENVv3 guard log required')
    labels, count = set(), 0
    with log.open('r') as stream:
        while True:
            line = stream.readline((1 << 20) + 1)
            if not line:
                break
            require(len(line) <= 1 << 20 and line.endswith('\n') and count < 32768, 'bounded complete ENVv3 event lines required')
            observation = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
            environment_observation(observation, contract, environment_contract)
            labels.add(observation['label'])
            count += 1
    required_labels = {'controller_admission', 'assert_inputs'}
    for job in plan['jobs']:
        for prefix in ('before_command_', 'immediately_before_Popen_', 'after_actual_wait_'):
            required_labels.add(prefix + job['name'])
    require(required_labels.issubset(labels), 'all24 apps need actual named before/Popen/after-wait environment guards')
    return environment_contract, dict(admission=info(path), events=info(log), checked_events=count,
                                      all_required_new24_boundary_labels_present=True)


def producer_controller_guard(contract, root, complete, plan):
    """Check successful latched controller guard and every bounded statvfs receipt."""
    require(complete['controller_guard_failure'] is None
            and not any(p.name.endswith('CONTROLLER_GUARD_FAILURE.json') for p in root.rglob('*')),
            'any latched controller hard guard failure prevents science success, even if receipt persistence failed')
    require(plan['protocol']['controller_guard_amendment']['version'] == 'latched_disk_guard_v4',
            'new terminal controller guard protocol required')
    reference = plan['protocol']['controller_guard_amendment']['storage_admission']
    require(reference['artifact'] == 'ROOT_STORAGE_AMENDMENT.json'
            and reference['sha256'] == contract['root_storage_amendment_sha256']
            and reference['actual_admission_sha256'] == contract['root_actual_storage_admission_sha256'],
            'protocol storage reference must match the actual outer-bound immutable proofs')
    path = root / 'disk_observations.jsonl'
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 8 * MIB,
            'nonempty bounded8MiB disk history required')
    expected_keys = {'observed_monotonic_ns', 'path', 'device', 'bytes_available', 'bytes_total',
                     'inodes_available', 'inodes_total', 'inodes_reported', 'mount_readonly', 'scope'}
    count, first, last, minimum_bytes, minimum_inodes, exposed = 0, None, None, None, None, 0
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate disk observation JSON key'); result[key] = value
        return result
    with path.open('r') as stream:
        while True:
            line = stream.readline(4097)
            if not line:
                break
            require(len(line.encode()) <= 4096 and line.endswith('\n'), 'complete disk history line must be<=4096bytes')
            value = json.loads(line, object_pairs_hook=unique,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite disk JSON')))
            require(set(value) == expected_keys and type(value['observed_monotonic_ns']) is int
                    and value['observed_monotonic_ns'] > 0 and (last is None or value['observed_monotonic_ns'] > last),
                    'exact original disk observation schema/ordered clocks required')
            require(value['path'] == str(root) and type(value['device']) is int
                    and value['device'] == contract['expected_storage_device'],
                    'every runtime disk observation must use the admitted output PVC device')
            require(all(type(value[key]) is int and value[key] >= 0 for key in
                        ('bytes_available', 'bytes_total', 'inodes_available', 'inodes_total'))
                    and 4 * (1 << 30) <= value['bytes_available'] <= value['bytes_total']
                    and value['mount_readonly'] is False
                    and value['inodes_reported'] is (value['inodes_total'] > 0)
                    and (not value['inodes_reported'] or 0 < value['inodes_available'] <= value['inodes_total']),
                    'observed readonly/lowbytes/exposed inode exhaustion cannot heal into successful science')
            require(isinstance(value['scope'], str) and 0 < len(value['scope']) <= 1024, 'bounded disk scope required')
            first = value['observed_monotonic_ns'] if first is None else first
            last = value['observed_monotonic_ns']; count += 1
            minimum_bytes = value['bytes_available'] if minimum_bytes is None else min(minimum_bytes, value['bytes_available'])
            if value['inodes_reported']:
                exposed += 1
                minimum_inodes = value['inodes_available'] if minimum_inodes is None else min(minimum_inodes, value['inodes_available'])
    require(count > 0, 'at least one original disk observation required')
    latest_path = root / 'output_inventory_observation.json'
    require(info(latest_path, 4096)['bytes'] <= 4096, 'bounded latest output inventory receipt required')
    latest = read(latest_path)
    require(set(latest) == {'observed_monotonic_ns', 'disk_observed_monotonic_ns', 'observed_non_runtime_bytes',
                          'observed_non_runtime_files', 'vanished_entries', 'complete_entry_observations', 'scope'}
            and all(type(latest[key]) is int and latest[key] >= 0 for key in
                    ('observed_monotonic_ns', 'disk_observed_monotonic_ns', 'observed_non_runtime_bytes',
                     'observed_non_runtime_files', 'vanished_entries'))
            and latest['disk_observed_monotonic_ns'] == last <= latest['observed_monotonic_ns']
            and latest['observed_non_runtime_bytes'] <= 2 * (1 << 30)
            and latest['complete_entry_observations'] is (latest['vanished_entries'] == 0)
            and isinstance(latest['scope'], str) and 0 < len(latest['scope']) <= 1024,
            'latest non-atomic output cap receipt must bind last disk observation and retain vanished-entry scope')
    return dict(controller_guard_failure=None, disk_history=info(path), disk_observations=count,
                first_observed_monotonic_ns=first, last_observed_monotonic_ns=last,
                minimum_recorded_bytes_available=minimum_bytes, inode_reporting_observations=exposed,
                minimum_recorded_inodes_available=minimum_inodes, latest_output_inventory=info(latest_path, 4096),
                expected_storage_device=contract['expected_storage_device'],
                scope='Every raw recorded disk threshold/device/readwrite checked. Runtime4GiB/positive exposed inode guards differ from initial80GiB/64K admission. Reported byte availability derives from pinned statvfs code; block counters are not serialized. Inventory is sampled/non-atomic, not an instantaneous quota or reservation.')


def scientific_terminal(contract, exit_receipt, child_launch, reader=process):
    successful_exit(exit_receipt)
    require(exit_receipt['pid'] == child_launch['pid'] and exit_receipt['start_ticks'] == child_launch['start_ticks'],
            'actual child PID/start/exit differs')
    root = Path(contract['science_out'])
    complete, terminal = read(root / 'COMPLETE.json'), read(root / 'SCIENTIFIC_TERMINAL.json')
    require(all(complete.get(k) is True for k in ('controller_finished', 'success', 'owned_tree_quiescent'))
            and complete.get('failure') is None and type(complete.get('YT_writes')) is int and complete['YT_writes'] == 0
            and type(complete.get('statefulset_actions')) is int and complete['statefulset_actions'] == 0
            and complete['source_manifest_sha256'] == contract['packet_manifest_sha256'],
            'successful original scientific terminal and source pin required')
    controller = complete['controller_identity']
    require(controller['pid'] == child_launch['pid'] and controller['start_ticks'] == child_launch['start_ticks']
            and controller['source_manifest_sha256'] == contract['packet_manifest_sha256'], 'producer ownership identity differs')
    records, plan = read(root / 'job_records.json'), read(root / 'plan.json')
    guard_proof = producer_controller_guard(contract, root, complete, plan)
    environment_contract, environment_proof = producer_environment(contract, root, complete, plan)
    require(isinstance(records, list) and len(records) == 24 and complete['records'] == records
            and terminal['records'] == records, 'all24 original job records required')
    require(plan['execute'] is True and len(plan['jobs']) == 24
            and plan['protocol_sha256'] == contract['protocol_sha256']
            and hashlib.sha256(json.dumps(plan['protocol'], sort_keys=True).encode()).hexdigest() == contract['protocol_sha256']
            and plan['source_manifest_sha256'] == contract['packet_manifest_sha256']
            and plan['friendster_root'] == contract['friendster_root'] and plan['out'] == contract['science_out']
            and plan['coordination_dir'] == contract['coordination'] and 0 < plan['max_hours'] <= 4,
            'predeclared full protocol/source/paths/budget differs')
    expected_grid = {(h, w, rep, arm) for h in HORIZONS for w in (1, 8) for rep in (1, 2, 3) for arm in ARMS}
    jobs = {j['name']: j for j in plan['jobs']}
    require(len(jobs) == 24 and {(j['horizon'], j['workers'], j['repetition'], j['arm']) for j in jobs.values()} == expected_grid
            and len({r['name'] for r in records}) == 24 and set(jobs) == {r['name'] for r in records}, 'exact full horizon/worker/repetition/arm grid required')
    report_digests, streams, short_prefixes = {}, {}, set()
    for record in records:
        job = jobs[record['name']]
        require(all(record[k] == job[k] for k in ('horizon', 'workers', 'repetition', 'arm')), 'job record/grid identity differs')
        horizon, config = HORIZONS[job['horizon']], job['config']
        require(config['environment_contract'] == environment_contract, 'every new app must inherit exact same ENVv3 contract')
        require(job['name'] == f"{job['horizon']}_w{job['workers']}_r{job['repetition']}_{job['arm']}"
                and config['horizon'] == job['horizon'] and config['steps'] == horizon['steps']
                and config['checkpoints'] == horizon['checkpoints'] and config['batch_size'] == 32
                and config['warmup_pairs'] == 256 and config['num_workers'] == job['workers']
                and config['arm'] == job['arm'], 'exact job config differs')
        successful_exit(record['exit'])
        folder = root / 'jobs' / job['name']
        require(info(folder / 'report.json')['sha256'] == record['report_sha256'], 'original report SHA differs')
        report = read(folder / 'report.json')
        for key in ('environment_start', 'environment_end'):
            environment_observation(report[key], contract, environment_contract)
            for worker in report['worker_receipts']:
                environment_observation(worker[key], contract, environment_contract)
        require(report['status'] == 'complete' and report['config'] == config
                and report['consumed_graphs'] == horizon['consumed'] and report['measured_graphs'] == horizon['measured']
                and [p['total_trained_graphs'] for p in report['checkpoints']] == horizon['checkpoints']
                and report['model']['initial_state_sha256'] == MODEL_SHA and len(report['losses']) == horizon['steps']
                and all(type(v) in (int, float) and math.isfinite(v) for v in report['losses']), 'original complete report endpoint/model/checkpoints differ')
        streams.setdefault(job['horizon'], set()).add(report['input_stream_sha256'])
        prefix = report['input_stream_sha256'] if job['horizon'] == 'short320' else report['semantic_input_prefixes']['576']
        short_prefixes.add(prefix)
        if job['arm'] == ARMS[1]:
            postcheck, cleanup_receipt = read(folder / 'native_post_exit_integrity.json'), read(folder / 'native_scratch_cleanup.json')
            require(postcheck['outside_application_lifetime'] is True and cleanup_receipt['after_actual_exit_and_hash'] is True
                    and cleanup_receipt['owned_files_removed'] == 3 and not os.path.lexists(folder / 'runtime/native'),
                    'successful native post-exit integrity/cleanup receipt and absent full scratch required')
        report_digests[job['name']] = record['report_sha256']
    require(all(len(v) == 1 for v in streams.values()) and len(short_prefixes) == 1,
            'within-horizon whole consumed SHA and cross-horizon exact consumed576 prefix required')
    comparisons = terminal['stream_comparisons']
    require(terminal['success'] is True and terminal['jobs'] == 24 and terminal['no_new_quality_claim'] is True
            and type(terminal['YT_writes']) is int and terminal['YT_writes'] == 0 and len(comparisons) == 12
            and {(x['horizon'], x['workers'], x['repetition']) for x in comparisons}
                == {(h, w, rep) for h in HORIZONS for w in (1, 8) for rep in (1, 2, 3)}
            and all(x['all_consumed_inputs_exact'] is True and x['consumed_graphs'] == HORIZONS[x['horizon']]['consumed']
                    and x['stream_sha256'] in streams[x['horizon']] for x in comparisons), 'all12 actual consumed input comparisons required')
    cross = terminal['cross_horizon_input_prefix576']
    require(cross['consumed_input_graphs'] == 576 and cross['all_arms_workers_repetitions_exact'] is True
            and cross['short_final_stream_sha256'] == cross['long_prefix_stream_sha256']
            and cross['short_final_stream_sha256'] in short_prefixes, 'original exact cross-horizon prefix proof required')
    summary = read(root / 'scientific_summary.json')
    require(summary['status'] == 'complete' and summary['jobs'] == 24 and len(summary['rows']) == 8
            and len(summary['paired_compiled_over_native']) == 4 and len(summary['sampled_PSS']) == 24
            and finite_numbers(summary), 'complete finite24-job summary required')
    require({(r['horizon'], r['workers'], r['arm']) for r in summary['rows']}
                == {(h, w, arm) for h in HORIZONS for w in (1, 8) for arm in ARMS}
            and {(r['horizon'], r['workers']) for r in summary['paired_compiled_over_native']}
                == {(h, w) for h in HORIZONS for w in (1, 8)}
            and {r['name'] for r in summary['sampled_PSS']} == set(jobs), 'summary horizon/grid identities differ')
    input_identity = read(root / 'input_identity.json')
    require(input_identity['whole_hashes'] is True
            and set(input_identity['files']) == {str(Path(contract['friendster_root']) / name) for name in FR3}
            and all(row['bytes'] == FR3[Path(name).name]['bytes'] and row['sha256'] == FR3[Path(name).name]['sha256']
                    for name, row in input_identity['files'].items()), 'producer independent whole-input identities differ')
    require(read(root / 'status.json')['phase'] == 'complete', 'final producer complete status required')
    require(read(root / 'hardware.json')['target']['uuid'] == contract['gpu_uuid'], 'actual target GPU UUID differs')
    quiet_after = quiet(child_launch, reader)
    require(quiet_after is not None, 'actual child must be quiescent after wait')
    return dict(scientific_success=True, declared_jobs=24, actual_terminated_horizons=list(HORIZONS),
                environment_policy='fixed27_env_v3', environment_proof=environment_proof, controller_guard_proof=guard_proof,
                short_and_long_prefix576_exact=True, report_sha256=report_digests,
                complete_identity=info(root / 'COMPLETE.json'), summary_identity=info(root / 'scientific_summary.json'),
                scientific_terminal_identity=info(root / 'SCIENTIFIC_TERMINAL.json'), process=quiet_after,
                timing_lock=lock_available(contract['coordination']))


def supervise_child(contract, support, timeout, evidence, *, reader=process, argv=None):
    packet = Path(contract['packet'])
    source = packet / 'run_lifecycle.py'
    command = argv or [contract['python'], '-B', '-u', str(source), '--friendster-root', contract['friendster_root'],
        '--out', contract['science_out'], '--coordination-dir', contract['coordination'],
        '--max-hours', str(contract['producer_max_seconds'] / 3600),
        '--source-manifest-sha256', contract['packet_manifest_sha256'], '--execute']
    launch = {}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
               WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    env['E25_EXPECTED_CPU_AFFINITY'] = json.dumps(contract['expected_cpu_affinity'], separators=(',', ':'))
    # Only the producer creates its own resource-baseline contract; never reuse
    # a qualification process's inherited baseline as new controller admission.
    env.pop('E25_ENV_CONTRACT', None)
    def popen(*args, **kwargs):
        kwargs['env'] = env
        child = subprocess.Popen(*args, **kwargs)
        observed = reader(child.pid)
        require(observed is not None, 'actual launched child process identity unavailable')
        launch.update(pid=child.pid, start_ticks=observed['start_ticks'], entrypoint=str(source), argv=command,
                      source_sha256=sha(source), packet_manifest_sha256=contract['packet_manifest_sha256'],
                      started_unix=time.time(), stdin_dependency=False, start_new_session=True)
        launch['expected_cpu_affinity'] = contract['expected_cpu_affinity']
        save_new(evidence / 'child_launch.json', launch)
        return child
    result = support.run_owned(command, evidence / 'scientific.log', cwd=packet, timeout=timeout, popen=popen)
    result.update(start_ticks=launch['start_ticks'], finished_unix=time.time(), actual_wait_returned=True)
    save_new(evidence / 'child_exit.json', result)
    return launch, result


def tiny_pair_identity(path):
    identity = info(path, 4096)
    metadata = read(path.with_name(path.name + '.json'))
    with path.open('rb') as stream:
        require(stream.read(6) == b'\x93NUMPY', 'small NPY workload header required')
        version = stream.read(2)
        require(version in (b'\x01\x00', b'\x02\x00'), 'bounded recognized NPY header required')
        count = 2 if version[0] == 1 else 4
        size = struct.unpack('<H' if count == 2 else '<I', stream.read(count))[0]
        require(size <= 1024, 'tiny NPY header cap')
        header = ast.literal_eval(stream.read(size).decode('latin1').strip())
        require(header == dict(descr='<i8', fortran_order=False, shape=(128, 3))
                and stream.tell() + 128 * 3 * 8 == identity['bytes'], 'exact tiny128x3 int64 workload required')
    require(metadata['graph']['path'] == 'fixture' and metadata['n_positives'] == 64,
            'exact tiny qualification workload metadata required')
    return identity


def collect_files(contract, evidence):
    files, excluded = {}, {}
    for key, label in (('packet', 'packet'), ('ingress_source', 'ingress')):
        root = Path(contract[key])
        manifest = read(root / 'SOURCE_MANIFEST.json')['files']
        rows = manifest if isinstance(manifest, dict) else {r['path']: r for r in manifest}
        for name in list(rows) + ['SOURCE_MANIFEST.json']:
            files['e25/sources/' + label + '/' + name] = root / name
    for path in HERE.iterdir():
        require(not path.is_symlink() and path.is_file(), 'outer source inventory must remain flat regular files')
        files['e25/sources/outer/' + path.name] = path
    science = Path(contract['science_out'])
    require(not any(p.name.endswith('CONTROLLER_GUARD_FAILURE.json') for p in science.rglob('*')),
            'controller hard failure cannot enter a successful final raw archive')
    tiny_names = {f'qualification/w{w}_{arm}/pairs.npy' for w in (1, 8) for arm in ARMS}
    for directory, subdirs, names in os.walk(science, topdown=True, followlinks=False):
        directory = Path(directory)
        for name in subdirs + names:
            require(not (directory / name).is_symlink(), 'scientific result symlink refused')
        for name in list(subdirs):
            path = directory / name
            rel = path.relative_to(science)
            is_cache = name == 'shared_numba_cache'
            is_runtime = name == 'runtime' and ((len(rel.parts) == 3 and rel.parts[0] == 'jobs')
                                               or (len(rel.parts) == 3 and rel.parts[0] == 'qualification'))
            if is_cache or is_runtime:
                excluded[rel.as_posix() + '/'] = dict(excluded_tree=True, reason='cache or known owned runtime; graph/native scratch bytes are not archived')
                subdirs.remove(name)
        for name in names:
            path = directory / name
            require(path.is_file(), 'regular scientific output required')
            relative = path.relative_to(science).as_posix()
            if relative == 'build/native_assoc_v1.so':
                excluded[relative] = dict(**info(path), reason='native binary; textual build receipt retained')
            elif relative == 'friendster_pairs.npy':
                require(info(path) == WORKLOAD, 'result workload differs from exact small frozen source workload')
                excluded[relative] = dict(**WORKLOAD, reason='duplicate binary workload; exact source workload is archived')
            elif relative in tiny_names:
                excluded[relative] = dict(**tiny_pair_identity(path), reason='tiny binary fixture copy; only metadata and report archived')
            else:
                require(path.suffix in TEXT_SUFFIXES, 'unexpected binary/source under result tree: ' + relative)
                files['e25/results/' + relative] = path
    save_new(evidence / 'EXCLUDED_BINARY_IDENTITIES.json', dict(files=excluded,
        policy='Scientific raw archive contains bounded textual evidence only. One exact192128B source workload is included; no graph, native scratch, binaries, caches, models or credentials.'))
    for path in evidence.iterdir():
        require(not path.is_symlink() and path.is_file() and path.suffix in ('.json', '.log'), 'flat bounded textual outer evidence required')
        require(not path.name.endswith('CONTROLLER_GUARD_FAILURE.json'), 'controller hard failure evidence prevents successful archive')
        files['e25/envelope/' + path.name] = path
    for name in ('plan.json', 'COMPLETE.json', 'status.json', 'downloads.json'):
        files['e25/ingress/' + name] = Path(contract['ingress_out']) / name
    files['e25/envelope/actual_outer_launch.json'] = Path(contract['outer_launch'])
    prior_aborted_binding(contract)
    files['e25/prior_aborted/E25_ENV_V3_EXTERNAL_STOP_DIAGNOSTIC_v1.tgz'] = Path(contract['prior_aborted_archive'])
    files['e25/prior_aborted/PRESERVATION_RECEIPT.json'] = HERE / 'PRIOR_ABORTED_PRESERVATION_RECEIPT.json'
    return files



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
    phase = 'waiting_for_verified_ingress'
    def interrupted(signum, frame):
        raise RuntimeError('bounded envelope interrupted by signal ' + str(signum))
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, contract['max_total_seconds'] - 30)
    try:
        wait_until = start + contract['ingress_wait_seconds']
        ingress = None
        while ingress is None:
            require(time.monotonic() < wait_until, 'ingress admission wait deadline reached')
            ingress = ingress_ready(contract)
            if ingress is None:
                time.sleep(min(10, max(.1, wait_until - time.monotonic())))
        save_new(out / 'ingress_admitted.json', ingress)
        phase = 'running_complete_scientific_grid'
        source_packet(Path(contract['packet']), 'SOURCE_MANIFEST.json', contract['packet_manifest_sha256'])
        save_new(out / 'CPU_ENV_BEFORE_PRODUCER.json', cpu_binding(contract, 'before_new24_producer'))
        lock_available(contract['coordination'])
        require(not os.path.lexists(contract['science_out']), 'fresh science output required immediately before launch')
        # All24 jobs get the complete frozen4h producer budget, plus bounded exit cleanup.
        # The finite90min ingress wait and15min reserve fit inside the6h envelope.
        timeout = contract['producer_max_seconds'] + 90
        require(deadline - time.monotonic() - contract['final_reserve_seconds'] >= timeout,
                'insufficient bounded time for complete4h controller; no partial-grid launch')
        launch, result = supervise_child(contract, support, timeout, out)
        save_new(out / 'CPU_ENV_AFTER_PRODUCER.json', cpu_binding(contract, 'after_new24_producer_actual_wait'))
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
        source_packet(Path(contract['packet']), 'SOURCE_MANIFEST.json', contract['packet_manifest_sha256'])
        source_packet(HERE, 'SOURCE_MANIFEST.json', args.source_manifest_sha256)
        source_packet(Path(contract['ingress_source']), 'SOURCE_MANIFEST.json', contract['ingress_source_manifest_sha256'])
        require(all(info(Path(contract['ingress_out']) / name) == pin for name, pin in ingress['receipt_files'].items()),
                'admitted ingress evidence changed before final packaging')
        receipt = archive_results(contract['archive'], collect_files(contract, out))
        save_new(out / 'archive_local_receipt.json', receipt)
        require(time.monotonic() < deadline - 660, 'final persistence reserve exhausted; preserve local archive')
        phase = 'single_final_YT_transaction'
        pointer = upload_once(contract['archive'], receipt, contract['yt_final'])
        save_new(out / 'archive_pointer.json', pointer)
        success = True
    except BaseException as exc:
        # Never serialize arbitrary YT exception message/configuration.
        failure = dict(type=type(exc).__name__, phase=phase,
                       scope='Envelope failed; preserve bounded local evidence. Scientific success is separate; no automatic retries.')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        quiescent = cleanup(support)
        save_new(out / 'COMPLETE.json', dict(controller_finished=True, success=success and quiescent,
            scientific_success=scientific_success, owned_tree_quiescent=quiescent, failure=failure,
            archive_pointer=pointer, started_unix=binding['admission_unix'], completed_unix=time.time(),
            elapsed_seconds=time.monotonic() - start,
            final_YT_write_file_calls=1 if pointer else '0 or1; transaction may have failed; no retry',
            statefulset_lifecycle_actions=0, package_installs=0, automatic_retry=False,
            archive_completion_scope='Archive was made after actual scientific child exit and before this future outer COMPLETE/upload pointer. Root must collect these separately and observe outer exit.'))
        for sig, old in previous.items():
            signal.signal(sig, old)
    return 0 if success and quiescent else 1


if __name__ == '__main__':
    raise SystemExit(main())
