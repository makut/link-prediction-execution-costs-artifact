#!/usr/bin/env python3
"""One bounded closure block, genuine owned-child wait and compact raw archive."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, importlib.util, io, json, os, signal, socket, subprocess, sys, tarfile, time

MIB = 1 << 20
ARMS = {'reference_union_compiled', 'reference_union_native_fresh', 'reference_union_native_reuse'}
TINY_QUALIFICATION_PAIRS = {'bytes': 3200, 'sha256': '049906aff7fd79aca8aa79800e45b39f61104d71b58d49bccff56811e4089913'}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(MIB), b''):
            h.update(chunk)
    return h.hexdigest()


def info(path, maximum=32 * MIB):
    p = Path(path)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size <= maximum, 'bounded regular file required: ' + str(p))
    return {'bytes': p.stat().st_size, 'sha256': sha(p)}


def read(path):
    info(path)
    return json.loads(Path(path).read_text())


def save(path, value):
    p = Path(path)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    need(len(raw) <= 32 * MIB and not os.path.lexists(p), 'fresh bounded output required')
    with p.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def copy_small(source, target):
    expected = info(source)
    raw = source.read_bytes()
    need({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected, 'source changed during metadata copy')
    need(not target.exists(), 'fresh metadata copy required')
    with target.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    need(info(target) == expected, 'metadata copy differs')
    return expected


def source_packet(root, expected):
    need(info(root / 'SOURCE_MANIFEST.json')['sha256'] == expected, 'source manifest differs')
    rows = read(root / 'SOURCE_MANIFEST.json')['files']
    need(isinstance(rows, dict) and 0 < len(rows) < 512, 'source inventory bounds')
    total = 0
    for name, pin in rows.items():
        rel = PurePosixPath(name)
        need(not rel.is_absolute() and '..' not in rel.parts and str(rel) == name, 'unsafe source member')
        p = root / name
        need(p.resolve() == p and info(p) == {k: pin[k] for k in ('bytes', 'sha256')}, 'source payload differs: ' + name)
        total += pin['bytes']
    need(total <= 32 * MIB, 'source size bound')
    observed = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'SOURCE_MANIFEST.json'}
    need(observed == set(rows), 'unexpected source file')
    return rows


def successful_exit(value):
    need(value.get('returncode') == 0 and value.get('timed_out') is False
         and value.get('orphan_cleanup') is False and value.get('owned_tree_quiescent') is True
         and value.get('actual_wait_returned') is True and value.get('owned_pgids_signaled') == []
         and value.get('guard_abort_reason') is None, 'genuine clean wait0 required')


def collect(args):
    files = {}
    for prefix, root in [('closure/sources/packet/', args.packet),
                         ('closure/sources/envelope/', Path(__file__).parent)]:
        for p in root.rglob('*'):
            need(not p.is_symlink(), 'source symlink refused')
            if p.is_file():
                files[prefix + str(p.relative_to(root))] = p
    root = args.out / 'producer'
    tiny_pairs = {Path('qualification') / ('w8_' + arm) / 'pairs.npy' for arm in ARMS}
    for directory, subdirs, names in os.walk(root, followlinks=False):
        subdirs[:] = [x for x in subdirs if x not in ('runtime', 'shared_numba_cache', '__pycache__')]
        for name in names:
            p = Path(directory) / name
            need(not p.is_symlink(), 'result symlink refused')
            if p.suffix in ('.json', '.jsonl', '.log', '.txt', '.csv', '.tsv'):
                files['closure/results/' + str(p.relative_to(root))] = p
            elif p.suffix == '.npy':
                if p.relative_to(root) in tiny_pairs:
                    need(info(p) == TINY_QUALIFICATION_PAIRS, 'tiny public qualification workload differs')
                    continue
                original = args.packet / 'workloads' / (args.dataset + '_pairs.npy')
                need(p.relative_to(root) == Path('pairs.npy') and info(p) == info(original), 'only exact public source workload allowed')
                files['closure/results/' + str(p.relative_to(root))] = p
    need('closure/results/pairs.npy' in files, 'exact public source workload required')
    for name in ('platform_admission.json', 'preflight.json', 'child_launch.json',
                 'child_exit.json', 'SCIENTIFIC_TERMINAL.json', 'producer.log',
                 'native_preparation_READY.json', 'native_preparation_launch.json',
                 'native_preparation_exit.json', 'native_preparation_driver.py'):
        files['closure/envelope/' + name] = args.out / name
    return files


def archive(path, files):
    need(not path.exists() and 0 < len(files) < 4096, 'fresh bounded archive')
    pins = {name: info(p) for name, p in files.items()}
    need(sum(x['bytes'] for x in pins.values()) <= 256 * MIB, 'raw result bound')
    manifest = (json.dumps({'files': pins, 'graphs_included': False, 'native_arrays_included': False},
                           sort_keys=True, indent=2) + '\n').encode()
    with tarfile.open(path, 'x:gz') as tar:
        for name, p in sorted(files.items()):
            rel = PurePosixPath(name)
            need(name.startswith('closure/') and not rel.is_absolute() and '..' not in rel.parts, 'archive path')
            need(info(p) == pins[name], 'payload changed before archive')
            member = tarfile.TarInfo(name); member.size = pins[name]['bytes']; member.mode = 0o644
            with p.open('rb') as stream:
                tar.addfile(member, stream)
            need(info(p) == pins[name], 'payload changed during archive')
        member = tarfile.TarInfo('SHA256_MANIFEST.json'); member.size = len(manifest); member.mode = 0o644
        tar.addfile(member, io.BytesIO(manifest))
    need(path.stat().st_size <= 64 * MIB, 'compressed result bound')
    seen = set()
    with tarfile.open(path, 'r:gz') as tar:
        for member in tar:
            need(member.isfile() and member.name not in seen and member.size <= 32 * MIB, 'regular unique bounded member')
            seen.add(member.name)
            with tar.extractfile(member) as stream:
                h = hashlib.sha256(); size = 0
                for chunk in iter(lambda: stream.read(MIB), b''):
                    h.update(chunk); size += len(chunk)
            expected = {'bytes': len(manifest), 'sha256': hashlib.sha256(manifest).hexdigest()} if member.name == 'SHA256_MANIFEST.json' else pins[member.name]
            need({'bytes': size, 'sha256': h.hexdigest()} == expected, 'archive member rehash failed')
    need(seen == set(pins) | {'SHA256_MANIFEST.json'}, 'archive inventory failed')
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    return {'path': str(path), **info(path, 64 * MIB), 'members': len(seen), 'all_members_rehashed': True}


def cleanup(support):
    groups = support.owned_groups(os.getpid())
    if groups:
        support.signal_confirmed(groups, signal.SIGTERM)
        end = time.monotonic() + 5
        while time.monotonic() < end and support.owned_groups(os.getpid()):
            time.sleep(.1)
        support.signal_confirmed(support.owned_groups(os.getpid()), signal.SIGKILL)
    return not support.owned_groups(os.getpid())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('friendster', 'citation2'), required=True)
    parser.add_argument('--block', type=int, choices=(1, 2, 3), required=True)
    for name in ('packet', 'graph-root', 'native-reuse-root', 'platform-admission-json', 'out', 'coordination'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('packet-manifest-sha256', 'protocol-sha256', 'native-reuse-receipt-sha256', 'platform-admission-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    for key in ('packet', 'graph_root', 'native_reuse_root', 'platform_admission_json', 'out', 'coordination'):
        p = getattr(args, key); need(p.is_absolute() and p.resolve() == p, 'canonical absolute path required')
    source_packet(args.packet, args.packet_manifest_sha256)
    need(hashlib.sha256(json.dumps(read(args.packet / 'PROTOCOL.json'), sort_keys=True).encode()).hexdigest()
         == args.protocol_sha256, 'canonical protocol object SHA differs')
    need(sha(args.platform_admission_json) == args.platform_admission_sha256, 'actual platform SHA differs')
    proof = read(args.platform_admission_json)
    need(proof['kind'] == 'CLOSURE_V1_ACTUAL_BLOCK_PLATFORM_ADMISSION'
         and proof['dataset_id'] == args.dataset and proof['block'] == args.block
         and proof['pod_uid'] == os.environ['POD_UID'] and proof['hostname'] == socket.gethostname()
         and proof['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
         and proof['actual_scientific_jobs_started'] == 0, 'actual allocation/dataset binding differs')
    need(sha(args.native_reuse_root / 'READY.json') == args.native_reuse_receipt_sha256, 'reusable source receipt differs')
    need(not args.out.exists(), 'fresh block output required'); args.out.mkdir()
    preparation_base = args.native_reuse_root.parent.parent
    need(preparation_base == Path(proof['base']) / 'closure_controls_20261127', 'owned preparation directory differs')
    preparation_exit = preparation_base / 'ops' / ('prepare_' + args.dataset + '_actual_exit.json')
    prep_exit = read(preparation_exit); successful_exit(prep_exit)
    need(prep_exit['dataset_id'] == args.dataset and prep_exit['source_manifest_sha256'] == args.packet_manifest_sha256
         and prep_exit['ready_receipt'] == info(args.native_reuse_root / 'READY.json'), 'genuine original preparation wait/READY binding differs')
    preparation_sources = {
        'native_preparation_READY.json': args.native_reuse_root / 'READY.json',
        'native_preparation_launch.json': preparation_base / 'ops' / ('prepare_' + args.dataset + '_actual_launch.json'),
        'native_preparation_exit.json': preparation_exit,
        'native_preparation_driver.py': preparation_base / 'source/prep_run.py'}
    need(sha(preparation_sources['native_preparation_driver.py']) == prep_exit['prep_driver_source_sha256'], 'preparation driver differs')
    preparation_pins = {name: copy_small(path, args.out / name) for name, path in preparation_sources.items()}
    save(args.out / 'platform_admission.json', proof)
    save(args.out / 'preflight.json', {'dataset_id': args.dataset, 'repetition_block': args.block,
         'source_hashes': {'packet_manifest_sha256': args.packet_manifest_sha256,
             'protocol_sha256': args.protocol_sha256, 'wrapper_sha256': sha(Path(__file__)),
             'platform_admission_sha256': args.platform_admission_sha256},
         'native_reuse_receipt_sha256': args.native_reuse_receipt_sha256,
         'native_preparation_files': preparation_pins,
         'max_seconds': 4200, 'producer_max_seconds': 3900, 'YT_writes': 0})
    spec = importlib.util.spec_from_file_location('closure_support', args.packet / 'support/owned_supervision.py')
    support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support); support.enable_subreaper()
    launch = {}; success = False; failure = None; archived = None; quiet = False
    def interrupted(*_):
        raise RuntimeError('closure envelope interrupted')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, 4180)
    try:
        command = [sys.executable, '-B', '-u', str(args.packet / 'run_lifecycle.py'),
            '--dataset', args.dataset, '--graph-root', str(args.graph_root),
            '--native-reuse-root', str(args.native_reuse_root),
            '--native-reuse-receipt-sha256', args.native_reuse_receipt_sha256,
            '--out', str(args.out / 'producer'), '--coordination-dir', str(args.coordination),
            '--max-hours', str(3900 / 3600), '--repetition-block', str(args.block),
            '--platform-admission-json', str(args.platform_admission_json),
            '--platform-admission-sha256', args.platform_admission_sha256,
            '--source-manifest-sha256', args.packet_manifest_sha256, '--execute']
        def popen(*values, **kwargs):
            kwargs['env'] = dict(kwargs['env'], PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                E25_EXPECTED_CPU_AFFINITY=json.dumps(proof['cpu_affinity'], separators=(',', ':')))
            began = time.monotonic_ns(); child = subprocess.Popen(*values, **kwargs)
            actual = support.process_identity(child.pid); need(actual is not None, 'actual producer identity absent')
            launch.update(pid=child.pid, start_ticks=actual['start_ticks'], entrypoint=str(args.packet / 'run_lifecycle.py'),
                          launch_monotonic_ns=began, argv=command)
            save(args.out / 'child_launch.json', launch); return child
        result = support.run_owned(command, args.out / 'producer.log', cwd=args.packet, timeout=3960, popen=popen)
        result.update(actual_wait_returned=True, start_ticks=launch['start_ticks'],
                      wait_returned_monotonic_ns=time.monotonic_ns(), guard_abort_reason=None)
        save(args.out / 'child_exit.json', result); successful_exit(result)
        need(not support.owned_groups(os.getpid()), 'owned descendants remain after actual wait')
        root = args.out / 'producer'; done = read(root / 'COMPLETE.json'); records = read(root / 'job_records.json')
        need(done['success'] is True and done['controller_finished'] is True and done['failure'] is None
             and done['owned_tree_quiescent'] is True and done['source_manifest_sha256'] == args.packet_manifest_sha256,
             'complete source-bound producer required')
        need(len(records) == 6 and {(r['horizon'], r['workers'], r['repetition'], r['arm']) for r in records}
             == {(h, 8, args.block, arm) for h in ('short320', 'long6400') for arm in ARMS}, 'exact six-app grid required')
        for record in records:
            successful_exit(record['exit'])
            report = root / 'jobs' / record['name'] / 'report.json'
            need(sha(report) == record['report_sha256'] and read(report)['status'] == 'complete', 'original report differs')
        source_packet(args.packet, args.packet_manifest_sha256)
        save(args.out / 'SCIENTIFIC_TERMINAL.json', {'kind': 'CLOSURE_V1_SIX_APP_ENVELOPE_SUCCESS',
             'success': True, 'dataset_id': args.dataset, 'repetition_block': args.block,
             'jobs': 6, 'source_manifest_sha256': args.packet_manifest_sha256,
             'producer_exit': result, 'independent_raw_audit_required': True, 'YT_writes': 0})
        archived = archive(args.out / 'block_raw.tgz', collect(args)); success = True
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)[:2000], 'partial_block_excluded': True}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); quiet = cleanup(support)
        save(args.out / 'COMPLETE.json', {'kind': 'CLOSURE_V1_BLOCK_ENVELOPE_TERMINAL', 'success': success and quiet,
             'dataset_id': args.dataset, 'repetition_block': args.block, 'block_jobs': 6,
             'owned_tree_quiescent': quiet, 'archive': archived, 'failure': failure,
             'automatic_retry': False, 'YT_writes': 0})
        for sig, old in previous.items():
            signal.signal(sig, old)
    return 0 if success and quiet else 1


if __name__ == '__main__':
    raise SystemExit(main())
