"""Public strict-clean Citation2 CPU contract and bounded identity helpers."""
from pathlib import Path
import ast, hashlib, importlib.util, json, math, os, platform, sys
from portable_ast_v1 import VERSION as PORTABLE_AST_VERSION, canonical_sha256

HERE = Path(__file__).resolve().parent
BASE = Path('/home/jovyan/seal_revision_20261124')
DEFAULT_GRAPH_PATH = BASE/'public_source_repro_v1/execution_v2_runs/citation2_public_repro_20261125_145307_fbdd1d46/dataset/prepared/train_csr'
NAME = 'citation2_cpu_matched_v1_20261126'
GIB, MIB = 2**30, 2**20
METHODS = ('scan', 'pairwise')
N, M = 2927963, 60703502
FULL_FILES = {'edge_starts.bin': {'bytes': 23423704, 'sha256': 'f06070adc368892f8441ea15e691dce400f9d44b6ec3e6e4505134c276c1072d'},
              'edge_ends.bin': {'bytes': 242814016, 'sha256': 'd3d8d1e2016675321c1d3d9ae817d2c6dca4a9dd2d5cdbba3e638ea7201057a4'}}
SCIENCE_PINS = {'original/native/prepare_replay_v1.py': '676dde1982e7f121b8b8343365d44e4fe0ee160ad3a0fd439d9f795585ce456f',
 'original/native/native_adapter_v1.py': '4aa4f51fcc6621ec42c6890be42439ef98132cc0554efbde3295e1d1f5e29549',
 'original/native/replay_worker_v1.py': '09a628149a8b06baabbb7b551bd98f00f41a8a44b5578ceb7a7d9e659cf147ea',
 'original/native/fixtures_v1.py': '28b2114495850fe9fa217de81072b531c1cab1611bc1e6d20f7d2e5fc6029c35',
 'original/native/run_parent_replay_v1.py': '239d499ffff748799207076db782b4f90120dac8f0965d93e7e791a7a77a62b2',
 'original/native/prepare_native_only_v1.py': '40992a0bb790caaf77194ce568c8c1f88c0040d20a276b8e49beea8837f2e241',
 'original/native/build_native_v1.py': '09f183d17fcc44416fa109ac05eb7e9b59f509c4443f2c5de9798091005a39e6',
 'original/native/native_assoc_v1.cpp': '67a9e651b7bad7889a73751bff3b33b4309fb2ca8ddfe2660acf851141045d20',
 'original/external_operator_benchmark.py': '525e88c904fd03e10fc617853c8d733e88af90487cbbdd9122ced97962df112c',
 'scientific_repo/scripts/seal_revision/exact_benchmark.py': '3ea7075dd80453a61eb42e8e990d21bf153635e6eff2e26379ba21788ad22544',
 'scientific_repo/seal/graph_csr_kernels.py': '19f517b874cfebc9f9388f094c17cf91b559936cd36f81976292c76025cfb2c5',
 'third_party/owned_supervision_v3.py': 'ec07bddc13f69cbbc1c5ba63a3ede390cfea254bcc02b811422338f5b9604aaf',
 'third_party/memory_guard_v3.py': 'e5bc616b518c17d6804c72f376872155514b902b490f43639526680ca5a516c8'}


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*MIB), b''): h.update(block)
    return h.hexdigest()


def info(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'regular immutable file required: '+str(p))
    return {'bytes': p.stat().st_size, 'sha256': sha(p)}


def read(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= 32*MIB, 'bounded regular JSON required')
    value = json.loads(p.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    require(isinstance(value, dict), 'JSON object required')
    return value


def save(path, value):
    p = Path(path); tmp = p.with_name(p.name+'.tmp')
    with tmp.open('x') as f: f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def layout(base):
    base = Path(base)
    return {'out': base/'results'/NAME, 'archive': base/(NAME+'.tgz'),
            'full': Path(os.environ.get('LPP_C2_MATCHED_GRAPH_PATH', str(DEFAULT_GRAPH_PATH))),
            'cohort': base/'results'/NAME/'cohort',
            'cohort_archive': base/'results'/NAME/'cohort/citation2_exact_int32.json',
            'cohort_plan': base/'results'/NAME/'cohort/FROZEN_128_SELECTION.json',
            'venv': base/'citation2_cpu_matched_v1_env', 'wheels': base/'citation2_cpu_matched_v1_wheels',
            'wheel_archive': base/'cpu_pt22_native_wheels_v1.tgz',
            'native': base/'citation2_cpu_matched_v1_data/native_i64',
            'build': base/'citation2_cpu_matched_v1_build', 'cache': base/'citation2_cpu_matched_v1_cache'}


def verify_sources():
    manifest = read(HERE/'SOURCE_MANIFEST.json'); files = manifest['files']; names = set(); bindings = {}
    require(0 < len(files) <= 80, 'bounded source inventory required')
    for item in files:
        rel = Path(item['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and str(rel) not in names, 'safe unique source name required')
        names.add(str(rel)); p = HERE/rel
        require(p.resolve().is_relative_to(HERE) and info(p) == {k: item[k] for k in ('bytes', 'sha256')}, 'source packet changed')
        bindings[str(p)] = info(p)
    for name, pin in SCIENCE_PINS.items():
        require(name in names and sha(HERE/name) == pin, 'original scientific/helper pin changed: '+name)
    portable = read(HERE/'PORTABLE_AST_PINS.json')
    require(portable['canonicalizer'] == PORTABLE_AST_VERSION, 'explicit portable AST schema required')
    for name in ('scientific_functions_v1.py', 'execution_v1.py'):
        proof = read(HERE/(name+'.lineage.json')); tree = ast.parse((HERE/name).read_text())
        entry = portable['files'][name]
        require(info(HERE/name) == entry['extracted_file_identity'] and proof == entry['unchanged_original_lineage'],
                'frozen extracted function bytes/original raw AST lineage changed')
        for fn, pin in entry['portable_functions'].items():
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
            require(canonical_sha256(node) == pin, 'original portable function AST changed')
    bindings[str(HERE/'SOURCE_MANIFEST.json')] = info(HERE/'SOURCE_MANIFEST.json')
    return bindings


def assert_bindings(bindings):
    require(all(info(path) == value for path, value in bindings.items()), 'bound source/environment file changed')


def stat_identity(path):
    p = Path(path); require(p.is_file() and not p.is_symlink(), 'regular numeric payload required')
    s = p.stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'device': s.st_dev, 'inode': s.st_ino}


def admit_original(folder, expected=FULL_FILES, dimensions=(N, M)):
    folder = Path(folder); files = {}
    for name, pin in expected.items():
        p = folder/name; before = stat_identity(p)
        require(before['bytes'] == pin['bytes'] and sha(p) == pin['sha256'], 'whole public fullCSR payload differs')
        require(before == stat_identity(p), 'fullCSR changed during full SHA verification')
        files[str(p)] = {**before, 'sha256': pin['sha256']}
    with (folder/'edge_ends.bin').open('rb') as f: count = int.from_bytes(f.read(8), 'little', signed=True)
    require(((folder/'edge_starts.bin').stat().st_size//8, count) == dimensions, 'fullCSR dimensions changed; trainCSR rejected')
    return {'kind': 'whole_public_strict_clean_CSR_SHA_v1', 'nodes': dimensions[0], 'stored_entries': dimensions[1], 'files': files,
            'expected_native_hashes': derive_native_hashes(folder, dimensions)}


def stable_original(admission):
    for path, value in admission['files'].items():
        require(stat_identity(path) == {k: value[k] for k in ('bytes', 'mtime_ns', 'device', 'inode')}, 'original fullCSR stat continuity failed')


def host_receipt():
    text = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    require(len(text) <= 2*MIB, 'bounded CPU description required')
    return {'hostname': platform.node(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip() if Path('/proc/sys/kernel/random/boot_id').exists() else None,
            'cpu_models': sorted({r.split(':', 1)[1].strip() for r in text.splitlines() if r.startswith('model name')}),
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}


def verify_host(expected, actual):
    require(all(actual.get(k) and actual[k] == expected[k] for k in ('hostname', 'boot_id', 'cpu_models', 'cpu_affinity')), 'same documented CPU host/boot/affinity required')


def clean(exit):
    return (type(exit.get('returncode')) is int and exit['returncode'] == 0 and exit.get('timed_out') is False
            and exit.get('owned_tree_quiescent') is True and exit.get('orphan_cleanup') is False and exit.get('guard_abort_reason') is None)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def frozen_plan_match(plan):
    expected = read(layout(BASE)['cohort_plan']); actual = dict(plan)
    expected['archive'] = dict(expected['archive']); actual['archive'] = dict(actual['archive'])
    expected['archive'].pop('original_local_path', None); actual['archive'].pop('original_local_path', None)
    require(actual == expected, 'exact independently frozen128 selection/protocol required')


def conversion_binding(native, parent, original):
    receipt, exit = read(Path(parent)/'native_conversion_receipt.json'), read(Path(parent)/'native_conversion_exit.json')
    require(receipt['status'] == 'complete' and type(exit['returncode']) is int and exit['returncode'] == 0 and exit['timed_out'] is False, 'successful original converter required')
    meta = read(Path(native)/'manifest.json')
    require(meta == receipt['conversion'] and all(meta[k] == v for k, v in original['expected_native_hashes'].items()), 'converted native hashes differ from independently streamed original-array promotion')
    require((meta['graph']['n'], meta['graph']['m']) == (N, M) and meta['sentinel_added'] is True and meta['native_dtype'] == 'little-endian int64', 'lossless native representation differs')
    source = [{'path': p, 'bytes': value['bytes'], 'mtime_ns': value['mtime_ns']} for p, value in original['files'].items() if Path(p).name in ('edge_starts.bin', 'edge_ends.bin')]
    require(meta['graph']['source_files'] == source, 'native conversion not bound to actually verified fullCSR')
    return meta


def derive_native_hashes(folder, dimensions):
    """Independent stdlib streaming promotion; never trust self-reported converter hashes."""
    import array, struct
    require(sys.byteorder == 'little', 'reviewed little-endian runtime required')
    n, m = dimensions; folder = Path(folder)
    rowptr = hashlib.sha256()
    with (folder/'edge_starts.bin').open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''): rowptr.update(block)
    rowptr.update(struct.pack('<q', m)); col = hashlib.sha256(); entries = 0
    with (folder/'edge_ends.bin').open('rb') as stream:
        require(stream.read(8) == struct.pack('<q', m), 'original entry header changed')
        for raw in iter(lambda: stream.read(MIB), b''):
            require(len(raw) % 4 == 0, 'aligned int32 input chunk required')
            values = array.array('i'); values.frombytes(raw)
            require(values.itemsize == 4, 'native int32 width required')
            promoted = array.array('q', values)
            require(promoted.itemsize == 8, 'native int64 width required')
            col.update(promoted.tobytes()); entries += len(values)
    require(entries == m, 'complete original promotion required')
    return {'rowptr_sha256': rowptr.hexdigest(), 'col_sha256': col.hexdigest()}
