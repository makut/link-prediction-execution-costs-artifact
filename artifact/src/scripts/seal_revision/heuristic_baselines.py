#!/usr/bin/env python3
"""Fixed Friendster degree/PA/CN/AA/RA baselines, CPU-only, query-row aware.

No training, HPO, negatives generation, Torch or GPU imports. Repeated physical
sources remain independent original queries; ties use midpoint ranks. Test
scoring is blocked until all six validation-selected SEAL checkpoints are jointly
frozen under one verified quality plan. With --query-ids use the exact frozen
row-ID file; otherwise evaluate every provided query row.

    python -m scripts.seal_revision.heuristic_baselines \
      --prepared-root DATA/friendster/prepared --quality-root RESULTS/quality_friendster \
      --split test --query-ids RESULTS/quality_friendster/subsets/test_query_ids.npy \
      --frozen-checkpoints-manifest RESULTS/quality_friendster/FROZEN_CHECKPOINTS.json \
      --threads 8 --out RESULTS/friendster_heuristics/test

Scores: degree_only=degree(dst), preferential_attachment=degree(src)*degree(dst),
common_neighbors=|N(src) intersection N(dst)|, adamic_adar=sum(1/log(degree(w))),
resource_allocation=sum(1/degree(w)). Graph must be sorted, simple, symmetric.
These are simple structural/degree-bias controls, not modern SOTA comparisons.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time
import uuid

import numba as nb
import numpy as np

METHODS = ('degree_only', 'preferential_attachment', 'common_neighbors', 'adamic_adar', 'resource_allocation')
LOG = logging.getLogger('friendster_heuristics')
GIB = 2**30


class OperationBudgetExceeded(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def rss_status():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current = None
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                current = int(line.split()[1]) * 1024
    return {'peak_rss_bytes': int(peak if sys.platform == 'darwin' else peak * 1024), 'current_rss_bytes': current}


class CSR:
    """Read-only LE mmap; N offsets without sentinel, int64-count/int32-neighbors."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        offsets, neighbors = self.root / 'edge_starts.bin', self.root / 'edge_ends.bin'
        if not offsets.stat().st_size or offsets.stat().st_size % 8:
            raise ValueError('invalid CSR offset file')
        self.starts = np.memmap(offsets, dtype='<i8', mode='r')
        with neighbors.open('rb') as stream:
            self.m = struct.unpack('<q', stream.read(8))[0]
        if self.m < 0 or neighbors.stat().st_size != 8 + self.m * 4:
            raise ValueError('CSR neighbor count/file-size mismatch')
        self.col = np.memmap(neighbors, dtype='<i4', offset=8, shape=(self.m,), mode='r') if self.m else np.empty(0, dtype=np.int32)
        self.n = int(self.starts.size)
        if int(self.starts[0]) != 0 or not self.starts.dtype.isnative or not self.col.dtype.isnative:
            raise ValueError('expected native-endian little CSR with first offset zero')
        self.logical_bytes = offsets.stat().st_size + neighbors.stat().st_size


@nb.njit(inline='always')
def _end(starts, m, u):
    return starts[u + 1] if u + 1 < len(starts) else m


@nb.njit(inline='always')
def _degree(starts, m, u):
    if u < 0 or u >= len(starts):
        return -1
    lo, hi = starts[u], _end(starts, m, u)
    if lo < 0 or hi < lo or hi > m:
        return -1
    return hi - lo


@nb.njit(inline='always')
def _binary_steps_bound(length):
    steps = 0
    while length:
        steps += 1
        length //= 2
    return steps + 1  # final membership comparison after lower_bound


@nb.njit(inline='always')
def pair_operation_bound(du, dv):
    if du <= 0 or dv <= 0:
        return 0
    small, large = (du, dv) if du <= dv else (dv, du)
    return min(du + dv, small * _binary_steps_bound(large))


@nb.njit(cache=True, parallel=True)
def estimate_query_operations(starts, m, sources, targets, negatives, query_ids):
    estimates = np.zeros(len(query_ids), dtype=np.int64)
    invalid = np.zeros(len(query_ids), dtype=np.uint8)
    for row in nb.prange(len(query_ids)):
        q = query_ids[row]; u = sources[q]
        du = _degree(starts, m, u)
        if du < 0:
            invalid[row] = 1
            continue
        total = 0
        for slot in range(negatives.shape[1] + 1):
            v = targets[q] if slot == 0 else negatives[q, slot - 1]
            dv = _degree(starts, m, v)
            if dv < 0 or u == v:
                invalid[row] = 1
                break
            total += pair_operation_bound(du, dv)
        estimates[row] = total
    return estimates, invalid


@nb.njit(inline='always')
def _weights(starts, m, w):
    degree = _degree(starts, m, w)
    if degree < 2:
        return 0.0, 0.0, 1  # impossible common neighbor in a simple symmetric graph
    return 1.0 / math.log(degree), 1.0 / degree, 0


@nb.njit(cache=True)
def score_pair(starts, col, u, v):
    """Intersect sorted rows with merge or shorter-row binary search; no row copies."""
    du, dv = _degree(starts, len(col), u), _degree(starts, len(col), v)
    scores = np.zeros(5, dtype=np.float64)
    if du < 0 or dv < 0 or u == v:
        return scores, 0, 1
    scores[0], scores[1] = dv, float(du) * dv
    if not du or not dv:
        return scores, 0, 0
    us, ue, vs, ve = starts[u], _end(starts, len(col), u), starts[v], _end(starts, len(col), v)
    small, large = (du, dv) if du <= dv else (dv, du)
    operations, error = 0, 0
    if small * _binary_steps_bound(large) < du + dv:
        if du > dv:
            us, ue, vs, ve = vs, ve, us, ue
        for at in range(us, ue):
            w = int(col[at]); lo, hi = vs, ve
            while lo < hi:
                operations += 1
                mid = (lo + hi) // 2
                if col[mid] < w:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < ve:
                operations += 1
                if col[lo] == w:
                    aa, ra, bad = _weights(starts, len(col), w)
                    scores[2] += 1; scores[3] += aa; scores[4] += ra
                    error |= bad
    else:
        a, b = us, vs
        while a < ue and b < ve:
            operations += 1
            left, right = int(col[a]), int(col[b])
            if left == right:
                aa, ra, bad = _weights(starts, len(col), left)
                scores[2] += 1; scores[3] += aa; scores[4] += ra
                error |= bad
                a += 1; b += 1
            elif left < right:
                a += 1
            else:
                b += 1
    return scores, operations, error


@nb.njit(cache=True, parallel=True)
def score_query_rows(starts, col, sources, targets, negatives, query_ids):
    """One score for each positive and ALL provided negatives, independently per row."""
    ranks = np.zeros((len(query_ids), 5), dtype=np.float64)
    operations = np.zeros(len(query_ids), dtype=np.int64)
    invalid = np.zeros(len(query_ids), dtype=np.uint8)
    for row in nb.prange(len(query_ids)):
        q = query_ids[row]; u = sources[q]
        positive, count, bad = score_pair(starts, col, u, targets[q])
        operations[row] += count; invalid[row] |= bad
        greater, equal = np.zeros(5, np.int64), np.zeros(5, np.int64)
        for slot in range(negatives.shape[1]):
            scores, count, bad = score_pair(starts, col, u, negatives[q, slot])
            operations[row] += count; invalid[row] |= bad
            for method in range(5):
                greater[method] += scores[method] > positive[method]
                equal[method] += scores[method] == positive[method]
        for method in range(5):
            ranks[row, method] = 1 + greater[method] + .5 * equal[method]
    return ranks, operations, invalid


def normalize_config(config):
    normalized = copy.deepcopy(config)
    normalized.pop('output', None)
    normalized['training'].pop('seed', None)
    normalized['train_loader'].pop('seed', None)
    normalized['train_loader']['negative_sampling'].pop('seed', None)
    normalized['seal'].pop('graph_csr_use_per_vertex_sampling', None)
    return normalized


def load_quality_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan['protocol']['dataset'] != 'friendster_colleague_80_10_10':
        raise ValueError('quality plan is not the fixed colleague Friendster protocol')
    expected = {(variant, seed) for variant in ('union', 'pv') for seed in (42, 43, 44)}
    if len(plan['jobs']) != 6 or {(j['variant'], j['seed']) for j in plan['jobs']} != expected or len({j['name'] for j in plan['jobs']}) != 6:
        raise ValueError('quality plan must contain exactly union/pv x seeds42/43/44')
    return plan


def verify_test_gate(frozen_path, plan_path=None):
    """No scoring/data-loader work occurs until this gate returns successfully."""
    if frozen_path is None:
        raise ValueError('--frozen-checkpoints-manifest is required before ANY test scoring')
    frozen_path = Path(frozen_path).resolve()
    plan_path = Path(plan_path).resolve() if plan_path else frozen_path.with_name('plan.json')
    frozen, plan = json.loads(frozen_path.read_text()), load_quality_plan(plan_path)
    if frozen.get('selection_uses_test') is not False or frozen.get('plan_sha256') != sha256_file(plan_path):
        raise ValueError('joint freeze is not validation-only or its plan hash changed')
    expected = {(v, seed) for v in ('union', 'pv') for seed in (42, 43, 44)}
    jobs = frozen.get('jobs', [])
    if len(jobs) != 6 or {(j['variant'], j['seed']) for j in jobs} != expected or len({j['name'] for j in jobs}) != 6:
        raise ValueError('joint freeze must contain all six unique variant/seed checkpoints')
    planned = {j['name']: j for j in plan['jobs']}
    normalized, initial = None, {}
    for job in jobs:
        if job['name'] not in planned:
            raise ValueError('frozen job absent from plan')
        entry = planned[job['name']]
        for key in ('seed', 'variant', 'config', 'config_sha256'):
            if job[key] != entry[key]:
                raise ValueError(f'frozen job differs from plan: {job["name"]}/{key}')
        if sha256_file(job['config']) != job['config_sha256'] or sha256_file(job['checkpoint']) != job['checkpoint_sha256']:
            raise ValueError('frozen config/checkpoint checksum mismatch')
        config = json.loads(Path(job['config']).read_text())
        if (config['training']['seed'] != job['seed'] or config['train_loader']['seed'] != job['seed']
                or config['train_loader']['negative_sampling']['seed'] != job['seed']
                or config['seal']['graph_csr_use_per_vertex_sampling'] != (job['variant'] == 'pv')
                or job['selection']['selection_split'] != 'validation'
                or job['selection']['metric'] != 'ogb_mrr'
                or job['training'].get('test_scored') is not False
                or job['training']['steps'] != plan['protocol']['max_steps']
                or config['training']['max_steps'] != plan['protocol']['max_steps']):
            raise ValueError('frozen jobs are not the same complete validation-only training protocol')
        key = json.dumps(normalize_config(config), sort_keys=True)
        if normalized is not None and key != normalized:
            raise ValueError('six checkpoint jobs do not share one normalized protocol')
        normalized = key
        if job['seed'] in initial and job['initial_model_sha256'] != initial[job['seed']]:
            raise ValueError('paired initialization hashes differ')
        initial[job['seed']] = job['initial_model_sha256']
    for subset in plan['subsets'].values():
        for item in subset['outputs'].values():
            if sha256_file(item['path']) != item['sha256']:
                raise ValueError('frozen quality query subset changed')
    return plan, {'frozen_manifest': str(frozen_path), 'frozen_manifest_sha256': sha256_file(frozen_path),
                  'plan_path': str(plan_path), 'plan_sha256': sha256_file(plan_path), 'verified_checkpoints': 6,
                  'same_normalized_protocol': True, 'selection_uses_test': False}


def resolve_query_ids(total, requested_path=None, *, plan=None, split='test'):
    if requested_path is None:
        return np.arange(total, dtype=np.int64), {'selection': 'all_original_query_rows', 'matches_quality_subset': False}
    if plan is None:
        raise ValueError('--query-ids requires --quality-root/--quality-plan or a verified frozen manifest')
    key = 'val' if split == 'valid' else 'test'
    expected = plan['subsets'][key]['outputs']['query_ids']
    if sha256_file(requested_path) != expected['sha256']:
        raise ValueError('requested query IDs differ from EXACT frozen quality row-ID file')
    ids = np.load(requested_path, mmap_mode='r', allow_pickle=False)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or ids.size == 0 or np.unique(ids).size != ids.size:
        raise ValueError('query IDs must be a nonempty unique integer vector')
    if int(ids.min()) < 0 or int(ids.max()) >= total:
        raise ValueError('query ID outside original query rows')
    return np.asarray(ids, dtype=np.int64), {'selection': 'exact_frozen_quality_query_rows',
             'path': str(Path(requested_path).resolve()), 'sha256': sha256_file(requested_path), 'matches_quality_subset': True}


def load_inputs(prepared, split, *, plan=None):
    prepared = Path(prepared).resolve()
    manifest_path = prepared / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads((prepared / 'PREPARED_COMPLETE.json').read_text())
    if sha256_file(manifest_path) != completion['manifest_sha256'] or manifest.get('dataset') != 'friendster':
        raise ValueError('not a completed verified Friendster preparation')
    if plan is not None:
        expected = plan['dataset_provenance']
        if sha256_file(manifest_path) != expected['sha256'] or prepared / 'train_csr' != Path(plan['graph_root']).resolve():
            raise ValueError('heuristic graph/preparation differs from frozen SEAL quality plan')
        for name, expected_stat in plan['graph_input_stats'].items():
            file = Path(name)
            if file.stat().st_size != expected_stat['bytes'] or file.stat().st_mtime_ns != expected_stat['mtime_ns']:
                raise ValueError('frozen training graph/pairs metadata changed')
    graph = CSR(prepared / 'train_csr')
    if graph.n != manifest['num_nodes'] or graph.m != manifest['train_stored_entries']:
        raise ValueError('graph dimensions differ from prepared manifest')
    for name in ('edge_starts.bin', 'edge_ends.bin'):
        item = manifest['files']['train_csr/' + name]
        if (graph.root / name).stat().st_size != item['bytes']:
            raise ValueError('graph file size differs from immutable preparation manifest')
    folder = prepared / 'evaluation' / split
    arrays, input_files = {}, {}
    for name in ('source_node', 'target_node', 'target_node_neg', 'warm_query'):
        path = folder / (name + '.npy')
        expected = manifest['files'][f'evaluation/{split}/{name}.npy']
        digest = sha256_file(path)
        if digest != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError('provided original queries changed after preparation')
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        input_files[name] = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
    q = arrays['source_node'].size
    k = arrays['target_node_neg'].shape[1] if arrays['target_node_neg'].ndim == 2 else 0
    if (arrays['source_node'].shape != (q,) or arrays['target_node'].shape != (q,) or arrays['warm_query'].shape != (q,)
            or arrays['target_node_neg'].shape != (q, k) or q < 1 or k < 1
            or any(a.dtype.kind not in 'iu' for name, a in arrays.items() if name != 'warm_query')
            or arrays['warm_query'].dtype.kind != 'b'):
        raise ValueError('invalid query-row input shapes/dtypes')
    return graph, arrays, manifest, {'prepared_manifest_sha256': sha256_file(manifest_path), 'query_files': input_files,
              'graph': {'path': str(graph.root), 'nodes': graph.n, 'stored_entries': graph.m,
                        'logical_csr_bytes': graph.logical_bytes, 'offset_dtype': str(graph.starts.dtype),
                        'index_dtype': str(graph.col.dtype), 'mapping': 'read-only mmap; no global degree array',
                        'identity_scope': 'sealed preparation manifest plus frozen size/mtime; large payloads not rehashed here'}}


def enforce_memory_budget(graph, arrays, queries, chunk_queries, memory_budget_gib):
    # The logical graph is an upper bound on graph-mmap resident pages. Query
    # scratch is per worker/query; no Q*K*methods score tensor is retained.
    estimate = graph.logical_bytes + sum(a.nbytes for a in arrays.values()) + queries * 160 + chunk_queries * 65536 + 2 * GIB
    if estimate > memory_budget_gib * GIB:
        raise MemoryError(f'conservative logical/runtime footprint {estimate} exceeds memory budget')
    return {'conservative_bytes': estimate, 'budget_bytes': int(memory_budget_gib * GIB),
            'graph_mmap_included_at_full_logical_size': True, 'runtime_reserve_bytes': 2 * GIB}


def metrics(ranks, warm):
    reciprocal = 1.0 / ranks
    result = {}
    for column, name in enumerate(METHODS):
        row = {'mrr': float(reciprocal[:, column].mean()), 'hits_at_10': float(np.mean(ranks[:, column] <= 10))}
        for label, mask in (('warm', warm), ('cold', ~warm)):
            row[label + '_queries'] = int(mask.sum())
            row[label + '_mrr'] = float(reciprocal[mask, column].mean()) if np.any(mask) else None
        result[name] = row
    return result


def run(prepared, split, out, *, query_ids_path=None, quality_plan_path=None, frozen_path=None,
        threads=8, max_operations=20_000_000_000, memory_budget_gib=64, chunk_queries=64):
    call_started = time.perf_counter()
    if split not in ('valid', 'test'):
        raise ValueError('split must be valid or test')
    plan, gate = None, None
    if split == 'test':
        plan, gate = verify_test_gate(frozen_path, quality_plan_path)
    elif quality_plan_path is not None:
        plan = load_quality_plan(quality_plan_path)
    # Crucially, no test query load, score kernel or output artifact precedes gate.
    graph, arrays, dataset_manifest, provenance = load_inputs(prepared, split, plan=plan)
    ids, selection = resolve_query_ids(len(arrays['source_node']), query_ids_path, plan=plan, split=split)
    if threads < 1 or threads > nb.config.NUMBA_NUM_THREADS or chunk_queries < 1 or max_operations < 0 or memory_budget_gib <= 0:
        raise ValueError('invalid thread, chunk, operation or memory budget')
    nb.set_num_threads(threads)
    memory_guard = enforce_memory_budget(graph, arrays, len(ids), chunk_queries, memory_budget_gib)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'no overwrite or silent resume: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + '.partial-' + uuid.uuid4().hex[:8]); temp.mkdir()
    started = time.perf_counter()
    state = {'status': 'preflight', 'split': split, 'methods': list(METHODS), 'gate': gate,
             'selection': selection, 'queries': len(ids), 'negatives_per_query': arrays['target_node_neg'].shape[1],
             'max_operations': max_operations, 'memory_guard': memory_guard, 'provenance': provenance,
             'processed_queries': 0, 'timings': {'gate_and_input_validation_seconds': started - call_started},
             'partial_directory': str(temp),
             'comparison_scope': 'simple degree/structural controls; no training/HPO and no modern SOTA claim'}
    atomic_json(temp / 'status.json', state)
    try:
        t0 = time.perf_counter()
        bounds, invalid = estimate_query_operations(graph.starts, graph.m, arrays['source_node'], arrays['target_node'],
                                                    arrays['target_node_neg'], ids)
        state['timings']['operation_preflight_seconds'] = time.perf_counter() - t0
        if np.any(invalid):
            raise ValueError('invalid candidate endpoint, self pair or CSR row bounds')
        total_bound = sum(int(x) for x in bounds)
        state['estimated_search_steps_upper_bound'] = total_bound
        if max_operations and total_bound > max_operations:
            raise OperationBudgetExceeded(f'{total_bound} bounded adjacency search steps exceed --max-operations {max_operations}; no query scored')
        ranks = np.lib.format.open_memmap(temp / 'midpoint_ranks.npy', mode='w+', dtype=np.float64, shape=(len(ids), len(METHODS)))
        ranks[:] = np.nan
        actual_operations = 0
        t0 = time.perf_counter()
        for lo in range(0, len(ids), chunk_queries):
            hi = min(len(ids), lo + chunk_queries)
            block, count, invalid = score_query_rows(graph.starts, graph.col, arrays['source_node'], arrays['target_node'],
                                                    arrays['target_node_neg'], ids[lo:hi])
            if np.any(invalid) or np.any(count > bounds[lo:hi]):
                raise ValueError('simple symmetric graph contract or proven operation bound violated')
            ranks[lo:hi] = block
            actual_operations += sum(int(x) for x in count)
            state.update(status='scoring', processed_queries=hi, actual_search_steps=actual_operations, memory=rss_status())
            if state['memory']['current_rss_bytes'] is not None and state['memory']['current_rss_bytes'] > memory_guard['budget_bytes']:
                raise MemoryError('sampled process RSS exceeded the memory budget')
            atomic_json(temp / 'status.json', state)
            LOG.info('SCORED %s/%s query rows; search steps=%s', hi, len(ids), actual_operations)
        state['timings']['scoring_seconds_including_first_kernel_compilation'] = time.perf_counter() - t0
        ranks.flush()
        if not np.all(np.isfinite(ranks)) or np.any(ranks < 1):
            raise ValueError('incomplete/nonfinite rank results')
        rr = 1.0 / np.asarray(ranks)
        np.save(temp / 'reciprocal_ranks.npy', rr, allow_pickle=False)
        np.save(temp / 'query_ids.npy', ids, allow_pickle=False)
        source, target = np.asarray(arrays['source_node'][ids]), np.asarray(arrays['target_node'][ids])
        np.save(temp / 'source_node.npy', source, allow_pickle=False)
        np.save(temp / 'target_node.npy', target, allow_pickle=False)
        warm = np.asarray(arrays['warm_query'][ids], dtype=bool)
        # Check the archived warm definition against this exact training graph.
        computed_warm = np.array([_degree(graph.starts, graph.m, int(u)) > 0 and _degree(graph.starts, graph.m, int(v)) > 0
                                  for u, v in zip(source, target)], dtype=bool)
        if not np.array_equal(warm, computed_warm):
            raise ValueError('archived warm mask differs from exact training graph')
        np.save(temp / 'warm_query.npy', warm, allow_pickle=False)
        state.update(status='complete', metrics=metrics(np.asarray(ranks), warm), memory=rss_status(),
                     total_seconds=time.perf_counter() - call_started,
                     source_rows=len(source), unique_sources=int(np.unique(source).size),
                     tie_policy='rank=1+#negative_scores>positive+0.5*#negative_scores==positive; exact float64 comparison, no rounding',
                     operation_definition='one merge iteration, one binary-search iteration, or final binary membership check; excludes row-degree lookup, math/log and bookkeeping',
                     code_sha256=sha256_file(Path(__file__)), python=sys.version, numpy=np.__version__, numba=nb.__version__,
                     numba_threads=nb.get_num_threads(), torch_imported='torch' in sys.modules,
                     methods_definitions=dict(zip(METHODS, ('degree(dst)', 'degree(src)*degree(dst)',
                         '|N(src) intersection N(dst)|', 'sum 1/log(degree(common))', 'sum 1/degree(common)'))))
        # Final identity checks catch changes during long CPU evaluation.
        if gate is not None:
            _, final_gate = verify_test_gate(frozen_path, quality_plan_path)
            if final_gate != gate:
                raise ValueError('joint checkpoint freeze/config/checkpoints changed during scoring')
        if sha256_file(Path(prepared) / 'manifest.json') != provenance['prepared_manifest_sha256']:
            raise ValueError('prepared manifest changed during scoring')
        for item in provenance['query_files'].values():
            if sha256_file(item['path']) != item['sha256']:
                raise ValueError('original query arrays changed during scoring')
        if plan is not None:
            for name, expected_stat in plan['graph_input_stats'].items():
                file = Path(name)
                if file.stat().st_size != expected_stat['bytes'] or file.stat().st_mtime_ns != expected_stat['mtime_ns']:
                    raise ValueError('frozen topology metadata changed during scoring')
        state.pop('partial_directory', None)
        state['output_directory'] = str(out)
        state['outputs'] = {path.name: {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
                            for path in temp.iterdir() if path.is_file() and path.name != 'status.json'}
        atomic_json(temp / 'status.json', state)
        atomic_json(temp / 'summary.json', state)
        atomic_json(temp / 'COMPLETE.json', {'summary_sha256': sha256_file(temp / 'summary.json'), 'completed_unix': time.time()})
        temp.rename(out)
        return state
    except BaseException as error:
        state.update(status='failed', error=type(error).__name__ + ': ' + str(error), total_seconds=time.perf_counter() - call_started)
        atomic_json(temp / 'status.json', state)
        atomic_json(temp / 'FAILED.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--quality-root', type=Path, help='directory containing frozen quality plan.json')
    parser.add_argument('--quality-plan', type=Path)
    parser.add_argument('--split', choices=('valid', 'test'), default='test')
    parser.add_argument('--query-ids', type=Path)
    parser.add_argument('--frozen-checkpoints-manifest', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-operations', type=int, default=20_000_000_000, help='upper-bounded search steps;0 means unlimited, never silently drops queries')
    parser.add_argument('--memory-budget-gib', type=float, default=64)
    parser.add_argument('--chunk-queries', type=int, default=64)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    plan_path = args.quality_plan or (args.quality_root / 'plan.json' if args.quality_root else None)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    report = run(args.prepared_root, args.split, args.out, query_ids_path=args.query_ids, quality_plan_path=plan_path,
                 frozen_path=args.frozen_checkpoints_manifest, threads=args.threads, max_operations=args.max_operations,
                 memory_budget_gib=args.memory_budget_gib, chunk_queries=args.chunk_queries)
    print(json.dumps({'out': str(args.out.resolve()), 'status': report['status'], 'queries': report['queries'],
                      'methods': report['methods'], 'torch_imported': report['torch_imported']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
