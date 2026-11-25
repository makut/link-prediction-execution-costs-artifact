#!/usr/bin/env python3
"""Positive-query structural audit of before-cap versus retained-only visitation.

No training, model scores, latency benchmark, Torch, GPU, downloads or YT writes.
Per-node pseudorandom priorities couple shared candidate IDs, not array indices.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

if __name__ == '__main__':
    for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[_name] = '1'

import numpy as np

VERSION = 'coupled_node_priority_visitation_v1'
SOURCES = ('scripts/seal_revision/exact_benchmark.py', 'seal/graph_csr_kernels.py', 'seal/graph_csr_extraction.py')
CASES = ('h2_cap', 'h1_cap', 'h2_no_cap')
GIB = 2**30


class Guard(RuntimeError):
    pass


class Deadline(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    digest = hashlib.sha256()
    flat = np.asarray(array).reshape(-1)
    for at in range(0, flat.size, 131072):
        digest.update(np.asarray(flat[at:at + 131072], dtype='<i8').tobytes())
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    require(path.is_file() and path.stat().st_size <= 16 << 20, 'small JSON required: ' + str(path))
    value = json.loads(path.read_text())
    require(isinstance(value, dict), 'JSON object required')
    return value


def load_exact(repo):
    repo = repo.resolve()
    existing = sys.modules.get('seal.graph_csr_kernels')
    require(existing is None or Path(existing.__file__).resolve() == repo / SOURCES[1], 'different repo kernels cached; use fresh process')
    sys.path.insert(0, str(repo))
    path = repo / SOURCES[0]
    name = '_semantic_frozen_exact_' + sha(path)[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    require(Path(module.kernels.__file__).resolve() == repo / SOURCES[1], 'kernel import escaped --repo')
    require(module.kernels.NUMBA_AVAILABLE, 'Numba required for bounded CPU audit')
    return module


def priorities(nodes, seed, query_id, hop):
    """SplitMix64 bijection of nodeID xor a SHA-derived query/hop salt.

    Distinct IDs have distinct keys. This is a PRF-style diagnostic permutation,
    not a proof of perfectly uniform subsets or upstream RNG realizations.
    """
    salt = int.from_bytes(hashlib.sha256(f'{seed}:{query_id}:{hop}'.encode()).digest()[:8], 'little')
    with np.errstate(over='ignore'):
        values = np.asarray(nodes, dtype=np.uint64) ^ np.uint64(salt)
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


def choose(nodes, cap, seed, query_id, hop):
    if cap is None or len(nodes) <= cap:
        return np.asarray(nodes, dtype=np.int64).copy()
    keys = priorities(nodes, seed, query_id, hop)
    selected = np.argpartition(keys, cap - 1)[:cap]
    return np.sort(np.asarray(nodes, dtype=np.int64)[selected])


class Expansions:
    def __init__(self, exact, graph, *, max_neighbors, max_nodes, working_bytes, deadline):
        self.exact, self.graph = exact, graph
        self.max_neighbors, self.max_nodes, self.working_bytes = max_neighbors, max_nodes, working_bytes
        self.deadline, self.cache, self.validated = deadline, {}, set()

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise Deadline('audit wall-clock budget reached')

    def expand(self, frontier):
        self.check_time()
        if len(frontier) > self.max_nodes:
            raise Guard('frontier exceeds max-nodes before expansion')
        key = tuple(int(n) for n in frontier)
        if key in self.cache:
            return self.cache[key]
        degree = self.graph.degree_sum(frontier)
        cached = sum(a.nbytes for a in self.cache.values())
        if degree > self.max_neighbors:
            raise Guard(f'expansion adjacency entries {degree} exceed max-neighbors {self.max_neighbors}')
        if degree * 256 + len(frontier) * 256 + cached + (16 << 20) > self.working_bytes:
            raise Guard('conservative local working-set allowance exceeded before expansion')
        for node in frontier:
            if int(node) not in self.validated:
                self.graph.validate_row(int(node)); self.validated.add(int(node))
        values = self.exact.kernels._fill_neighbors_numba(self.graph.starts, self.graph.indices, self.graph.m,
                                                        np.asarray(frontier, dtype=np.int64))
        result = np.unique(values).astype(np.int64, copy=False)
        self.check_time()
        self.cache[key] = result
        return result


def coupled_nodes(expansions, pair, query_id, seed, hops, cap):
    """Exactly two visitation policies; both use common per-node hop priorities."""
    initial = np.asarray(pair, dtype=np.int64)
    chunks = [[initial], [initial]]
    visited = [np.sort(initial), np.sort(initial)]
    frontiers = [initial, initial]
    first_candidates = first_selected = None
    details = {}
    for hop in range(1, hops + 1):
        for policy in range(2):
            expansions.check_time()
            values = expansions.expand(frontiers[policy])
            candidates = np.setdiff1d(values, visited[policy], assume_unique=True)
            retained = choose(candidates, cap, seed, query_id, hop)
            if hop == 1:
                if policy == 0:
                    first_candidates, first_selected = candidates, retained
                    details.update(hop1_candidates=len(candidates), hop1_retained=len(retained),
                                   hop1_cap_binding=cap is not None and len(candidates) > cap)
                else:
                    require(np.array_equal(candidates, first_candidates) and np.array_equal(retained, first_selected),
                            'common first-hop priority draw diverged')
                    details['hop1_draw_identical'] = True
            if hop == 2 and policy == 1:
                discarded = np.setdiff1d(first_candidates, first_selected, assume_unique=True)
                details['discarded_hop1_return_candidates'] = int(np.intersect1d(discarded, candidates, assume_unique=True).size)
                details['discarded_hop1_retained_at_hop2'] = int(np.intersect1d(discarded, retained, assume_unique=True).size)
            visited[policy] = np.union1d(visited[policy], candidates if policy == 0 else retained)
            if retained.size:
                chunks[policy].append(retained)
            frontiers[policy] = retained
    nodes = [np.concatenate(parts) for parts in chunks]
    require(all(np.unique(n).size == len(n) for n in nodes), 'duplicate node in sampled S')
    return nodes[0], nodes[1], details


def global_edges(adjacency, nodes):
    coo = adjacency.tocoo()
    source, target = nodes[coo.row], nodes[coo.col]
    keep = source < target
    edges = np.column_stack((source[keep], target[keep])).astype(np.int64, copy=False)
    if edges.size:
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    return np.ascontiguousarray(edges)


def edge_jaccard(left, right):
    dtype = np.dtype([('u', '<i8'), ('v', '<i8')])
    common = np.intersect1d(left.astype('<i8', copy=False).view(dtype).reshape(-1),
                            right.astype('<i8', copy=False).view(dtype).reshape(-1), assume_unique=True).size
    union = len(left) + len(right) - common
    return {'intersection': int(common), 'union': int(union), 'jaccard': float(common / union) if union else 1.0}


def compare_structures(exact, graph, left, right, expansions):
    expansions.check_time()
    maximum = max(len(left), len(right))
    if maximum > expansions.max_nodes:
        raise Guard('sampled S exceeds max-nodes before induction')
    if maximum * maximum * 64 + sum(a.nbytes for a in expansions.cache.values()) > expansions.working_bytes:
        raise Guard('local adjacency/DRNL workspace guard before induction')
    common, left_index, right_index = np.intersect1d(left, right, assume_unique=True, return_indices=True)
    union = len(left) + len(right) - len(common)
    matrices, edges, labels = [], [], []
    for nodes in (left, right):
        adjacency = exact.production_induced(graph, nodes, 'pairwise')
        require(not adjacency[0, 1] and not adjacency[1, 0] and not (adjacency != adjacency.T).nnz,
                'induced graph violates symmetric target-removed contract')
        matrices.append(adjacency)
        edges.append(global_edges(adjacency, nodes))
        labels.append(exact.drnl_labels(adjacency))
        expansions.check_time()
    mismatches = labels[0][left_index] != labels[1][right_index]
    edge = edge_jaccard(*edges)
    return {'global_node_intersection': len(common), 'global_node_union': union,
        'global_node_jaccard': float(len(common) / union) if union else 1.0,
        'global_undirected_edge_intersection': edge['intersection'], 'global_undirected_edge_union': edge['union'],
        'global_undirected_edge_jaccard': edge['jaccard'], 'reference_edges': len(edges[0]), 'retained_only_edges': len(edges[1]),
        'drnl_common_global_nodes': len(common), 'drnl_disagreements_aligned_global_ids': int(mismatches.sum()),
        'drnl_disagreement_fraction': float(mismatches.mean()) if len(common) else None,
        'reference_global_edges_sha256': array_sha(edges[0]), 'retained_only_global_edges_sha256': array_sha(edges[1])}


def audit_query(exact, graph, pair, query_id, args, deadline):
    expansions = Expansions(exact, graph, max_neighbors=args.max_neighbors, max_nodes=args.max_nodes,
                            working_bytes=args.max_working_set_mib * 2**20, deadline=deadline)
    output = {'query_id': int(query_id), 'source_node': int(pair[0]), 'target_node': int(pair[1]), 'cases': {}}
    for name, hops, cap in [('h2_cap', 2, args.cap), ('h1_cap', 1, args.cap), ('h2_no_cap', 2, None)]:
        try:
            left, right, detail = coupled_nodes(expansions, pair, query_id, args.seed, hops, cap)
            same = np.array_equal(np.sort(left), np.sort(right))
            item = {'status': 'complete', 'hops': hops, 'cap': cap, **detail,
                'reference_nodes': len(left), 'retained_only_nodes': len(right), 'global_node_identity': bool(same),
                'reference_node_set_sha256': array_sha(np.sort(left)), 'retained_only_node_set_sha256': array_sha(np.sort(right))}
            if name in ('h1_cap', 'h2_no_cap'):
                require(same, 'h1/no-cap node-identity control failed')
            if name == 'h2_no_cap' and max(len(left), len(right)) > args.max_control_induction_nodes:
                item.update(induction_status='guarded_large_control_S', structural_metrics=None)
            else:
                item.update(induction_status='complete', structural_metrics=compare_structures(exact, graph, left, right, expansions))
            output['cases'][name] = item
        except Guard as error:
            output['cases'][name] = {'status': 'guarded', 'reason': str(error), 'hops': hops, 'cap': cap}
    return output


def load_inputs(args):
    prepared, quality = args.prepared_root.resolve(), args.quality_root.resolve()
    manifest_path, plan_path = prepared / 'manifest.json', quality / 'plan.json'
    manifest, plan = read(manifest_path), read(plan_path)
    marker = prepared / 'PREPARED_COMPLETE.json'
    require(read(marker)['manifest_sha256'] == sha(manifest_path) == plan['dataset_provenance']['sha256'], 'prepared manifest identity changed')
    require(manifest['dataset'] == 'friendster' and manifest['variant'] == 'exact_colleague_80_10_10', 'expected prepared colleague Friendster train graph')
    require(Path(plan['graph_root']).resolve() == prepared / 'train_csr', 'quality graph root differs')
    subset = plan['subsets']['val']
    require(subset['query_count'] == args.expected_query_count == plan['protocol']['validation_query_count'], 'validation query count differs')
    files = {'plan': plan_path, 'prepared_manifest': manifest_path, 'prepared_complete': marker}
    arrays = {}
    for name in ('query_ids', 'positive'):
        item = subset['outputs'][name]
        path = quality / 'subsets' / ('valid_query_ids.npy' if name == 'query_ids' else 'valid_edge.npy')
        require(sha(path) == item['sha256'], 'frozen validation ' + name + ' hash changed')
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False); files[name] = path
    ids, positive = arrays['query_ids'], arrays['positive']
    require(ids.shape == (args.expected_query_count,) and positive.shape == (len(ids), 2)
        and ids.dtype.kind in 'iu' and positive.dtype.kind in 'iu' and np.unique(ids).size == len(ids), 'original query-row shapes/identities')
    originals = []
    for name in ('source_node', 'target_node'):
        path = prepared / 'evaluation/valid' / (name + '.npy')
        require(sha(path) == manifest['files']['evaluation/valid/' + name + '.npy']['sha256'], 'original validation endpoints changed')
        originals.append(np.load(path, mmap_mode='r', allow_pickle=False)); files[name] = path
    require(originals[0].ndim == originals[1].ndim == 1 and originals[0].shape == originals[1].shape
        and all(a.dtype.kind in 'iu' for a in originals), 'invalid original validation endpoint arrays')
    require(ids.min() >= 0 and ids.max() < len(originals[0])
        and np.array_equal(positive[:, 0], originals[0][ids]) and np.array_equal(positive[:, 1], originals[1][ids]), 'positive endpoints are not aligned with original query rows')
    source_hashes = {name: sha(args.repo / name) for name in SOURCES}
    for name in SOURCES[1:]:
        require(plan['code'][str((args.repo / name).resolve())] == source_hashes[name], 'frozen quality source hash differs: ' + name)
    exact = load_exact(args.repo)
    graph = exact.load_graph(prepared / 'train_csr', 'little')
    require(graph.n == manifest['num_nodes'] and graph.m == manifest['train_stored_entries'], 'prepared train dimensions differ')
    require(isinstance(graph.starts, np.memmap) and isinstance(graph.indices, np.memmap)
        and not graph.starts.flags.writeable and not graph.indices.flags.writeable, 'graph must be existing read-only mmap')
    require(positive.min() >= 0 and positive.max() < graph.n and np.all(positive[:, 0] != positive[:, 1]), 'invalid positive endpoints')
    stats = {}
    for name in ('edge_starts.bin', 'edge_ends.bin'):
        path = prepared / 'train_csr' / name
        value = {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
        require(value == plan['graph_input_stats'][str(path)], 'frozen graph metadata differs')
        stats[str(path)] = value
    return exact, graph, ids, positive, {str(path): sha(path) for path in files.values()}, source_hashes, stats


def aggregate(records):
    result = {}
    for name in CASES:
        rows = [row['cases'][name] for row in records]
        complete = [row for row in rows if row['status'] == 'complete']
        metrics = [row['structural_metrics'] for row in complete if row.get('structural_metrics') is not None]
        summary = {'provided_query_rows': len(rows), 'complete_queries': len(complete),
            'guarded_queries': sum(row['status'] == 'guarded' for row in rows),
            'deadline_unprocessed_queries': sum(row['status'] == 'not_processed_deadline' for row in rows),
            'failure_unprocessed_queries': sum(row['status'] == 'not_processed_failure' for row in rows),
            'induction_guarded_queries': sum(row.get('induction_status') == 'guarded_large_control_S' for row in complete),
            'node_identity_queries': sum(row['global_node_identity'] for row in complete), 'structural_metric_queries': len(metrics)}
        if complete:
            summary['hop1_cap_binding_fraction_among_complete'] = sum(row['hop1_cap_binding'] for row in complete) / len(complete)
            summary['discarded_hop1_retained_at_hop2_total'] = sum(row.get('discarded_hop1_retained_at_hop2', 0) for row in complete)
        for key in ('global_node_jaccard', 'global_undirected_edge_jaccard', 'drnl_disagreement_fraction'):
            values = [row[key] for row in metrics if row[key] is not None]
            summary[key] = {'n': len(values), 'mean': statistics.mean(values) if values else None,
                            'median': statistics.median(values) if values else None}
        result[name] = summary
    return result


def binding_summary(records):
    known = [row['cases']['h1_cap'] for row in records if row['cases']['h1_cap']['status'] == 'complete']
    count = sum(row['hop1_cap_binding'] for row in known)
    return {'measured_from': 'h1_cap control, avoiding conditioning on second-hop completion',
            'provided_queries': len(records), 'known_queries': len(known), 'binding_queries': count,
            'unknown_queries': len(records) - len(known),
            'percent_among_known': count / len(known) * 100 if known else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'prepared-root', 'quality-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-query-count', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20261125)
    parser.add_argument('--cap', type=int, default=200)
    parser.add_argument('--max-neighbors', type=int, default=2_000_000)
    parser.add_argument('--max-nodes', type=int, default=10000)
    parser.add_argument('--max-control-induction-nodes', type=int, default=1000)
    parser.add_argument('--max-working-set-mib', type=int, default=4096)
    parser.add_argument('--max-wall-seconds', type=float, default=3600)
    args = parser.parse_args(argv)
    require(0 < args.expected_query_count <= 512 and 0 < args.cap <= 200, 'bounded query/cap contract')
    require(0 < args.max_neighbors <= 20_000_000 and 2 <= args.max_nodes <= 10000
        and 2 <= args.max_control_induction_nodes <= args.max_nodes and 0 < args.max_working_set_mib <= 4096
        and 0 < args.max_wall_seconds <= 3600, 'bounded resource contract')
    args.repo = args.repo.resolve(); args.out = args.out.resolve()
    require(not args.out.exists(), 'output exists; no overwrite or resume')
    torch_before = 'torch' in sys.modules
    started = time.monotonic(); deadline = started + args.max_wall_seconds
    exact, graph, ids, positive, inputs, sources, graph_stats = load_inputs(args)
    import numba
    numba.set_num_threads(1)
    args.out.mkdir(parents=True)
    report = {'schema_version': 1, 'implementation_version': VERSION, 'status': 'running',
        'scope': 'Positive validation rows only; structural diagnostic, no MRR/training/latency/time-to-quality claim.',
        'coupling': 'Per original query/hop: SplitMix64(nodeID xor SHA256-derived salt), top-k shared priorities. PRF-style diagnostic subset law; not byte-identical upstream/production RNG realizations. Shared IDs keep their priorities when candidate pools differ.',
        'edge_unit': 'undirected global vertex-ID pairs; target removed; no comparison of local CSR indices',
        'drnl_unit': 'labels aligned by common global node IDs, including endpoints',
        'configuration': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'input_sha256': inputs, 'source_sha256': sources, 'graph_input_stats': graph_stats,
        'graph_identity_scope': 'sealed preparation + frozen size/mtime; no full payload rehash or N-sized scratch',
        'query_rows': len(ids), 'unique_sources': int(np.unique(positive[:, 0]).size),
        'records': [], 'torch_imported_before': torch_before, 'implementation_sha256': sha(Path(__file__))}
    previous_handler = None
    if hasattr(signal, 'setitimer'):
        previous_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Deadline('wall-clock alarm')))
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    try:
        for row, (qid, pair) in enumerate(zip(ids, positive)):
            report['records'].append(audit_query(exact, graph, pair, int(qid), args, deadline))
            if row % 16 == 0:
                write(args.out / 'status.json', {'status': 'running', 'processed_query_rows': len(report['records']), 'expected': len(ids)})
        for path, expected in inputs.items():
            require(sha(Path(path)) == expected, 'input changed during audit')
        for name, expected in sources.items():
            require(sha(args.repo / name) == expected, 'frozen source changed during audit')
        require(sha(Path(__file__)) == report['implementation_sha256'], 'audit implementation changed during audit')
        for path, expected in graph_stats.items():
            stat = Path(path).stat()
            require({'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns} == expected, 'graph metadata changed during audit')
        require(('torch' in sys.modules) == torch_before, 'audit introduced Torch')
        report['status'] = 'complete'
        report['input_identity_rechecked'] = True
    except Deadline:
        for qid, pair in zip(ids[len(report['records']):], positive[len(report['records']):]):
            report['records'].append({'query_id': int(qid), 'source_node': int(pair[0]), 'target_node': int(pair[1]),
                'cases': {name: {'status': 'not_processed_deadline'} for name in CASES}})
        report['status'] = 'deadline_exceeded'
        report['input_identity_rechecked'] = False
    except BaseException as error:
        for qid, pair in zip(ids[len(report['records']):], positive[len(report['records']):]):
            report['records'].append({'query_id': int(qid), 'source_node': int(pair[0]), 'target_node': int(pair[1]),
                'cases': {name: {'status': 'not_processed_failure'} for name in CASES}})
        report.update(status='failed', input_identity_rechecked=False,
                      error=type(error).__name__ + ': ' + str(error)[:240])
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous_handler)
    report['summary'] = aggregate(report['records'])
    report['hop1_cap_binding'] = binding_summary(report['records'])
    report['torch_imported_after'] = 'torch' in sys.modules
    report['wall_seconds_budget_accounting_only'] = time.monotonic() - started
    write(args.out / 'report.json', report)
    write(args.out / 'status.json', {'status': report['status'], 'all_query_rows_accounted_for': len(report['records']) == len(ids)})
    if report['status'] == 'complete':
        write(args.out / 'COMPLETE.json', {'complete': True, 'query_rows': len(ids), 'report_sha256': sha(args.out / 'report.json')})
    print(json.dumps({'status': report['status'], 'out': str(args.out), 'queries': len(ids), 'torch_imported': report['torch_imported_after']}))
    return 0 if report['status'] == 'complete' else 2 if report['status'] == 'deadline_exceeded' else 1


if __name__ == '__main__':
    raise SystemExit(main())
