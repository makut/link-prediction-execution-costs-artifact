#!/usr/bin/env python3
"""Replay archived frozen node sets; isolate Python-harness vs production routing.

No sampling, stock SciPy N-sized workspace, Torch, remote calls or queue control.
All adjacency constructors come from --repo's frozen exact_benchmark helper.
The NumPy route matches production's routing algebra, not the full SEAL pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')
for _key in THREAD_KEYS:
    os.environ[_key] = '1'

VERSION = 'frozen_s_dispatch_v1'
BASE_METHODS = ('fixed_pairwise', 'fixed_scan', 'exact_python_auto', 'production_vectorized_auto')
LOWER_BOUND = 'precomputed_route_lower_bound'
MAX_INPUT_BYTES = 32 << 20
SOURCE_PATHS = ('scripts/seal_revision/exact_benchmark.py', 'seal/graph_csr_kernels.py',
                'seal/graph_csr_extraction.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_exact(repo):
    """Import only the requested frozen repo, rejecting an incompatible cache."""
    repo = Path(repo).resolve()
    source = repo / SOURCE_PATHS[0]
    require(source.is_file(), 'missing frozen exact_benchmark.py in --repo')
    existing = sys.modules.get('seal.graph_csr_kernels')
    if existing is not None:
        require(Path(existing.__file__).resolve() == repo / SOURCE_PATHS[1],
                'different repository kernels already imported; use a fresh process')
    sys.path.insert(0, str(repo))
    name = '_frozen_dispatch_exact_' + digest_file(source)[:16]
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # required by dataclasses during execution
    spec.loader.exec_module(module)
    require(Path(module.kernels.__file__).resolve() == repo / SOURCE_PATHS[1], 'kernel import escaped --repo')
    return module


def vectorized_degree_sum(graph, nodes):
    """Production expression, including eager np.where-safe terminal indexing."""
    import numpy as np
    starts = graph.starts
    n_starts = int(starts.size)
    row_start = starts[nodes]
    row_end = np.where(nodes + 1 < n_starts,
                       starts[np.minimum(nodes + 1, n_starts - 1)], int(graph.m))
    return int((row_end - row_start).sum())


def route_from_degree(count, degree_sum, ratio=1.0, directed=False):
    pairs = count * (count - 1) // 2
    pair_cost = 2 * pairs if directed else pairs
    return pair_cost <= float(ratio) * max(degree_sum, 1)


def python_route(graph, nodes, ratio=1.0):
    return route_from_degree(int(nodes.size), graph.degree_sum(nodes), ratio)


def vectorized_route(graph, nodes, ratio=1.0):
    return route_from_degree(int(nodes.size), vectorized_degree_sum(graph, nodes), ratio)


def balanced_cases(records, limit, seed):
    """Optional bounded subset, balanced over sampler/cap/workload cells."""
    if limit == 0 or limit >= len(records):
        return list(enumerate(records))
    cells = defaultdict(list)
    for index, record in enumerate(records):
        cells[(record['sampler'], record['cap'], record['workload'])].append((index, record))
    rng = random.Random(seed)
    for cell in cells.values():
        rng.shuffle(cell)
    selected = []
    keys = sorted(cells)
    while len(selected) < limit:
        added = False
        for key in keys:
            if cells[key] and len(selected) < limit:
                selected.append(cells[key].pop())
                added = True
        if not added:
            break
    return selected


def checked_nodes(exact, graph, record, args):
    import numpy as np
    raw = record['nodes']
    require(isinstance(raw, list) and all(type(n) is int for n in raw), 'selected nodes must be JSON integers')
    nodes = np.asarray(raw, dtype=np.int64)
    require(nodes.ndim == 1 and 2 <= nodes.size <= args.max_subgraph_nodes, 'selected S size guard')
    require(len(set(raw)) == len(raw), 'duplicate node in frozen S')
    require(int(nodes.min()) >= 0 and int(nodes.max()) < graph.n, 'selected node out of graph bounds')
    require(raw[:2] == record['pair'] and record['num_nodes'] == nodes.size, 'frozen pair/S identity mismatch')
    require(exact.digest(nodes) == record['node_set_sha256'], 'frozen S hash mismatch')
    degree = vectorized_degree_sum(graph, nodes)
    require(degree == record['sum_out_degrees'], 'graph degrees differ from archived frozen S')
    require(degree <= args.max_neighbors, 'selected degree sum exceeds --max-neighbors')
    estimate = degree * 256 + nodes.size * 256
    require(estimate <= args.max_working_set_mib * (1 << 20), 'conservative local workspace guard')
    pairs = int(nodes.size * (nodes.size - 1) // 2)
    require(pairs <= args.max_pairwise_checks, 'pairwise work guard')
    for node in nodes:
        graph.validate_row(int(node))
    return nodes, degree, estimate


def validate_result(exact, actual, record):
    require(actual[0, 1] == 0 and actual[1, 0] == 0, 'target edge survived removal')
    require(exact.digest(exact.canonical_edges(actual)) == record['canonical_edges_sha256'], 'canonical edge hash differs from archive')
    require(exact.digest(exact.drnl_labels(actual)) == record['drnl_sha256'], 'DRNL hash differs from archive')


def constructors(exact, graph, nodes, ratio, include_lower_bound):
    def fixed(pairwise):
        return exact.production_induced(graph, nodes, 'pairwise' if pairwise else 'scan')
    def numpy_auto():
        return fixed(vectorized_route(graph, nodes, ratio))
    methods = {'fixed_pairwise': lambda: fixed(True), 'fixed_scan': lambda: fixed(False),
               'exact_python_auto': lambda: exact.production_induced(graph, nodes, 'auto', auto_ratio=ratio),
               'production_vectorized_auto': numpy_auto}
    routes = {'exact_python_auto': lambda: python_route(graph, nodes, ratio),
              'production_vectorized_auto': lambda: vectorized_route(graph, nodes, ratio)}
    if include_lower_bound:
        chosen = vectorized_route(graph, nodes, ratio)
        methods[LOWER_BOUND] = lambda: fixed(chosen)
        routes[LOWER_BOUND] = lambda: chosen
    return methods, routes


def measure_case(exact, graph, record, args, input_index):
    nodes, degree, estimate = checked_nodes(exact, graph, record, args)
    chosen = vectorized_route(graph, nodes, args.auto_ratio)
    require(chosen == python_route(graph, nodes, args.auto_ratio), 'Python/vectorized route mismatch')
    if args.auto_ratio == 1.0:
        require(record['auto_selected'] == ('pairwise' if chosen else 'scan'), 'archived route mismatch')
    methods, routes = constructors(exact, graph, nodes, args.auto_ratio, args.include_precomputed_lower_bound)
    times = {name: [] for name in methods}
    route_times = {name: [] for name in routes}
    order_log, route_order_log = [], []
    rng = random.Random(args.seed + input_index * 1000003)
    for repeat in range(args.warmup + args.repeats):
        order = list(methods)
        rng.shuffle(order)
        order_log.append(order)
        for name in order:
            before = time.perf_counter_ns()
            actual = methods[name]()
            after = time.perf_counter_ns()
            # All results, including warmups, checked against archived hashes.
            # Hashing, DRNL and row validation are outside the edge timer.
            validate_result(exact, actual, record)
            del actual
            if repeat >= args.warmup:
                times[name].append((after - before) / 1e9)
        route_order = list(routes)
        rng.shuffle(route_order)
        route_order_log.append(route_order)
        for name in route_order:
            before = time.perf_counter_ns()
            selected = routes[name]()
            after = time.perf_counter_ns()
            require(selected == chosen, 'timed route changed decision')
            if repeat >= args.warmup:
                route_times[name].append((after - before) / 1e9)
    return {'input_index': input_index, 'pair_index': record['pair_index'], 'pair': record['pair'],
            'sampler': record['sampler'], 'cap': record['cap'], 'workload': record['workload'],
            'num_nodes': int(nodes.size), 'degree_sum': degree,
            'degree_sum_over_s_squared': degree / int(nodes.size)**2,
            'conservative_workspace_estimate_bytes': int(estimate),
            'selected_route': 'pairwise' if chosen else 'scan',
            'node_set_sha256': record['node_set_sha256'], 'canonical_edges_sha256': record['canonical_edges_sha256'],
            'drnl_sha256': record['drnl_sha256'], 'all_results_match_archived_edges_and_drnl': True,
            'methods': {name: {'edge_seconds': values, 'edge_summary': exact.stats(values)} for name, values in times.items()},
            'route_only': {name: {'seconds': values, 'summary': exact.stats(values)} for name, values in route_times.items()},
            'method_order_by_repeat_including_warmup': order_log,
            'route_order_by_repeat_including_warmup': route_order_log}


def summarize(records):
    if not records:
        return {}
    methods = list(records[0]['methods'])
    sums = {name: sum(statistics.median(r['methods'][name]['edge_seconds']) for r in records) for name in methods}
    route_sums = {name: sum(statistics.median(r['route_only'][name]['seconds']) for r in records)
                  for name in records[0]['route_only']}
    return {'records': len(records), 'auto_route_counts': dict(Counter(r['selected_route'] for r in records)),
            'edge_sum_of_per_record_medians_seconds': sums, 'route_only_sum_of_per_record_medians_seconds': route_sums,
            'python_auto_over_vectorized_auto_sum_ratio': sums['exact_python_auto'] / sums['production_vectorized_auto'],
            'hypothetical_lower_bound_included': LOWER_BOUND in methods,
            'aggregation': 'Sum of per-record medians, not a timed concatenated run; same fixed S and randomized method order.'}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--input-json', type=Path, required=True, help='Completed exact JSON containing archived nodes and fingerprints')
    p.add_argument('--graph-path', type=Path, required=True, help='Read-only existing GraphCSR directory; no downloads or conversion')
    p.add_argument('--out', type=Path, required=True, help='New directory; any existing path is refused')
    p.add_argument('--limit', type=int, default=0, help='0=all768; otherwise balanced sampler/cap/workload subset')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--seed', type=int, default=20261124)
    p.add_argument('--auto-ratio', type=float, default=1.0)
    p.add_argument('--max-neighbors', type=int, default=20_000_000)
    p.add_argument('--max-working-set-mib', type=float, default=4096)
    p.add_argument('--max-pairwise-checks', type=int, default=20_000_000)
    p.add_argument('--max-subgraph-nodes', type=int, default=10000)
    p.add_argument('--include-precomputed-lower-bound', action='store_true', help='Hypothetical: route decision supplied outside timer')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(not args.out.exists(), '--out exists; no overwrite or automatic resume')
    require(args.warmup >= 1 and args.repeats >= 1 and args.limit >= 0, 'invalid repetition/limit budget')
    require(math.isfinite(args.auto_ratio) and args.auto_ratio > 0, 'invalid auto ratio')
    require(math.isfinite(args.max_working_set_mib) and args.max_working_set_mib > 0
            and args.max_neighbors > 0 and args.max_pairwise_checks > 0 and args.max_subgraph_nodes >= 2, 'invalid workspace/work guards')
    args.repo = args.repo.resolve()
    args.input_json = args.input_json.resolve()
    args.graph_path = args.graph_path.resolve()
    require(args.input_json.is_file() and args.input_json.stat().st_size <= MAX_INPUT_BYTES, 'raw JSON missing or over32MiB')
    raw = json.loads(args.input_json.read_text())
    require(raw['schema_version'] == 1 and 1 <= len(raw['records']) <= 768, 'unknown, empty or over768 exact case schema')
    require(raw['configuration']['directed'] is False, 'v1 replay is restricted to undirected Friendster')
    require(raw['configuration']['file_endian'] == sys.byteorder, 'requires native-endian frozen graph')
    for path in SOURCE_PATHS:
        require(digest_file(args.repo / path) == raw['provenance']['source_sha256'][path], 'frozen repo source mismatch: ' + path)
    torch_before = 'torch' in sys.modules
    exact = load_exact(args.repo)
    require(exact.kernels.NUMBA_AVAILABLE, 'Numba is required; refusing an uncompiled speed run')
    import numba
    from threadpoolctl import threadpool_limits, threadpool_info
    numba.set_num_threads(1)
    require(args.graph_path.is_dir(), 'requires an existing read-only mmap GraphCSR directory')
    graph = exact.load_graph(args.graph_path, raw['configuration']['file_endian'])
    require(graph.n == raw['provenance']['graph']['nodes'] and graph.m == raw['provenance']['graph']['stored_entries'], 'graph shape/count mismatch')
    import numpy as np
    require(isinstance(graph.starts, np.memmap) and isinstance(graph.indices, np.memmap)
            and not graph.starts.flags.writeable and not graph.indices.flags.writeable, 'graph is not read-only mmap')
    selected = balanced_cases(raw['records'], args.limit, args.seed)
    identity = [(r['pair_index'], r['sampler'], r['cap'], r['workload']) for _,r in selected]
    require(len(identity) == len(set(identity)), 'duplicate frozen case identity')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'schema_version': 1, 'implementation_version': VERSION, 'status': 'running',
        'scope': 'Frozen-S CSR-construction microbenchmark. Production-matched means NumPy routing algebra only; constructors are the same frozen exact helper. Not full SEAL or a GPU pipeline.',
        'interpretation': 'The old18% auto gap belongs to the Python exact harness; actual production is already vectorized. Precomputed route, if present, is a hypothetical lower bound.',
        'input': {'path': str(args.input_json), 'bytes': args.input_json.stat().st_size, 'sha256': digest_file(args.input_json)},
        'source_sha256': {path: digest_file(args.repo/path) for path in SOURCE_PATHS},
        'benchmark_sha256': digest_file(Path(__file__)), 'repo': str(args.repo),
        'graph': {'path': str(args.graph_path), 'nodes': graph.n, 'stored_entries': graph.m,
                  'read_only_mmap': True, 'content_identity': 'Shape/count, selected degree sums and every archived S/edge/DRNL fingerprint; no full-graph hash.'},
        'configuration': {key: value for key,value in vars(args).items() if not isinstance(value, Path)},
        'selected_input_indices': [index for index,_ in selected],
        'sampling': 'No resampling. Optional limit balances pre-existing sampler/cap/workload cells; same archived nodes and order.',
        'timing': 'CSR construction only; hash, DRNL, touched-row validation, degree guard and route-only timers are separate. Randomized method order for every warmup/measured repeat; JIT excluded by at least one warmup.',
        'thread_environment': {key: os.environ[key] for key in THREAD_KEYS},
        'numba_threads': numba.get_num_threads(), 'records': []}
    try:
        with threadpool_limits(limits=1):
            report['threadpools'] = threadpool_info()
            with (args.out/'records.jsonl').open('x') as stream:
                for completed, (index, record) in enumerate(selected, 1):
                    result = measure_case(exact, graph, record, args, index)
                    report['records'].append(result)
                    stream.write(json.dumps(result, allow_nan=False)+'\n')
                    stream.flush()
                    if completed == 1 or completed % 16 == 0 or completed == len(selected):
                        write_json(args.out/'status.json', {'status':'running','completed':completed,'expected':len(selected),'elapsed_seconds':time.monotonic()-started})
        require(('torch' in sys.modules) == torch_before, 'benchmark introduced a Torch import')
        require(digest_file(args.input_json) == report['input']['sha256'], 'archived input changed during replay')
        for path, digest in report['source_sha256'].items():
            require(digest_file(args.repo/path) == digest, 'frozen source changed during replay: ' + path)
        report.update(status='complete', summary=summarize(report['records']), elapsed_seconds=time.monotonic()-started,
                      torch_imported_before=torch_before, torch_imported_after='torch' in sys.modules)
        write_json(args.out/'report.json', report)
        write_json(args.out/'status.json', {'status':'complete','completed':len(selected),'expected':len(selected),'elapsed_seconds':report['elapsed_seconds']})
        # Publish success last, after every output and final source check.
        write_json(args.out/'COMPLETE.json', {'success':True,'records':len(selected),'report_sha256':digest_file(args.out/'report.json')})
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error), elapsed_seconds=time.monotonic()-started)
        write_json(args.out/'report.json', report)
        write_json(args.out/'status.json', {'status':'failed','completed':len(report['records']),'error':report['error']})
        raise
    # A closed caller stdout must not rewrite already-published success artifacts.
    print(json.dumps({'status':'complete','out':str(args.out.resolve()),'records':len(selected),'elapsed_seconds':report['elapsed_seconds']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
