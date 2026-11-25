from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.seal_revision import dispatch_benchmark_v1 as bench

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def exact():
    return bench.load_exact(REPO)


def arguments(**changes):
    values = dict(max_subgraph_nodes=10000, max_neighbors=20_000_000,
                  max_working_set_mib=4096, max_pairwise_checks=20_000_000,
                  auto_ratio=1.0, include_precomputed_lower_bound=True,
                  seed=42, warmup=1, repeats=2)
    values.update(changes)
    return SimpleNamespace(**values)


def archived_record(exact, graph, nodes):
    nodes = np.asarray(nodes, dtype=np.int64)
    oracle = exact.oracle_induced(graph, nodes)
    degree = graph.degree_sum(nodes)
    return {'nodes': nodes.tolist(), 'num_nodes': len(nodes), 'pair': nodes[:2].tolist(),
            'pair_index': 0, 'sampler': 'reference_union', 'cap': 50, 'workload': 'positive',
            'sum_out_degrees': degree, 'node_set_sha256': exact.digest(nodes),
            'canonical_edges_sha256': exact.digest(exact.canonical_edges(oracle)),
            'drnl_sha256': exact.digest(exact.drnl_labels(oracle)),
            'auto_selected': 'pairwise' if bench.route_from_degree(len(nodes), degree) else 'scan'}


def test_vectorized_sum_matches_python_unsorted_terminal_nonempty_row(exact):
    graph = exact.CSRGraph(np.array([0, 2, 3, 4], dtype=np.int64),
                           np.array([1, 3, 0, 3, 0, 2], dtype=np.int32))
    nodes = np.array([3, 1, 0], dtype=np.int64)
    assert bench.vectorized_degree_sum(graph, nodes) == graph.degree_sum(nodes) == 5
    assert bench.vectorized_route(graph, nodes) == bench.python_route(graph, nodes)


def test_vectorized_sum_preserves_int64_degrees_above_int32():
    graph = SimpleNamespace(starts=np.array([0, 2**31+7, 2**31+9], dtype=np.int64), m=2**31+14)
    assert bench.vectorized_degree_sum(graph, np.array([2, 0], dtype=np.int64)) == 2**31+12


def test_single_isolate_zero_degrees_and_directed_tie_boundary():
    graph = SimpleNamespace(starts=np.array([0], dtype=np.int64), m=0)
    assert bench.vectorized_degree_sum(graph, np.array([0], dtype=np.int64)) == 0
    assert bench.route_from_degree(1, 0)
    assert bench.route_from_degree(2, 0)  # max(sum_degree,1), <= boundary
    assert bench.route_from_degree(3, 3, directed=False)
    assert not bench.route_from_degree(3, 3, directed=True)
    assert bench.route_from_degree(3, 6, directed=True)
    assert bench.route_from_degree(3, 3, ratio=.999) is False


@pytest.mark.parametrize('corruption,match', [('duplicate', 'duplicate node'), ('hash', 'S hash mismatch'),
                                             ('degree', 'degrees differ')])
def test_frozen_selected_identity_and_guards_fail_closed(exact, corruption, match):
    graph = exact.fixture_graph()
    record = archived_record(exact, graph, [0, 1, 2, 3, 8])
    if corruption == 'duplicate':
        record['nodes'][-1] = 0
    elif corruption == 'hash':
        record['node_set_sha256'] = '0'*64
    else:
        record['sum_out_degrees'] += 1
    with pytest.raises(ValueError, match=match):
        bench.checked_nodes(exact, graph, record, arguments())


def test_all_routes_construct_same_archived_edges_and_drnl(exact):
    graph = exact.fixture_graph()
    record = archived_record(exact, graph, [0, 1, 2, 3, 4, 8])
    result = bench.measure_case(exact, graph, record, arguments(), 0)
    assert set(result['methods']) == set(bench.BASE_METHODS) | {bench.LOWER_BOUND}
    assert result['all_results_match_archived_edges_and_drnl'] is True
    assert result['canonical_edges_sha256'] == record['canonical_edges_sha256']
    assert result['drnl_sha256'] == record['drnl_sha256']
    assert all(len(m['edge_seconds']) == 2 for m in result['methods'].values())
    assert all(len(m['seconds']) == 2 for m in result['route_only'].values())
    assert len(result['method_order_by_repeat_including_warmup']) == 3
    assert len({tuple(order) for order in result['method_order_by_repeat_including_warmup']}) > 1


def test_balanced_limit_uses_only_existing_frozen_cases():
    records = [{'sampler': sampler, 'cap': cap, 'workload': workload, 'pair_index': pair}
               for sampler in ('a', 'b', 'c') for cap in (50, 200)
               for workload in ('positive', 'twohop_nonedge') for pair in range(20)]
    selected = bench.balanced_cases(records, 128, 42)
    assert len(selected) == len({index for index,_ in selected}) == 128
    assert all(record is records[index] for index,record in selected)
    counts = Counter((r['sampler'], r['cap'], r['workload']) for _,r in selected)
    assert max(counts.values()) - min(counts.values()) == 1
    assert [i for i,_ in selected] == [i for i,_ in bench.balanced_cases(records,128,42)]
    assert bench.balanced_cases(records,0,42) == list(enumerate(records))


def fixture_inputs(tmp_path, exact):
    graph = exact.fixture_graph()
    graph_path = tmp_path/'graph'
    graph_path.mkdir()
    graph.starts.astype('<i8').tofile(graph_path/'edge_starts.bin')
    with (graph_path/'edge_ends.bin').open('wb') as stream:
        stream.write(graph.m.to_bytes(8,'little',signed=True))
        stream.write(graph.indices.astype('<i4').tobytes())
    raw = {'schema_version':1, 'configuration':{'directed':False,'file_endian':'little'},
           'provenance':{'source_sha256':{p:bench.digest_file(REPO/p) for p in bench.SOURCE_PATHS},
                         'graph':{'nodes':graph.n,'stored_entries':graph.m}},
           'records':[archived_record(exact,graph,[0,1,2,3,4,8])]}
    raw_path = tmp_path/'raw.json'
    raw_path.write_text(json.dumps(raw))
    return graph_path, raw_path


def test_mmap_fixture_cli_completes_and_refuses_existing_output(tmp_path, exact):
    graph, raw = fixture_inputs(tmp_path, exact)
    out = tmp_path/'out'
    args = ['--repo',str(REPO),'--input-json',str(raw),'--graph-path',str(graph),
            '--out',str(out),'--repeats','2']
    assert bench.main(args) == 0
    report_bytes = (out/'report.json').read_bytes()
    report = json.loads(report_bytes)
    assert report['status'] == 'complete' and len(report['records']) == 1
    assert report['numba_threads'] == 1
    assert report['torch_imported_before'] == report['torch_imported_after']
    assert report['summary']['hypothetical_lower_bound_included'] is False
    complete = json.loads((out/'COMPLETE.json').read_text())
    assert complete['report_sha256'] == bench.digest_file(out/'report.json')
    with pytest.raises(ValueError, match='no overwrite'):
        bench.main(args)
    assert (out/'report.json').read_bytes() == report_bytes


def test_wrong_frozen_repo_digest_stops_before_output_creation(tmp_path, exact):
    graph, raw_path = fixture_inputs(tmp_path, exact)
    raw = json.loads(raw_path.read_text())
    raw['provenance']['source_sha256']['seal/graph_csr_extraction.py'] = '0'*64
    raw_path.write_text(json.dumps(raw))
    out = tmp_path/'never-created'
    with pytest.raises(ValueError, match='frozen repo source mismatch'):
        bench.main(['--repo',str(REPO),'--input-json',str(raw_path),'--graph-path',str(graph),'--out',str(out)])
    assert not out.exists()


def test_failed_final_status_write_never_publishes_success_marker(tmp_path, exact, monkeypatch):
    graph, raw_path = fixture_inputs(tmp_path, exact)
    out = tmp_path/'failed-output'
    original = bench.write_json
    def fail_final_status(path, data):
        if path.name == 'status.json' and data['status'] == 'complete':
            raise OSError('simulated output failure')
        return original(path, data)
    monkeypatch.setattr(bench, 'write_json', fail_final_status)
    with pytest.raises(OSError, match='simulated output failure'):
        bench.main(['--repo',str(REPO),'--input-json',str(raw_path),'--graph-path',str(graph),
                    '--out',str(out),'--repeats','1'])
    assert not (out/'COMPLETE.json').exists()
    assert json.loads((out/'report.json').read_text())['status'] == 'failed'
