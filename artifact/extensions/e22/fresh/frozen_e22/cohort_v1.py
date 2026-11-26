#!/usr/bin/env python3
"""Freeze a new public Citation2 cohort before every physical timing arm.

This is preparation, not a benchmark. Candidate selection never reads timings,
model outputs, held-out test examples, or the relative cost of any method.
"""
from cpu_common_v1 import *
import argparse


def candidate_grid(protocol):
    return [
        {'sampler': sampler, 'sampler_index': si, 'cap': cap,
         'pair_index': pi, 'workload': 'positive' if pi % 2 == 0 else 'twohop_nonedge'}
        for si, sampler in enumerate(protocol['samplers'])
        for cap in protocol['caps']
        for pi in range(protocol['pair_count'])
    ]


def choose_candidates(protocol, protocol_sha, prepare):
    candidates = candidate_grid(protocol)
    require(len(candidates) == protocol['candidate_count'], 'complete predeclared candidate grid required')
    selected, policy = prepare.choose(candidates, protocol['selected_count'], protocol_sha,
                                      protocol['candidate_selection_seed'])
    require(len(selected) == 128 and selected == sorted(set(selected)), '128 unique ordered candidate indices required')
    return [(i, candidates[i]) for i in selected], policy


def build_archive(graph, protocol, protocol_sha, np, exact, prepare):
    """Actual original samplers and independent edge/DRNL oracle, no timings."""
    require((graph.n, graph.m) == (protocol['nodes'], protocol['stored_entries']), 'strict-clean graph dimensions required')
    pairs = exact.deterministic_pairs(graph, protocol['pair_count'], protocol['pair_seed'], protocol['pair_kind'])
    require(len(pairs) == 128 and all((graph.has_edge(*pair)) is (i % 2 == 0)
                                    for i, pair in enumerate(pairs)), 'mixed edge/twohop-nonedge workload changed')
    selected, policy = choose_candidates(protocol, protocol_sha, prepare)
    records = []
    for candidate_index, meta in selected:
        pair = pairs[meta['pair_index']]
        seed = int(np.random.SeedSequence([protocol['pair_seed'], meta['sampler_index'],
                                          meta['cap'], meta['pair_index']]).generate_state(1)[0])
        nodes = exact.sample_node_set(graph, pair, protocol['hops'], meta['cap'],
                                      np.random.default_rng(seed), meta['sampler'],
                                      oversample=protocol['oversample'], max_neighbors=protocol['max_neighbors'],
                                      max_nodes=protocol['max_subgraph_nodes'])
        degree = graph.degree_sum(nodes)
        require(degree <= protocol['max_neighbors'], 'selected-row neighbor guard exceeded; no automatic cohort replacement')
        require(degree * 256 + nodes.size * 256 <= 6 * GIB, 'selected workspace guard exceeded')
        require(nodes.size * (nodes.size - 1) // 2 <= 20_000_000, 'selected pair-work guard exceeded')
        oracle = exact.oracle_induced(graph, nodes)
        require((oracle != oracle.T).nnz == 0 and not oracle[0, 1] and not oracle[1, 0],
                'symmetric target-removed independent oracle required')
        edges, labels = exact.canonical_edges(oracle), exact.drnl_labels(oracle)
        record = {
            'nodes': nodes.tolist(), 'pair': list(pair), 'pair_index': meta['pair_index'],
            'candidate_record_index': candidate_index, 'sampler': meta['sampler'], 'cap': meta['cap'],
            'workload': meta['workload'], 'sampling_seed': seed, 'added_isolates': 0,
            'virtual_graph_nodes': graph.n, 'sum_out_degrees': degree,
            'induced_directed_entries': int(oracle.nnz), 'node_set_sha256': exact.digest(nodes),
            'canonical_edges_sha256': exact.digest(edges), 'drnl_sha256': exact.digest(labels),
        }
        prepare.valid_record(record, graph.n)
        records.append(record)
    return {
        'schema_version': 1, 'kind': 'fresh_citation2_fixed_S_preparation_v1',
        'provenance': {'graph': {'nodes': graph.n, 'stored_entries': graph.m},
                       'public_input_pins': protocol['input_files'], 'cohort_protocol_sha256': protocol_sha},
        'configuration': {'stock_index_dtype': 'int32', 'timing_methods_executed': 0,
                          'candidate_grid_count': len(candidate_grid(protocol)),
                          'selected_candidate_indices': [i for i, _ in selected], 'selection_policy': policy,
                          'pair_seed': protocol['pair_seed'], 'candidate_selection_seed': protocol['candidate_selection_seed']},
        'records': records,
        'scope': 'New deterministic public graph queries, frozen before all physical timing arms. '
                 'No original Citation2 fixed-S archive is claimed; no timing/quality outputs were consulted.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args(argv)
    require(args.base.resolve() == BASE and sys.platform.startswith('linux'), 'canonical Linux CPU base required')
    bindings = verify_sources(); L = layout(args.base)
    original = read(L['out']/'full_input_before.json'); stable_original(original)
    require((original['nodes'], original['stored_entries']) == (N, M), 'actual whole-input admission required')
    require(not L['cohort'].exists(), 'fresh cohort directory required; preserve any failed preparation')
    protocol = read(HERE/'COHORT_PROTOCOL.json'); protocol_sha = sha(HERE/'COHORT_PROTOCOL.json')
    require(protocol['input_files'] == FULL_FILES, 'predeclared public graph pins changed')
    import cpu_worker_v1
    np, exact, prepare, replay, functions = cpu_worker_v1.load_runtime()
    graph = exact.load_graph(L['full'], 'little')
    archive = build_archive(graph, protocol, protocol_sha, np, exact, prepare)
    L['cohort'].mkdir()
    save(L['cohort_archive'], archive)
    plan = prepare.prepare(L['cohort_archive'], 128, protocol['native_plan_seed'])
    save(L['cohort_plan'], plan)
    require(len(prepare.load_selected(plan, L['cohort_archive'])) == 128, 'actual selected archive verification required')
    stable_original(original); assert_bindings(bindings)
    ready = {'status': 'FROZEN_BEFORE_ALL_MEASUREMENTS', 'record_count': 128,
             'archive': info(L['cohort_archive']), 'plan': info(L['cohort_plan']),
             'protocol': info(HERE/'COHORT_PROTOCOL.json'), 'stratum_counts': plan['stratum_counts'],
             'candidate_count': 768, 'new_sampling_performed_only_in_preparation': True,
             'timing_or_model_results_consulted': False, 'replacement_on_guard_failure': False}
    save(L['cohort']/'COHORT_READY.json', ready)
    print(json.dumps(ready)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
