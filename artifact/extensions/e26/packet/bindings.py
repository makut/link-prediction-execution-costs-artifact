"""Small source-pinned dataset bindings; no data acquisition or remote I/O."""
from common import HERE, read, require, sha

WORKLOADS = {
    'friendster': '340e61cdd1eb8d329d52be164f919480e6a41feb4c620852df0393c2e9a192f0',
    'citation2': '648332c22cb6a25825e8487e2e82e39e2cb5dc53c8890cb3acb71b6324d7b750',
}


def binding(dataset):
    require(dataset in WORKLOADS, 'unknown prospective dataset')
    graph = read(HERE / 'pins' / (dataset + '_graph_expected.json'))
    native = read(HERE / 'pins' / (dataset + '_native_expected.json'))
    require(graph['num_nodes'] == native['nodes'] and graph['stored_entries'] == native['stored_entries'],
            'graph/native source-pinned dimensions differ')
    require(native['rowptr_bytes'] == 8 * (native['nodes'] + 1) and
            native['col_bytes'] == 8 * native['stored_entries'], 'native byte formulas differ')
    workload = HERE / 'workloads' / (dataset + '_pairs.npy')
    require(workload.stat().st_size == 192128 and sha(workload) == WORKLOADS[dataset],
            'original frozen E23 workload differs')
    return {'dataset_id': dataset, 'graph': graph, 'native': native,
            'workload_path': str(workload), 'workload_sha256': WORKLOADS[dataset]}
