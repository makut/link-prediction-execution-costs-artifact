import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_csr.serializer import GraphCSRSerializer


def _log(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[GRAPH CSR DATASET {timestamp}] {message}', flush=True)


class GraphCSRData:
    def __init__(
        self,
        root,
        num_nodes,
        graph_root,
        file_endian='big',
        use_mmap=True,
        allow_non_native=True,
        chunk_bytes=256 * 1024 * 1024,
        x=None,
        directed_out_root=None,
        directed_in_root=None,
    ):
        self.root = root
        self.num_nodes = int(num_nodes)
        self.graph_root = graph_root
        self.file_endian = file_endian
        self.use_mmap = use_mmap
        self.allow_non_native = allow_non_native
        self.chunk_bytes = chunk_bytes
        self.x = x
        self.directed_out_root = directed_out_root
        self.directed_in_root = directed_in_root


def load_graph_csr_data(
    graph_root,
    *,
    file_endian='big',
    use_mmap=True,
    allow_non_native=True,
    chunk_bytes=256 * 1024 * 1024,
    node_features_path=None,
    directed_out_root=None,
    directed_in_root=None,
):
    _log(f'Loading graph metadata graph_root={graph_root}')
    with GraphCSRSerializer.deserialize(
        graph_root,
        use_mmap=use_mmap,
        file_endian=file_endian,
        writable=False,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    ) as graph:
        num_nodes = int(graph.edge_starts.numpy().size)
        num_positive_edges = int(graph.edge_ends.size)

    x = None
    if node_features_path is not None:
        import numpy as np

        x = np.load(str(node_features_path), mmap_mode='r')
        if x.ndim != 2 or int(x.shape[0]) != num_nodes:
            raise ValueError(
                f'node features at {node_features_path} must be [num_nodes={num_nodes}, F], got {x.shape}'
            )
        _log(f'Loaded node features shape={tuple(x.shape)} dtype={x.dtype} (mmap)')

    data = GraphCSRData(
        root=str(graph_root),
        num_nodes=num_nodes,
        graph_root=str(graph_root),
        file_endian=file_endian,
        use_mmap=use_mmap,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
        x=x,
        directed_out_root=None if directed_out_root is None else str(directed_out_root),
        directed_in_root=None if directed_in_root is None else str(directed_in_root),
    )
    if directed_out_root is not None or directed_in_root is not None:
        if directed_out_root is None or directed_in_root is None:
            raise ValueError('both directed_out_root and directed_in_root must be set')
        for name, root in (('out', directed_out_root), ('in', directed_in_root)):
            with GraphCSRSerializer.deserialize(
                str(root), use_mmap=use_mmap, file_endian=file_endian, writable=False,
                allow_non_native=allow_non_native, chunk_bytes=chunk_bytes,
            ) as dg:
                n = int(dg.edge_starts.numpy().size)
                if n != num_nodes:
                    raise ValueError(f'directed {name} CSR has num_nodes={n}, expected {num_nodes}')
                _log(f'Directed {name} CSR {root}: edges={int(dg.edge_ends.size)}')
    meta = {'num_positive_edges': num_positive_edges}
    _log(
        f'Loaded graph metadata num_nodes={num_nodes} train_edges={num_positive_edges}'
    )
    return data, meta
