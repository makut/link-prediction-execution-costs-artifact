from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

from graph_csr.graph import GraphCSR
from graph_csr.large_int32 import LargeInt32Array
from graph_csr.large_int64_raw import LargeInt64RawArray
from seal import graph_csr_extraction
from seal.config import SEALExtractionConfig, load_config


def _make_graph(adjacency: dict[int, list[int]]) -> GraphCSR:
    num_nodes = max(
        [*adjacency, *(neighbor for neighbors in adjacency.values() for neighbor in neighbors)]
    ) + 1
    edge_starts = []
    edge_ends = []
    for node in range(num_nodes):
        edge_starts.append(len(edge_ends))
        edge_ends.extend(sorted(adjacency.get(node, [])))
    return GraphCSR(
        edge_ends=LargeInt32Array(size=len(edge_ends), arr=np.asarray(edge_ends, dtype=np.int32)),
        edge_starts=LargeInt64RawArray(arr=np.asarray(edge_starts, dtype=np.int64)),
    )


def _extract(
    graph: GraphCSR,
    *,
    use_per_vertex_sampling: bool = True,
    use_pairwise_subgraph: bool = True,
    max_nodes_per_hop=None,
):
    return graph_csr_extraction._k_hop_subgraph_graph_csr(
        0,
        1,
        1,
        graph,
        max_nodes_per_hop=max_nodes_per_hop,
        graph_csr_use_per_vertex_sampling=use_per_vertex_sampling,
        graph_csr_use_pairwise_subgraph=use_pairwise_subgraph,
    )


class GraphCSRExtractionModesTest(unittest.TestCase):
    def test_optimized_algorithms_are_enabled_by_default(self) -> None:
        config = SEALExtractionConfig()
        self.assertTrue(config.graph_csr_use_per_vertex_sampling)
        self.assertTrue(config.graph_csr_use_pairwise_subgraph)

    def test_flags_are_loaded_from_yaml(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "seal.yaml"
            config_path.write_text(
                """
dataset:
  graph_csr_root: /tmp/graph
seal:
  graph_csr_use_per_vertex_sampling: false
  graph_csr_use_pairwise_subgraph: false
""",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertFalse(config.seal.graph_csr_use_per_vertex_sampling)
        self.assertFalse(config.seal.graph_csr_use_pairwise_subgraph)

    def test_modes_produce_same_regular_undirected_subgraph(self) -> None:
        graph = _make_graph({
            0: [1, 2],
            1: [0, 2],
            2: [0, 1, 3],
            3: [2],
        })

        optimized = _extract(graph)
        legacy_subgraph = _extract(graph, use_pairwise_subgraph=False)
        auto_subgraph = _extract(graph, use_pairwise_subgraph="auto")

        self.assertEqual(optimized[0], legacy_subgraph[0])
        self.assertEqual(optimized[2], legacy_subgraph[2])
        np.testing.assert_array_equal(optimized[1].toarray(), legacy_subgraph[1].toarray())
        self.assertEqual(optimized[0], auto_subgraph[0])
        np.testing.assert_array_equal(optimized[1].toarray(), auto_subgraph[1].toarray())

    def test_auto_mode_switches_on_pair_count_vs_degree_sum(self) -> None:
        # star: hub 0 with 6 leaves. S = {0, 1, 2..6} -> |S|=7, pairs=21, sum deg(S)=12.
        graph = _make_graph({0: [1, 2, 3, 4, 5, 6], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0]})
        for ratio, expect_scan in ((1.0, True), (4.0, False)):
            with self.subTest(ratio=ratio):
                with mock.patch.object(
                    graph_csr_extraction,
                    "_build_induced_edges_neighbor_scan",
                    wraps=graph_csr_extraction._build_induced_edges_neighbor_scan,
                ) as build_neighbor_scan:
                    graph_csr_extraction._k_hop_subgraph_graph_csr(
                        0, 1, 1, graph, 1.0, None,
                        graph_csr_use_per_vertex_sampling=False,
                        graph_csr_use_pairwise_subgraph="auto",
                        pairwise_auto_ratio=ratio,
                    )
                self.assertEqual(build_neighbor_scan.call_count, int(expect_scan))
        self.assertEqual(graph_csr_extraction.resolve_pairwise_mode("auto"), "auto")
        self.assertIs(graph_csr_extraction.resolve_pairwise_mode("false"), False)
        with self.assertRaises(ValueError):
            graph_csr_extraction.resolve_pairwise_mode("sometimes")

    def test_flags_select_algorithms_independently(self) -> None:
        graph = _make_graph({
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [0, 1],
            3: [0, 1],
        })

        for use_per_vertex_sampling in (False, True):
            for use_pairwise_subgraph in (False, True):
                with self.subTest(
                    use_per_vertex_sampling=use_per_vertex_sampling,
                    use_pairwise_subgraph=use_pairwise_subgraph,
                ):
                    with (
                        mock.patch.object(
                            graph_csr_extraction,
                            "_collect_neighbors_sampled",
                            wraps=graph_csr_extraction._collect_neighbors_sampled,
                        ) as collect_sampled,
                        mock.patch.object(
                            graph_csr_extraction,
                            "_build_induced_edges_neighbor_scan",
                            wraps=graph_csr_extraction._build_induced_edges_neighbor_scan,
                        ) as build_neighbor_scan,
                    ):
                        _extract(
                            graph,
                            use_per_vertex_sampling=use_per_vertex_sampling,
                            use_pairwise_subgraph=use_pairwise_subgraph,
                            max_nodes_per_hop=1,
                        )

                    self.assertEqual(collect_sampled.call_count, int(use_per_vertex_sampling))
                    self.assertEqual(build_neighbor_scan.call_count, int(not use_pairwise_subgraph))

    def test_neighbor_scan_subgraph_preserves_self_loops(self) -> None:
        graph = _make_graph({
            0: [1, 2],
            1: [0],
            2: [0, 2],
        })

        optimized = _extract(graph)[1].toarray()
        neighbor_scan = _extract(graph, use_pairwise_subgraph=False)[1].toarray()

        self.assertEqual(optimized[2, 2], 0)
        self.assertEqual(neighbor_scan[2, 2], 1)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Directed mode vs. the SEAL-OGB reference semantics (directed=True):
# fringe = out ∪ in, induced subgraph keeps direction, target link removed both ways.
# ---------------------------------------------------------------------------

def _make_directed_pair(out_adj: dict[int, list[int]], num_nodes: int):
    """Return (out_csr, in_csr) GraphCSR objects for a directed adjacency dict."""
    def build(adj):
        starts, ends = [], []
        for u in range(num_nodes):
            starts.append(len(ends))
            ends.extend(sorted(adj.get(u, [])))
        return _serialize_graph_obj(starts, ends)
    in_adj: dict[int, list[int]] = {}
    for u, vs in out_adj.items():
        for v in vs:
            in_adj.setdefault(v, []).append(u)
    return build(out_adj), build(in_adj)


def _serialize_graph_obj(starts, ends):
    from graph_csr.graph import GraphCSR
    from graph_csr.large_int32 import LargeInt32Array
    from graph_csr.large_int64_raw import LargeInt64RawArray
    return GraphCSR(
        edge_ends=LargeInt32Array(size=len(ends), arr=np.asarray(ends, dtype=np.int32)),
        edge_starts=LargeInt64RawArray(arr=np.asarray(starts, dtype=np.int64)),
        timestamps=None,
    )


def _reference_k_hop_directed(src, dst, num_hops, A, A_csc):
    """Verbatim SEAL-OGB utils.k_hop_subgraph logic for directed=True without sampling."""
    nodes = [src, dst]
    visited = {src, dst}
    fringe = {src, dst}
    for _ in range(1, num_hops + 1):
        out_neighbors = set(A[list(fringe)].indices)
        in_neighbors = set(A_csc[:, list(fringe)].indices)  # csc: .indices are ROW ids (in-neighbours)
        fringe = (out_neighbors | in_neighbors) - visited
        visited = visited | fringe
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
    subgraph = A[nodes, :][:, nodes]
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0
    return nodes, subgraph


class DirectedExtractionTest(unittest.TestCase):
    # citation-like directed graph: 0 cites 1,2 ; 2 cites 1,3 ; 3 cites 0 ; 4 cites 2 ; 5 cites 5? (no self loops) ; 1 cites 4
    OUT = {0: [1, 2], 1: [4], 2: [1, 3], 3: [0], 4: [2], 5: [3]}
    N = 6

    def _adjacency_sets(self, nodes, subgraph):
        rows, cols = subgraph.nonzero()
        return {(int(nodes[r]), int(nodes[c])) for r, c in zip(rows, cols)}

    def test_directed_matches_reference_semantics(self) -> None:
        import scipy.sparse as ssp
        out_csr, in_csr = _make_directed_pair(self.OUT, self.N)
        rows, cols = zip(*[(u, v) for u, vs in self.OUT.items() for v in vs])
        A = ssp.csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=(self.N, self.N))
        A_csc = A.tocsc()
        for src, dst in ((0, 1), (0, 2), (2, 1), (4, 3), (5, 0)):
            for hops in (1, 2):
                for mode in ("auto", True, False):
                    with self.subTest(src=src, dst=dst, hops=hops, mode=mode):
                        ref_nodes, ref_sub = _reference_k_hop_directed(src, dst, hops, A, A_csc)
                        nodes, sub, dists, _, _ = graph_csr_extraction._k_hop_subgraph_graph_csr(
                            src, dst, hops, out_csr, 1.0, None,
                            graph_csr_use_per_vertex_sampling=False,
                            graph_csr_use_pairwise_subgraph=mode,
                            graph_in=in_csr, directed=True,
                        )
                        self.assertEqual(sorted(nodes), sorted(ref_nodes))
                        self.assertEqual(nodes[:2], [src, dst])
                        self.assertEqual(self._adjacency_sets(nodes, sub), self._adjacency_sets(ref_nodes, ref_sub))
                        # direction is really kept (asymmetric unless mutual)
                        edges = self._adjacency_sets(nodes, sub)
                        for (u, v) in edges:
                            self.assertIn(v, self.OUT.get(u, []))

    def test_symmetrised_directed_equals_undirected_extraction(self) -> None:
        out_csr, in_csr = _make_directed_pair(self.OUT, self.N)
        sym: dict[int, list[int]] = {}
        for u, vs in self.OUT.items():
            for v in vs:
                sym.setdefault(u, []).append(v)
                sym.setdefault(v, []).append(u)
        sym = {u: sorted(set(vs)) for u, vs in sym.items()}
        sym_graph = _make_graph(sym)
        for src, dst in ((0, 1), (2, 1), (4, 3)):
            with self.subTest(src=src, dst=dst):
                d_nodes, d_sub, *_ = graph_csr_extraction._k_hop_subgraph_graph_csr(
                    src, dst, 2, out_csr, 1.0, None, graph_csr_use_per_vertex_sampling=False,
                    graph_csr_use_pairwise_subgraph=True, graph_in=in_csr, directed=True,
                )
                u_nodes, u_sub, *_ = graph_csr_extraction._k_hop_subgraph_graph_csr(
                    src, dst, 2, sym_graph, 1.0, None, graph_csr_use_per_vertex_sampling=False,
                    graph_csr_use_pairwise_subgraph=True,
                )
                self.assertEqual(sorted(d_nodes), sorted(u_nodes))
                d_edges = self._adjacency_sets(d_nodes, d_sub)
                d_sym = d_edges | {(v, u) for (u, v) in d_edges}
                self.assertEqual(d_sym, self._adjacency_sets(u_nodes, u_sub))

    def test_directed_per_vertex_sampling_draws_from_out_and_in(self) -> None:
        out_csr, in_csr = _make_directed_pair(self.OUT, self.N)
        seen = set()
        for _ in range(50):
            nodes, *_ = graph_csr_extraction._k_hop_subgraph_graph_csr(
                0, 1, 1, out_csr, 1.0, 1, graph_csr_use_per_vertex_sampling=True,
                graph_csr_use_pairwise_subgraph=True, graph_in=in_csr, directed=True,
            )
            seen.update(nodes[2:])
        # neighbours of {0,1}: out(0)={1,2}, in(0)={3}, out(1)={4}, in(1)={0,2} -> candidates {2,3,4}
        self.assertTrue(seen <= {2, 3, 4} and len(seen) >= 2, seen)
