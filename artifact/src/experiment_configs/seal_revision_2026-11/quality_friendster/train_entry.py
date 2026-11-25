"""Friendster-only evaluation adapter around the frozen SEAL trainer.

Physical endpoints feed extraction. Original query-row IDs feed ranking metrics.
No production module is edited, and no query is merged because sources repeat.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch_geometric.data import Batch

from query_data import build_query_dataset
from seal.reproducibility import edge_extraction_seed, seed_extraction_rng


class QueryEvalCollator:
    def __init__(self, extractor, *, seed=None):
        self.extractor, self.seed = extractor, seed

    def __call__(self, batch):
        graphs = []
        for edge, label, query_id, slot in batch:
            src, dst = (int(v) for v in edge.tolist())
            if self.seed is not None:
                seed_extraction_rng(edge_extraction_seed(self.seed, src, dst))
            graph = self.extractor.build_graph(src, dst, float(label.item()))
            # The frozen accumulator groups by edge_src. In this isolated entry
            # point that channel intentionally carries query identity; physical
            # source remains available separately and in the selected query file.
            graph.edge_src = torch.tensor([query_id], dtype=torch.int64)
            graph.edge_dst = torch.tensor([dst], dtype=torch.int64)
            graph.edge_original_src = torch.tensor([src], dtype=torch.int64)
            graph.edge_candidate_slot = torch.tensor([slot], dtype=torch.int64)
            graphs.append(graph)
        return Batch.from_data_list(graphs)


def main():
    from seal import seal_link_pred as trainer
    trainer.build_eval_dataset = build_query_dataset
    trainer.SEALEvalCollator = QueryEvalCollator
    trainer.main()


if __name__ == "__main__":
    main()
