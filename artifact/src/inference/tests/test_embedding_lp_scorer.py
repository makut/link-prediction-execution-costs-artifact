from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from embedding_lp.scoring import EmbeddingLPScorer
from embedding_lp.trainer import LogisticRegression, lookup_edge_features
from embedding_table_utils import EmbeddingTableConfig, ReadOnlyEmbeddingStore, create_embedding_table


def _make_embedding_checkpoint(tmp_path: Path, *, num_nodes: int, dim: int) -> str:
    config = EmbeddingTableConfig(backend="vanilla", num_embeddings=num_nodes, embedding_dim=dim, device="cpu")
    table = create_embedding_table(config, device=torch.device("cpu"))
    emb_dir = tmp_path / "embedding_table"
    table.save_local(str(emb_dir))
    return str(emb_dir)


class EmbeddingLPScorerTests(unittest.TestCase):
    def test_scorer_matches_manual_forward(self) -> None:
        num_nodes, dim = 64, 8
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            emb_dir = _make_embedding_checkpoint(tmp_path, num_nodes=num_nodes, dim=dim)

            torch.manual_seed(0)
            classifier = LogisticRegression(dim)
            ckpt_path = tmp_path / "best.pt"
            torch.save(
                {
                    "model": classifier.state_dict(),
                    "config": {"lp_operator": "hadamard", "treat_as_undirected_for_lp": True},
                    "embedding_checkpoint_dir": emb_dir,
                },
                ckpt_path,
            )

            scorer = EmbeddingLPScorer.from_checkpoint(ckpt_path, device=device)
            edges = np.array([[1, 2], [5, 3], [10, 10], [63, 0]], dtype=np.int64)
            scores = scorer.score_batch(edges)

            self.assertEqual(scores.shape, (edges.shape[0],))
            self.assertEqual(scores.dtype, np.float32)
            self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))

            store = ReadOnlyEmbeddingStore.from_checkpoint(emb_dir, device=device)
            features = lookup_edge_features(
                store,
                torch.as_tensor(edges, dtype=torch.int64),
                operator="hadamard",
                treat_as_undirected=True,
                device=device,
            )
            with torch.no_grad():
                expected = torch.sigmoid(classifier(features).float()).numpy()
            np.testing.assert_allclose(scores, expected.astype(np.float32), rtol=1e-5, atol=1e-6)

    def test_scorer_embedding_dir_override(self) -> None:
        num_nodes, dim = 32, 4
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            emb_dir = _make_embedding_checkpoint(tmp_path, num_nodes=num_nodes, dim=dim)
            ckpt_path = tmp_path / "latest.pt"
            torch.save(
                {
                    "model": LogisticRegression(dim).state_dict(),
                    "config": {"lp_operator": "hadamard"},
                    "embedding_checkpoint_dir": "/nonexistent/path",
                },
                ckpt_path,
            )
            scorer = EmbeddingLPScorer.from_checkpoint(ckpt_path, device=device, embedding_checkpoint_dir=emb_dir)
            scores = scorer.score_batch(np.array([[0, 1], [2, 3]], dtype=np.int64))
            self.assertEqual(scores.shape, (2,))


if __name__ == "__main__":
    unittest.main()
