from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference.io import load_pairs, write_scores


class LoadPairsTests(unittest.TestCase):
    def test_load_pairs_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pairs = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
            path = Path(tmp) / "pairs.npy"
            np.save(path, pairs)
            loaded = load_pairs(path, use_mmap=False)
            np.testing.assert_array_equal(np.asarray(loaded), pairs)
            self.assertEqual(np.asarray(loaded).dtype, np.int64)

    def test_load_pairs_casts_non_int64(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npy"
            np.save(path, np.array([[1, 2], [3, 4]], dtype=np.int32))
            loaded = load_pairs(path, use_mmap=False)
            self.assertEqual(np.asarray(loaded).dtype, np.int64)

    def test_load_pairs_rejects_bad_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npy"
            np.save(path, np.arange(9).reshape(3, 3))
            with self.assertRaises(ValueError):
                load_pairs(path, use_mmap=False)


class WriteScoresTests(unittest.TestCase):
    def test_write_scores_writes_array_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scores = np.array([0.1, 0.9, 0.5], dtype=np.float64)
            manifest = {"method": "verse", "num_pairs": 3}
            scores_path, manifest_path = write_scores(Path(tmp) / "out", scores=scores, manifest=manifest)

            written = np.load(scores_path)
            self.assertEqual(written.dtype, np.float32)
            np.testing.assert_allclose(written, scores.astype(np.float32))
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)


if __name__ == "__main__":
    unittest.main()
