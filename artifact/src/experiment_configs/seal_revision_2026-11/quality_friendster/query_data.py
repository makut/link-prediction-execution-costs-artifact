"""Preserve original query rows; repeated source vertices are distinct queries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_order(query_id: int, negatives: int, seed: int) -> np.ndarray:
    # This permutation uses query identity, never labels or model scores.
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(query_id)])).permutation(negatives + 1).astype(np.int32)


def freeze_query_subset(*, input_dir: Path, output_dir: Path, name: str, count: int,
                        seed: int, candidate_order_seed: int, expected_negatives: int = 1000,
                        expected_queries: int = 10000, chunk_queries: int = 128) -> dict:
    paths = {key: input_dir / f"{key}.npy" for key in ("source_node", "target_node", "target_node_neg", "warm_query")}
    source, target, negative, warm = [np.load(paths[key], mmap_mode="r", allow_pickle=False) for key in paths]
    total = len(source)
    if source.shape != (total,) or target.shape != (total,) or warm.shape != (total,):
        raise ValueError("Expected one source, target and warm flag per original query row")
    if total != expected_queries or negative.shape != (total, expected_negatives):
        raise ValueError(f"Wrong provided query layout: {total} rows / {negative.shape}")
    if any(array.dtype.kind not in "iu" for array in (source, target, negative)):
        raise ValueError("Provided query endpoints must be integers")
    if count <= 0 or count > total or chunk_queries <= 0:
        raise ValueError("Invalid query-subset or chunk size")
    query_ids = np.sort(np.random.default_rng(seed).choice(total, count, replace=False)).astype(np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"positive": output_dir / f"{name}_edge.npy",
               "negative": output_dir / f"{name}_edge_neg.npy",
               "query_ids": output_dir / f"{name}_query_ids.npy",
               "warm_query": output_dir / f"{name}_warm_query.npy"}
    positive_out = np.lib.format.open_memmap(outputs["positive"], mode="w+", dtype=np.int64, shape=(count, 2))
    negative_out = np.lib.format.open_memmap(outputs["negative"], mode="w+", dtype=np.int64,
                                              shape=(count * expected_negatives, 2))
    for lo in range(0, count, chunk_queries):
        hi = min(count, lo + chunk_queries)
        ids = query_ids[lo:hi]
        positive_out[lo:hi, 0], positive_out[lo:hi, 1] = source[ids], target[ids]
        block = negative_out[lo * expected_negatives:hi * expected_negatives].reshape(hi - lo, expected_negatives, 2)
        block[:, :, 0] = np.asarray(source[ids])[:, None]
        block[:, :, 1] = negative[ids]
    positive_out.flush(); negative_out.flush()
    del positive_out, negative_out
    np.save(outputs["query_ids"], query_ids, allow_pickle=False)
    np.save(outputs["warm_query"], warm[query_ids], allow_pickle=False)
    chosen_sources = np.asarray(source[query_ids])
    manifest = {
        "split": name, "selection_unit": "original query row, including repeated source IDs",
        "subset_seed": int(seed), "candidate_order_seed": int(candidate_order_seed),
        "query_count": int(count), "original_query_count": int(total),
        "unique_source_count": int(np.unique(chosen_sources).size),
        "original_unique_source_count": int(np.unique(source).size),
        "negatives_per_query": int(expected_negatives), "all_provided_negatives_retained": True,
        "query_id_definition": "zero-based row in colleague evaluation source_node/target_node/target_node_neg arrays",
        "inputs": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()},
        "outputs": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in outputs.items()},
        "metric": "custom_sampled_mrr", "is_official_ogb_benchmark": False,
    }
    (output_dir / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class QueryRowDataset:
    """Map-style dataset with contiguous *query* groups, not source groups."""
    def __init__(self, positive, negative, query_ids, *, candidate_order_seed: int):
        self.positive = np.asarray(positive)
        self.negative = np.asarray(negative)
        self.query_ids = np.asarray(query_ids, dtype=np.int64)
        self.count = len(self.positive)
        if self.count < 1 or self.positive.shape != (self.count, 2) or self.negative.ndim != 2 or self.negative.shape[1] != 2:
            raise ValueError("Expected [Q,2] positives and [Q*K,2] negatives")
        if len(self.negative) % self.count or self.query_ids.shape != (self.count,) or np.unique(self.query_ids).size != self.count:
            raise ValueError("Query IDs must be unique and negative rows must form complete query blocks")
        self.negatives = len(self.negative) // self.count
        if self.negatives < 1:
            raise ValueError("No provided negative candidates")
        for lo in range(0, self.count, 128):
            hi = min(lo + 128, self.count)
            block = self.negative[lo * self.negatives:hi * self.negatives].reshape(hi - lo, self.negatives, 2)
            if not np.all(block[:, :, 0] == self.positive[lo:hi, 0, None]):
                raise ValueError("Provided negative rows are not aligned with their original query source")
        self.orders = np.stack([candidate_order(int(q), self.negatives, candidate_order_seed) for q in self.query_ids])

    def __len__(self):
        return self.count * (self.negatives + 1)

    def __getitem__(self, index):
        import torch
        query, offset = divmod(int(index), self.negatives + 1)
        slot = int(self.orders[query, offset])
        edge = self.positive[query] if slot == 0 else self.negative[query * self.negatives + slot - 1]
        return (torch.tensor(np.asarray(edge), dtype=torch.int64),
                torch.tensor(float(slot == 0), dtype=torch.float32),
                int(self.query_ids[query]), slot)


def build_query_dataset(*, dataset_config, split_name, pos_edges, neg_edges):
    prefix = "valid" if split_name == "val" else "test"
    positive_path = Path(dataset_config.valid_edge_path if split_name == "val" else dataset_config.test_edge_path)
    directory = positive_path.parent
    manifest = json.loads((directory / f"{prefix}_manifest.json").read_text())
    ids = np.load(directory / f"{prefix}_query_ids.npy", mmap_mode="r", allow_pickle=False)
    return QueryRowDataset(pos_edges, neg_edges, ids, candidate_order_seed=manifest["candidate_order_seed"])
