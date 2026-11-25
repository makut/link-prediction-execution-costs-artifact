"""Opt-in SEAL RNG initialization; this does not promise bitwise CUDA determinism."""
from __future__ import annotations

import random
import hashlib

import numpy as np
import torch

try:
    from .graph_csr_kernels import seed_numba_rng
except ImportError:  # script-style trainer invocation
    from graph_csr_kernels import seed_numba_rng


def seed_extraction_rng(seed: int) -> None:
    """Seed both RNGs used by Python and compiled neighbourhood sampling."""
    value = int(seed) % (2**32)
    np.random.seed(value)
    seed_numba_rng(value)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    seed_extraction_rng(seed)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seed_seal_worker(worker_id: int) -> None:
    """Picklable DataLoader initializer, deriving each worker RNG from its generator.

    Do not initialize CUDA in forked data workers. Torch already sets the worker
    CPU seed; explicitly reset its CPU generator along with Python/NumPy/Numba.
    """
    del worker_id
    seed = int(torch.initial_seed())
    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    seed_extraction_rng(seed)


def configure_seeded_loader(loader, seed: int) -> None:
    # Assign before the first iterator; the shared loader factory remains unchanged.
    loader.generator = torch.Generator().manual_seed(int(seed))
    loader.worker_init_fn = seed_seal_worker


def edge_extraction_seed(seed: int, src: int, dst: int) -> int:
    """Stable ordered-pair hash independent of Python hash salt and worker order."""
    value = (int(seed) ^ (int(src) * 0x9E3779B97F4A7C15) ^ (int(dst) * 0xBF58476D1CE4E5B9)) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    return int((value ^ (value >> 31)) & 0xFFFFFFFF)


def model_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
