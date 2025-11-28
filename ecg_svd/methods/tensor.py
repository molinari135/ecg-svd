import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, parafac
from loguru import logger
from typing import Tuple


def run_tucker(
    tensor: np.ndarray,
    rank: Tuple[int, ...]
) -> Tuple[np.ndarray, list]:
    core, factors = tucker(tensor, rank=rank, init='random', n_iter_max=100, tol=1e-3, random_state=42, verbose=False)
    logger.debug(f"Tucker decomposition performed with rank {rank}.")
    return core, factors


def run_parafac(
    tensor: np.ndarray,
    rank: int
) -> tl.cp_tensor.CPTensor:
    cp_tensor = parafac(tensor, rank=rank, init='random', n_iter_max=100, tol=1e-3, random_state=42, verbose=False)
    logger.debug(f"PARAFAC decomposition performed with rank {rank}.")
    return cp_tensor
