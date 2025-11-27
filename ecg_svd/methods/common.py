import numpy as np
import torch
from loguru import logger
from scipy.linalg import hankel
from typing import List


def lower_peaks(
    signal: np.ndarray,
    peaks: np.ndarray,
    reduction_factor: float = 0.2,
    neighborhood: int = 70
) -> np.ndarray:
    clean_signal = signal.copy()

    for peak_idx in peaks:
        left = max(0, peak_idx - neighborhood)
        right = min(len(signal), peak_idx + neighborhood)

        clean_signal[left:right] *= reduction_factor

    logger.debug(f"Applied peak suppression on {len(peaks)} peaks.")
    return clean_signal


def diagonal_averaging(hankel_matrix, on_cuda: bool = False) -> np.ndarray:
    sums = 0
    counts = 0

    device = torch.device("cuda" if on_cuda and torch.cuda.is_available() else "cpu")

    if on_cuda:
        if isinstance(hankel_matrix, np.ndarray):
            hankel_matrix = torch.from_numpy(hankel_matrix).float().to(device)
        else:
            hankel_matrix = hankel_matrix.float().to(device)

        L, K = hankel_matrix.shape
        N = L + K - 1

        row_indices = torch.arange(L, device=device).unsqueeze(1)
        col_indices = torch.arange(K, device=device).unsqueeze(0)
        indices = row_indices + col_indices

        indices_flat = indices.flatten().long()
        weights_flat = hankel_matrix.flatten()

        sums = torch.bincount(indices_flat, weights=weights_flat, minlength=N)
        counts = torch.bincount(indices_flat, minlength=N)
        counts[counts == 0] = 1e-8
    else:
        L, K = hankel_matrix.shape

        # index matrix of anti-diagonals
        idx = np.arange(L)[:, None] + np.arange(K)
        sums = np.zeros(L + K - 1, dtype=hankel_matrix.dtype)
        counts = np.zeros(L + K - 1, dtype=hankel_matrix.dtype)

        # accumulate matrix values from anti-diagonals
        np.add.at(sums, idx, hankel_matrix)

        # count anti-diagonal elements
        np.add.at(counts, idx, 1)
    return sums / counts


def reconstruct_channels(H_block: np.ndarray, on_cuda: bool = False) -> List[np.ndarray]:
    L, K, C = H_block.shape
    if on_cuda:
        rec = torch.stack([diagonal_averaging(H_block[:, :, c], on_cuda=True) for c in range(C)], dim=1)
    else:
        rec = [diagonal_averaging(H_block[:, :, c]) for c in range(C)]
    return rec


def create_hankel_matrix(signal: np.ndarray, L_samples: int) -> np.ndarray:
    K_cols = len(signal) - L_samples + 1
    if K_cols <= 0:
        logger.error("Signal length is insufficient for the window size L.")
        raise ValueError("Invalid Hankel parameters.")

    hankel_matrix = hankel(signal[:L_samples], signal[L_samples - 1:])

    logger.debug(f"Hankel matrix created: {hankel_matrix.shape}")
    return hankel_matrix
