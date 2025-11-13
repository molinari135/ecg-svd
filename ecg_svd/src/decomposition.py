import numpy as np
import torch
import gc
import tensorly as tl
from tensorly.decomposition import tucker, parafac
from sklearn.decomposition import FastICA
from scipy.linalg import svd, hankel
from loguru import logger
from typing import Tuple, Optional, List, Dict, Any


# set tensorly backend
tl.set_backend('numpy')


def create_hankel_matrix(signal: np.ndarray, L_samples: int) -> np.ndarray:
    K_cols = len(signal) - L_samples + 1
    if K_cols <= 0:
        logger.error("Signal length is insufficient for the window size L.")
        raise ValueError("Invalid Hankel parameters.")

    hankel_matrix = hankel(signal[:L_samples], signal[L_samples - 1:])

    logger.debug(f"Hankel matrix created: {hankel_matrix.shape}")
    return hankel_matrix


def diagonal_averaging(hankel_matrix: np.ndarray) -> np.ndarray:
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


def diagonal_averaging_torch(hankel_matrix, on_cuda: bool = True):
    device = torch.device("cuda" if on_cuda and torch.cuda.is_available() else "cpu")

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
    return sums / counts


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


def hankel_with_svd(
    signal: np.ndarray,
    window_length: int = 625 * 2,
    cvp: float = 0.75
) -> np.ndarray:
    H = create_hankel_matrix(signal, L_samples=window_length)
    U, S, Vt = svd(H, full_matrices=False)

    variances = S**2
    explained_variance = variances / np.sum(variances)
    k = np.argmax(np.cumsum(explained_variance) >= cvp) + 1
    logger.info(f"SVD performed. Rank k={k} chosen (CVP: {cvp * 100:.1f}%)")

    R = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    reconstructed_signal = diagonal_averaging(R)
    return reconstructed_signal


def run_fastica(
    data_matrix: np.ndarray,
    n_components: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if n_components is None:
        n_components = min(data_matrix.shape)
        logger.info(f"Using default n_components: {n_components}")

    ica = FastICA(n_components=n_components, max_iter=1000, random_state=42)
    sources = ica.fit_transform(data_matrix)
    mixing_matrix = ica.mixing_

    logger.success(f"FastICA performed. Extracted {n_components} components.")
    return sources, mixing_matrix


def run_tucker(
    tensor: np.ndarray,
    rank: Tuple[int, ...]
) -> Tuple[np.ndarray, list]:
    core, factors = tucker(tensor, rank=rank, init='random', n_iter_max=500, tol=1e-2, verbose=False)
    logger.success(f"Tucker decomposition performed with rank {rank}.")
    return core, factors


def run_parafac(
    tensor: np.ndarray,
    rank: int
) -> tl.cp_tensor.CPTensor:
    cp_tensor = parafac(tensor, rank=rank, init='random', n_iter_max=500, tol=1e-2, verbose=False)
    logger.success(f"PARAFAC decomposition performed with rank {rank}.")
    return cp_tensor


def get_tucker_rank(signal_list: List[Dict[str, Any]], signal_quality_func) -> int:

    L = 625 * 2
    ranks = []

    for signal_data in signal_list:
        signal = signal_data['segment']
        H = create_hankel_matrix(signal, L_samples=L)
        S = svd(H, full_matrices=False)[1]

        variances = S**2
        explained_variance = variances / np.sum(variances)
        cumulative_variance = np.cumsum(explained_variance)

        best_cvp = signal_quality_func(signal)['best_cvp']
        k = np.searchsorted(cumulative_variance, best_cvp) + 1
        ranks.append(k)

        del H, S
        gc.collect()

    return int(np.round(np.mean(ranks)))


def reconstruct_channels(H_block: np.ndarray) -> List[np.ndarray]:
    L, K, C = H_block.shape
    reconstructed = []
    for c in range(C):  # iterate over channels
        x_rec = diagonal_averaging(H_block[:, :, c])  # perform diagonal averaging on the L x K slice
        reconstructed.append(x_rec)
    return reconstructed


def hankel_torch(signal: torch.Tensor, L_samples: int) -> torch.Tensor:
    N = signal.size(0)
    K = N - L_samples + 1
    if K <= 0:
        return torch.empty((0, 0), device=signal.device)

    indices = torch.arange(L_samples, device=signal.device).unsqueeze(1) + torch.arange(K, device=signal.device)
    return signal[indices]


def reconstruct_channels_torch(H_block):
    L, K, C = H_block.shape
    rec = torch.stack([diagonal_averaging_torch(H_block[:, :, c]) for c in range(C)], dim=1)
    return rec
