import numpy as np
import torch
import tensorly as tl
from sklearn.decomposition import FastICA
from scipy.linalg import svd
from loguru import logger
from typing import Tuple, Optional

from ecg_svd.methods.common import diagonal_averaging, create_hankel_matrix

# set tensorly backend
tl.set_backend('numpy')


def run_ssa(
    signal: np.ndarray,
    window_length: int = 625 * 2,
    cvp: float = 0.75,
    on_cuda: bool = False
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = create_hankel_matrix(signal, L_samples=window_length)

    if on_cuda:
        H_torch = torch.from_numpy(H).float().to(device)
        U, S, Vt = torch.svd(H_torch, full_matrices=False)
        variances = S**2
        explained_variance = variances / torch.sum(variances)
        k_torch = torch.argmax(torch.cumsum(explained_variance, dim=0) >= cvp) + 1
        k = k_torch.item()
        S_diag = S[:k].unqueeze(0)
        U_S_k = U[:, :k] * S_diag
        R = torch.matmul(U_S_k, Vt[:k, :]).cpu().numpy()
    else:
        U, S, Vt = svd(H, full_matrices=False)
        variances = S**2
        explained_variance = variances / np.sum(variances)
        k = np.argmax(np.cumsum(explained_variance) >= cvp) + 1
        R = np.dot(U[:, :k] * S[:k], Vt[:k, :])
    logger.debug(f"SVD performed with rank k={k} (CVP: {cvp * 100:.1f}%)")
    reconstructed_signal = diagonal_averaging(R)
    return reconstructed_signal


def run_fastica(
    data_matrix: np.ndarray,
    n_components: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if n_components is None:
        n_components = min(data_matrix.shape)
        logger.debug(f"Using default n_components: {n_components}")

    ica = FastICA(n_components=n_components, max_iter=100, random_state=42)
    sources = ica.fit_transform(data_matrix)
    mixing_matrix = ica.mixing_

    logger.debug(f"FastICA performed with {n_components} components in {ica.n_iter_} iterations.")
    return sources, mixing_matrix


def hankel_torch(signal: torch.Tensor, L_samples: int) -> torch.Tensor:
    N = signal.size(0)
    K = N - L_samples + 1
    if K <= 0:
        return torch.empty((0, 0), device=signal.device)

    indices = torch.arange(L_samples, device=signal.device).unsqueeze(1) + torch.arange(K, device=signal.device)
    return signal[indices]
