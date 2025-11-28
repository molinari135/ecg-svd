import numpy as np
import neurokit2 as nk
from tqdm import tqdm
from scipy.linalg import svd
from loguru import logger
from typing import Dict, Any, List, Tuple

from ecg_svd.methods.common import diagonal_averaging
from ecg_svd.methods.matrix import create_hankel_matrix


def get_classification_report(ground_truth: np.ndarray, prediction: np.ndarray, epsilon: float = 0.15) -> Dict[str, Any]:
    if len(ground_truth) == 0 and len(prediction) == 0:
        logger.warning("Both Ground Truth and Prediction arrays are empty. Returning 100% accuracy (Trivial case).")
        return {
            "TP": 0, "FN": 0, "FP": 0, "TN": 0,
            "accuracy": 100.0, "precision": 100.0, "recall": 100.0
        }

    if len(prediction) == 0:
        logger.warning("Prediction array is empty. Accuracy and Precision will be low/zero.")

    # identify True Positives (TP) and False Negatives (FN)
    is_gt_found = [
        np.any(np.abs(gt_element - prediction) <= epsilon)
        for gt_element in ground_truth
    ]

    # identify False Positives (FP)
    is_pred_correct = [
        np.any(np.abs(p_element - ground_truth) <= epsilon)
        for p_element in prediction
    ]

    TP = np.sum(is_gt_found)
    FN = len(ground_truth) - TP
    TN = 0
    FP = len(prediction) - np.sum(is_pred_correct)

    total_samples = TP + TN + FP + FN

    # calculate metrics, handling potential ZeroDivisionError
    accuracy = (TP + TN) / total_samples if total_samples > 0 else 0
    precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0
    recall = (TP / (TP + FN)) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": int(TP),
        "FN": int(FN),
        "FP": int(FP),
        "TN": int(TN),
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100
    }


def signal_quality(
    signal: np.ndarray,
    sampling_rate: int = 1000,
    cvp_to_test: List[float] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    window_length: int = 625 * 2
) -> Dict[str, Any]:

    # guard against empty or invalid inputs
    if signal is None or len(signal) == 0:
        logger.warning("Empty signal provided to signal_quality; returning defaults.")
        return {
            "qualities": {},
            "best_cvp": None,
            "best_quality": 0.0,
            "singular_values": np.array([])
        }

    if window_length is None or window_length <= 0:
        # fallback to a reasonable default based on signal length
        window_length = max(1, len(signal) // 2)

    # create hankel matrix and guard against failures / empty matrices
    try:
        H = create_hankel_matrix(signal, L_samples=window_length)
    except Exception as e:
        logger.warning(f"Failed to create Hankel matrix: {e}; returning defaults.")
        return {
            "qualities": {},
            "best_cvp": None,
            "best_quality": 0.0,
            "singular_values": np.array([])
        }

    if H is None or getattr(H, "size", 0) == 0:
        logger.warning("Hankel matrix is empty; returning defaults.")
        return {
            "qualities": {},
            "best_cvp": None,
            "best_quality": 0.0,
            "singular_values": np.array([])
        }

    try:
        U, S, Vt = svd(H, full_matrices=False)
    except Exception as e:
        logger.warning(f"SVD failed: {e}; returning defaults.")
        return {
            "qualities": {},
            "best_cvp": None,
            "best_quality": 0.0,
            "singular_values": np.array([])
        }

    S = np.asarray(S)
    variances = S**2
    total_variance = np.sum(variances)
    if total_variance == 0:
        logger.warning("Total variance of singular values is zero. Explained variance set to zeros to avoid division by zero.")
        explained_variance = np.zeros_like(variances)
    else:
        explained_variance = variances / total_variance
    cumulative_explained_variance = np.cumsum(explained_variance)

    qualities = {}

    for cvp in cvp_to_test:
        # find smallest k such that cumulative explained variance >= cvp
        # if cumulative_explained_variance is empty, fallback to k = 1
        if cumulative_explained_variance.size == 0:
            k = 1
        else:
            k = int(np.argmax(cumulative_explained_variance >= cvp) + 1)
            # ensure at least one component is selected
            if k <= 0:
                k = 1
        # if S has fewer elements than k, adjust k
        if k > len(S):
            k = len(S) if len(S) > 0 else 1

        # reconstruct and compute quality, guarding ecg_quality against errors (including division by zero inside that function)
        try:
            R = np.dot(U[:, :k] * S[:k], Vt[:k, :])
            reconstructed_signal = diagonal_averaging(R)
            # ecg_quality may raise errors for very short/invalid signals, handle gracefully
            try:
                q = nk.ecg_quality(reconstructed_signal, sampling_rate=sampling_rate)
                quality_mean = float(np.nanmean(q)) if q is not None else 0.0
            except Exception as e:
                logger.warning(f"ecg_quality failed for cvp={cvp}: {e}; using quality 0.0")
                quality_mean = 0.0
        except Exception as e:
            logger.warning(f"Reconstruction failed for cvp={cvp}: {e}; using quality 0.0")
            quality_mean = 0.0

        qualities[cvp] = quality_mean

    if len(qualities) == 0:
        logger.warning("No qualities were computed; returning defaults.")
        return {
            "qualities": {},
            "best_cvp": None,
            "best_quality": 0.0,
            "singular_values": S
        }

    best_cvp = max(qualities, key=qualities.get)
    best_quality = qualities[best_cvp]

    return {
        "qualities": qualities,
        "best_cvp": best_cvp,
        "best_quality": best_quality,
        "singular_values": S
    }


def get_signal_weights_and_qualities(
    signal_list: List[Dict[str, Any]],
    verbose: bool = False
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:

    all_quality_data = []
    quality_values = []

    # show progress bar when verbose is True
    iterator = tqdm(
        signal_list,
        desc="Calculating Weights",
        unit="ch",
        disable=not verbose
    )

    for signal_data in iterator:
        # guard access to expected key
        segment = signal_data.get('segment') if isinstance(signal_data, dict) else None
        quality_data = signal_quality(segment) if segment is not None else {
            "qualities": {},
            "best_cvp": None,
            "best_quality": 0.0,
            "singular_values": np.array([])
        }

        all_quality_data.append(quality_data)
        # ensure best_quality is numeric
        best_q = quality_data.get('best_quality', 0.0)
        if best_q is None or (isinstance(best_q, float) and np.isnan(best_q)):
            best_q = 0.0
        quality_values.append(best_q)

    quality_array = np.array(quality_values, dtype=float)

    sum_quality = np.sum(quality_array)
    if sum_quality == 0:
        if verbose:
            logger.warning("Total quality is zero. Returning uniform weights.")
        # avoid division by zero when length is zero
        if quality_array.size == 0:
            weights = np.array([])
        else:
            weights = np.ones_like(quality_array, dtype=float) / float(quality_array.size)
    else:
        weights = quality_array / float(sum_quality)

    return weights, all_quality_data


def get_tucker_rank(quality_results: List[Dict[str, Any]]) -> int:
    ranks = []

    for quality_data in quality_results:
        S = np.asarray(quality_data.get('singular_values', np.array([])))
        best_cvp = quality_data.get('best_cvp')
        if best_cvp is None:
            best_cvp = 0.0
        # ensure numeric type for cvp
        try:
            best_cvp = float(best_cvp)
        except Exception:
            best_cvp = 0.0

        variances = S**2
        sum_variances = np.sum(variances)
        if sum_variances == 0 or variances.size == 0:
            explained_variance = np.zeros_like(variances)
        else:
            explained_variance = variances / sum_variances
        cumulative_variance = np.cumsum(explained_variance)

        # find minimal k such that cumulative_variance >= best_cvp
        if cumulative_variance.size == 0:
            k = 1
        else:
            k = int(np.searchsorted(cumulative_variance, best_cvp) + 1)
            if k <= 0:
                k = 1
        if k > len(S):
            k = len(S) if len(S) > 0 else 1
        ranks.append(k)

    if len(ranks) == 0:
        # fallback rank if no data was provided
        return 1

    return int(np.round(np.mean(ranks)))
