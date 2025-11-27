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

    H = create_hankel_matrix(signal, L_samples=window_length)
    U, S, Vt = svd(H, full_matrices=False)

    variances = S**2
    total_variance = np.sum(variances)
    explained_variance = variances / total_variance
    cumulative_explained_variance = np.cumsum(explained_variance)

    qualities = {}

    for cvp in cvp_to_test:
        k = np.argmax(cumulative_explained_variance >= cvp) + 1
        R = np.dot(U[:, :k] * S[:k], Vt[:k, :])
        reconstructed_signal = diagonal_averaging(R)

        quality_mean = nk.ecg_quality(reconstructed_signal, sampling_rate=sampling_rate).mean()
        qualities[cvp] = quality_mean
        del R
        del reconstructed_signal

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

    iterator = tqdm(
        signal_list,
        desc="Calculating Weights",
        unit="ch",
        disable=verbose
    )

    for signal_data in iterator:
        quality_data = signal_quality(signal_data['segment'])

        all_quality_data.append(quality_data)
        quality_values.append(quality_data['best_quality'])

    quality_array = np.array(quality_values)

    sum_quality = np.sum(quality_array)
    if sum_quality == 0:
        if verbose:
            logger.warning("Total quality is zero. Returning uniform weights.")
        weights = np.ones_like(quality_array) / len(quality_array)
    else:
        weights = quality_array / sum_quality

    return weights, all_quality_data


def get_tucker_rank(quality_results: List[Dict[str, Any]]) -> int:
    ranks = []

    for quality_data in quality_results:
        S = quality_data['singular_values']
        best_cvp = quality_data['best_cvp']

        variances = S**2
        explained_variance = variances / np.sum(variances)
        cumulative_variance = np.cumsum(explained_variance)

        k = np.searchsorted(cumulative_variance, best_cvp) + 1
        ranks.append(k)

    return int(np.round(np.mean(ranks)))
