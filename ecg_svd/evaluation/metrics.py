import warnings
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
from scipy.linalg import svd
from loguru import logger
from typing import Dict, Any, List, Tuple

from ecg_svd.methods.common import diagonal_averaging
from ecg_svd.methods.matrix import create_hankel_matrix


def get_classification_report(ground_truth: np.ndarray, prediction: np.ndarray, epsilon: float = 0.15) -> Dict[str, Any]:
    # no labels on ground truth or prediction
    if len(ground_truth) == 0 and len(prediction) == 0:
        logger.info("Both Ground Truth and Prediction arrays are empty. Returning 100% accuracy (Trivial case).")
        return {
            "TP": 0, "FN": 0, "FP": 0, "TN": 0,
            "accuracy": 100.0, "precision": 100.0, "recall": 100.0, "f1": 100.0
        }

    # empty prediction but non-empty ground truth
    if len(prediction) == 0:
        num_gt = len(ground_truth)
        logger.warning(f"Prediction array is empty. {num_gt} GT events missed.")

        return {
            "TP": 0,
            "FN": num_gt,
            "FP": 0,
            "TN": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    # empty ground truth but non-empty prediction
    if len(ground_truth) == 0:
        num_pred = len(prediction)
        logger.warning(f"Ground Truth array is empty. {num_pred} predictions are False Positives.")

        return {
            "TP": 0,
            "FN": 0,
            "FP": num_pred,
            "TN": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 100.0,
            "f1": 0.0
        }

    gt = ground_truth.copy()
    pred = prediction.copy()

    TP = 0
    FP = 0

    for gt_event in gt:
        # compute temporal differences
        time_diffs = np.abs(pred - gt_event)

        # find closest prediction
        if len(time_diffs) > 0:
            closest_idx = np.argmin(time_diffs)
            min_diff = time_diffs[closest_idx]

            if min_diff <= epsilon:
                TP += 1
                pred = np.delete(pred, closest_idx)

    FP = len(pred)
    FN = len(gt) - TP
    TN = 0

    precision_denom = TP + FP
    precision = (TP / precision_denom) * 100.0 if precision_denom > 0 else 0.0

    recall_denom = TP + FN
    recall = (TP / recall_denom) * 100.0 if recall_denom > 0 else 0.0

    f1_denom = precision + recall
    f1 = (2 * precision * recall) / f1_denom if f1_denom > 0 else 0.0

    accuracy = recall

    return {
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "TN": TN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
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

    # compute rank k for each cvp
    cvp_k_map = {}
    for cvp in cvp_to_test:
        # np.argmax returns the first index
        k = np.argmax(cumulative_explained_variance >= cvp) + 1
        cvp_k_map[cvp] = k

    # sort unique k values
    unique_k_sorted = sorted(list(set(cvp_k_map.values())))

    reconstructed_signals = {}
    current_k = 0
    current_reconstructed_signal = np.zeros_like(signal, dtype=S.dtype)

    # loop over unique ranks to incremetally reconstruct signals
    for k_next in unique_k_sorted:
        k_prev = current_k

        # compute rank difference
        k_diff = k_next - k_prev

        if k_diff <= 0:
            # skip if no new components to add
            current_k = k_next
            continue

        # partial reconstruction U[:, k_prev:k_next] * S[k_prev:k_next] @ Vt[k_prev:k_next, :]
        R_partial = np.dot(
            U[:, k_prev:k_next] * S[k_prev:k_next],
            Vt[k_prev:k_next, :]
        )

        partial_reconstructed_signal = diagonal_averaging(R_partial)
        # accumulate partial reconstructions
        current_reconstructed_signal += partial_reconstructed_signal
        reconstructed_signals[k_next] = current_reconstructed_signal.copy()
        current_k = k_next  # update starting rank for next iteration

    qualities = {}
    for cvp in cvp_to_test:
        k = cvp_k_map[cvp]
        reconstructed_signal = reconstructed_signals[k]
        quality_mean = 0.0

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                q_result = nk.ecg_quality(reconstructed_signal, sampling_rate=sampling_rate)
                quality_mean = q_result.mean()

                if np.isnan(quality_mean) or np.isinf(quality_mean):
                    quality_mean = 0.0

        except (ZeroDivisionError, TypeError) as e:
            if isinstance(e, TypeError) and "cannot convert float NaN to integer" in str(e):
                quality_mean = 0.0

        except Exception:
            quality_mean = 0.0

        qualities[cvp] = quality_mean

    if not qualities or all(np.isnan(list(qualities.values()))):
        best_cvp = None
        best_quality = np.nan
    else:
        best_cvp = max(qualities, key=qualities.get)
        best_quality = qualities[best_cvp]

    return {
        "qualities": qualities,
        "best_cvp": best_cvp,
        "best_quality": best_quality,
        "cumulative_explained_variance": cumulative_explained_variance
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
        cumulative_variance = quality_data['cumulative_explained_variance']
        best_cvp = quality_data['best_cvp']
        # np.searchsorted finds the index where best_cvp would fit to maintain order
        k = np.searchsorted(cumulative_variance, best_cvp) + 1
        ranks.append(k)
    return int(np.round(np.mean(ranks)))
