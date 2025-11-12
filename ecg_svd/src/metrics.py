import numpy as np
import neurokit2 as nk
import gc
from loguru import logger
from typing import Dict, Any, List

from .decomposition import hankel_with_svd


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

    return {
        "TP": int(TP),
        "FN": int(FN),
        "FP": int(FP),
        "TN": int(TN),
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100
    }


def extract_neurokit_r_peaks(signal: np.ndarray, sampling_rate: int = 1000) -> np.ndarray:
    # use the negative signal for MECG extraction if needed, but standard is positive for ECG peaks
    _, info = nk.ecg_peaks(signal, sampling_rate=sampling_rate, correct_artifacts=True)

    # convert R-peak indices to seconds
    r_peaks_seconds = info.get('ECG_R_Peaks', []) / sampling_rate
    return r_peaks_seconds


def signal_quality(
    signal: np.ndarray,
    sampling_rate: int = 1000,
    cvp_to_test: List[float] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    window_length: int = 625 * 2
) -> Dict[str, Any]:
    qualities = {}

    # iterate over CVP values
    for cvp in cvp_to_test:
        # use the composite decomposition function to get the reconstructed signal
        extracted_signal = hankel_with_svd(signal, window_length=window_length, cvp=cvp)

        # calculate quality using NeuroKit2
        # note: NeuroKit returns a quality index series; we take the mean.
        signal_quality = nk.ecg_quality(extracted_signal, sampling_rate=sampling_rate)
        qualities[cvp] = signal_quality.mean()

        del extracted_signal
        gc.collect()

    # find the CVP that yielded the best quality
    best_cvp = max(qualities, key=qualities.get)
    best_quality = qualities[best_cvp]

    return {
        "qualities": qualities,
        "best_cvp": best_cvp,
        "best_quality": best_quality
    }


def get_signal_weights(signal_list: List[Dict[str, Any]]) -> np.ndarray:
    quality = []

    for signal_data in signal_list:
        # calculate the best quality for each signal segment
        best_quality = signal_quality(signal_data['segment'])['best_quality']
        quality.append(best_quality)

    quality_array = np.array(quality)

    # return normalized weights
    if np.sum(quality_array) == 0:
        logger.warning("Total quality is zero. Returning uniform weights.")
        return np.ones_like(quality_array) / len(quality_array)

    return quality_array / np.sum(quality_array)
