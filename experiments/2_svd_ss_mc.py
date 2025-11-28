import typer
import sys
import time
import numpy as np
import neurokit2 as nk
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import List

from ecg_svd.config import RAW_DATA_DIR
from ecg_svd.data.io import get_edf_reader, close_edf_reader, save_results
from ecg_svd.data.preprocessing import get_signal_segment
from ecg_svd.methods.common import lower_peaks
from ecg_svd.methods.matrix import run_ssa
from ecg_svd.evaluation.metrics import get_classification_report, get_signal_weights_and_qualities

app = typer.Typer(help="Runs SVD separation on multiple segments/channels with weighted summation.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channels: List[int] = [1, 2, 3, 4],
    gt_channel: int = 0,
    segment_duration: float = 5.0,
    mecg_cvp: float = 0.75,
    fecg_cvp: float = 0.95,
    window_length: int = 625 * 2,
    verbose: bool = False
):
    # configure logger
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="SUCCESS")

    edf_path = RAW_DATA_DIR / filename
    start_time = time.time()

    try:
        # --- Loading Data ---
        edf = get_edf_reader(edf_path)

        # load ground truth
        gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=segment_duration)
        gt_onsets = gt_data['onsets']
        sampling_rate = gt_data['sampling_rate']

        # load all segments
        segments_data = [
            get_signal_segment(edf, ch_number=ch, end_time=segment_duration)
            for ch in target_channels
        ]

        logger.info(f"Processing {len(target_channels)} channels...")

        # weight calculation
        weights, _ = get_signal_weights_and_qualities(segments_data)
        logger.info(f"Calculated normalized weights: {weights}")

        # iterative SVD separation
        mecg_list = []
        fecg_list = []

        # --- 2. Channel Processing Loop with TQDM ---
        channel_iterator = tqdm(
            enumerate(segments_data),
            total=len(segments_data),
            desc="Processing Channels",
            unit="ch",
            disable=verbose
        )

        for i, data in channel_iterator:
            segment = data['segment']
            segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)  # center data

            # mECG extraction
            mecg_i = run_ssa(segment, window_length=window_length, cvp=mecg_cvp)

            # identify MECG peaks
            _, mecg_info_i = nk.ecg_peaks(mecg_i, sampling_rate=sampling_rate, correct_artifacts=True)
            mecg_peaks_indices = mecg_info_i.get('ECG_R_Peaks', [])

            # fECG extraction
            cleaned_signal = lower_peaks(segment, peaks=mecg_peaks_indices)
            fecg_i = run_ssa(cleaned_signal, window_length=window_length, cvp=fecg_cvp)

            mecg_list.append(mecg_i)
            fecg_list.append(fecg_i)

            logger.debug(f"Channel {target_channels[i]} processed.")

        # signal combination
        # mecg_mean = np.mean(mecg_list, axis=0)
        fecg_mean = np.mean(fecg_list, axis=0)

        # weighted sum
        min_len = min(len(s) for s in fecg_list)
        fecg_stack = np.stack([s[:min_len] for s in fecg_list], axis=-1)
        mecg_stack = np.stack([s[:min_len] for s in mecg_list], axis=-1)

        mecg_weighted = np.average(mecg_stack, axis=-1, weights=weights)
        fecg_weighted = np.average(fecg_stack, axis=-1, weights=weights)

        _, info_mean = nk.ecg_peaks(fecg_mean, sampling_rate=sampling_rate, correct_artifacts=True)
        fecg_mean_peaks_sec = info_mean.get('ECG_R_Peaks', []) / sampling_rate
        report_mean = get_classification_report(gt_onsets, fecg_mean_peaks_sec)
        logger.info(f"Mean Sum Result Accuracy: {report_mean['accuracy']:.2f}%")

        _, info_weighted = nk.ecg_peaks(fecg_weighted, sampling_rate=sampling_rate, correct_artifacts=True)
        fecg_weighted_peaks_sec = info_weighted.get('ECG_R_Peaks', []) / sampling_rate
        report = get_classification_report(gt_onsets, fecg_weighted_peaks_sec)

        elapsed_time = time.time() - start_time
        experiment_name = Path(sys.argv[0]).stem

        data_to_save = {
            'mecg': mecg_weighted,
            'fecg': fecg_weighted
        }

        experiment_report = {
            "experiment_id": experiment_name,
            "execution_time_seconds": elapsed_time,
            "target_channel": target_channels,
            "segment_duration": segment_duration,
            "mecg_cvp": mecg_cvp,
            "fecg_cvp": fecg_cvp,
            "window_length": window_length,
            "results": report
        }

        save_results(filename, experiment_name, data_to_save, experiment_report)
        logger.success(f"fECG from {filename} extracted in {round(elapsed_time, 2)} seconds (accuracy: {report['accuracy']:.2f}%)")

    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
