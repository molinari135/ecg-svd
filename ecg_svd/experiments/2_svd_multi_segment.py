import typer
import sys
import json
import time
import numpy as np
import neurokit2 as nk
from loguru import logger
from typing import List
from pathlib import Path

from ecg_svd.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from ecg_svd.src.data_io import get_edf_reader, get_signal_segment, close_edf_reader
from ecg_svd.src.decomposition import hankel_with_svd, lower_peaks
from ecg_svd.src.metrics import get_classification_report, get_signal_weights_and_qualities

app = typer.Typer(help="Runs SVD separation on multiple segments/channels with weighted summation.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channels: List[int] = [1, 2, 3, 4],
    gt_channel: int = 0,
    segment_duration: float = 5.0,
    mecg_cvp: float = 0.75,
    fecg_cvp: float = 0.95,
    window_length: int = 625 * 2
):
    edf_path = RAW_DATA_DIR / filename
    start_time = time.time()

    try:
        # initialization and data loading
        edf = get_edf_reader(edf_path)

        # load ground truth
        gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=segment_duration)
        gt_onsets = gt_data['onsets']
        sampling_rate = gt_data['sampling_rate']

        # load all segments (required for processing and weight calculation)
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

        for i, data in enumerate(segments_data):
            segment = data['segment']

            # mECG extraction
            mecg_i = hankel_with_svd(segment, window_length=window_length, cvp=mecg_cvp)

            # identify MECG peaks
            _, mecg_info_i = nk.ecg_peaks(mecg_i, sampling_rate=sampling_rate, correct_artifacts=True)
            mecg_peaks_indices = mecg_info_i.get('ECG_R_Peaks', [])

            # fECG extraction
            cleaned_signal = lower_peaks(segment, peaks=mecg_peaks_indices)
            fecg_i = hankel_with_svd(cleaned_signal, window_length=window_length, cvp=fecg_cvp)

            mecg_list.append(mecg_i)
            fecg_list.append(fecg_i)

            logger.debug(f"Channel {target_channels[i]} processed.")

        # signal combination
        # mecg_mean = np.mean(mecg_list, axis=0)
        fecg_mean = np.mean(fecg_list, axis=0)

        # weighted sum
        mecg_weighted = np.average(np.stack(mecg_list, axis=-1), axis=-1, weights=weights)
        fecg_weighted = np.average(np.stack(fecg_list, axis=-1), axis=-1, weights=weights)

        # metrics calculations

        # --- simple mean ---
        _, info_mean = nk.ecg_peaks(fecg_mean, sampling_rate=sampling_rate, correct_artifacts=True)
        fecg_mean_peaks_sec = info_mean.get('ECG_R_Peaks', []) / sampling_rate
        report_mean = get_classification_report(gt_onsets, fecg_mean_peaks_sec)
        logger.info(f"Mean Sum Result Accuracy: {report_mean['accuracy']:.2f}%")

        # --- weighted sum ---
        _, info_weighted = nk.ecg_peaks(fecg_weighted, sampling_rate=sampling_rate, correct_artifacts=True)
        fecg_weighted_peaks_sec = info_weighted.get('ECG_R_Peaks', []) / sampling_rate
        report_weighted = get_classification_report(gt_onsets, fecg_weighted_peaks_sec)
        logger.success(f"Weighted Sum Result Accuracy: {report_weighted['accuracy']:.2f}%")

        elapsed_time = time.time() - start_time
        experiment_name = Path(sys.argv[0]).stem
        data_to_save = {
            'original_segment': segments_data,
            'mecg': mecg_weighted,
            'fecg': fecg_weighted
        }

        experiment_report = {
            "experiment_id": experiment_name,
            "execution_time_seconds": elapsed_time,
            "filename": filename,
            "target_channel": target_channels,
            "segment_duration": segment_duration,
            "mecg_cvp": mecg_cvp,
            "fecg_cvp": fecg_cvp,
            "window_length": window_length,
            "results": report_weighted
        }

        np.save(PROCESSED_DATA_DIR / f"{experiment_name}.npy", data_to_save)

        json_output_path = REPORTS_DIR / f"{experiment_name}.json"
        with open(json_output_path, 'w') as f:
            json.dump(experiment_report, f, indent=4)
        logger.info(f"Report saved to {json_output_path}")

    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
