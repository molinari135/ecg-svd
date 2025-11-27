import typer
import sys
import time
import neurokit2 as nk
import numpy as np
from pathlib import Path
from loguru import logger

from ecg_svd.config import RAW_DATA_DIR
from ecg_svd.data.io import get_edf_reader, close_edf_reader, save_results
from ecg_svd.data.preprocessing import get_signal_segment
from ecg_svd.evaluation.metrics import get_classification_report


app = typer.Typer(help="Runs SVD-based separation on a single ECG segment.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channel: int = 0,
    gt_channel: int = 0,
    segment_duration: float = 5.0,
    verbose: bool = False
):

    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="SUCCESS")

    edf_path = RAW_DATA_DIR / filename
    start_time = time.time()

    try:
        # initialization and data loading
        edf = get_edf_reader(edf_path)

        # load ground truth and target signal (single segment)
        gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=segment_duration)
        target_data = get_signal_segment(edf, ch_number=target_channel, end_time=segment_duration)

        target_segment = target_data['segment']
        target_segment = (target_segment - np.mean(target_segment)) / np.std(target_segment + 1e-8)  # center data
        gt_onsets = gt_data['onsets']
        sampling_rate = target_data['sampling_rate']  # use sampling rate from segment_data

        # identify mECG peaks
        _, fecg_info = nk.ecg_peaks(target_segment, sampling_rate=sampling_rate, correct_artifacts=True)
        fecg_peaks_seconds = fecg_info.get('ECG_R_Peaks', []) / sampling_rate  # convert to seconds
        report = get_classification_report(gt_onsets, fecg_peaks_seconds)

        elapsed_time = time.time() - start_time
        experiment_name = Path(sys.argv[0]).stem
        data_to_save = {}
        experiment_report = {
            "execution_time_seconds": elapsed_time,
            "target_channel": target_channel,
            "segment_duration": segment_duration,
            "results": report
        }

        save_results(filename, experiment_name, data_to_save, experiment_report)
        logger.success(f"Experiment completed in {round(elapsed_time, 2)} seconds. Final fECG accuracy: {report['accuracy']:.2f}%")

    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
