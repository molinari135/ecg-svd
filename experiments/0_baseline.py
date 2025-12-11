import typer
import sys
import time
import neurokit2 as nk
from pathlib import Path
from loguru import logger
from tqdm import tqdm

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
    # configure logger
    logger.remove()
    if verbose:
        # show debug logs and hide progress bar
        logger.add(sys.stderr, level="DEBUG")
    else:
        # show only success/error and progress bar
        logger.add(sys.stderr, level="SUCCESS")

    edf_path = RAW_DATA_DIR / filename
    start_time = time.time()

    # number of steps for progress bar
    TOTAL_STEPS = 4

    try:
        # initialize progress bar
        with tqdm(total=TOTAL_STEPS, disable=verbose, unit="step", desc="Initializing") as pbar:

            # --- STEP 1: Data Loading ---
            pbar.set_description("Loading Data")
            edf = get_edf_reader(edf_path)

            gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=segment_duration)
            target_data = get_signal_segment(edf, ch_number=target_channel, end_time=segment_duration)
            target_segment = target_data['segment']

            gt_onsets = gt_data['onsets']
            sampling_rate = target_data['sampling_rate']
            pbar.update(1)  # 25%

            # --- STEP 2: Processing (Peak Detection) ---
            pbar.set_description("Processing Signal")
            _, fecg_info = nk.ecg_peaks(target_segment, sampling_rate=sampling_rate, correct_artifacts=True)
            fecg_peaks_seconds = fecg_info.get('ECG_R_Peaks', []) / sampling_rate
            pbar.update(1)  # 50%

            # --- STEP 3: Metrics ---
            pbar.set_description("Calculating Metrics")
            report = get_classification_report(gt_onsets, fecg_peaks_seconds)
            pbar.update(1)  # 75%

            # --- STEP 4: Saving ---
            pbar.set_description("Saving Results")
            elapsed_time = time.time() - start_time
            experiment_name = Path(sys.argv[0]).stem
            data_to_save = {}

            experiment_report = {
                "experiment_id": experiment_name,
                "execution_time_seconds": elapsed_time,
                "target_channel": target_channel,
                "segment_duration": segment_duration,
                "results": report
            }

            save_results(filename, experiment_name, data_to_save, experiment_report)
            pbar.update(1)  # 100%
            pbar.set_description("Done")

        logger.success(f"fECG from {filename} extracted in {round(elapsed_time, 2)} seconds (Accuracy: {report['accuracy']:.2f}%)")

    except Exception as e:
        logger.exception(f"An error occurred during the experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
