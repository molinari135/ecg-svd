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
from ecg_svd.methods.common import lower_peaks
from ecg_svd.methods.matrix import run_ssa
from ecg_svd.evaluation.metrics import get_classification_report


app = typer.Typer(help="Runs SVD-based separation on a single ECG segment on a single channel.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channel: int = 4,
    gt_channel: int = 0,
    segment_duration: float = 5.0,
    mecg_cvp: float = 0.75,
    fecg_cvp: float = 0.90,
    window_length: int = 625 * 2,
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
    TOTAL_STEPS = 6

    try:
        # initialize progress bar
        with tqdm(total=TOTAL_STEPS, disable=verbose, unit="step", desc="Initializing") as pbar:

            # --- STEP 1: Data Loading & Preprocessing ---
            pbar.set_description("Loading Data")
            edf = get_edf_reader(edf_path)

            # load ground truth and target signal (single segment)
            gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=segment_duration)
            target_data = get_signal_segment(edf, ch_number=target_channel, end_time=segment_duration)

            target_segment = target_data['segment']
            gt_onsets = gt_data['onsets']
            sampling_rate = target_data['sampling_rate']

            logger.info(f"Starting SVD separation on Channel {target_channel} ({segment_duration}s segment)...")
            pbar.update(1)

            # --- STEP 2: mECG Extraction ---
            pbar.set_description("Extracting mECG")
            mecg = run_ssa(target_segment, window_length=window_length, cvp=mecg_cvp)
            pbar.update(1)

            # --- STEP 3: Peak Detection & Suppression ---
            pbar.set_description("Suppressing mECG")
            _, mecg_info = nk.ecg_peaks(mecg, sampling_rate=sampling_rate, correct_artifacts=True)
            mecg_peaks_indices = mecg_info.get('ECG_R_Peaks', [])

            # safety check (only log if verbose, otherwise it breaks the bar)
            if len(mecg_peaks_indices) == 0:
                logger.warning("No mECG peaks detected! Peak suppression will be skipped.")

            # mECG peaks suppression
            cleaned_signal = lower_peaks(target_segment, peaks=mecg_peaks_indices)
            pbar.update(1)

            # --- STEP 4: fECG Extraction ---
            pbar.set_description("Extracting fECG")
            fecg = run_ssa(cleaned_signal, window_length=window_length, cvp=fecg_cvp)
            pbar.update(1)

            # --- STEP 5: Metrics ---
            pbar.set_description("Calculating Metrics")
            _, fecg_info = nk.ecg_peaks(fecg, sampling_rate=sampling_rate, correct_artifacts=True)

            fecg_peaks_seconds = fecg_info.get('ECG_R_Peaks', []) / sampling_rate  # convert to seconds
            report = get_classification_report(gt_onsets, fecg_peaks_seconds)
            pbar.update(1)

            # --- STEP 6: Saving ---
            pbar.set_description("Saving Results")
            elapsed_time = time.time() - start_time
            experiment_name = Path(sys.argv[0]).stem

            data_to_save = {
                'mecg': mecg,
                'fecg': fecg
            }

            experiment_report = {
                "experiment_id": experiment_name,
                "execution_time_seconds": elapsed_time,
                "target_channel": target_channel,
                "segment_duration": segment_duration,
                "mecg_cvp": mecg_cvp,
                "fecg_cvp": fecg_cvp,
                "window_length": window_length,
                "results": report
            }

            save_results(filename, experiment_name, data_to_save, experiment_report)
            pbar.update(1)
            pbar.set_description("Done")

        logger.success(f"fECG from {filename} extracted in {round(elapsed_time, 2)} seconds (Accuracy: {report['accuracy']:.2f}%)")
        edf.close()

    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
