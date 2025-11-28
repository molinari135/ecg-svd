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
from ecg_svd.methods.matrix import run_fastica
from ecg_svd.evaluation.metrics import get_classification_report

app = typer.Typer(help="Runs FastICA on all raw signals segments for Blind Source Separation.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channels: List[int] = [1, 2, 3, 4],
    gt_channel: int = 0,
    segment_duration: float = 5.0,
    n_sources: int = 3,  # mECG, fECG, noise
    mecg_component_idx: int = 2,  # assuming that the 1st component is mECG
    fecg_component_idx: int = 1,  # assuming that the 3rd component is fECG
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
    TOTAL_STEPS = 5

    try:
        # initialize progress bar
        with tqdm(total=TOTAL_STEPS, disable=verbose, unit="step", desc="Initializing") as pbar:

            # --- STEP 1: Data Loading ---
            pbar.set_description("Loading Data")
            edf = get_edf_reader(edf_path)

            gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=segment_duration)
            gt_onsets = gt_data['onsets']
            sampling_rate = gt_data['sampling_rate']

            # load all segments
            signals_list = [
                get_signal_segment(edf, ch_number=ch, end_time=segment_duration)['segment']
                for ch in target_channels
            ]

            # stack segments: (channels, samples) -> transpose to (samples, channels) for ICA
            stacked_matrix = np.stack(signals_list, axis=0)
            X = stacked_matrix.T  # input matrix X (N_samples, N_channels)

            # center data (whitening is handled by FastICA)
            X = X - np.mean(X, axis=0)
            logger.info(f"Input Matrix for ICA shape: {X.shape}. Running ICA with {n_sources} sources.")
            pbar.update(1)

            # --- STEP 2: FastICA Decomposition ---
            pbar.set_description("Running FastICA")
            # S_ (sources) shape: (N_samples, n_sources); A_ (mixing) shape: (N_channels, n_sources)
            S_sources, A_mixing = run_fastica(X, n_components=n_sources)
            pbar.update(1)

            # --- STEP 3: Component Cleaning ---
            pbar.set_description("Processing Components")
            # mECG component (assumed/selected)
            mecg_component = S_sources[:, mecg_component_idx]
            # raw fECG component (assumed/selected)
            fecg_component_raw = S_sources[:, fecg_component_idx]
            # identify mECG peaks
            _, mecg_info = nk.ecg_peaks(mecg_component, sampling_rate=sampling_rate, correct_artifacts=True)
            mecg_peaks_indices = mecg_info.get('ECG_R_Peaks', [])
            logger.info(f"mECG peaks identified from component {mecg_component_idx + 1}: {len(mecg_peaks_indices)}")

            # clean the fECG component by suppressing the mECG peaks
            ica_fecg = lower_peaks(fecg_component_raw, peaks=mecg_peaks_indices, neighborhood=65)
            pbar.update(1)

            # --- STEP 4: Metrics ---
            pbar.set_description("Calculating Metrics")
            try:
                _, ica_fecg_info = nk.ecg_peaks(ica_fecg, sampling_rate=sampling_rate, correct_artifacts=True)
                fecg_peaks_seconds = ica_fecg_info.get('ECG_R_Peaks', []) / sampling_rate
            except Exception:
                fecg_peaks_seconds = np.array([])  # Handle failure case

            report = get_classification_report(gt_onsets, fecg_peaks_seconds)
            pbar.update(1)

            # --- STEP 5: Saving ---
            pbar.set_description("Saving Results")
            elapsed_time = time.time() - start_time
            experiment_name = Path(sys.argv[0]).stem

            data_to_save = {
                'mecg': mecg_component,
                'fecg': ica_fecg,
            }

            experiment_report = {
                "experiment_id": experiment_name,
                "execution_time_seconds": elapsed_time,
                "segment_duration": segment_duration,
                "n_sources": n_sources,
                "mecg_component_idx": mecg_component_idx,
                "fecg_component_idx": fecg_component_idx,
                "results": report
            }

            save_results(filename, experiment_name, data_to_save, experiment_report)

            pbar.update(1)
            pbar.set_description("Done")

        logger.success(f"fECG from {filename} extracted in {round(elapsed_time, 2)} seconds (accuracy: {report['accuracy']:.2f}%)")

    except Exception as e:
        logger.error(f"An error occurred during the FastICA experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
