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
from ecg_svd.src.decomposition import run_fastica, lower_peaks
from ecg_svd.src.metrics import get_classification_report

app = typer.Typer(help="Runs FastICA on all raw signals for Blind Source Separation.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channels: List[int] = [1, 2, 3, 4],
    gt_channel: int = 0,
    segment_duration: float = 5.0,
    n_sources: int = 3,  # mECG, fECG, noise
    mecg_component_idx: int = 2,  # assuming that the 1st component is mECG
    fecg_component_idx: int = 1,  # assuming that the 3rd component is fECG
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

        # load all segments and stack them into the input matrix
        signals_list = [
            get_signal_segment(edf, ch_number=ch, end_time=segment_duration)['segment']
            for ch in target_channels
        ]

        # stack segments: (channels, samples) -> transpose to (samples, channels) for ICA
        stacked_matrix = np.stack(signals_list, axis=0)
        X = stacked_matrix.T  # Input matrix X (N_samples, N_channels)

        logger.info(f"Input Matrix for ICA shape: {X.shape}. Running ICA with {n_sources} sources.")

        # FastICA decomposition
        # S_ (sources) shape: (N_samples, n_sources); A_ (mixing) shape: (N_channels, n_sources)
        S_sources, A_mixing = run_fastica(X, n_components=n_sources)

        # component analysis and cleaning
        # mECG component (assumed)
        mecg_component = S_sources[:, mecg_component_idx]

        # raw fECG component (assumed)
        fecg_component_raw = S_sources[:, fecg_component_idx]

        # identify mECG peaks from the assumed mECG component
        _, mecg_info = nk.ecg_peaks(mecg_component, sampling_rate=sampling_rate, correct_artifacts=True)
        mecg_peaks_indices = mecg_info.get('ECG_R_Peaks', [])

        logger.info(f"mECG peaks identified from component {mecg_component_idx + 1}: {len(mecg_peaks_indices)}")

        # clean the fECG component by suppressing the mECG peaks
        ica_fecg = lower_peaks(fecg_component_raw, peaks=mecg_peaks_indices, neighborhood=65)

        # metrics calculations
        # identify final fECG R-peaks (prediction)
        _, ica_fecg_info = nk.ecg_peaks(ica_fecg, sampling_rate=sampling_rate, correct_artifacts=True)

        fecg_peaks_seconds = ica_fecg_info.get('ECG_R_Peaks', []) / sampling_rate  # convert peaks to seconds

        report = get_classification_report(gt_onsets, fecg_peaks_seconds)
        logger.success(f"Experiment completed. Final Accuracy: {report['accuracy']:.2f}%")

        elapsed_time = time.time() - start_time
        experiment_name = Path(sys.argv[0]).stem
        data_to_save = {
            'input_matrix': stacked_matrix,
            'mecg': mecg_component,
            'fecg': ica_fecg,
            'sampling_rate': sampling_rate
        }

        npy_output_path = PROCESSED_DATA_DIR / f"{experiment_name}_signals.npy"
        np.save(npy_output_path, data_to_save)
        logger.info(f"Signals and Matrix saved to {npy_output_path}")

        # 3. Prepara il report per JSON
        experiment_report = {
            "experiment_id": experiment_name,
            "execution_time_seconds": elapsed_time,
            "filename": filename,
            "target_channels": target_channels,
            "segment_duration": segment_duration,
            "n_sources": n_sources,
            "mecg_component_idx": mecg_component_idx,
            "fecg_component_idx": fecg_component_idx,
            "results": report
        }

        json_output_path = REPORTS_DIR / f"{experiment_name}_report.json"
        with open(json_output_path, 'w') as f:
            json.dump(experiment_report, f, indent=4)
        logger.info(f"Report saved to {json_output_path}")

    except Exception as e:
        logger.error(f"An error occurred during the FastICA experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
