import typer
import sys
import time
import json
import numpy as np
import neurokit2 as nk
import tensorly as tl
from loguru import logger
from typing import List
from pathlib import Path

from ecg_svd.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from ecg_svd.data.io import get_edf_reader, close_edf_reader
from ecg_svd.data.preprocessing import get_signal_segment, create_segment_tensor
from ecg_svd.methods.common import reconstruct_channels, create_hankel_matrix
from ecg_svd.methods.tensor import run_parafac
from ecg_svd.evaluation.metrics import get_classification_report, get_signal_weights_and_qualities

tl.set_backend('numpy')

app = typer.Typer(help="Runs PARAFAC Decomposition on the 3D Hankel tensor for source separation.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channels: List[int] = [1, 2, 3, 4],
    gt_channel: int = 0,
    segment_duration: float = 5.0,
    window_length: int = 625 * 2,
    parafac_rank: int = 4,
):
    edf_path = RAW_DATA_DIR / filename
    start_time = time.time()

    try:
        # initialization and data loading
        edf = get_edf_reader(edf_path)

        gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=segment_duration)
        gt_onsets = gt_data['onsets']
        sampling_rate = gt_data['sampling_rate']

        segments_data = []
        for ch in target_channels:
            sig = get_signal_segment(edf, ch_number=ch, end_time=segment_duration)

            # center data
            sig['segment'] = (sig['segment'] - np.mean(sig['segment'])) / (np.std(sig['segment']) + 1e-8)
            segments_data.append(sig)

        # tensor and weights
        hankel_matrices = [
            create_hankel_matrix(data['segment'], L_samples=window_length)
            for data in segments_data
        ]
        segment_tensor = create_segment_tensor(hankel_matrices)

        weights, _ = get_signal_weights_and_qualities(segments_data)
        logger.info(f"Calculated channel weights: {weights}")
        logger.info(f"PARAFAC decomposition starting with rank: {parafac_rank}")

        # parafac
        cp_tensor = run_parafac(segment_tensor, rank=parafac_rank)
        parafac_weights, factors = cp_tensor
        U_L, U_K, U_C = factors  # time, trajectory, channel factor matrices

        # component reconstructions
        components_list = []
        for r in range(parafac_rank):
            comp = (parafac_weights[r] * np.outer(U_L[:, r], U_K[:, r])[:, :, None] * U_C[None, None, r])
            components_list.append(comp)

        # components shape: (L, K, C, rank)
        components = np.stack(components_list, axis=-1)

        # subspace selection
        S_mecg = np.sum(components[:, :, :, [0]], axis=-1)
        S_fecg = np.sum(components[:, :, :, [1, 2]], axis=-1)

        S_noise = segment_tensor - (S_mecg + S_fecg)
        logger.info(f"Components selected: mECG={np.array([0])}, fECG={np.array([1, 2])}")

        mECG_signals_list = reconstruct_channels(S_mecg)
        fECG_signals_list = reconstruct_channels(S_fecg)
        noise_signals_list = reconstruct_channels(S_noise)

        mECG_combined = np.sum(np.array(mECG_signals_list) * weights[:, None], axis=0)
        fECG_combined = np.sum(np.array(fECG_signals_list) * weights[:, None], axis=0)
        noise_combined = np.sum(np.array(noise_signals_list) * weights[:, None], axis=0)

        # metrics calculations
        N_original = len(segments_data[0]['segment'])
        fecg_to_test = noise_combined[:N_original]  # noise_combined is actually the fECG

        _, info = nk.ecg_peaks(fecg_to_test, sampling_rate=sampling_rate, correct_artifacts=True)
        fecg_peaks_seconds = info.get('ECG_R_Peaks', []) / sampling_rate
        report = get_classification_report(gt_onsets, fecg_peaks_seconds)

        logger.success(f"Experiment completed. Final fECG Accuracy: {report['accuracy']:.2f}%")

        elapsed_time = time.time() - start_time
        experiment_name = Path(sys.argv[0]).stem
        data_to_save = {
            'mecg_combined': mECG_combined,
            'fecg_combined_selected': noise_combined,
            'fecg_combined_residual': fECG_combined,
            'parafac_factors': factors,
            'sampling_rate': sampling_rate
        }

        experiment_report = {
            "experiment_id": experiment_name,
            "execution_time_seconds": elapsed_time,
            "filename": filename,
            "target_channels": target_channels,
            "segment_duration": segment_duration,
            "window_length": window_length,
            "parafac_rank": parafac_rank,
            "weights_channels": weights.tolist(),
            "selected_mecg_components": [0],
            "selected_fecg_components": [1, 2],
            "results": report
        }

        np.save(PROCESSED_DATA_DIR / f"{experiment_name}.npy", data_to_save)

        json_output_path = REPORTS_DIR / f"{experiment_name}.json"
        with open(json_output_path, 'w') as f:
            json.dump(experiment_report, f, indent=4)
        logger.info(f"Report saved to {json_output_path}")

    except Exception as e:
        logger.error(f"An error occurred during the PARAFAC experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
