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
from ecg_svd.data.preprocessing import get_signal_segment, create_segment_tensor
from ecg_svd.methods.common import diagonal_averaging
from ecg_svd.methods.matrix import create_hankel_matrix
from ecg_svd.evaluation.metrics import get_classification_report

app = typer.Typer(help="Runs Multichannel Singular Spectrum Analysis (MSSA) on the unfolded Hankel tensor of single segments.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channels: List[int] = [1, 2, 3, 4],
    gt_channel: int = 0,
    segment_duration: float = 5.0,
    window_length: int = 625 * 2,
    mecg_cvp_threshold: float = 0.75,
    fecg_cvp_threshold: float = 0.95,
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
            segments_data = []

            # load and center all channels
            for ch in target_channels:
                sig = get_signal_segment(edf, ch_number=ch, end_time=segment_duration)['segment']
                segments_data.append(sig)
            pbar.update(1)

            # --- STEP 2: Tensor Unfolding ---
            pbar.set_description("Unfolding Tensor")
            hankel_matrices = [
                create_hankel_matrix(data, L_samples=window_length)
                for data in segments_data
            ]
            segment_tensor = create_segment_tensor(hankel_matrices)
            L, K, N_channels = segment_tensor.shape

            # Unfold: (L, K*C)
            H = segment_tensor.reshape(L, K * N_channels)
            logger.info(f"Unfolded matrix H shape: {H.shape}")
            pbar.update(1)

            # --- STEP 3: SVD Decomposition ---
            pbar.set_description("Running SVD")
            U, S, Vt = np.linalg.svd(H, full_matrices=False)

            # rank determination
            var_explained = S**2 / np.sum(S**2)
            cumulative_var = np.cumsum(var_explained)
            k1 = np.argmax(cumulative_var >= mecg_cvp_threshold) + 1
            k2 = np.argmax(cumulative_var >= fecg_cvp_threshold) + 1
            k2 = max(k2, k1 + 1)

            logger.info(f"SVD Ranks selected: mECG (1..{k1}), fECG ({k1 + 1}..{k2})")
            pbar.update(1)

            # --- STEP 4: Reconstruction ---
            pbar.set_description("Reconstructing Signals")
            H_mecg_reco = U[:, :k1] @ np.diag(S[:k1]) @ Vt[:k1, :]
            H_fecg_reco = U[:, k1:k2] @ np.diag(S[k1:k2]) @ Vt[k1:k2, :]

            # reshape back to tensor: (L, K, N)
            fecg_tensor_reco = H_fecg_reco.reshape(L, K, N_channels)
            mecg_tensor_reco = H_mecg_reco.reshape(L, K, N_channels)

            fecg_signals_list = []
            mecg_signals_list = []

            # diagonal averaging per channel
            for ch_idx in range(N_channels):
                # fECG
                H_fecg_ch = fecg_tensor_reco[:, :, ch_idx]
                fecg_ch_signal = diagonal_averaging(H_fecg_ch)
                fecg_signals_list.append(fecg_ch_signal)

                # mECG
                H_mecg_ch = mecg_tensor_reco[:, :, ch_idx]
                mecg_ch_signal = diagonal_averaging(H_mecg_ch)
                mecg_signals_list.append(mecg_ch_signal)

            # Weighted mean could be used here, but simple mean is standard for MSSA
            fecg_combined = np.mean(fecg_signals_list, axis=0)
            mecg_combined = np.mean(mecg_signals_list, axis=0)
            pbar.update(1)

            # --- STEP 5: Metrics & Saving ---
            pbar.set_description("Saving Results")
            N_original = len(segments_data[0])
            fecg_to_test = fecg_combined[:N_original]

            _, info = nk.ecg_peaks(fecg_to_test, sampling_rate=sampling_rate, correct_artifacts=True)
            fecg_peaks_seconds = info.get('ECG_R_Peaks', []) / sampling_rate
            report = get_classification_report(gt_onsets, fecg_peaks_seconds)

            elapsed_time = time.time() - start_time
            experiment_name = Path(sys.argv[0]).stem

            data_to_save = {
                'mecg': mecg_combined,
                'fecg': fecg_combined,
            }

            experiment_report = {
                "experiment_id": experiment_name,
                "execution_time_seconds": elapsed_time,
                "target_channels": target_channels,
                "segment_duration": segment_duration,
                "window_length": window_length,
                "mecg_cvp_threshold": mecg_cvp_threshold,
                "fecg_cvp_threshold": fecg_cvp_threshold,
                "k1_rank_mecg": int(k1),
                "k2_rank_fecg": int(k2),
                "results": report
            }

            save_results(filename, experiment_name, data_to_save, experiment_report)
            pbar.update(1)
            pbar.set_description("Done")
        logger.success(f"fECG from {filename} extracted in {round(elapsed_time, 2)} seconds (Accuracy: {report['accuracy']:.2f}%)")

    except Exception as e:
        logger.error(f"An error occurred during the MSSA experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
