import typer
import time
import sys
import numpy as np
import neurokit2 as nk
import tensorly as tl
from pathlib import Path
from loguru import logger
from typing import List, Tuple
from tqdm import tqdm

from ecg_svd.config import RAW_DATA_DIR
from ecg_svd.data.io import get_edf_reader, close_edf_reader, save_results
from ecg_svd.data.preprocessing import get_signal_segment, create_segment_tensor
from ecg_svd.methods.tensor import run_tucker
from ecg_svd.methods.common import reconstruct_channels, create_hankel_matrix
from ecg_svd.evaluation.metrics import get_classification_report, get_signal_weights_and_qualities, get_tucker_rank

tl.set_backend('numpy')

app = typer.Typer(help="Runs Tucker Decomposition on the 3D Hankel tensor for source separation.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channels: List[int] = [1, 2, 3, 4],
    gt_channel: int = 0,
    segment_duration: float = 5.0,
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
    TOTAL_STEPS = 6

    try:
        # initialize progress bar
        with tqdm(total=TOTAL_STEPS, disable=verbose, unit="step", desc="Initializing") as pbar:

            # --- STEP 1: Loading & Preprocessing ---
            pbar.set_description("Loading Data")
            edf = get_edf_reader(edf_path)

            gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=segment_duration)
            gt_onsets = gt_data['onsets']
            sampling_rate = gt_data['sampling_rate']

            segments_data = []
            for ch in target_channels:
                sig = get_signal_segment(edf, ch_number=ch, end_time=segment_duration)
                # Z-score normalization
                sig['segment'] = (sig['segment'] - np.mean(sig['segment'])) / (np.std(sig['segment']) + 1e-8)
                segments_data.append(sig)

            hankel_matrices = [
                create_hankel_matrix(data['segment'], L_samples=window_length)
                for data in segments_data
            ]
            segment_tensor = create_segment_tensor(hankel_matrices)
            pbar.update(1)

            # --- STEP 2: Adaptive Parameters (Weights & Rank) ---
            pbar.set_description("Estimating Rank")

            weights, quality_results = get_signal_weights_and_qualities(segments_data, verbose=verbose)
            if verbose:
                logger.debug(f"Calculated channel weights: {weights}")

            tucker_rank = get_tucker_rank(quality_results)

            # Rank tuple: (Time, Trajectory, Channels)
            rank_tucker: Tuple[int, int, int] = (tucker_rank, tucker_rank, len(target_channels))
            if verbose:
                logger.debug(f"Tucker rank: {rank_tucker}")
            pbar.update(1)

            # --- STEP 3: Decomposition ---
            pbar.set_description("Running Tucker")
            core, factors = run_tucker(segment_tensor, rank=rank_tucker)
            U_L, U_K, U_C = factors
            pbar.update(1)

            # --- STEP 4: Reconstruction & Fusion ---
            pbar.set_description("Reconstructing Signals")
            S_mecg = np.zeros_like(core)
            S_mecg[:, :, 0] = core[:, :, 0]  # mECG is Component 0

            S_fecg = np.zeros_like(core)
            S_fecg[:, :, 1:] = core[:, :, 1:]  # fECG is Components 1+

            # reconstruct partial tensors
            H_mecg = tl.tucker_to_tensor((S_mecg, factors))
            H_fecg = tl.tucker_to_tensor((S_fecg, factors))
            H_noise = segment_tensor - H_mecg - H_fecg  # Residual

            # diagonal averaging
            mECG_signals_list = reconstruct_channels(H_mecg)
            fECG_signals_list = reconstruct_channels(H_fecg)
            noise_signals_list = reconstruct_channels(H_noise)

            # weighted combination
            mECG_combined = np.average(np.stack(mECG_signals_list, axis=-1), axis=-1, weights=weights)
            fECG_combined = np.average(np.stack(fECG_signals_list, axis=-1), axis=-1, weights=weights)
            noise_combined = np.average(np.stack(noise_signals_list, axis=-1), axis=-1, weights=weights)
            pbar.update(1)

            # --- STEP 5: Metrics ---
            pbar.set_description("Calculating Metrics")
            N_original = len(segments_data[0]['segment'])
            fecg_to_test = noise_combined[:N_original]

            try:
                _, info = nk.ecg_peaks(fecg_to_test, sampling_rate=sampling_rate, correct_artifacts=True)
                fecg_peaks_seconds = info.get('ECG_R_Peaks', []) / sampling_rate
            except Exception:
                fecg_peaks_seconds = np.array([])

            report = get_classification_report(gt_onsets, fecg_peaks_seconds)
            pbar.update(1)

            # --- STEP 6: Saving ---
            pbar.set_description("Saving Results")
            elapsed_time = time.time() - start_time
            experiment_name = Path(sys.argv[0]).stem

            data_to_save = {
                'mecg': mECG_combined,
                'fecg': fECG_combined,
                'noise': noise_combined,
            }

            experiment_report = {
                "experiment_id": experiment_name,
                "execution_time_seconds": elapsed_time,
                "target_channel": target_channels,
                "segment_duration": segment_duration,
                "window_length": window_length,
                "tucker_rank_L": int(rank_tucker[0]),
                "tucker_rank_K": int(rank_tucker[1]),
                "tucker_rank_C": int(rank_tucker[2]),
                "weights": weights.tolist(),
                "results": report
            }

            save_results(filename, experiment_name, data_to_save, experiment_report)

            pbar.update(1)
            pbar.set_description("Done")

        logger.success(f"fECG from {filename} extracted in {round(elapsed_time, 2)} seconds (accuracy: {report['accuracy']:.2f}%)")

    except Exception as e:
        logger.error(f"An error occurred during the Tucker experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
