import typer
import sys
import time
import numpy as np
import neurokit2 as nk
import tensorly as tl
from pathlib import Path
from loguru import logger
from typing import List
from tqdm import tqdm

from ecg_svd.config import RAW_DATA_DIR
from ecg_svd.data.io import get_edf_reader, close_edf_reader, save_results
from ecg_svd.data.preprocessing import get_signal_segment, create_segment_tensor
from ecg_svd.methods.tensor import run_parafac
from ecg_svd.methods.common import reconstruct_channels, create_hankel_matrix
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

            # --- STEP 2: Weights Calculation ---
            pbar.set_description("Calculating Weights")
            weights, _ = get_signal_weights_and_qualities(segments_data, verbose=verbose)
            if verbose:
                logger.debug(f"Calculated channel weights: {weights}")
            if verbose:
                logger.debug(f"PARAFAC rank: {parafac_rank}")
            pbar.update(1)

            # --- STEP 3: Decomposition ---
            pbar.set_description("Running PARAFAC")
            cp_tensor = run_parafac(segment_tensor, rank=parafac_rank)
            parafac_weights, factors = cp_tensor
            U_L, U_K, U_C = factors
            pbar.update(1)

            # --- STEP 4: Reconstruction & Fusion ---
            pbar.set_description("Reconstructing Signals")
            components_list = []
            for r in range(parafac_rank):
                comp = (parafac_weights[r] * np.outer(U_L[:, r], U_K[:, r])[:, :, None] * U_C[None, None, r])
                components_list.append(comp)

            # components shape: (L, K, C, rank)
            components = np.stack(components_list, axis=-1)

            # subspace selection
            S_mecg = np.sum(components[:, :, :, [0]], axis=-1)  # Component 0 = mECG
            S_fecg = np.sum(components[:, :, :, [1, 2]], axis=-1)  # Components 1, 2 = fECG
            S_noise = segment_tensor - (S_mecg + S_fecg)  # Residual

            if verbose:
                logger.debug("Selected components: mECG=[0], fECG=[1, 2]")

            # diagonal averaging
            mECG_signals_list = reconstruct_channels(S_mecg)
            fECG_signals_list = reconstruct_channels(S_fecg)
            noise_signals_list = reconstruct_channels(S_noise)

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
                'noise': noise_combined,  # NOTE fECG
            }

            experiment_report = {
                "experiment_id": experiment_name,
                "execution_time_seconds": elapsed_time,
                "target_channels": target_channels,
                "segment_duration": segment_duration,
                "window_length": window_length,
                "parafac_rank": parafac_rank,
                "weights_channels": weights.tolist(),
                "selected_mecg_components": [0],
                "selected_fecg_components": [1, 2],
                "results": report
            }

            save_results(filename, experiment_name, data_to_save, experiment_report)
            pbar.update(1)
            pbar.set_description("Done")

        logger.success(f"fECG from {filename} extracted in {round(elapsed_time, 2)} seconds (accuracy: {report['accuracy']:.2f}%)")

    except Exception as e:
        logger.error(f"An error occurred during the PARAFAC experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
