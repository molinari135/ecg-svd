import typer
import sys
import time
import json
import neurokit2 as nk
import torch
import tensorly as tl
import numpy as np
from loguru import logger
from typing import List
from scipy.signal.windows import hann
from scipy.linalg import hankel
from pathlib import Path
import warnings

from ecg_svd.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from ecg_svd.src.data_io import get_edf_reader, get_signal_segment, close_edf_reader
from ecg_svd.src.decomposition import run_parafac, reconstruct_channels_torch
from ecg_svd.src.metrics import get_classification_report, get_signal_weights


# set pytorch for tensorly and GPU optimization
tl.set_backend('pytorch')


app = typer.Typer(help="Runs PARAFAC Decomposition on a 3D Hankel tensor using a sliding window and GPU.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channels: List[int] = [1, 2, 3, 4],
    gt_channel: int = 0,
    segment_length: float = 5.0,
    overlap: float = 0.5,
    window_length: int = 625 * 2,
    parafac_rank: int = 4,
):
    edf_path = RAW_DATA_DIR / filename
    start_time = time.time()

    # setup cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    warnings.filterwarnings("ignore", category=UserWarning, module='torch')

    try:
        # initialization and data loading
        edf = get_edf_reader(edf_path)

        full_gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=300)
        full_gt_onsets = full_gt_data['onsets']
        total_duration = full_gt_data['time'][-1]
        sampling_rate = full_gt_data['sampling_rate']

        signal_data_ref = get_signal_segment(edf, ch_number=target_channels[-1], end_time=total_duration)
        signal_full_len = len(signal_data_ref['segment'])

        logger.info(f"Full signal length: {signal_full_len}")

        full_mecg = torch.zeros(signal_full_len, device=device, dtype=torch.float32)
        full_fecg = torch.zeros_like(full_mecg, device=device)
        full_noise = torch.zeros_like(full_mecg, device=device)
        weights_final = torch.zeros_like(full_mecg, device=device)

        # compute channel weights
        initial_segments_data = [get_signal_segment(edf, ch_number=ch, end_time=segment_length) for ch in target_channels]
        weights_np = get_signal_weights(initial_segments_data)
        weights = torch.tensor(weights_np, device=device, dtype=torch.float32)
        logger.info(f"Calculated channel weights: {weights_np}")

        L = window_length
        rank_parafac = parafac_rank

        curr_start = 0.0
        segment_step = segment_length - overlap
        segment_count = 0

        # slidind window
        while curr_start < total_duration:
            curr_end = min(curr_start + segment_length, total_duration)
            segment_count += 1
            logger.debug(f"Processing segment {segment_count}: {curr_start:.2f}s to {curr_end:.2f}s")

            hankels = []

            # Hankel tensor
            for ch in target_channels:
                segment = get_signal_segment(edf, ch_number=ch, start_time=curr_start, end_time=curr_end)['segment']

                segment_len = len(segment)
                if segment_len < L:
                    logger.warning(f"Segment length ({segment_len}) < window length ({L}). Stopping.")
                    break

                H_np = hankel(segment[:L], segment[L - 1:])
                hankels.append(torch.tensor(H_np, device=device, dtype=torch.float32))

            if len(hankels) < len(target_channels):
                break  # interrupt if channel is uncomplete

            segment_tensor = torch.stack(hankels, dim=2).to(device)
            L_curr, K, C = segment_tensor.shape

            # parafac
            cp_tensor = run_parafac(segment_tensor, rank=rank_parafac)
            parafac_weights, factors = cp_tensor
            U_L, U_K, U_C = factors

            # component reconstruction
            components_list = []
            for r in range(rank_parafac):
                outer_LK = torch.outer(U_L[:, r], U_K[:, r])
                comp = (parafac_weights[r] * outer_LK.unsqueeze(2) * U_C[None, None, r])
                components_list.append(comp)

            # components shape: (L, K, C, rank_parafac)
            components = torch.stack(components_list, dim=-1)

            # select subspaces
            S_mecg = torch.sum(components[:, :, :, [0]], dim=-1)
            S_fecg = torch.sum(components[:, :, :, [1, 2]], dim=-1)
            H_noise = segment_tensor - (S_mecg + S_fecg)

            mECG_signals = reconstruct_channels_torch(S_mecg).to(device)
            fECG_signals = reconstruct_channels_torch(S_fecg).to(device)
            noise_signals = reconstruct_channels_torch(H_noise).to(device)

            mECG_seg = torch.sum(mECG_signals * weights[None, :], dim=1)
            fECG_seg = torch.sum(fECG_signals * weights[None, :], dim=1)
            noise_seg = torch.sum(noise_signals * weights[None, :], dim=1)

            window = torch.tensor(hann(len(mECG_seg)), device=device, dtype=torch.float32)
            start_idx = int(curr_start * sampling_rate)
            segment_rec_len = len(mECG_seg)

            end_idx = min(start_idx + segment_rec_len, full_mecg.size(0))
            w = window[:end_idx - start_idx]

            mECG_seg_final = mECG_seg[:len(w)]
            fECG_seg_final = fECG_seg[:len(w)]
            noise_seg_final = noise_seg[:len(w)]

            # weighted sum
            full_mecg[start_idx:end_idx] += mECG_seg_final * w
            full_fecg[start_idx:end_idx] += fECG_seg_final * w
            full_noise[start_idx:end_idx] += noise_seg_final * w
            weights_final[start_idx:end_idx] += w

            # next segment
            curr_start += segment_step

        logger.info(f"Finished processing {segment_count} segments.")

        # final normalization
        full_mecg /= torch.maximum(weights_final, torch.tensor(1e-8, device=device))  # periodicity
        full_fecg /= torch.maximum(weights_final, torch.tensor(1e-8, device=device))  # mECG
        full_noise /= torch.maximum(weights_final, torch.tensor(1e-8, device=device))  # fECG
        fecg_to_test = full_noise.cpu().numpy()

        # metrics computation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice.", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
            _, info = nk.ecg_peaks(fecg_to_test, sampling_rate=sampling_rate, correct_artifacts=True)

        fecg_peaks_seconds = info.get('ECG_R_Peaks', []) / sampling_rate
        report = get_classification_report(full_gt_onsets, fecg_peaks_seconds)

        logger.success(f"Experiment completed. Final fECG Accuracy: {report['accuracy']:.2f}%")

        elapsed_time = time.time() - start_time
        experiment_name = Path(sys.argv[0]).stem
        data_to_save = {
            'mecg_combined_full': full_mecg.cpu().numpy(),
            'fecg_combined_selected_full': full_fecg.cpu().numpy(),
            'fecg_combined_residual_full': full_noise.cpu().numpy(),
            'sampling_rate': sampling_rate
        }

        experiment_report = {
            "experiment_id": experiment_name,
            "execution_time_seconds": elapsed_time,
            "filename": filename,
            "target_channels": target_channels,
            "segment_length": segment_length,
            "overlap": overlap,
            "window_length": window_length,
            "parafac_rank": parafac_rank,
            "weights_channels": weights_np.tolist(),
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
        logger.error(f"An error occurred during the PARAFAC GPU experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
