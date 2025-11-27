import typer
import sys
import json
import time
import neurokit2 as nk
import torch
import tensorly as tl
import numpy as np
from loguru import logger
from typing import List
from scipy.signal.windows import hann
from pathlib import Path
import warnings

from ecg_svd.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from ecg_svd.data.io import get_edf_reader, close_edf_reader
from ecg_svd.data.preprocessing import get_signal_segment
from ecg_svd.methods.common import reconstruct_channels
from ecg_svd.methods.tensor import run_tucker
from ecg_svd.evaluation.metrics import get_classification_report, get_signal_weights_and_qualities, get_tucker_rank


# set pytorch for tensorly
tl.set_backend('pytorch')


app = typer.Typer(help="Runs Tucker HOSVD on a 3D Hankel tensor using a sliding window and GPU.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channels: List[int] = [1, 2, 3, 4],
    gt_channel: int = 0,
    segment_length: float = 5.0,
    overlap: float = 0.5,
    window_length: int = 625 * 2,
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

        signal_data_4 = get_signal_segment(edf, ch_number=target_channels[-1], end_time=300)
        signal_full_len = len(signal_data_4['segment'])

        logger.info(f"Full signal length: {signal_full_len}")

        # zero tensors
        full_mecg = torch.zeros(signal_full_len, device=device, dtype=torch.float32)
        full_fecg = torch.zeros_like(full_mecg, device=device)
        full_noise = torch.zeros_like(full_mecg, device=device)
        weights_final = torch.zeros_like(full_mecg, device=device)

        segments_data = [get_signal_segment(edf, ch_number=ch, end_time=segment_length) for ch in target_channels]
        weights_np, quality_results = get_signal_weights_and_qualities(segments_data)

        # move on GPU
        weights = torch.tensor(weights_np, device=device, dtype=torch.float32)

        # tucker rank
        tucker_rank = get_tucker_rank(quality_results)
        L = window_length
        rank_tucker = [tucker_rank, tucker_rank, len(target_channels)]

        curr_start = 0.0
        segment_step = segment_length - overlap
        segment_count = 0

        # define Hann window
        L_seg_samples = int(segment_length * sampling_rate)
        window_cpu = hann(L_seg_samples)
        window_gpu_full = torch.tensor(window_cpu, device=device, dtype=torch.float32)

        # sliding window
        with torch.no_grad():
            while curr_start < total_duration:
                curr_end = min(curr_start + segment_length, total_duration)
                segments_list_gpu = []
                for ch in target_channels:
                    seg_np = get_signal_segment(edf, ch_number=ch, start_time=curr_start, end_time=curr_end)['segment']
                    seg_t = torch.tensor(seg_np, device=device, dtype=torch.float32)

                    # center data
                    seg_t = (seg_t - seg_t.mean()) / (seg_t.std() + 1e-8)
                    segments_list_gpu.append(seg_t)

                current_seg_len = segments_list_gpu[0].shape[0]
                if current_seg_len < L:  # skip if segment smaller than Hankel window
                    break

                segment_count += 1
                if segment_count % 10 == 0:
                    logger.debug(f"Processing segment {segment_count}: {curr_start:.2f}s")

                hankels = []
                for seg_t in segments_list_gpu:
                    # unfold creates a sliding window view: (Batch, Channel, Time) -> (1, 1, N)
                    # unfold(dimension, size, step)
                    # PyTorch unfold gives: (N_windows, Window_Size) -> (K, L)
                    H = seg_t.unfold(0, L, 1).T
                    hankels.append(H)

                segment_tensor = torch.stack(hankels, dim=2).to(device)
                L_curr, K, C = segment_tensor.shape

                core, factors = run_tucker(segment_tensor, rank=rank_tucker)
                # core, factors = run_tucker(segment_tensor, rank=[64, 64, 4])
                U_L, U_K, U_C = factors

                # component selection
                S_mecg = torch.zeros_like(core, device=device)
                S_mecg[:, :, 0] = core[:, :, 0]

                S_fecg = torch.zeros_like(core, device=device)
                S_fecg[:, :, 1:] = core[:, :, 1:]

                # partial reconstruction
                H_mecg = tl.tucker_to_tensor((S_mecg, factors))
                H_fecg = tl.tucker_to_tensor((S_fecg, factors))
                H_noise = segment_tensor - (H_mecg + H_fecg)

                # diagonal averaging
                mECG_signals = reconstruct_channels(H_mecg, on_cuda=True).to(device)
                fECG_signals = reconstruct_channels(H_fecg, on_cuda=True).to(device)
                noise_signals = reconstruct_channels(H_noise, on_cuda=True).to(device)

                # weighted combination
                mECG_seg = torch.sum(mECG_signals * weights[None, :], dim=1)
                fECG_seg = torch.sum(fECG_signals * weights[None, :], dim=1)
                noise_seg = torch.sum(noise_signals * weights[None, :], dim=1)

                if current_seg_len == L_seg_samples:
                    w = window_gpu_full
                else:
                    w = window_gpu_full[:current_seg_len]

                # Overlap-Add accumulation
                start_idx = int(curr_start * sampling_rate)
                end_idx = start_idx + current_seg_len

                # safety clip
                end_idx = min(end_idx, signal_full_len)
                len_w = end_idx - start_idx
                w = w[:len_w]

                full_mecg[start_idx:end_idx] += mECG_seg[:len_w] * w
                full_fecg[start_idx:end_idx] += fECG_seg[:len_w] * w
                full_noise[start_idx:end_idx] += noise_seg[:len_w] * w
                weights_final[start_idx:end_idx] += w

                # next segment
                curr_start += segment_step

        logger.info(f"Finished processing {segment_count} segments.")

        # final normalization
        full_mecg /= torch.maximum(weights_final, torch.tensor(1e-8, device=device))  # periodicity
        full_fecg /= torch.maximum(weights_final, torch.tensor(1e-8, device=device))  # mECG
        full_noise /= torch.maximum(weights_final, torch.tensor(1e-8, device=device))  # fECG
        fecg_to_test = full_noise.cpu().numpy()

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
            'mecg': full_fecg.cpu().numpy(),
            'fecg': full_noise.cpu().numpy(),
            'noise': full_mecg.cpu().numpy(),
            'sampling_rate': sampling_rate
        }

        experiment_report = {
            "experiment_id": experiment_name,
            "execution_time_seconds": elapsed_time,
            "filename": filename,
            "target_channels": target_channels,
            "window_length": window_length,
            "tucker_rank_L": int(rank_tucker[0]),
            "tucker_rank_K": int(rank_tucker[1]),
            "tucker_rank_C": int(rank_tucker[2]),
            "weights": weights.tolist(),
            "results": report
        }

        np.save(PROCESSED_DATA_DIR / f"{experiment_name}.npy", data_to_save)

        json_output_path = REPORTS_DIR / f"{experiment_name}.json"
        with open(json_output_path, 'w') as f:
            json.dump(experiment_report, f, indent=4)
        logger.info(f"Report saved to {json_output_path}")

    except Exception as e:
        logger.error(f"An error occurred during the Tucker GPU experiment: {e}")
        # raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
