import scipy
import typer
import sys
import time
import neurokit2 as nk
import torch
import tensorly as tl
import numpy as np
from loguru import logger
from typing import List, Dict, Any
from scipy.signal.windows import hann
from pathlib import Path
from tqdm import tqdm
import warnings

from ecg_svd.config import RAW_DATA_DIR
from ecg_svd.data.io import get_edf_reader, close_edf_reader, save_results
from ecg_svd.data.preprocessing import get_signal_segment
from ecg_svd.methods.tensor import run_tucker
from ecg_svd.methods.common import reconstruct_channels
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

    # setup cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device: {device}")
    warnings.filterwarnings("ignore", category=UserWarning, module='torch')
    
    try:
        if verbose:
            logger.debug("Initializing data...")
        edf = get_edf_reader(edf_path)

        full_gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=300)
        full_gt_onsets = full_gt_data['onsets']
        total_duration = full_gt_data['time'][-1]
        sampling_rate = full_gt_data['sampling_rate']

        signal_data_full = get_signal_segment(edf, ch_number=target_channels[-1], end_time=300)
        signal_full_len = len(signal_data_full['segment'])

        if verbose:
            logger.info(f"Full signal length: {signal_full_len}")

        # zero tensors
        full_mecg = torch.zeros(signal_full_len, device=device, dtype=torch.float32)
        full_fecg = torch.zeros_like(full_mecg, device=device)
        full_noise = torch.zeros_like(full_mecg, device=device)
        weights_final = torch.zeros_like(full_mecg, device=device)

        segments_data = [get_signal_segment(edf, ch_number=ch, end_time=segment_length) for ch in target_channels]
        
        for signal in segments_data:
            # apply high-pass filter to remove baseline wander
            lowpassed = scipy.ndimage.gaussian_filter1d(signal['segment'], sigma=0.2 * 1000, order=0)
            signal['segment'] = signal['segment'] - lowpassed

        weights_np, quality_results = get_signal_weights_and_qualities(segments_data, verbose=False)

        weights = torch.tensor(weights_np, device=device, dtype=torch.float32)
        tucker_rank = get_tucker_rank(quality_results)
        L = window_length
        rank_tucker = [tucker_rank, tucker_rank, len(target_channels)]

        curr_start = 0.0
        segment_step = segment_length - overlap
        segment_count = 0

        # estimate total segments for tqdm
        est_total_segments = int(np.ceil((total_duration - segment_length) / segment_step)) + 1

        # define Hann window
        L_seg_samples = int(segment_length * sampling_rate)
        window_cpu = hann(L_seg_samples)
        window_gpu_full = torch.tensor(window_cpu, device=device, dtype=torch.float32)

        # sliding window loop with tqdm
        with tqdm(total=est_total_segments, desc="Tucker GPU", unit="win", disable=verbose) as pbar:
            with torch.no_grad():
                while curr_start < total_duration:
                    curr_end = min(curr_start + segment_length, total_duration)

                    segments_list_gpu = []
                    current_seg_len = 0

                    for ch in target_channels:
                        seg_np = get_signal_segment(edf, ch_number=ch, start_time=curr_start, end_time=curr_end)['segment']
                        seg_t = torch.tensor(seg_np, device=device, dtype=torch.float32)

                        # center data
                        seg_t = (seg_t - seg_t.mean()) / (seg_t.std() + 1e-8)
                        segments_list_gpu.append(seg_t)

                        if ch == target_channels[-1]:
                            current_seg_len = seg_t.shape[0]

                    if current_seg_len < L:  # skip if segment smaller than Hankel window
                        break

                    segment_count += 1
                    if verbose:
                        logger.debug(f"Processing segment {segment_count}: {curr_start:.2f}s")

                    hankels = []
                    for seg_t in segments_list_gpu:
                        H = seg_t.unfold(0, L, 1).T
                        hankels.append(H)

                    segment_tensor = torch.stack(hankels, dim=2)

                    core, factors = run_tucker(segment_tensor, rank=rank_tucker)

                    # component selection
                    S_mecg = torch.zeros_like(core, device=device)
                    S_mecg[:, :, 0] = core[:, :, 0]

                    S_fecg = torch.zeros_like(core, device=device)
                    S_fecg[:, :, 1] = core[:, :, 1]

                    S_noise = torch.zeros_like(core, device=device)
                    if rank_tucker[1] > 2:
                        S_noise[:, :, 2:] = core[:, :, 2:]
                    else:
                        S_noise = core - S_mecg - S_fecg

                    # partial reconstruction
                    H_mecg = tl.tucker_to_tensor((S_mecg, factors))
                    H_fecg = tl.tucker_to_tensor((S_fecg, factors))
                    H_noise = tl.tucker_to_tensor((S_noise, factors))

                    # diagonal averaging
                    mECG_signals = reconstruct_channels(H_mecg, on_cuda=True)
                    fECG_signals = reconstruct_channels(H_fecg, on_cuda=True)
                    noise_signals = reconstruct_channels(H_noise, on_cuda=True)

                    # weighted combination
                    mECG_seg = torch.sum(mECG_signals * weights[None, :], dim=1)
                    fECG_seg = torch.sum(fECG_signals * weights[None, :], dim=1)
                    noise_seg = torch.sum(noise_signals * weights[None, :], dim=1)

                    if current_seg_len == L_seg_samples:
                        w = window_gpu_full
                    else:
                        w = window_gpu_full[:current_seg_len]

                    # overlap-add accumulation
                    start_idx = int(curr_start * sampling_rate)
                    end_idx = min(start_idx + current_seg_len, signal_full_len)
                    len_w = end_idx - start_idx
                    w_slice = w[:len_w]

                    full_mecg[start_idx:end_idx] += mECG_seg[:len_w] * w_slice
                    full_fecg[start_idx:end_idx] += fECG_seg[:len_w] * w_slice
                    full_noise[start_idx:end_idx] += noise_seg[:len_w] * w_slice
                    weights_final[start_idx:end_idx] += w_slice

                    # next segment
                    curr_start += segment_step
                    pbar.update(1)

        if verbose:
            logger.info(f"Finished processing {segment_count} segments.")

        # final normalization
        epsilon = 1e-8
        full_mecg /= torch.maximum(weights_final, torch.tensor(epsilon, device=device))
        full_fecg /= torch.maximum(weights_final, torch.tensor(epsilon, device=device))
        full_noise /= torch.maximum(weights_final, torch.tensor(epsilon, device=device))

        fecg_to_test = full_noise.cpu().numpy()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            _, info = nk.ecg_peaks(fecg_to_test, sampling_rate=sampling_rate, correct_artifacts=True)

        fecg_peaks_seconds = info.get('ECG_R_Peaks', []) / sampling_rate
        report = get_classification_report(full_gt_onsets, fecg_peaks_seconds)

        # saving results
        elapsed_time = time.time() - start_time
        experiment_name = Path(sys.argv[0]).stem

        data_to_save = {
            'mecg': full_fecg.cpu().numpy(),
            'fecg': full_noise.cpu().numpy(),
            'noise': full_mecg.cpu().numpy(),
        }

        experiment_report: Dict[str, Any] = {
            "experiment_id": experiment_name,
            "execution_time_seconds": elapsed_time,
            "target_channels": target_channels,
            "segment_length": segment_length,
            "overlap": overlap,
            "window_length": window_length,
            "tucker_rank_L": int(rank_tucker[0]),
            "tucker_rank_K": int(rank_tucker[1]),
            "tucker_rank_C": int(rank_tucker[2]),
            "weights": weights.cpu().tolist(),
            "results": report
        }

        save_results(filename, experiment_name, data_to_save, experiment_report)
        logger.success(f"Experiment completed in {round(elapsed_time, 2)}s (Accuracy: {report['accuracy']:.2f}%)")

    except Exception as e:
        logger.error(f"An error occurred during the Tucker GPU experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
