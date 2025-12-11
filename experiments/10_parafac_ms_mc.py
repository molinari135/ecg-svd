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
from ecg_svd.methods.tensor import run_parafac
from ecg_svd.methods.common import reconstruct_channels
from ecg_svd.evaluation.metrics import get_classification_report


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
        # initialization and data loading
        if verbose:
            logger.debug("Initializing data...")
        edf = get_edf_reader(edf_path)

        full_gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=300)
        full_gt_onsets = full_gt_data['onsets']
        total_duration = full_gt_data['time'][-1]
        sampling_rate = full_gt_data['sampling_rate']

        signal_data_ref = get_signal_segment(edf, ch_number=target_channels[-1], end_time=300)
        signal_full_len = len(signal_data_ref['segment'])

        if verbose:
            logger.info(f"Full signal length: {signal_full_len}")

        full_mecg = torch.zeros(signal_full_len, device=device, dtype=torch.float32)
        full_fecg = torch.zeros_like(full_mecg, device=device)
        full_noise = torch.zeros_like(full_mecg, device=device)
        weights_final = torch.zeros_like(full_mecg, device=device)

        # initial_segments_data = [get_signal_segment(edf, ch_number=ch, end_time=segment_length) for ch in target_channels]
        # weights_np, _ = get_signal_weights_and_qualities(initial_segments_data, verbose=False)
        # weights = torch.tensor(weights_np, device=device, dtype=torch.float32)

        L = window_length
        parafac_rank = parafac_rank

        curr_start = 0.0
        segment_step = segment_length - overlap
        segment_count = 0

        # define Hann window once
        L_seg_samples = int(segment_length * sampling_rate)
        window_cpu = hann(L_seg_samples)
        window_gpu_full = torch.tensor(window_cpu, device=device, dtype=torch.float32)

        # estimate total segments for tqdm
        est_total_segments = int(np.ceil((total_duration - segment_length) / segment_step)) + 1

        # sliding window loop with tqdm
        with tqdm(total=est_total_segments, desc="PARAFAC GPU", unit="win", disable=verbose) as pbar:
            with torch.no_grad():
                while curr_start < total_duration:
                    curr_end = min(curr_start + segment_length, total_duration)

                    # load segments to GPU
                    segments_list_gpu = []
                    for ch in target_channels:
                        seg_np = get_signal_segment(edf, ch_number=ch, start_time=curr_start, end_time=curr_end)['segment']
                        seg_t = torch.tensor(seg_np, device=device, dtype=torch.float32)
                        segments_list_gpu.append(seg_t)

                    current_seg_len = segments_list_gpu[0].shape[0]
                    if current_seg_len < L:
                        break  # stop if segment too short

                    segment_count += 1
                    if verbose:
                        logger.debug(f"Processing segment {segment_count}: {curr_start:.2f}s")

                    hankels = []
                    for seg_t in segments_list_gpu:
                        H = seg_t.unfold(0, L, 1).T
                        hankels.append(H)

                    segment_tensor = torch.stack(hankels, dim=2)

                    # parafac
                    cp_tensor = run_parafac(segment_tensor, rank=parafac_rank)
                    parafac_weights, factors = cp_tensor
                    U_L, U_K, U_C = factors

                    # component reconstruction
                    components_list = []
                    for r in range(parafac_rank):
                        # Outer product: L x K x C (componente r)
                        outer_LK = torch.outer(U_L[:, r], U_K[:, r])
                        comp = (parafac_weights[r] * outer_LK.unsqueeze(2) * U_C[None, None, r])
                        components_list.append(comp)

                    # components shape: (L, K, C, rank_parafac)
                    components = torch.stack(components_list, dim=-1)

                    # select subspaces
                    S_mecg = torch.sum(components[:, :, :, [0]], dim=-1)
                    S_fecg = torch.sum(components[:, :, :, [1, 2]], dim=-1)
                    H_noise = segment_tensor - (S_mecg + S_fecg)

                    # diagonal averaging
                    mECG_signals = reconstruct_channels(S_mecg, on_cuda=True)
                    fECG_signals = reconstruct_channels(S_fecg, on_cuda=True)
                    noise_signals = reconstruct_channels(H_noise, on_cuda=True)

                    # weighted combination
                    mECG_seg = torch.sum(mECG_signals, dim=1)  # * weights[None, :]
                    fECG_seg = torch.sum(fECG_signals, dim=1)  # * weights[None, :]
                    noise_seg = torch.sum(noise_signals, dim=1)  # * weights[None, :]

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
            "parafac_rank": parafac_rank,
            # "weights_channels": weights.cpu().tolist(),
            "selected_mecg_components": [0],
            "selected_fecg_components": [1, 2],
            "results": report
        }

        save_results(filename, experiment_name, data_to_save, experiment_report)
        logger.success(f"fECG from {filename} extracted in {round(elapsed_time, 2)} seconds (Accuracy: {report['accuracy']:.2f}%)")

    except Exception as e:
        logger.error(f"An error occurred during the PARAFAC GPU experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
