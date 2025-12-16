import warnings
import typer
import sys
import time
import neurokit2 as nk
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import List, Dict, Any
from scipy.signal.windows import hann

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
    overlap: float = 0.5,
    n_sources: int = 3,
    mecg_component_idx: int = 0,  # 0 if using deflation (default), 2 otherwise
    fecg_component_idx: int = 1,  # same for deflation and parallel
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

    try:
        # data loading
        if verbose:
            logger.debug("Initializing EDF Reader...")
        edf = get_edf_reader(edf_path)

        full_gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=300)
        full_gt_onsets = full_gt_data['onsets']
        total_duration = full_gt_data['time'][-1]
        sampling_rate = full_gt_data['sampling_rate']

        full_signals_list = []
        for ch in target_channels:
            sig = get_signal_segment(edf, ch_number=ch, end_time=300)['segment']
            full_signals_list.append(sig)

        full_stacked_matrix = np.stack(full_signals_list, axis=0)
        full_length = full_stacked_matrix.shape[1]

        logger.info(f"Total duration: {total_duration:.1f}s. Full length: {full_length} samples. Channels: {len(target_channels)}")

        # overlap-add initialization
        combined_mecg = np.zeros(full_length)
        combined_fecg = np.zeros(full_length)
        weights = np.zeros(full_length)

        curr_start = 0.0
        segment_step = segment_duration - overlap

        L_seg_samples = int(segment_duration * sampling_rate)
        window_full = hann(L_seg_samples)

        est_total_segments = int(np.ceil((total_duration - segment_duration) / segment_step)) + 1
        segment_count = 0

        # sliding window loop with tqdm
        with tqdm(total=est_total_segments, desc="Processing Segments (ICA)", unit="win", disable=verbose) as pbar:
            while curr_start < total_duration:
                curr_end = min(curr_start + segment_duration, total_duration)

                start_sample = int(curr_start * sampling_rate)
                end_sample = int(curr_end * sampling_rate)

                stacked_matrix_segment = full_stacked_matrix[:, start_sample:end_sample]
                current_seg_len = stacked_matrix_segment.shape[1]

                if current_seg_len < int(segment_step * sampling_rate):
                    break

                X = stacked_matrix_segment.T
                X = X - np.mean(X, axis=0)

                segment_count += 1
                logger.debug(f"Processing segment {segment_count}: {curr_start:.2f}s to {curr_end:.2f}s")

                try:
                    S_sources, _ = run_fastica(X, n_components=n_sources)
                except Exception as e:
                    logger.warning(f"ICA failed on segment {segment_count}: {e}")
                    curr_start += segment_step
                    pbar.update(1)
                    continue

                mecg_component = S_sources[:, mecg_component_idx]
                fecg_component_raw = S_sources[:, fecg_component_idx]

                # mECG peak identification
                _, mecg_info = nk.ecg_peaks(mecg_component, sampling_rate=sampling_rate, correct_artifacts=True)
                mecg_peaks_indices = mecg_info.get('ECG_R_Peaks', [])

                # clean the fECG component
                ica_fecg = lower_peaks(fecg_component_raw, peaks=mecg_peaks_indices, neighborhood=65)
                if current_seg_len == L_seg_samples:
                    w = window_full
                else:
                    w = hann(current_seg_len)

                slice_len = min(current_seg_len, full_length - start_sample)
                w_slice = w[:slice_len]

                combined_mecg[start_sample:end_sample] += mecg_component[:slice_len] * w_slice
                combined_fecg[start_sample:end_sample] += ica_fecg[:slice_len] * w_slice
                weights[start_sample:end_sample] += w_slice

                # next segment
                curr_start += segment_step
                pbar.update(1)

        logger.info(f"Finished processing {segment_count} segments.")
        combined_mecg /= np.maximum(weights, 1e-8)
        combined_fecg /= np.maximum(weights, 1e-8)

        full_fecg = combined_fecg

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            _, info = nk.ecg_peaks(full_fecg, sampling_rate=sampling_rate, correct_artifacts=True)

        fecg_peaks_seconds = info.get('ECG_R_Peaks', []) / sampling_rate
        report = get_classification_report(full_gt_onsets, fecg_peaks_seconds)

        # saving results
        elapsed_time = time.time() - start_time
        experiment_name = Path(sys.argv[0]).stem

        data_to_save = {
            'mecg': combined_mecg,
            'fecg': combined_fecg,
        }

        experiment_report: Dict[str, Any] = {
            "experiment_id": experiment_name,
            "execution_time_seconds": elapsed_time,
            "segment_duration": segment_duration,
            "overlap": overlap,
            "n_sources": n_sources,
            "mecg_component_idx": mecg_component_idx,
            "fecg_component_idx": fecg_component_idx,
            "segment_count": segment_count,
            "results": report
        }

        save_results(filename, experiment_name, data_to_save, experiment_report)
        logger.success(f"fECG from {filename} extracted in {round(elapsed_time, 2)} seconds (Accuracy: {report['accuracy']:.2f}%)")

    except Exception as e:
        logger.error(f"An error occurred during the FastICA experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
