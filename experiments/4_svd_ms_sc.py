import typer
import sys
import time
import neurokit2 as nk
import numpy as np
from pathlib import Path
from loguru import logger
from scipy.signal.windows import hann
from tqdm import tqdm
from typing import Dict, Any

from ecg_svd.config import RAW_DATA_DIR
from ecg_svd.data.io import get_edf_reader, close_edf_reader, save_results
from ecg_svd.data.preprocessing import get_signal_segment
from ecg_svd.methods.common import lower_peaks
from ecg_svd.methods.matrix import run_ssa
from ecg_svd.evaluation.metrics import get_classification_report

app = typer.Typer(help="Runs two-step SVD separation on a single channel of a single segment using a sliding window approach.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channel: int = 4,
    gt_channel: int = 0,
    segment_length: float = 5.0,
    overlap: float = 0.5,
    mecg_cvp: float = 0.75,
    fecg_cvp: float = 0.90,
    window_length_svd: int = 1250,
    verbose: bool = False
):
    # configure loader
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="SUCCESS")

    edf_path = RAW_DATA_DIR / filename
    start_time = time.time()

    try:
        # loading data
        if verbose:
            logger.debug("Initializing EDF Reader...")
        edf = get_edf_reader(edf_path)

        full_gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=300)
        full_target_data = get_signal_segment(edf, ch_number=target_channel, end_time=300)

        full_gt_onsets = full_gt_data['onsets']
        total_duration = full_gt_data['time'][-1]
        sampling_rate = full_gt_data['sampling_rate']
        full_length = len(full_target_data['segment'])

        logger.info(f"Total duration: {total_duration:.1f}s. Full length: {full_length} samples.")

        # overlap-add initialization
        combined_mecg = np.zeros(full_length)
        combined_fecg = np.zeros(full_length)
        weights = np.zeros(full_length)

        curr_start = 0.0
        segment_step = segment_length - overlap

        # define Hann window
        L_seg_samples = int(segment_length * sampling_rate)
        window_full = hann(L_seg_samples)

        # estimate total segments for the progress bar
        est_total_segments = int(np.ceil((total_duration - segment_length) / segment_step)) + 1
        segment_count = 0

        # liding window loop with tqdm
        with tqdm(total=est_total_segments, desc="Processing Segments", unit="win", disable=verbose) as pbar:
            while curr_start < total_duration:
                curr_end = min(curr_start + segment_length, total_duration)

                start_sample = int(curr_start * sampling_rate)
                end_sample = int(curr_end * sampling_rate)

                segment_signal = full_target_data['segment'][start_sample:end_sample]
                current_seg_len = len(segment_signal)

                # skip if segment is too short
                if current_seg_len < int(segment_step * sampling_rate):
                    break

                segment_count += 1
                logger.debug(f"Processing segment {segment_count}: {curr_start:.2f}s to {curr_end:.2f}s, length={current_seg_len}")

                # mECG extraction
                mecg = run_ssa(segment_signal, cvp=mecg_cvp, window_length=window_length_svd, on_cuda=True)

                if mecg is None or len(mecg) == 0 or np.all(np.isnan(mecg)):
                    if verbose:
                        logger.warning(f"Skipping segment {segment_count}: invalid mECG reconstruction.")
                    curr_start += segment_step
                    pbar.update(1)
                    continue

                mecg = np.nan_to_num(mecg)

                # mECG peak identification
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        _, mecg_info = nk.ecg_peaks(mecg, sampling_rate=sampling_rate, correct_artifacts=True)
                    mecg_peaks_indices = mecg_info.get("ECG_R_Peaks", [])
                except Exception as e:
                    if verbose:
                        logger.warning(f"Peak detection failed on segment {segment_count}: {e}")
                    curr_start += segment_step
                    pbar.update(1)
                    continue

                # fECG extraction
                residual = lower_peaks(segment_signal, peaks=mecg_peaks_indices)
                fecg = run_ssa(residual, cvp=fecg_cvp, window_length=window_length_svd, on_cuda=True)

                if fecg is None or len(fecg) == 0 or np.all(np.isnan(fecg)):
                    curr_start += segment_step
                    pbar.update(1)
                    continue

                fecg = np.nan_to_num(fecg)
                if current_seg_len == L_seg_samples:
                    w = window_full
                else:
                    # handle edge case (last segment)
                    w = hann(current_seg_len)

                start_idx = int(curr_start * sampling_rate)
                end_idx = min(start_idx + current_seg_len, full_length)
                slice_len = end_idx - start_idx
                w_slice = w[:slice_len]

                combined_mecg[start_idx:end_idx] += mecg[:slice_len] * w_slice
                combined_fecg[start_idx:end_idx] += fecg[:slice_len] * w_slice
                weights[start_idx:end_idx] += w_slice

                # next segment
                curr_start += segment_step
                pbar.update(1)

        logger.info(f"Finished processing {segment_count} segments.")

        # final normalization
        combined_mecg /= np.maximum(weights, 1e-8)
        combined_fecg /= np.maximum(weights, 1e-8)

        full_fecg = combined_fecg

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice.", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
            _, info = nk.ecg_peaks(full_fecg, sampling_rate=sampling_rate, correct_artifacts=True)

        fecg_peaks_seconds = info.get('ECG_R_Peaks', []) / sampling_rate
        report = get_classification_report(full_gt_onsets, fecg_peaks_seconds)

        # saving results
        elapsed_time = time.time() - start_time
        experiment_name = Path(sys.argv[0]).stem

        data_to_save = {
            'mecg': combined_mecg,
            'fecg': combined_fecg
        }

        experiment_report: Dict[str, Any] = {
            "experiment_id": experiment_name,
            "execution_time_seconds": elapsed_time,
            "target_channel": target_channel,
            "segment_length": segment_length,
            "overlap": overlap,
            "window_length_svd": window_length_svd,
            "mecg_cvp": mecg_cvp,
            "fecg_cvp": fecg_cvp,
            "segment_count": segment_count,
            "results": report
        }

        save_results(filename, experiment_name, data_to_save, experiment_report)
        logger.success(f"fECG from {filename} extracted in {round(elapsed_time, 2)} seconds (Accuracy: {report['accuracy']:.2f}%)")

    except Exception as e:
        logger.error(f"An error occurared during the Sliding Window experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
