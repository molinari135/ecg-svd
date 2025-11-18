import typer
import sys
import json
import time
import numpy as np
import neurokit2 as nk
from loguru import logger
from scipy.signal.windows import hann
from pathlib import Path

from ecg_svd.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from ecg_svd.src.data_io import get_edf_reader, get_signal_segment, close_edf_reader
from ecg_svd.src.decomposition import hankel_with_svd, lower_peaks
from ecg_svd.src.metrics import get_classification_report

app = typer.Typer(help="Runs two-step SVD separation on a single channel using a sliding window approach.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channel: int = 4,
    gt_channel: int = 0,
    segment_length: float = 5.0,
    overlap: float = 0.5,
    mecg_cvp: float = 0.75,
    fecg_cvp: float = 0.90,
    window_length_svd: int = 625 * 2
):
    edf_path = RAW_DATA_DIR / filename
    start_time = time.time()

    try:
        # initialization and data loading
        edf = get_edf_reader(edf_path)

        full_gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=300)
        full_gt_onsets = full_gt_data['onsets']
        total_duration = full_gt_data['time'][-1]
        sampling_rate = full_gt_data['sampling_rate']
        full_length = len(full_gt_data['segment'])

        logger.info(f"Total duration: {total_duration:.1f}s. Full length: {full_length} samples.")

        combined_mecg = np.zeros(full_length)
        combined_fecg = np.zeros(full_length)
        weights = np.zeros(full_length)

        curr_start = 0.0
        segment_step = segment_length - overlap

        # define Hann window
        L_seg_samples = int(segment_length * sampling_rate)
        window_full = hann(L_seg_samples)

        # sliding window
        segment_count = 0
        while curr_start < total_duration:
            curr_end = min(curr_start + segment_length, total_duration)

            # target segment
            segment_data = get_signal_segment(
                edf, ch_number=target_channel, start_time=curr_start, end_time=curr_end
            )
            segment_signal = segment_data['segment']
            segment_signal = (segment_signal - np.mean(segment_signal)) / (np.std(segment_signal) + 1e-8)  # center data

            if len(segment_signal) < 1000:  # skip if segment is too short
                break

            segment_count += 1
            logger.debug(f"Processing segment {segment_count}: {curr_start:.2f}s to {curr_end:.2f}s")

            # mECG extraction
            mecg = hankel_with_svd(segment_signal, cvp=mecg_cvp, window_length=window_length_svd)

            if mecg is None or len(mecg) == 0 or np.all(np.isnan(mecg)):
                logger.warning(f"Skipping segment {segment_count}: invalid mECG reconstruction.")
                curr_start += segment_step
                continue

            # handling nan values
            mecg = np.nan_to_num(mecg)

            # mECG peak identification
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice.", category=RuntimeWarning)
                    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
                    _, mecg_info = nk.ecg_peaks(-mecg, sampling_rate=sampling_rate, correct_artifacts=True)

                mecg_peaks_indices = mecg_info.get("ECG_R_Peaks", [])
            except Exception as e:
                logger.warning(f"Peak detection failed on segment {segment_count}: {e}")
                curr_start += segment_step
                continue

            # fECG extraction
            residual = lower_peaks(segment_signal, peaks=mecg_peaks_indices)
            fecg = hankel_with_svd(residual, cvp=fecg_cvp, window_length=window_length_svd)

            if fecg is None or len(fecg) == 0 or np.all(np.isnan(fecg)):
                logger.warning(f"Skipping segment {segment_count}: invalid fECG extraction.")
                curr_start += segment_step
                continue

            fecg = np.nan_to_num(fecg)

            current_seg_len = len(segment_signal)
            if current_seg_len == L_seg_samples:
                w = window_full
            else:
                # handle edge case (last segment)
                w = hann(current_seg_len)

            # weighted sum
            start_idx = int(curr_start * sampling_rate)
            end_idx = min(start_idx + len(mecg), full_length)
            combined_mecg[start_idx:end_idx] += mecg[:len(w)] * w
            combined_fecg[start_idx:end_idx] += fecg[:len(w)] * w
            weights[start_idx:end_idx] += w

            # next segment
            curr_start += segment_step

        logger.info(f"Finished processing {segment_count} segments.")

        # final normalization
        combined_mecg /= np.maximum(weights, 1e-8)
        combined_fecg /= np.maximum(weights, 1e-8)

        full_fecg = combined_fecg
        _, info = nk.ecg_peaks(full_fecg, sampling_rate=sampling_rate, correct_artifacts=True)
        fecg_peaks_seconds = info.get('ECG_R_Peaks', []) / sampling_rate  # in seconds

        report = get_classification_report(full_gt_onsets, fecg_peaks_seconds)
        logger.success(f"Experiment 6 (Sliding Window SVD) Completed. Final FECG Accuracy: {report['accuracy']:.2f}%")

        elapsed_time = time.time() - start_time
        experiment_name = Path(sys.argv[0]).stem
        data_to_save = {
            'mecg_combined': combined_mecg,
            'fecg_combined': combined_fecg,
            'sampling_rate': sampling_rate
        }

        experiment_report = {
            "experiment_id": experiment_name,
            "execution_time_seconds": elapsed_time,
            "filename": filename,
            "target_channel": target_channel,
            "segment_length": segment_length,
            "overlap": overlap,
            "window_length_svd": window_length_svd,
            "mecg_cvp": mecg_cvp,
            "fecg_cvp": fecg_cvp,
            "segment_count": segment_count,
            "results": report
        }

        np.save(PROCESSED_DATA_DIR / f"{experiment_name}.npy", data_to_save)

        json_output_path = REPORTS_DIR / f"{experiment_name}.json"
        with open(json_output_path, 'w') as f:
            json.dump(experiment_report, f, indent=4)
        logger.info(f"Report saved to {json_output_path}")

    except Exception as e:
        logger.error(f"An error occurred during the Sliding Window experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
