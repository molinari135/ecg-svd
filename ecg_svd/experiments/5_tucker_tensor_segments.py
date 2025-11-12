import typer
import numpy as np
import neurokit2 as nk
import tensorly as tl
from loguru import logger
from typing import List, Tuple

from ecg_svd.config import RAW_DATA_DIR
from ecg_svd.src.data_io import get_edf_reader, get_signal_segment, close_edf_reader, create_segment_tensor
from ecg_svd.src.decomposition import run_tucker, reconstruct_channels, create_hankel_matrix, get_tucker_rank
from ecg_svd.src.metrics import get_classification_report, get_signal_weights, signal_quality

tl.set_backend('numpy')

app = typer.Typer(help="Runs Tucker Decomposition on the 3D Hankel tensor for source separation.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channels: List[int] = [1, 2, 3, 4],
    gt_channel: int = 0,
    segment_duration: float = 5.0,
    window_length: int = 625 * 2,
):
    edf_path = RAW_DATA_DIR / filename

    try:
        # initialization and data loading
        edf = get_edf_reader(edf_path)

        gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=segment_duration)
        gt_onsets = gt_data['onsets']
        sampling_rate = gt_data['sampling_rate']

        segments_data = [
            get_signal_segment(edf, ch_number=ch, end_time=segment_duration)
            for ch in target_channels
        ]

        # tensor and weights
        hankel_matrices = [
            create_hankel_matrix(data['segment'], L_samples=window_length)
            for data in segments_data
        ]
        segment_tensor = create_segment_tensor(hankel_matrices)

        # calculate weights based on signal quality
        weights = get_signal_weights(segments_data)
        logger.info(f"Calculated channel weights: {weights}")

        # rank determination based on the median optimal rank across channels
        tucker_rank = get_tucker_rank(segments_data, signal_quality_func=signal_quality)

        # tucker rank is (R_L, R_K, R_C). R_L and R_K are based on time/trajectory subspace (k=tucker_rank).
        # R_C (channels) is fixed to the number of channels (4 in this case)
        rank_tucker: Tuple[int, int, int] = (tucker_rank, tucker_rank, len(target_channels))
        logger.info(f"Tucker decomposition starting with rank: {rank_tucker}")

        # tucker decomposition
        core, factors = run_tucker(segment_tensor, rank=rank_tucker)
        U_L, U_K, U_C = factors  # time, trajectory, channel factor matrices

        # component selection

        # core for mECG
        S_mecg = np.zeros_like(core)
        S_mecg[:, :, 0] = core[:, :, 0]

        # core for fECG
        S_fecg = np.zeros_like(core)
        S_fecg[:, :, 1:] = core[:, :, 1:]

        # reconstruct partial tensors (H_mecg, H_fecg)
        H_mecg = tl.tucker_to_tensor((S_mecg, factors))
        H_fecg = tl.tucker_to_tensor((S_fecg, factors))

        H_noise = segment_tensor - H_mecg - H_fecg
        logger.info("Partial tensors reconstructed.")

        # diagonal averaging per channel
        mECG_signals_list = reconstruct_channels(H_mecg)
        fECG_signals_list = reconstruct_channels(H_fecg)
        noise_signals_list = reconstruct_channels(H_noise)

        # weighted combination
        # noise_combined is actually the fECG
        mECG_combined = np.sum(np.array(mECG_signals_list) * weights[:, None], axis=0)
        fECG_combined = np.sum(np.array(fECG_signals_list) * weights[:, None], axis=0)
        noise_combined = np.sum(np.array(noise_signals_list) * weights[:, None], axis=0)

        # metrics calculation
        N_original = len(segments_data[0]['segment'])
        fecg_to_test = noise_combined[:N_original]

        _, info = nk.ecg_peaks(fecg_to_test, sampling_rate=sampling_rate, correct_artifacts=True)

        fecg_peaks_seconds = info.get('ECG_R_Peaks', []) / sampling_rate
        report = get_classification_report(gt_onsets, fecg_peaks_seconds)

        logger.success(f"Experiment 5 (Tucker) Completed. Final FECG Accuracy: {report['accuracy']:.2f}%")

    except Exception as e:
        logger.error(f"An error occurred during the Tucker experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
