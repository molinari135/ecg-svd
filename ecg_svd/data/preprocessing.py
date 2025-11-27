import pyedflib
import numpy as np
from loguru import logger


def get_signal_segment(
    edf_reader: pyedflib.EdfReader,
    ch_number: int = 0,
    start_time: float = 0.0,
    end_time: float = 5.0
) -> dict:
    sampling_rate = edf_reader.getSampleFrequency(ch_number)
    signal = edf_reader.readSignal(ch_number)
    ons, _, _ = edf_reader.readAnnotations()

    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)

    end_idx = min(end_idx, len(signal))

    segment = signal[start_idx:end_idx]
    time = np.linspace(start_time, end_time, len(segment))

    onsets = ons[(ons >= start_time) & (ons <= end_time)]
    values = np.interp(onsets, time, segment)

    logger.debug(f"Segment ch {ch_number} extracted: {len(segment)} samples")

    return {
        "sampling_rate": sampling_rate,
        "segment": segment,
        "time": time,
        "onsets": onsets,
        "values": values
    }


def create_segment_tensor(hankel_matrices: list[np.ndarray]) -> np.ndarray:
    if not hankel_matrices:
        raise ValueError("The list of Hankel matrices cannot be empty")

    first_shape = hankel_matrices[0].shape
    if not all(m.shape == first_shape for m in hankel_matrices):
        logger.error("Hankel matrices must have the same (L, K) dimension for stacking")
        raise ValueError("Inconsistent matrix dimensions")

    segment_tensor = np.stack(hankel_matrices, axis=2)
    logger.debug(f"3D Tensor created with shape (L, K, Channels): {segment_tensor.shape}")
    return segment_tensor
