from pathlib import Path
import pyedflib
import numpy as np
from loguru import logger

from ecg_svd.config import INTERIM_DATA_DIR


_EDF_READER = None


def get_edf_reader(edf_path: Path) -> pyedflib.EdfReader:
    global _EDF_READER

    if _EDF_READER is None:
        try:
            _EDF_READER = pyedflib.EdfReader(str(edf_path))
            logger.info(f"EDF Reader initialized for: {edf_path.name}")
        except Exception as e:
            logger.error(f"Error during EdfReader initialization: {e}")
            raise
    return _EDF_READER


def close_edf_reader():
    global _EDF_READER
    if _EDF_READER:
        _EDF_READER.close()
        del _EDF_READER
        _EDF_READER = None
        logger.info("EDF Reader closed")


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


def save_numpy_array(data: np.ndarray, filename: str, data_dir: Path = INTERIM_DATA_DIR):
    file_path = data_dir / filename
    np.save(file_path, data)
    logger.success(f"Array saved to: {file_path} with shape {data.shape}")
