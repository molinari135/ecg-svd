import json
import pyedflib
import numpy as np
from pathlib import Path
from loguru import logger

from ecg_svd.config import PROCESSED_DATA_DIR, REPORTS_DIR

# global EDF reader instance
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


def save_npy_json(filename, experiment_name, data_to_save, experiment_report):
    np_output_path = PROCESSED_DATA_DIR / f"{experiment_name}"
    if not np_output_path.exists():
        np_output_path.mkdir(parents=True, exist_ok=True)
    np.save(np_output_path / f"{filename[:-4]}.npy", data_to_save)

    json_output_path = REPORTS_DIR / f"{experiment_name}.json"
    if json_output_path.exists():
        with open(json_output_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[filename] = experiment_report
    with open(json_output_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Report saved to {json_output_path}")
