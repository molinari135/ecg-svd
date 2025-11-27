from pathlib import Path
import pyedflib
from loguru import logger

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
