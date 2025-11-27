import json
from pathlib import Path
import pandas as pd

from ecg_svd.config import REPORTS_DIR


def consolidate_all_reports(edf_file: str = "r01.edf", results_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    all_data = []

    for report_file in results_dir.glob("*.json"):
        with open(report_file, 'r') as f:
            data = json.load(f)

            experiment_data = {
                "experiment_id": report_file.stem,
                "filename": edf_file,
                "execution_time_seconds": round(data[edf_file]["execution_time_seconds"], 2),
                "accuracy": data[edf_file]["results"]["accuracy"],
                "precision": data[edf_file]["results"]["precision"],
                "recall": data[edf_file]["results"]["recall"],
                "f1": data[edf_file]["results"]["f1"]
            }
            all_data.append(experiment_data)

    return pd.DataFrame(all_data)


if __name__ == "__main__":
    df_comparison = consolidate_all_reports()
    print(df_comparison.sort_values(by="accuracy", ascending=False))
