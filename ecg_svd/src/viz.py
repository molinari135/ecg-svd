import json
from pathlib import Path
import pandas as pd

from ecg_svd.config import REPORTS_DIR


def consolidate_all_reports(results_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    all_data = []

    for report_file in results_dir.glob("*.json"):
        with open(report_file, 'r') as f:
            data = json.load(f)
            experiment_data = {
                "experiment_id": data["experiment_id"],
                "execution_time_seconds": data["execution_time_seconds"],
                "filename": data["filename"],
                "accuracy": data["results"]["accuracy"],
                "precision": data["results"]["precision"],
                "recall": data["results"]["recall"]
            }
            all_data.append(experiment_data)

    return pd.DataFrame(all_data)


if __name__ == "__main__":
    df_comparison = consolidate_all_reports()
    print(df_comparison.sort_values(by="accuracy", ascending=False))
