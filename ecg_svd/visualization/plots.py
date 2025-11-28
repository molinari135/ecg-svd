import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import typer

from ecg_svd.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR
from ecg_svd.data.io import get_edf_reader, close_edf_reader
from ecg_svd.data.preprocessing import get_signal_segment

sys.path.append(".")
app = typer.Typer(help="Plot extracted mECG and fECG signals from processed results.")


@app.command()
def plot_mecg_fecg(edf_name):
    for name in os.listdir(PROCESSED_DATA_DIR):
        # name must start with a number
        if not name[0].isdigit():
            continue

        edf_path = RAW_DATA_DIR / f"{edf_name}.edf"
        res_path = PROCESSED_DATA_DIR / name / f"{edf_name}.npy"
        print(f"Processing {res_path} with {edf_path}...")
        edf = get_edf_reader(edf_path)

        np_data = np.load(res_path, allow_pickle=True)
        data_dict = np_data.item()

        if 'mecg' not in data_dict:
            continue

        mecg = data_dict['mecg'][:5000] * 30
        fecg = data_dict['fecg'][:5000] * 30
        aecg = get_signal_segment(edf, ch_number=1)['segment']

        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'lines.linewidth': 1.0
        })

        fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

        axes[0].plot(aecg, label='Original aECG', alpha=0.8)
        axes[0].set_title('Original Abdominal ECG (aECG)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')

        axes[1].plot(mecg, label='Extracted mECG', alpha=0.8)
        axes[1].set_title('Extracted Maternal ECG (mECG)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')

        axes[2].plot(fecg, label='Extracted fECG', alpha=0.8)
        axes[2].set_title('Extracted Fetal ECG (mECG)')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right')

        axes[-1].set_xlabel('Samples')
        plt.tight_layout()

        save_path = FIGURES_DIR / f"{name}/{edf_name}.png"
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        close_edf_reader()


if __name__ == "__main__":
    app()
