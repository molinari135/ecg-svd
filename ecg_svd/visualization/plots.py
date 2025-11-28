import numpy as np
import matplotlib.pyplot as plt
import sys, os

from ecg_svd.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR
from ecg_svd.data.io import get_edf_reader, close_edf_reader
from ecg_svd.data.preprocessing import get_signal_segment

sys.path.append(".")


def plot_mecg(edf_name):
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
        aecg = get_signal_segment(edf, ch_number=1)['segment']

        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'lines.linewidth': 1.0
        })

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        axes[0].plot(aecg, color='black', label='Real aECG', alpha=0.8)
        axes[0].set_title('Real Abdominal ECG (aECG)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')

        axes[1].plot(mecg, color='#1f77b4', label='Extracted mECG', alpha=0.8)
        axes[1].set_title('Extracted Maternal ECG (mECG)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')

        axes[-1].set_xlabel('Samples')
        plt.tight_layout()

        save_path = FIGURES_DIR / f"{name}/{edf_name}_mecg.png"
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        close_edf_reader()


def plot_fecg(edf_file: str = "r01.edf", npy_file: str = "r01.npy"):

    for name in os.listdir(PROCESSED_DATA_DIR):
        # name must start with a number
        if not name[0].isdigit():
            continue
        
        edf_path = RAW_DATA_DIR / edf_file
        res_path = PROCESSED_DATA_DIR / name / npy_file
        edf = get_edf_reader(edf_path)

        np_data = np.load(res_path, allow_pickle=True)
        data_dict = np_data.item()
        
        # if 'fecg' or 'mecg' not in data_dict, skip
        if 'fecg' not in data_dict:
            continue
        
        fecg = data_dict['fecg'][:5000] * 30
        aecg = get_signal_segment(edf)['segment']

        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'lines.linewidth': 1.0
        })
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        axes[0].plot(aecg, color='black', label='Real aECG', alpha=0.8)
        axes[0].set_title('Real Abdominal ECG (aECG)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')

        axes[1].plot(fecg, color='#d62728', label='Extracted fECG')
        axes[1].set_title('Extracted Fetal ECG (fECG)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')

        axes[-1].set_xlabel('Samples')
        plt.tight_layout()
        plt.show()
    close_edf_reader()


def plot_aecg_mecg_fecg(edf_file: str = "r04.edf", npy_path: str = "8_tucker_tensor_gpu/r04.npy"):
    edf_path = RAW_DATA_DIR / edf_file
    res_path = PROCESSED_DATA_DIR / npy_path
    edf = get_edf_reader(edf_path)

    np_data = np.load(res_path, allow_pickle=True)
    data_dict = np_data.item()
    fecg = data_dict['fecg'][:5000] * 30
    mecg = data_dict['mecg'][:5000] * 30
    aecg = get_signal_segment(edf)['segment']

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'lines.linewidth': 1.0
    })
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 2.5 * 3), sharex=True)

    axes[0].plot(aecg, color='black', label='Real aECG', alpha=0.8)
    axes[0].set_title('Real Abdominal ECG (aECG)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    axes[1].plot(mecg, color='#1f77b4', label='Extracted mECG', alpha=0.8)
    axes[1].set_title('Extracted Maternal ECG (mECG)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    axes[2].plot(fecg, color='#d62728', label='Extracted fECG')
    axes[2].set_title('Extracted Fetal ECG (fECG)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    axes[-1].set_xlabel('Samples')
    plt.tight_layout()
    plt.show()
    close_edf_reader()


def plot_qualitative_assessment():
    filename_edf = "r01.edf"
    filename_result = "8_tucker_tensor_gpu.npy"

    start_sec = 200
    duration = 4

    raw_channel_idx = 1
    edf_path = RAW_DATA_DIR / filename_edf
    edf = get_edf_reader(edf_path)

    gt_data = get_signal_segment(edf, ch_number=0, start_time=start_sec, end_time=start_sec + duration)
    gt_signal = gt_data['segment']
    sampling_rate = gt_data['sampling_rate']

    raw_data = get_signal_segment(edf, ch_number=raw_channel_idx, start_time=start_sec, end_time=start_sec + duration)
    raw_signal = raw_data['segment']

    close_edf_reader()

    res_path = PROCESSED_DATA_DIR / filename_result
    if not res_path.exists():
        print(f"Errore: File {res_path} non trovato.")
        return

    results = np.load(res_path, allow_pickle=True)

    try:
        if isinstance(results, np.lib.npyio.NpzFile):
            data_dict = results
        else:
            data_dict = results.item()

        full_fecg_est = data_dict['fecg']
        full_mecg_est = data_dict['mecg']

    except KeyError:
        print("Errore: Chiavi 'mecg' o 'fecg' non trovate nel file dei risultati.")
        return

    start_sample = int(start_sec * sampling_rate)
    end_sample = int((start_sec + duration) * sampling_rate)

    est_fecg = full_fecg_est[start_sample:end_sample]
    est_mecg = full_mecg_est[start_sample:end_sample]

    time_axis = np.linspace(0, duration, len(raw_signal))
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'lines.linewidth': 1.0
    })

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    def z_norm(sig):
        return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

    axes[0].plot(time_axis, z_norm(raw_signal), color='black', alpha=0.8, label='Composite Signal')
    axes[0].set_title(f'(a) Raw Abdominal ECG (Channel {raw_channel_idx})')
    axes[0].set_ylabel('Ampl. [a.u.]')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, z_norm(est_mecg), color='#1f77b4', label='Extracted mECG')  # Blue
    axes[1].set_title('(b) Extracted Maternal Component (Tucker)')
    axes[1].set_ylabel('Ampl. [a.u.]')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, z_norm(est_fecg), color='#d62728', label='Extracted fECG')  # Red
    axes[2].set_title('(c) Extracted Fetal Component (Tucker)')
    axes[2].set_ylabel('Ampl. [a.u.]')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_axis, z_norm(gt_signal), color='#2ca02c', label='Reference fECG')  # Green
    axes[3].set_title('(d) Ground Truth Reference (Direct Scalp)')
    axes[3].set_ylabel('Ampl. [a.u.]')
    axes[3].set_xlabel('Time [s]')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = "qualitative_comparison_full.png"
    # plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato come: {output_file}")
    plt.show()


if __name__ == "__main__":
    plot_mecg("r01")
    plot_mecg("r04")
    plot_mecg("r07")
    plot_mecg("r08")
    plot_mecg("r10")
