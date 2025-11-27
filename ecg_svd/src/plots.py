import numpy as np
import matplotlib.pyplot as plt
import sys

from ecg_svd.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from ecg_svd.src.data_io import get_edf_reader, get_signal_segment, close_edf_reader

sys.path.append(".")


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
    plot_qualitative_assessment()
