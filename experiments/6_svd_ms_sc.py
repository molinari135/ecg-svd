import typer
import sys
import time
import neurokit2 as nk
import numpy as np
from pathlib import Path
from loguru import logger
from scipy.signal.windows import hann
from tqdm import tqdm
from typing import List, Dict, Any

from ecg_svd.config import RAW_DATA_DIR
from ecg_svd.data.io import get_edf_reader, close_edf_reader, save_results
from ecg_svd.data.preprocessing import get_signal_segment
from ecg_svd.methods.common import lower_peaks
from ecg_svd.methods.matrix import run_ssa
from ecg_svd.evaluation.metrics import get_classification_report

app = typer.Typer(help="Runs two-step SVD separation on a single channel using a sliding window approach.")


@app.command()
def main(
    filename: str = "r01.edf",
    target_channel: int = 4,
    gt_channel: int = 0,
    segment_length: float = 5.0,
    overlap: float = 0.5,
    mecg_cvp: float = 0.75,
    fecg_cvp: float = 0.90,
    window_length_svd: int = 1250,
    verbose: bool = False
):
    # configure loader
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="SUCCESS")

    edf_path = RAW_DATA_DIR / filename
    start_time = time.time()

    try:
        # --- Loading Data (Fast) ---
        if verbose:
            logger.debug("Initializing EDF Reader...")
        edf = get_edf_reader(edf_path)

        # Carica tutti i dati di Ground Truth e il segnale target completo (PIÙ EFFICIENTE)
        full_gt_data = get_signal_segment(edf, ch_number=gt_channel, end_time=300)
        full_target_data = get_signal_segment(edf, ch_number=target_channel, end_time=300)

        full_gt_onsets = full_gt_data['onsets']
        total_duration = full_gt_data['time'][-1]
        sampling_rate = full_gt_data['sampling_rate']
        full_length = len(full_target_data['segment']) # Usa la lunghezza del segnale target

        logger.info(f"Total duration: {total_duration:.1f}s. Full length: {full_length} samples.")

        # Inizializzazione per l'Overlap-Add
        combined_mecg = np.zeros(full_length)
        combined_fecg = np.zeros(full_length)
        weights = np.zeros(full_length)

        # Lista per i report di classificazione di ogni segmento (NUOVO)
        segment_reports: List[Dict[str, float]] = []

        curr_start = 0.0
        # Calcolo dello step: Lunghezza - Overlap (es. 5.0 - 0.5 = 4.5s)
        segment_step = segment_length - overlap

        # define Hann window
        L_seg_samples = int(segment_length * sampling_rate)
        window_full = hann(L_seg_samples)

        # estimate total segments for the progress bar
        est_total_segments = int(np.ceil((total_duration - segment_length) / segment_step)) + 1
        segment_count = 0

        # --- Sliding Window Loop with TQDM ---
        with tqdm(total=est_total_segments, desc="Processing Segments", unit="win", disable=verbose) as pbar:
            while curr_start < total_duration:
                curr_end = min(curr_start + segment_length, total_duration)

                # Target segment estrazione usando lo slicing sull'array completo
                start_sample = int(curr_start * sampling_rate)
                end_sample = int(curr_end * sampling_rate)
                
                segment_signal = full_target_data['segment'][start_sample:end_sample]
                current_seg_len = len(segment_signal)

                # skip if segment is too short
                if current_seg_len < int(segment_step * sampling_rate):
                    break 

                # Z-score normalization
                segment_signal = (segment_signal - np.mean(segment_signal)) / (np.std(segment_signal) + 1e-8)

                segment_count += 1
                logger.debug(f"Processing segment {segment_count}: {curr_start:.2f}s to {curr_end:.2f}s, length={current_seg_len}")

                # mECG extraction
                mecg = run_ssa(segment_signal, cvp=mecg_cvp, window_length=window_length_svd)

                if mecg is None or len(mecg) == 0 or np.all(np.isnan(mecg)):
                    if verbose:
                        logger.warning(f"Skipping segment {segment_count}: invalid mECG reconstruction.")
                    curr_start += segment_step
                    pbar.update(1)
                    continue

                mecg = np.nan_to_num(mecg)

                # mECG peak identification
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        _, mecg_info = nk.ecg_peaks(mecg, sampling_rate=sampling_rate, correct_artifacts=True)
                    mecg_peaks_indices = mecg_info.get("ECG_R_Peaks", [])
                except Exception as e:
                    if verbose:
                        logger.warning(f"Peak detection failed on segment {segment_count}: {e}")
                    curr_start += segment_step
                    pbar.update(1)
                    continue

                # fECG extraction
                residual = lower_peaks(segment_signal, peaks=mecg_peaks_indices)
                fecg = run_ssa(residual, cvp=fecg_cvp, window_length=window_length_svd)

                if fecg is None or len(fecg) == 0 or np.all(np.isnan(fecg)):
                    curr_start += segment_step
                    pbar.update(1)
                    continue

                fecg = np.nan_to_num(fecg)

                # --- CALCOLO METRICHE PER SEGMENTO (NUOVO) ---
                
                # Picchi fECG predetti (in secondi relativi all'inizio del segmento)
                _, fecg_info = nk.ecg_peaks(fecg, sampling_rate=sampling_rate, correct_artifacts=True)
                fecg_peaks_seconds = fecg_info.get('ECG_R_Peaks', []) / sampling_rate

                # Onset GT filtrati
                gt_onsets_segment_absolute = full_gt_onsets[
                    (full_gt_onsets >= curr_start) & (full_gt_onsets < curr_end)
                ]
                
                # Calcola il report di classificazione per il segmento (usando tempo relativo 0s)
                try:
                    # Trasla gli onset GT a 0s
                    gt_onsets_segment_relative = gt_onsets_segment_absolute - curr_start
                    report_segment = get_classification_report(
                        gt_onsets_segment_relative, fecg_peaks_seconds
                    )
                    segment_reports.append(report_segment)
                except Exception as e:
                    logger.warning(f"Classification report failed for segment {segment_count}: {e}")
                    pass

                # Overlap-Add accumulation
                if current_seg_len == L_seg_samples:
                    w = window_full
                else:
                    # handle edge case (last segment)
                    w = hann(current_seg_len)

                start_idx = int(curr_start * sampling_rate)
                end_idx = min(start_idx + current_seg_len, full_length)
                slice_len = end_idx - start_idx
                w = w[:slice_len]

                combined_mecg[start_idx:end_idx] += mecg[:slice_len] * w
                combined_fecg[start_idx:end_idx] += fecg[:slice_len] * w
                weights[start_idx:end_idx] += w

                # next segment
                curr_start += segment_step
                pbar.update(1)

        logger.info(f"Finished processing {segment_count} segments. {len(segment_reports)} reports collected.")

        # final normalization
        combined_mecg /= np.maximum(weights, 1e-8)
        combined_fecg /= np.maximum(weights, 1e-8)
        
        # --- Calcolo della Media dei Report (FINALE) ---
        if not segment_reports:
            # Se la lista è vuota, il report è 0.0
            mean_report = {
                "TP": 0, "FN": 0, "FP": 0, "TN": 0,
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "reports_count": 0
            }
        else:
            # 1. Inizializza le liste per le metriche percentuali (da mediare)
            list_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
            # 2. Inizializza le somme per i conteggi (da sommare)
            sum_metrics = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
            
            # 3. Aggrega i dati
            for r in segment_reports:
                # Somma i conteggi
                sum_metrics['TP'] += r.get('TP', 0)
                sum_metrics['FN'] += r.get('FN', 0)
                sum_metrics['FP'] += r.get('FP', 0)
                sum_metrics['TN'] += r.get('TN', 0)
                
                # Raccogli le metriche percentuali (usando .get per sicurezza)
                list_metrics['accuracy'].append(r.get('accuracy', 0.0))
                list_metrics['precision'].append(r.get('precision', 0.0))
                list_metrics['recall'].append(r.get('recall', 0.0))
                list_metrics['f1'].append(r.get('f1', 0.0))

            # 4. Calcola il report finale
            mean_report = {
                # Conteggi (somma)
                "TP": sum_metrics['TP'],
                "FN": sum_metrics['FN'],
                "FP": sum_metrics['FP'],
                "TN": sum_metrics['TN'],
                # Percentuali (media)
                "accuracy": np.mean(list_metrics['accuracy']),
                "precision": np.mean(list_metrics['precision']),
                "recall": np.mean(list_metrics['recall']),
                "f1": np.mean(list_metrics['f1']),
                "reports_count": len(segment_reports)
            }
        
        report = mean_report  # La variabile finale da salvare

        # --- Saving ---
        elapsed_time = time.time() - start_time
        experiment_name = Path(sys.argv[0]).stem

        data_to_save = {
            'mecg': combined_mecg,
            'fecg': combined_fecg
        }

        experiment_report: Dict[str, Any] = {
            "experiment_id": experiment_name,
            "execution_time_seconds": elapsed_time,
            "target_channel": target_channel,
            "segment_length": segment_length,
            "overlap": overlap,
            "window_length_svd": window_length_svd,
            "mecg_cvp": mecg_cvp,
            "fecg_cvp": fecg_cvp,
            "segment_count": segment_count,
            "results": report  # Salviamo il report medio
        }

        save_results(filename, experiment_name, data_to_save, experiment_report)
        logger.success(f"Experiment completed in {round(elapsed_time, 2)}s (Accuracy MEDIA: {report['accuracy']:.2f}%)")

    except Exception as e:
        logger.error(f"An error occurared during the Sliding Window experiment: {e}")
        raise

    finally:
        # cleanup
        close_edf_reader()


if __name__ == "__main__":
    app()
