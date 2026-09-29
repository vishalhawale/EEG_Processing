import os
import re
import mne
import warnings
import numpy as np
from scipy.signal import welch

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

DATA_DIR = "/Users/apple/Downloads/EEG_Processing/data"

STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]


def parse_summary_file(summary_path):
    seizure_info = {}
    current_file = None
    if not os.path.exists(summary_path):
        return seizure_info

    with open(summary_path, 'r', encoding='latin-1') as f:
        for line in f:
            if "File Name:" in line:
                current_file = line.split(":")[-1].strip()
                seizure_info[current_file] = []
            elif "Seizure" in line and "Start Time" in line:
                match = re.search(r'\d+', line.split(":")[-1])
                if match and current_file:
                    seizure_info[current_file].append({'start': int(match.group()), 'end': None})
            elif "Seizure" in line and "End Time" in line:
                match = re.search(r'\d+', line.split(":")[-1])
                if match and current_file and seizure_info[current_file]:
                    seizure_info[current_file][-1]['end'] = int(match.group())

    return seizure_info


def sanitize_and_standardize(raw):
    cleaned_names = [ch.upper().replace('EEG ', '').strip() for ch in raw.ch_names]
    unique_names = []
    seen = set()
    for name in cleaned_names:
        clean_name = name.split('-0')[0]
        if clean_name not in seen:
            seen.add(clean_name)
            unique_names.append(clean_name)
        else:
            unique_names.append(f"{clean_name}_DUP")
            
    rename_dict = dict(zip(raw.ch_names, unique_names))
    raw.rename_channels(rename_dict)
    
    channel_mapping = {}
    for std_ch in STANDARD_CHANNELS:
        for ch_name in raw.ch_names:
            if std_ch == ch_name or std_ch in ch_name:
                channel_mapping[ch_name] = std_ch
                break

    if len(channel_mapping) < 18:
        return None

    raw.pick(list(channel_mapping.keys()))
    raw.rename_channels(channel_mapping)
    raw.reorder_channels(STANDARD_CHANNELS)
    return raw


def extract_window_features(window, sfreq=256):
    features = []
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }

    for ch_idx in range(window.shape[0]):
        sig = window[ch_idx, :]
        mean_val = np.mean(sig)
        std_val = np.std(sig)
        line_length = np.sum(np.abs(np.diff(sig)))
        activity = np.var(sig)
        
        freqs, psd = welch(sig, fs=sfreq, nperseg=min(len(sig), int(sfreq * 2)))
        band_powers = [np.sum(psd[(freqs >= fmin) & (freqs <= fmax)]) for fmin, fmax in bands.values()]
            
        features.extend([mean_val, std_val, line_length, activity] + band_powers)

    return np.array(features)


def extract_subject_dataset(subject_id, data_dir=DATA_DIR, window_sec=5, overlap_sec=2):
    summary_path = os.path.join(data_dir, f"{subject_id}-summary.txt")
    seizure_dict = parse_summary_file(summary_path)

    X_list, y_list = [], []
    edf_files = [f for f in os.listdir(data_dir) if f.startswith(subject_id) and f.endswith('.edf')]

    print(f"--> Extracting EEG features for Subject {subject_id} ({len(edf_files)} files)...")

    for file_name in sorted(edf_files):
        edf_path = os.path.join(data_dir, file_name)
        seizures = seizure_dict.get(file_name, [])

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)
            raw.notch_filter(freqs=50.0, verbose=False)
            
            raw = sanitize_and_standardize(raw)
            if raw is None:
                continue

            data = raw.get_data()
            sfreq = int(raw.info['sfreq'])
            
            step_samples = int((window_sec - overlap_sec) * sfreq)
            window_samples = int(window_sec * sfreq)
            total_samples = data.shape[1]

            for start_idx in range(0, total_samples - window_samples, step_samples):
                end_idx = start_idx + window_samples
                win_start_sec = start_idx / sfreq
                win_end_sec = end_idx / sfreq

                is_seizure = 0
                for sz in seizures:
                    if sz['end'] is not None:
                        overlap = max(0, min(win_end_sec, sz['end']) - max(win_start_sec, sz['start']))
                        if overlap >= (window_sec / 2):
                            is_seizure = 1
                            break

                feats = extract_window_features(data[:, start_idx:end_idx], sfreq=sfreq)
                X_list.append(feats)
                y_list.append(is_seizure)

        except Exception as e:
            print(f"    [Warning] Skipping {file_name}: {e}")

    return np.array(X_list), np.array(y_list)