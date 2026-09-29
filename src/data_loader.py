import os
import re
import mne

STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

def parse_summary_file(summary_path):
    """
    Parses CHB-MIT summary text files to extract start and end timestamps for seizures.
    """
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
    """
    Cleans raw EEG channel names and forces standard 18-channel bipolar montage ordering.
    """
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