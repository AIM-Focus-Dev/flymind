# File: training/scripts/MindReaderModel/preprocess_data.py

import mne
import numpy as np
import os
from pathlib import Path # For handling paths reliably
from .configs.config import PROJECT_ROOT, DATA_PATH, RESULTS_PATH, MODELS_PATH, RESULTS_PATH, MODELS_PATH, FS, LOW_FREQ, HIGH_FREQ, TMIN, TMAX, N_SUBJECTS, EVENT_DESC_MAP

def load_and_preprocess_subject_data(subject_id, session_type='T', data_path=DATA_PATH,
                                     l_freq=LOW_FREQ, h_freq=HIGH_FREQ, tmin=TMIN, tmax=TMAX,
                                     event_desc_map=EVENT_DESC_MAP, fs=FS):
    """
    Loads and preprocesses EEG data for a single subject and session from BCI IV 2a GDF files.
    Focuses on creating epochs based on cue markers (769-772) typically found in 'T' files.
    """
    if not 1 <= subject_id <= N_SUBJECTS:
        raise ValueError(f"Subject ID must be between 1 and {N_SUBJECTS}")
    if session_type not in ['T', 'E']:
        raise ValueError("session_type must be 'T' (Training) or 'E' (Evaluation)")

    file_name = f"A{subject_id:02d}{session_type}.gdf"
    file_path = data_path / file_name

    if not file_path.exists():
        print(f"Warning: Data file not found: {file_path}")
        return None

    print(f"Loading data from: {file_path}...")
    try:
        # Setting stim_channel=None forces MNE to rely on annotations for events
        raw = mne.io.read_raw_gdf(file_path, preload=True, stim_channel=None, verbose='WARNING')
    except Exception as e:
        print(f"Error loading GDF file {file_path}: {e}")
        return None

    # RuntimeWarning about duplicate channel names is noted, MNE handles it.

    if len(raw.ch_names) < 22:
        print(f"Warning: Expected at least 22 channels, found {len(raw.ch_names)}.")
        return None
    eeg_channels = raw.ch_names[:22]
    # raw.pick_channels(eeg_channels) # Apply pick later, after filtering if desired

    # Apply band-pass filter (Common for MI tasks)
    print(f"Applying band-pass filter ({l_freq}-{h_freq} Hz)...")
    # Filter only EEG channels
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=eeg_channels, fir_design='firwin', skip_by_annotation='edge', verbose='WARNING')

    # Select only EEG channels AFTER filtering
    raw.pick(picks=eeg_channels)

    # Optional: Set standard montage if channel names match standard layout
    # try:
    #     montage = mne.channels.make_standard_montage('standard_1005')
    #     raw.set_montage(montage, on_missing='warn')
    # except Exception as e:
    #      print(f"Could not set montage: {e}")

    # Find events from annotations
    print("Finding events from annotations...")
    try:
        events, event_ids_found_dict = mne.events_from_annotations(raw, verbose='WARNING')
        # event_ids_found_dict maps annotation descriptions (strings) like '769'
        # to MNE integer IDs like 7
        print(f"Found event IDs (Annotation Desc -> MNE ID): {event_ids_found_dict}")

        # Create the event_id dictionary needed for mne.Epochs
        # Map our desired task names ('left_hand', etc.) to the MNE integer IDs
        # found in *this specific file*.
        epochs_event_id = {}
        target_descs_found_in_file = []
        for desc_str, task_name in event_desc_map.items():
            if desc_str in event_ids_found_dict:
                epochs_event_id[task_name] = event_ids_found_dict[desc_str]
                target_descs_found_in_file.append(desc_str)

        if not epochs_event_id:
             print(f"No target annotation descriptions {list(event_desc_map.keys())} found in this file.")
             if session_type == 'E':
                 print("Note: Cue events 769-772 are typically absent in evaluation ('E') files.")
             return None # Return None if no target events found for epoching

        print(f"Found mapping for MNE IDs relevant for epoching: {epochs_event_id}")
        print(f"(Corresponding to annotation descriptions found: {target_descs_found_in_file})")

        # Check if any of these mapped MNE IDs actually exist in the events array
        event_codes_in_array = np.unique(events[:, 2])
        actual_event_ids_for_epochs = {k: v for k, v in epochs_event_id.items() if v in event_codes_in_array}

        if not actual_event_ids_for_epochs:
             print(f"Mapped MNE event IDs {list(epochs_event_id.values())} not found in events array data. Cannot create epochs.")
             if session_type == 'E':
                  print("Note: Cue events 769-772 are typically absent in evaluation ('E') files.")
             return None

        print(f"Creating epochs using event mapping: {actual_event_ids_for_epochs}")
        # Create Epochs based on the event IDs found in *this file*
        epochs = mne.Epochs(raw, events, event_id=actual_event_ids_for_epochs, tmin=tmin, tmax=tmax,
                            proj=False, picks='eeg', baseline=None, preload=True,
                            event_repeated='drop', verbose='WARNING')

    except Exception as e:
        print(f"Error finding events or creating epochs for subject {subject_id}{session_type}: {e}")
        return None

    print(f"Successfully created {len(epochs)} epochs for subject {subject_id}{session_type}.")
    print(f"Epoch event counts: {epochs.event_id}") # Show MNE IDs used

    return epochs

# --- Main execution block for testing and processing all subjects ---
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Results Path: {RESULTS_PATH}")
    print(f"Models Path: {MODELS_PATH}")


    # Dictionary to store all loaded epochs
    # Structure: all_subject_epochs[subject_id]['train'/'test'] = epochs_object
    all_subject_epochs = {sub: {'train': None, 'test': None} for sub in range(1, N_SUBJECTS + 1)}

    print("\n=== Starting Data Loading and Preprocessing ===")
    for subject_id in range(1, N_SUBJECTS + 1):
        print(f"\n--- Processing Subject {subject_id} ---")

        # Process Training Data
        print(f"--- Processing Training Data (Subject {subject_id}T) ---")
        epochs_train = load_and_preprocess_subject_data(subject_id, session_type='T')
        if epochs_train:
            print(f"Subject {subject_id}T: Successfully processed {len(epochs_train)} epochs.")
            all_subject_epochs[subject_id]['train'] = epochs_train
        else:
            print(f"Subject {subject_id}T: Failed to process or no target epochs found.")

        # Process Evaluation Data (attempting, may return None)
        print(f"\n--- Processing Evaluation Data (Subject {subject_id}E) ---")
        epochs_test = load_and_preprocess_subject_data(subject_id, session_type='E')
        if epochs_test:
            # This might not run if cue events 769-772 are absent
            print(f"Subject {subject_id}E: Successfully processed {len(epochs_test)} epochs based on available markers.")
            all_subject_epochs[subject_id]['test'] = epochs_test
        else:
            print(f"Subject {subject_id}E: No cue-based epochs created (likely expected for 'E' files).")

    print("\n=== Data Loading and Preprocessing Complete ===")

    # Example: Check loaded data for subject 1
    if all_subject_epochs[1]['train'] is not None:
        print("\nExample: Info for Subject 1 Training Epochs:")
        s1_train = all_subject_epochs[1]['train']
        print(s1_train.info)
        print(f"Shape: {s1_train.get_data().shape}")
        print(f"Events: {s1_train.event_id}") # Shows task_name -> MNE_ID mapping used
    else:
        print("\nExample: No training epochs loaded for Subject 1.")

    # Now `all_subject_epochs` dictionary holds the preprocessed data (MNE Epochs objects)
    # for all subjects where processing was successful (likely all 'T' files).
    # You can pass this dictionary or individual Epochs objects to your training scripts.
    # e.g., train_baseline(all_subject_epochs) or train_ae(all_subject_epochs)
