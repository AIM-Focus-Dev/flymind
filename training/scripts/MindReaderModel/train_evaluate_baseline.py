# File: training/scripts/MindReaderModel/train_evaluate_baseline.py
# Updated to include confusion matrix plotting and fix CSP import

import numpy as np
import pandas as pd
import mne
from pathlib import Path
import matplotlib.pyplot as plt # For plotting
import seaborn as sns           # For plotting confusion matrix
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from mne.decoding import CSP

# --- Import Preprocessing Function and Config ---
try:
    from .preprocess_data import (
        load_and_preprocess_subject_data, N_SUBJECTS, DATA_PATH,
        RESULTS_PATH # Import needed constants
    )
    print(f"Imported paths: DATA={DATA_PATH}, RESULTS={RESULTS_PATH}")
except ImportError:
    print("Could not import from sibling, importing preprocess_data directly.")
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
    except NameError:
        PROJECT_ROOT = Path('.').resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
    RESULTS_PATH = PROJECT_ROOT / "training" / "results"
    N_SUBJECTS = 9
    N_CHANNELS = 22
    N_SAMPLES = 751
    from preprocess_data import load_and_preprocess_subject_data
    print(f"Imported paths: DATA={DATA_PATH}, RESULTS={RESULTS_PATH}")


# --- Configuration ---
N_CSP_COMPONENTS = 4
N_SPLITS = 5
SHUFFLE_FOLDS = True
RANDOM_STATE = 42
METRIC = 'accuracy'
# ------------------------

def plot_confusion_matrix(y_true, y_pred, classes, subject_id, pipeline_name, results_path):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Subject {subject_id:02d} - {pipeline_name} Confusion Matrix (CV Preds)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    # Include pipeline name in filename
    plot_filename = results_path / f"subject_{subject_id:02d}_{pipeline_name}_confusion_matrix.png"
    plt.savefig(plot_filename)
    print(f"Saved confusion matrix plot to {plot_filename}")
    plt.close()

def run_baseline_evaluation():
    """
    Runs CSP+LDA and CSP+SVM baseline evaluations using cross-validation
    within the training ('T') session data for each subject.
    Includes confusion matrix plotting.
    """
    print("\n=== Starting Baseline Evaluation (CSP + LDA/SVM) with Plotting ===")
    all_results = []

    # --- Get Class Names ---
    temp_epochs = load_and_preprocess_subject_data(1, session_type='T')
    if temp_epochs is None or not hasattr(temp_epochs, 'event_id') or not temp_epochs.event_id:
        print("Error: Cannot determine event ID mapping. Need at least one subject's data.")
        return
    mne_id_to_task = {v: k for k, v in temp_epochs.event_id.items()}
    sorted_mne_ids = sorted(mne_id_to_task.keys())
    class_names = [mne_id_to_task[mne_id] for mne_id in sorted_mne_ids]
    print(f"Class names (order {sorted_mne_ids}): {class_names}")
    del temp_epochs
    # --- End Get Class Names ---

    for subject_id in range(1, N_SUBJECTS + 1):
        print(f"\n--- Processing Subject {subject_id} ---")

        # 1. Load preprocessed TRAINING data only
        epochs_train = load_and_preprocess_subject_data(subject_id, session_type='T')
        if epochs_train is None or len(epochs_train) == 0:
            print(f"Skipping Subject {subject_id} - no epochs.")
            continue

        # 2. Extract data (X) and labels (y)
        X_raw = epochs_train.get_data(copy=True)
        X = np.ascontiguousarray(X_raw, dtype=np.float64)
        y = epochs_train.events[:, -1] # MNE event IDs
        n_epochs, n_channels, n_times = X.shape
        print(f"Data shape: {X.shape}, Labels shape: {y.shape}, Unique labels: {np.unique(y)}")

        if n_epochs < N_SPLITS:
            print(f"Skipping Subject {subject_id} - only {n_epochs} epochs (< {N_SPLITS} folds).")
            continue

        # 3. Define pipelines
        csp = CSP(n_components=N_CSP_COMPONENTS, reg='ledoit_wolf', log=True, norm_trace=False)
        lda = LDA()
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')

        pipe_lda = Pipeline([('CSP', csp), ('LDA', lda)])
        pipe_svm = Pipeline([('CSP', csp), ('SVM', svm)])

        pipelines = {'CSP+LDA': pipe_lda, 'CSP+SVM': pipe_svm}

        # 4. Cross-validation strategy
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE_FOLDS, random_state=RANDOM_STATE)

        # 5. Run CV and Plotting
        for name, pipe in pipelines.items():
            print(f"Running {N_SPLITS}-fold CV for {name} (single-threaded)...")
            try:
                # Get cross-validated scores
                scores = cross_val_score(pipe, X, y, cv=cv, scoring=METRIC, n_jobs=1)
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"  Mean {METRIC.capitalize()}: {mean_score:.4f} (+/- {std_score:.4f})")

                # Get cross-validated predictions
                print(f"  Generating CV predictions for {name}...")
                y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)

                # Plot confusion matrix
                plot_confusion_matrix(y, y_pred, class_names, subject_id, name, RESULTS_PATH)

                # Store results
                all_results.append({
                    'subject': f'A{subject_id:02d}',
                    'pipeline': name,
                    f'mean_{METRIC}': mean_score,
                    f'std_{METRIC}': std_score,
                    'n_epochs': n_epochs
                })

            except ValueError as e:
                print(f"  Error during CV for {name} on Subject {subject_id}: {e}")
            except Exception as e:
                print(f"  Unexpected error during CV for {name} on Subject {subject_id}: {e}")

    # --- Process and Save Results ---
    if not all_results:
        print("\nNo baseline results were generated.")
        return

    results_df = pd.DataFrame(all_results)
    print("\n=== Baseline Evaluation Summary ===")
    print(results_df.to_string())

    avg_results = results_df.groupby('pipeline')[f'mean_{METRIC}'].agg(['mean', 'std'])
    print("\n--- Average Performance Across Subjects ---")
    print(avg_results)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    detailed_csv = RESULTS_PATH / f"baseline_csp_{METRIC}_results.csv"
    results_df.to_csv(detailed_csv, index=False, float_format='%.4f')
    print(f"\nDetailed results saved to {detailed_csv}")
    avg_csv = RESULTS_PATH / f"baseline_csp_{METRIC}_average.csv"
    avg_results.to_csv(avg_csv, float_format='%.4f')
    print(f"Average results saved to {avg_csv}")


if __name__ == "__main__":
    run_baseline_evaluation()
