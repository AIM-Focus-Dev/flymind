# File: training/scripts/MindReaderModel/train_evaluate_baseline.py

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, make_scorer
from mne.decoding import CSP

# --- Import Preprocessing Function and Config ---
# Try to import from sibling directory first, fallback for direct run
try:
    from .preprocess_data import (
        load_and_preprocess_subject_data,
        N_SUBJECTS,
        DATA_PATH,
        RESULTS_PATH
    )
    print(f"Imported paths: DATA={DATA_PATH}, RESULTS={RESULTS_PATH}")
except ImportError:
    print("Could not import from sibling, importing preprocess_data directly.")
    from preprocess_data import (
        load_and_preprocess_subject_data,
        N_SUBJECTS,
        DATA_PATH,
        RESULTS_PATH
    )
    print(f"Imported paths: DATA={DATA_PATH}, RESULTS={RESULTS_PATH}")

# --- Configuration ---
N_CSP_COMPONENTS = 4     # Number of CSP filters
N_SPLITS = 5             # Number of CV folds
SHUFFLE_FOLDS = True     # Shuffle before splitting
RANDOM_STATE = 42        # Random seed
METRIC = 'accuracy'      # Scoring metric
# ------------------------

def run_baseline_evaluation():
    """
    Runs CSP+LDA and CSP+SVM baseline evaluations using cross-validation
    within the training ('T') session data for each subject.
    """

    print("\n=== Starting Baseline Evaluation (CSP + LDA/SVM) ===")
    all_results = []

    for subject_id in range(1, N_SUBJECTS + 1):
        print(f"\n--- Processing Subject {subject_id} ---")

        # 1. Load preprocessed TRAINING data only
        epochs_train = load_and_preprocess_subject_data(subject_id, session_type='T')
        if epochs_train is None or len(epochs_train) == 0:
            print(f"Skipping Subject {subject_id} - no epochs.")
            continue

        # 2. Extract data (X) and labels (y)
        #    Ensure data is float64 and C‐contiguous to satisfy MNE CSP requirements
        X_raw = epochs_train.get_data(copy=True)  # (n_epochs, n_channels, n_times)
        # Force dtype and contiguity
        X = np.ascontiguousarray(X_raw, dtype=np.float64)
        y = epochs_train.events[:, -1]            # event IDs
        n_epochs, n_channels, n_times = X.shape
        print(f"Data shape: {X.shape}, Labels shape: {y.shape}, Unique labels: {np.unique(y)}")

        if n_epochs < N_SPLITS:
            print(f"Skipping Subject {subject_id} - only {n_epochs} epochs (< {N_SPLITS} folds).")
            continue

        # 3. Define pipelines
        csp = CSP(
            n_components=N_CSP_COMPONENTS,
            reg='ledoit_wolf',
            log=True,
            norm_trace=False
        )
        lda = LDA()
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')

        pipe_lda = Pipeline([('CSP', csp), ('LDA', lda)])
        pipe_svm = Pipeline([('CSP', csp), ('SVM', svm)])

        pipelines = {
            'CSP+LDA': pipe_lda,
            'CSP+SVM': pipe_svm
        }

        # 4. Cross‐validation strategy
        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=SHUFFLE_FOLDS,
            random_state=RANDOM_STATE
        )

        # 5. Run CV with n_jobs=1 to avoid multi‐process CSP bug
        for name, pipe in pipelines.items():
            print(f"Running {N_SPLITS}-fold CV for {name} (single‐threaded)...")
            try:
                scores = cross_val_score(
                    pipe, X, y,
                    cv=cv,
                    scoring=METRIC,
                    n_jobs=1                # <-- single‐threaded
                )
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"  Mean {METRIC.capitalize()}: {mean_score:.4f} (+/- {std_score:.4f})")

                all_results.append({
                    'subject': f'A{subject_id:02d}',
                    'pipeline': name,
                    f'mean_{METRIC}': mean_score,
                    f'std_{METRIC}': std_score,
                    'n_epochs': n_epochs
                })

            except ValueError as e:
                print(f"  Error during CV for {name} on Subject {subject_id}: {e}")
                print(f"  Check X shape {X.shape} and labels distribution.")
            except Exception as e:
                print(f"  Unexpected error during CV for {name} on Subject {subject_id}: {e}")

    # --- Process and Save Results ---
    if not all_results:
        print("\nNo baseline results were generated.")
        return

    results_df = pd.DataFrame(all_results)
    print("\n=== Baseline Evaluation Summary ===")
    print(results_df.to_string())

    # Average across subjects
    avg_results = results_df.groupby('pipeline')[f'mean_{METRIC}'].agg(['mean', 'std'])
    print("\n--- Average Performance Across Subjects ---")
    print(avg_results)

    # Ensure results directory exists
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed_csv = RESULTS_PATH / f"baseline_csp_{METRIC}_results.csv"
    results_df.to_csv(detailed_csv, index=False, float_format='%.4f')
    print(f"\nDetailed results saved to {detailed_csv}")

    # Save averages
    avg_csv = RESULTS_PATH / f"baseline_csp_{METRIC}_average.csv"
    avg_results.to_csv(avg_csv, float_format='%.4f')
    print(f"Average results saved to {avg_csv}")


if __name__ == "__main__":
    run_baseline_evaluation()
