# File: training/scripts/MindReaderModel/train_evaluate_eegnet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler # For scaling within folds if needed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

# --- Import Components ---
try:
    # Assuming execution from project root or configured PYTHONPATH
    from .preprocess_data import load_and_preprocess_subject_data, N_SUBJECTS, DATA_PATH, MODELS_PATH, RESULTS_PATH
    # *** Import EEGNet model ***
    from .supervised_models import EEGNet, N_CLASSES
    print(f"Imported paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}, RESULTS={RESULTS_PATH}")
except ImportError:
    # Fallback if running script directly from its own directory
    print("Could not import from siblings, importing directly.")
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
    except NameError:
        PROJECT_ROOT = Path('.').resolve().parents[2] # Adjust if running interactively from script dir
    DATA_PATH = PROJECT_ROOT / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
    MODELS_PATH = PROJECT_ROOT / "training" / "models"
    RESULTS_PATH = PROJECT_ROOT / "training" / "results"
    N_SUBJECTS = 9
    N_CHANNELS = 22
    N_SAMPLES = 751
    N_CLASSES = 4
    from preprocess_data import load_and_preprocess_subject_data
    # *** Import EEGNet model ***
    from supervised_models import EEGNet
    print(f"Imported paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}, RESULTS={RESULTS_PATH}")

# --- Configuration ---
# CV params
N_SPLITS = 5
SHUFFLE_FOLDS = True
RANDOM_STATE = 42
METRIC = 'accuracy'

# EEGNet Training parameters
LEARNING_RATE = 1e-3 # Learning rate might need tuning for EEGNet
BATCH_SIZE = 32    # Smaller batch size often good for CNNs
NUM_EPOCHS = 150   # EEGNet might need more epochs, use early stopping
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15 # Increase patience slightly

# Device setup
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
# --------------------

def train_eegnet_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, n_channels, n_samples, num_classes, device):
    """Trains the EEGNet model for a single cross-validation fold."""
    # Input X shape: (n_epochs, n_channels, n_samples)
    model = EEGNet(n_channels=n_channels, n_samples=n_samples, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Data Scaling within Fold ---
    # Apply standardization across the time dimension for each channel independently per epoch
    # This helps normalize signal power while preserving relative patterns
    scaler = StandardScaler() # Fit scaler only on training fold data

    # Reshape for scaling: (epochs * channels, samples)
    n_train, c_train, t_train = X_train_fold.shape
    X_train_reshaped = X_train_fold.reshape(n_train * c_train, t_train)
    scaler.fit(X_train_reshaped) # Fit on reshaped training data

    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(n_train, c_train, t_train) # Reshape back

    # Apply the same scaler to validation data
    n_val, c_val, t_val = X_val_fold.shape
    X_val_reshaped = X_val_fold.reshape(n_val * c_val, t_val)
    X_val_scaled_reshaped = scaler.transform(X_val_reshaped)
    X_val_scaled = X_val_scaled_reshaped.reshape(n_val, c_val, t_val) # Reshape back
    # --- End Scaling ---

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_fold, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val_fold, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    print(f"  Training EEGNet for up to {NUM_EPOCHS} epochs...")
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for epochs, labels in train_loader:
            # Input shape for EEGNet: (batch, 1, channels, samples)
            epochs, labels = epochs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(epochs) # Pass data directly
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for epochs, labels in val_loader:
                epochs, labels = epochs.to(device), labels.to(device)
                outputs = model(epochs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Simple print for progress
        if (epoch + 1) % 10 == 0:
             print(f'    Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Optional: Save best model state
            # torch.save(model.state_dict(), f"temp_best_eegnet_fold.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f'    Early stopping triggered at epoch {epoch+1}')
                break

    # Optional: Load best model state if saved
    # model.load_state_dict(torch.load(f"temp_best_eegnet_fold.pth"))
    print(f"  Finished training EEGNet for fold. Best Val Loss: {best_val_loss:.4f}")
    return model # Return the trained model for this fold

# --- Main Evaluation Function ---
def run_eegnet_evaluation():
    """
    Trains and evaluates EEGNet directly on preprocessed EEG epochs using cross-validation
    within the training ('T') session data for each subject.
    """
    print("\n=== Starting Supervised EEGNet Evaluation ===")

    # Store results per subject
    all_results = []
    # Map MNE event IDs back to simple 0, 1, 2, 3 labels
    temp_epochs = load_and_preprocess_subject_data(1, session_type='T')
    if temp_epochs is None:
        print("Error: Cannot determine event ID mapping.")
        return
    if not hasattr(temp_epochs, 'event_id') or not temp_epochs.event_id:
         print("Error: Loaded epochs object has no event_id dictionary.")
         return
    mne_id_to_task = {v: k for k, v in temp_epochs.event_id.items()}
    sorted_tasks = sorted(mne_id_to_task.keys())
    mne_id_map = {mne_id: i for i, mne_id in enumerate(sorted_tasks)}
    print(f"Mapping MNE event IDs to 0-based labels: {mne_id_map}")
    del temp_epochs

    for subject_id in range(1, N_SUBJECTS + 1):
        print(f"\n--- Processing Subject {subject_id} ---")

        # 1. Load preprocessed TRAINING data only
        epochs_train = load_and_preprocess_subject_data(subject_id, session_type='T')

        if epochs_train is None or len(epochs_train) == 0:
            print(f"Skipping Subject {subject_id} - No training epochs found/processed.")
            continue

        # 2. Extract data (X) and labels (y)
        # *** Use unflattened data directly ***
        X = epochs_train.get_data(copy=True).astype(np.float64) # Use float64 for scaling stability
        y_mne_ids = epochs_train.events[:, -1]
        try:
            y = np.array([mne_id_map[mne_id] for mne_id in y_mne_ids]) # Map to 0-based labels
        except KeyError as e:
            print(f"Error mapping event IDs for Subject {subject_id}. Missing key: {e}")
            continue

        n_epochs, n_channels_data, n_samples_data = X.shape
        print(f"Data shape: {X.shape}, Labels shape: {y.shape}, Unique labels: {np.unique(y)}")

        if n_epochs < N_SPLITS:
             print(f"Skipping Subject {subject_id} - Only {n_epochs} epochs, less than n_splits={N_SPLITS} for CV.")
             continue
        if n_channels_data != N_CHANNELS or n_samples_data != N_SAMPLES:
             print(f"Warning: Data dimensions ({n_channels_data}x{n_samples_data}) differ from config ({N_CHANNELS}x{N_SAMPLES}) for Subject {subject_id}. Skipping.")
             continue

        # 3. Define Cross-Validation Strategy
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE_FOLDS, random_state=RANDOM_STATE)

        # 4. Run cross-validation loop
        print(f"Running {N_SPLITS}-fold CV for EEGNet...")
        fold_scores = []
        fold_idx = 0
        for train_index, val_index in cv.split(X, y): # Split original data
            fold_idx += 1
            print(f"  Fold {fold_idx}/{N_SPLITS}...")
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Train the EEGNet model for this fold (includes internal scaling)
            trained_eegnet = train_eegnet_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                                n_channels=N_CHANNELS, n_samples=N_SAMPLES,
                                                num_classes=N_CLASSES, device=DEVICE)

            # Evaluate on the validation set for this fold
            trained_eegnet.eval()
            # --- Apply scaling to validation fold data using scaler fitted on train fold ---
            # This requires getting the scaler back from the training function or recalculating
            # For simplicity here, we re-apply the scaling logic as done inside train_eegnet_fold
            # NOTE: Ideally, the scaler object from training should be returned and reused
            temp_scaler = StandardScaler()
            n_train_f, c_train_f, t_train_f = X_train_fold.shape
            X_train_fold_reshaped = X_train_fold.reshape(n_train_f * c_train_f, t_train_f)
            temp_scaler.fit(X_train_fold_reshaped) # Fit scaler
            n_val_f, c_val_f, t_val_f = X_val_fold.shape
            X_val_fold_reshaped = X_val_fold.reshape(n_val_f * c_val_f, t_val_f)
            X_val_fold_scaled_reshaped = temp_scaler.transform(X_val_fold_reshaped)
            X_val_fold_scaled = X_val_fold_scaled_reshaped.reshape(n_val_f, c_val_f, t_val_f)
            # --- End scaling re-application ---

            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val_fold_scaled, dtype=torch.float32).to(DEVICE)
                outputs = trained_eegnet(X_val_tensor)
                _, predicted = torch.max(outputs.data, 1)
                y_pred_fold = predicted.cpu().numpy()

            acc = accuracy_score(y_val_fold, y_pred_fold)
            fold_scores.append(acc)
            print(f"  Fold {fold_idx} Accuracy: {acc:.4f}")

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f"  Overall Mean Accuracy (EEGNet): {mean_score:.4f} (+/- {std_score:.4f})")
        all_results.append({
            'subject': f'A{subject_id:02d}',
            'pipeline': 'EEGNet', # Pipeline name
            f'mean_{METRIC}': mean_score,
            f'std_{METRIC}': std_score,
            'n_epochs_trained': epoch + 1 if 'epoch' in locals() else NUM_EPOCHS, # Approx epochs if early stopped
            'n_epochs_data': n_epochs # Number of epochs in subject's data
        })


    # --- Process and Save Results ---
    if not all_results:
        print("\nNo EEGNet results were generated.")
        return

    results_df = pd.DataFrame(all_results)
    print("\n=== EEGNet Evaluation Summary (Within-Session CV) ===")
    print(results_df.to_string())

    avg_results = results_df.groupby('pipeline')[f'mean_{METRIC}'].agg(['mean', 'std'])
    print("\n--- Average Performance Across Subjects ---")
    print(avg_results)

    # Save detailed results to CSV
    results_filename = RESULTS_PATH / f"eegnet_{METRIC}_results.csv"
    results_df.to_csv(results_filename, index=False, float_format='%.4f')
    print(f"\nDetailed results saved to {results_filename}")

    # Save average results
    avg_results_filename = RESULTS_PATH / f"eegnet_{METRIC}_average.csv"
    avg_results.to_csv(avg_results_filename, float_format='%.4f')
    print(f"Average results saved to {avg_results_filename}")

# --- Run the evaluation ---
if __name__ == "__main__":
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    run_eegnet_evaluation()
