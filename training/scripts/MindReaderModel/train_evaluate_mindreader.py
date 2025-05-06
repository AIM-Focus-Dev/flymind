# File: training/scripts/MindReaderModel/train_evaluate_mindreader.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix

# --- Import Components ---
try:
    from .preprocess_data import load_and_preprocess_subject_data, N_SUBJECTS, DATA_PATH, MODELS_PATH, RESULTS_PATH
    # *** Import the renamed MindReaderModel ***
    from .supervised_models import MindReaderModel, N_CLASSES
    print(f"Imported paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}, RESULTS={RESULTS_PATH}")
except ImportError:
    print("Could not import from siblings, importing directly.")
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
    except NameError:
        PROJECT_ROOT = Path('.').resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
    MODELS_PATH = PROJECT_ROOT / "training" / "models"
    RESULTS_PATH = PROJECT_ROOT / "training" / "results"
    N_SUBJECTS = 9
    N_CHANNELS = 22
    N_SAMPLES = 751
    N_CLASSES = 4
    from preprocess_data import load_and_preprocess_subject_data
    # *** Import the renamed MindReaderModel ***
    from supervised_models import MindReaderModel
    print(f"Imported paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}, RESULTS={RESULTS_PATH}")

# --- Configuration ---
# CV params
N_SPLITS = 5
SHUFFLE_FOLDS = True
RANDOM_STATE = 42
METRIC = 'accuracy' # Primary metric for reporting average

# MindReaderModel Training parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 150 # Max epochs
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15

# Device setup
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
# --------------------

def plot_training_history(history, subject_id, results_path):
    """Plots training/validation loss and accuracy curves."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, history['train_loss'], color=color, linestyle='--', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color=color, label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, history['val_acc'], color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title(f'Subject {subject_id:02d} Training History (Avg over Folds)')
    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plot_filename = results_path / f"subject_{subject_id:02d}_training_history.png"
    plt.savefig(plot_filename)
    print(f"Saved training history plot to {plot_filename}")
    plt.close(fig) # Close the figure to free memory

def plot_confusion_matrix(y_true, y_pred, classes, subject_id, results_path):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Subject {subject_id:02d} Confusion Matrix (Combined Val Folds)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plot_filename = results_path / f"subject_{subject_id:02d}_confusion_matrix.png"
    plt.savefig(plot_filename)
    print(f"Saved confusion matrix plot to {plot_filename}")
    plt.close()

def train_mindreader_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                          n_channels, n_samples, num_classes, device):
    """Trains the MindReaderModel for a single fold and returns history."""
    model = MindReaderModel(n_channels=n_channels, n_samples=n_samples, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Data Scaling within Fold ---
    scaler = StandardScaler()
    n_train, c_train, t_train = X_train_fold.shape
    X_train_reshaped = X_train_fold.reshape(n_train * c_train, t_train)
    scaler.fit(X_train_reshaped)
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(n_train, c_train, t_train)

    n_val, c_val, t_val = X_val_fold.shape
    X_val_reshaped = X_val_fold.reshape(n_val * c_val, t_val)
    X_val_scaled_reshaped = scaler.transform(X_val_reshaped)
    X_val_scaled = X_val_scaled_reshaped.reshape(n_val, c_val, t_val)
    # --- End Scaling ---

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_fold, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val_fold, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    print(f"  Training MindReaderModel for up to {NUM_EPOCHS} epochs...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []} # Store history

    for epoch in range(NUM_EPOCHS):
        model.train()
        batch_train_losses = []
        for epochs, labels in train_loader:
            epochs, labels = epochs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(epochs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        avg_train_loss = np.mean(batch_train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        batch_val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for epochs, labels in val_loader:
                epochs, labels = epochs.to(device), labels.to(device)
                outputs = model(epochs)
                loss = criterion(outputs, labels)
                batch_val_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        avg_val_loss = np.mean(batch_val_losses)
        val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f'    Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # torch.save(model.state_dict(), f"temp_best_mindreader_fold.pth") # Optional save best
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f'    Early stopping triggered at epoch {epoch+1}')
                break

    # model.load_state_dict(torch.load(f"temp_best_mindreader_fold.pth")) # Optional load best
    print(f"  Finished training MindReaderModel for fold. Best Val Loss: {best_val_loss:.4f}")
    return model, history # Return model and history

# --- Main Evaluation Function ---
def run_mindreader_evaluation():
    """
    Trains and evaluates MindReaderModel using cross-validation with plotting.
    """
    print("\n=== Starting Supervised MindReaderModel Evaluation ===")

    all_subject_results = []
    # Map MNE event IDs back to simple 0, 1, 2, 3 labels and get class names
    temp_epochs = load_and_preprocess_subject_data(1, session_type='T')
    if temp_epochs is None or not hasattr(temp_epochs, 'event_id') or not temp_epochs.event_id:
        print("Error: Cannot determine event ID mapping.")
        return
    mne_id_to_task = {v: k for k, v in temp_epochs.event_id.items()}
    sorted_mne_ids = sorted(mne_id_to_task.keys())
    mne_id_map = {mne_id: i for i, mne_id in enumerate(sorted_mne_ids)}
    class_names = [mne_id_to_task[mne_id] for mne_id in sorted_mne_ids] # Get class names in order
    print(f"Mapping MNE event IDs to 0-based labels: {mne_id_map}")
    print(f"Class names (order 0, 1, 2, 3): {class_names}")
    del temp_epochs

    for subject_id in range(1, N_SUBJECTS + 1):
        print(f"\n--- Processing Subject {subject_id} ---")

        # 1. Load preprocessed TRAINING data only
        epochs_train = load_and_preprocess_subject_data(subject_id, session_type='T')
        if epochs_train is None or len(epochs_train) == 0: continue

        # 2. Extract data (X) and labels (y)
        X = epochs_train.get_data(copy=True).astype(np.float64)
        y_mne_ids = epochs_train.events[:, -1]
        try:
            y = np.array([mne_id_map[mne_id] for mne_id in y_mne_ids])
        except KeyError as e:
            print(f"Error mapping event IDs for Subject {subject_id}. Missing key: {e}")
            continue

        n_epochs, n_channels_data, n_samples_data = X.shape
        print(f"Data shape: {X.shape}, Labels shape: {y.shape}, Unique labels: {np.unique(y)}")
        if n_epochs < N_SPLITS: continue
        if n_channels_data != N_CHANNELS or n_samples_data != N_SAMPLES: continue

        # 3. Define Cross-Validation Strategy
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE_FOLDS, random_state=RANDOM_STATE)

        # 4. Run cross-validation loop
        print(f"Running {N_SPLITS}-fold CV for MindReaderModel...")
        fold_scores = []
        all_fold_histories = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        y_true_all_folds = []
        y_pred_all_folds = []
        fold_idx = 0

        for train_index, val_index in cv.split(X, y):
            fold_idx += 1
            print(f"  Fold {fold_idx}/{N_SPLITS}...")
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Train the MindReaderModel model for this fold
            trained_model, history = train_mindreader_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                n_channels=N_CHANNELS, n_samples=N_SAMPLES,
                num_classes=N_CLASSES, device=DEVICE
            )

            # Store history for averaging later
            # Pad shorter histories if early stopping occurred
            max_len = NUM_EPOCHS
            for key in all_fold_histories.keys():
                 padded_hist = history[key] + [np.nan] * (max_len - len(history[key]))
                 all_fold_histories[key].append(padded_hist[:max_len]) # Ensure fixed length

            # Evaluate on the validation set for this fold
            trained_model.eval()
            # Re-apply scaling for evaluation
            temp_scaler = StandardScaler()
            n_train_f, c_train_f, t_train_f = X_train_fold.shape
            X_train_fold_reshaped = X_train_fold.reshape(n_train_f * c_train_f, t_train_f)
            temp_scaler.fit(X_train_fold_reshaped)
            n_val_f, c_val_f, t_val_f = X_val_fold.shape
            X_val_fold_reshaped = X_val_fold.reshape(n_val_f * c_val_f, t_val_f)
            X_val_fold_scaled_reshaped = temp_scaler.transform(X_val_fold_reshaped)
            X_val_fold_scaled = X_val_fold_scaled_reshaped.reshape(n_val_f, c_val_f, t_val_f)

            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val_fold_scaled, dtype=torch.float32).to(DEVICE)
                outputs = trained_model(X_val_tensor)
                _, predicted = torch.max(outputs.data, 1)
                y_pred_fold = predicted.cpu().numpy()

            acc = accuracy_score(y_val_fold, y_pred_fold)
            fold_scores.append(acc)
            y_true_all_folds.extend(y_val_fold)
            y_pred_all_folds.extend(y_pred_fold)
            print(f"  Fold {fold_idx} Accuracy: {acc:.4f}")

        # Calculate average history across folds (ignoring NaNs from early stopping)
        avg_history = {k: np.nanmean(np.array(v), axis=0) for k, v in all_fold_histories.items()}

        # Plot training history averaged over folds
        plot_training_history(avg_history, subject_id, RESULTS_PATH)

        # Plot confusion matrix for all predictions from this subject's CV
        plot_confusion_matrix(y_true_all_folds, y_pred_all_folds, class_names, subject_id, RESULTS_PATH)

        # Store overall subject results
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f"  Overall Mean Accuracy (MindReaderModel): {mean_score:.4f} (+/- {std_score:.4f})")
        all_subject_results.append({
            'subject': f'A{subject_id:02d}',
            'pipeline': 'MindReaderModel', # Updated name
            f'mean_{METRIC}': mean_score,
            f'std_{METRIC}': std_score,
            'n_epochs_data': n_epochs
        })

        # Optional: Save the final model trained on one fold, or train a final model on all data
        # torch.save(trained_model.state_dict(), MODELS_PATH / f"mindreader_subject_{subject_id:02d}_final.pth")


    # --- Process and Save Overall Results ---
    if not all_subject_results:
        print("\nNo MindReaderModel results were generated.")
        return

    results_df = pd.DataFrame(all_subject_results)
    print("\n=== MindReaderModel Evaluation Summary (Within-Session CV) ===")
    print(results_df.to_string())

    avg_results = results_df.groupby('pipeline')[f'mean_{METRIC}'].agg(['mean', 'std'])
    print("\n--- Average Performance Across Subjects ---")
    print(avg_results)

    results_filename = RESULTS_PATH / f"mindreader_{METRIC}_results.csv" # Updated filename
    results_df.to_csv(results_filename, index=False, float_format='%.4f')
    print(f"\nDetailed results saved to {results_filename}")

    avg_results_filename = RESULTS_PATH / f"mindreader_{METRIC}_average.csv" # Updated filename
    avg_results.to_csv(avg_results_filename, float_format='%.4f')
    print(f"Average results saved to {avg_results_filename}")

# --- Run the evaluation ---
if __name__ == "__main__":
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    run_mindreader_evaluation()
