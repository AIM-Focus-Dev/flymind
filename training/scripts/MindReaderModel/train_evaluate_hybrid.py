# File: training/scripts/MindReaderModel/train_evaluate_hybrid.py
# Updated to use features from the EEGNetInspiredConvAE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

from training.scripts.MindReaderModel.train_autoencoder import BATCH_SIZE

# --- Import Components ---
try:
    from .preprocess_data import load_and_preprocess_subject_data, N_SUBJECTS, DATA_PATH, MODELS_PATH, RESULTS_PATH
    # *** Import the EEGNet-Inspired AE definition and its LATENT_DIM ***
    from .supervised_models import EEGNetInspiredConvAE, AE_LATENT_DIM, MLPClassifierPytorch, N_CLASSES
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
    # *** Import the EEGNet-Inspired AE definition and its LATENT_DIM ***
    from supervised_models import EEGNetInspiredConvAE, AE_LATENT_DIM, MLPClassifierPytorch
    print(f"Imported paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}, RESULTS={RESULTS_PATH}")

# --- Configuration ---
# *** Use the latent dimension and paths corresponding to EEGNetInspiredConvAE ***
CURRENT_LATENT_DIM = AE_LATENT_DIM
# *** Path to load the FULL AE model ***
AE_MODEL_LOAD_PATH = MODELS_PATH / f"eegnet_ae_latent{CURRENT_LATENT_DIM}_full.pth"
SCALER_LOAD_PATH = MODELS_PATH / f"eegnet_ae_scaler_latent{CURRENT_LATENT_DIM}.npz"

# CV params
N_SPLITS = 5
SHUFFLE_FOLDS = True
RANDOM_STATE = 42
METRIC = 'accuracy'

# PyTorch MLP Training parameters
PT_LEARNING_RATE = 1e-4
PT_BATCH_SIZE = 32
PT_NUM_EPOCHS = 100
PT_WEIGHT_DECAY = 1e-4
PT_EARLY_STOPPING_PATIENCE = 10

BATCH_SIZE = 64

# SVM parameters
SVM_C = 1.0
SVM_KERNEL = 'rbf'
SVM_GAMMA = 'scale'

# Device setup
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
# --------------------

# *** Modified function to load the FULL AE model ***
def load_full_ae_model(model_path=AE_MODEL_LOAD_PATH, n_channels=N_CHANNELS, n_samples=N_SAMPLES, latent_dim=CURRENT_LATENT_DIM, device=DEVICE):
    """Loads the state dictionary into the full EEGNetInspiredConvAE model."""
    model = EEGNetInspiredConvAE(n_channels=n_channels, n_samples=n_samples, latent_dim=latent_dim)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print(f"Full AE model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: AE model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading AE model: {e}")
        return None

# *** Modified function to use the full AE model's encode method ***
def extract_features_from_ae(ae_model, data_loader, device=DEVICE):
    """Passes data through the full AE model's encode method."""
    if ae_model is None: return None
    all_features = []
    print("Extracting features using loaded AE model's encode method...")
    with torch.no_grad():
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device)
            latent_features = ae_model.encode(batch_data) # Use the model's encode method
            all_features.append(latent_features.cpu().numpy())
    if not all_features: return None
    return np.concatenate(all_features, axis=0)

# --- Keep train_pytorch_mlp_fold function exactly the same ---
def train_pytorch_mlp_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, input_dim, num_classes, device):
    """Trains the PyTorch MLP for a single cross-validation fold."""
    model = MLPClassifierPytorch(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=PT_LEARNING_RATE, weight_decay=PT_WEIGHT_DECAY)
    train_dataset = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32), torch.tensor(y_train_fold, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_fold, dtype=torch.float32), torch.tensor(y_val_fold, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=PT_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=PT_BATCH_SIZE * 2, shuffle=False, num_workers=0)
    print(f"  Training MLP for up to {PT_NUM_EPOCHS} epochs...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(PT_NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PT_EARLY_STOPPING_PATIENCE:
                print(f'    Early stopping triggered at epoch {epoch+1}')
                break
    print(f"  Finished training MLP for fold. Best Val Loss: {best_val_loss:.4f}")
    return model
# -----------------------------------------------------------------------------

# --- Main Evaluation Function ---
def run_hybrid_evaluation():
    """
    Runs EEGNetInspiredConvAE+PyTorchMLP and EEGNetInspiredConvAE+SVM evaluations using CV.
    """
    print("\n=== Starting Hybrid Evaluation (EEGNetInspiredConvAE Features + PyTorch MLP/SVM) ===")

    # 1. Load Scaler
    try:
        scaler_data = np.load(SCALER_LOAD_PATH)
        scaler_mean = scaler_data['mean']
        scaler_std = scaler_data['std']
        print(f"Scaler loaded successfully from {SCALER_LOAD_PATH}")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {SCALER_LOAD_PATH}. Cannot proceed.")
        return
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return

    # 2. Load FULL Autoencoder Model
    # *** Use the function to load the full AE model ***
    ae_model = load_full_ae_model(n_channels=N_CHANNELS, n_samples=N_SAMPLES, latent_dim=CURRENT_LATENT_DIM)
    if ae_model is None:
        print("Could not load AE model. Exiting.")
        return

    # Store results
    all_results = []
    # Map MNE event IDs back to simple 0, 1, 2, 3 labels
    temp_epochs = load_and_preprocess_subject_data(1, session_type='T')
    if temp_epochs is None or not hasattr(temp_epochs, 'event_id') or not temp_epochs.event_id:
        print("Error: Cannot determine event ID mapping.")
        return
    mne_id_to_task = {v: k for k, v in temp_epochs.event_id.items()}
    sorted_tasks = sorted(mne_id_to_task.keys())
    mne_id_map = {mne_id: i for i, mne_id in enumerate(sorted_tasks)}
    print(f"Mapping MNE event IDs to 0-based labels: {mne_id_map}")
    del temp_epochs

    for subject_id in range(1, N_SUBJECTS + 1):
        print(f"\n--- Processing Subject {subject_id} ---")

        # 3. Load preprocessed TRAINING data only
        epochs_train = load_and_preprocess_subject_data(subject_id, session_type='T')
        if epochs_train is None or len(epochs_train) == 0: continue

        # 4. Prepare Data for Encoder
        X_np = epochs_train.get_data(copy=False).astype(np.float64)
        y_mne_ids = epochs_train.events[:, -1]
        try:
            y = np.array([mne_id_map[mne_id] for mne_id in y_mne_ids])
        except KeyError as e:
            print(f"Error mapping event IDs for Subject {subject_id}. Missing key: {e}")
            continue

        n_epochs, n_channels_data, n_samples_data = X_np.shape
        print(f"Original data shape: {X_np.shape}, Labels shape: {y.shape}, Unique labels: {np.unique(y)}")
        if n_epochs < N_SPLITS: continue

        # Apply Scaler Correctly
        print("Applying saved scaler (Flatten -> Scale -> Reshape)...")
        X_flat = X_np.reshape(n_epochs, -1)
        if X_flat.shape[1] != len(scaler_mean):
             print(f"Error: Scaler dimensions ({len(scaler_mean)}) do not match data ({X_flat.shape[1]}).")
             continue
        X_scaled_flat = ((X_flat - scaler_mean) / scaler_std).astype(np.float32)
        X_scaled = X_scaled_flat.reshape(n_epochs, n_channels_data, n_samples_data)
        print(f"Scaled & Reshaped data shape: {X_scaled.shape}")

        # Convert to Tensor and create DataLoader for feature extraction
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        dummy_targets = torch.zeros(n_epochs)
        feature_dataset = TensorDataset(X_tensor, dummy_targets)
        # *** Use defined BATCH_SIZE for feature loader ***
        feature_loader = DataLoader(feature_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # 5. Extract Latent Features using loaded EEGNet-Inspired AE
        # *** Use the function that takes the full AE model ***
        X_features = extract_features_from_ae(ae_model, feature_loader, device=DEVICE)
        if X_features is None: continue
        print(f"Extracted features shape: {X_features.shape}")
        actual_latent_dim = X_features.shape[1]
        if actual_latent_dim != CURRENT_LATENT_DIM:
             print(f"Warning: Extracted feature dim ({actual_latent_dim}) differs from config ({CURRENT_LATENT_DIM}). Using {actual_latent_dim} for MLP.")

        # 6. Define Cross-Validation Strategy
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE_FOLDS, random_state=RANDOM_STATE)

        # --- Evaluate EEGNetInspiredConvAE + PyTorch MLP ---
        print(f"Running {N_SPLITS}-fold CV for EEGNetAE+PyTorchMLP...") # Updated name
        fold_scores_mlp = []
        fold_idx = 0
        for train_index, val_index in cv.split(X_features, y):
            fold_idx += 1
            print(f"  Fold {fold_idx}/{N_SPLITS}...")
            X_train_fold, X_val_fold = X_features[train_index], X_features[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]
            scaler_feat = StandardScaler()
            X_train_fold = scaler_feat.fit_transform(X_train_fold)
            X_val_fold = scaler_feat.transform(X_val_fold)
            trained_mlp = train_pytorch_mlp_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                                 input_dim=actual_latent_dim, # Use actual dim
                                                 num_classes=N_CLASSES, device=DEVICE)
            trained_mlp.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(DEVICE)
                outputs = trained_mlp(X_val_tensor)
                _, predicted = torch.max(outputs.data, 1)
                y_pred_fold = predicted.cpu().numpy()
            acc = accuracy_score(y_val_fold, y_pred_fold)
            fold_scores_mlp.append(acc)
            print(f"  Fold {fold_idx} Accuracy: {acc:.4f}")
        mean_score_mlp = np.mean(fold_scores_mlp)
        std_score_mlp = np.std(fold_scores_mlp)
        print(f"  Overall Mean Accuracy (EEGNetAE+PyTorchMLP): {mean_score_mlp:.4f} (+/- {std_score_mlp:.4f})") # Updated name
        all_results.append({
            'subject': f'A{subject_id:02d}', 'pipeline': f'EEGNetAE{actual_latent_dim}+PyTorchMLP', # Updated name
            f'mean_{METRIC}': mean_score_mlp, f'std_{METRIC}': std_score_mlp,
            'n_epochs': n_epochs
        })

        # --- Evaluate EEGNetInspiredConvAE + SVM ---
        print(f"\nRunning {N_SPLITS}-fold CV for EEGNetAE+SVM...") # Updated name
        try:
            svm = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=False)
            pipeline_svm = Pipeline([('scaler', StandardScaler()), ('SVM', svm)])
            scores_svm = cross_val_score(pipeline_svm, X_features, y, cv=cv, scoring=METRIC, n_jobs=-1)
            mean_score_svm = np.mean(scores_svm)
            std_score_svm = np.std(scores_svm)
            print(f"  Mean Accuracy (EEGNetAE+SVM): {mean_score_svm:.4f} (+/- {std_score_svm:.4f})") # Updated name
            all_results.append({
                'subject': f'A{subject_id:02d}', 'pipeline': f'EEGNetAE{actual_latent_dim}+SVM', # Updated name
                f'mean_{METRIC}': mean_score_svm, f'std_{METRIC}': std_score_svm,
                'n_epochs': n_epochs
            })
        except Exception as e:
            print(f"  An unexpected error occurred during CV for EEGNetAE+SVM on Subject {subject_id}: {e}")


    # --- Process and Save Results ---
    if not all_results:
        print("\nNo hybrid EEGNetAE model results were generated.")
        return

    results_df = pd.DataFrame(all_results)
    print("\n=== Hybrid EEGNetAE Evaluation Summary (Within-Session CV) ===") # Updated name
    print(results_df.to_string())
    avg_results = results_df.groupby('pipeline')[f'mean_{METRIC}'].agg(['mean', 'std'])
    print("\n--- Average Performance Across Subjects ---")
    print(avg_results)
    # *** Update results filenames ***
    results_filename = RESULTS_PATH / f"hybrid_eegnet_ae_pytorch_{METRIC}_results.csv"
    results_df.to_csv(results_filename, index=False, float_format='%.4f')
    print(f"\nDetailed results saved to {results_filename}")
    avg_results_filename = RESULTS_PATH / f"hybrid_eegnet_ae_pytorch_{METRIC}_average.csv"
    avg_results.to_csv(avg_results_filename, float_format='%.4f')
    print(f"Average results saved to {avg_results_filename}")

# --- Run the evaluation ---
if __name__ == "__main__":
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    run_hybrid_evaluation()
