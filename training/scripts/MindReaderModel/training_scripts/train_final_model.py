# File: training/scripts/MindReaderModel/training_scripts/train_final_model.py
# Trains the final MindReaderModel on a single subject's full training data.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# --- Import Components ---
try:
    from ..preprocess_data import load_and_preprocess_subject_data
    from ..configs.config import N_SUBJECTS, DATA_PATH, MODELS_PATH, RESULTS_PATH
    from ..configs.sm_config import N_CHANNELS, N_CLASSES, N_SAMPLES
    # Import the final model definition 
    from ..supervised_models import MindReaderModel, N_CLASSES
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
    from supervised_models import MindReaderModel 
    print(f"Imported paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}, RESULTS={RESULTS_PATH}")

# --- Configuration ---
TARGET_SUBJECT_ID = 8 # Train on Subject A08
FINAL_MODEL_NAME = f"mindreader_subject{TARGET_SUBJECT_ID:02d}_final"

# Training parameters 
LEARNING_RATE = 5e-4 
BATCH_SIZE = 32
NUM_EPOCHS = 100 
WEIGHT_DECAY = 1e-4

# Device setup
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Ensure save directories exist
MODELS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# --- Main Training Function ---
def train_final_subject_model(subject_id=TARGET_SUBJECT_ID):
    """
    Loads data for a single subject, trains the MindReaderModel on all
    training data for that subject, and saves the model and scaler.
    """
    print(f"\n=== Starting Final Model Training for Subject {subject_id} ===")

    # 1. Load preprocessed TRAINING data for the target subject
    epochs_train = load_and_preprocess_subject_data(subject_id, session_type='T')

    if epochs_train is None or len(epochs_train) == 0:
        print(f"Error: No training epochs found for Subject {subject_id}. Exiting.")
        return

    # --- Get Class Mapping ---
    if not hasattr(epochs_train, 'event_id') or not epochs_train.event_id:
         print("Error: Loaded epochs object has no event_id dictionary.")
         return
    mne_id_to_task = {v: k for k, v in epochs_train.event_id.items()}
    sorted_mne_ids = sorted(mne_id_to_task.keys())
    mne_id_map = {mne_id: i for i, mne_id in enumerate(sorted_mne_ids)}
    class_names = [mne_id_to_task[mne_id] for mne_id in sorted_mne_ids]
    print(f"Using Subject {subject_id}'s event mapping: {mne_id_map}")
    print(f"Class names (order 0-{N_CLASSES-1}): {class_names}")

    # 2. Extract data (X) and labels (y)
    X_np = epochs_train.get_data(copy=True).astype(np.float64) # Copy for scaling
    y_mne_ids = epochs_train.events[:, -1]
    try:
        y = np.array([mne_id_map[mne_id] for mne_id in y_mne_ids]) # Map to 0-based labels
    except KeyError as e:
        print(f"Error mapping event IDs for Subject {subject_id}. Missing key: {e}")
        return

    n_epochs, n_channels_data, n_samples_data = X_np.shape
    print(f"Training Data shape: {X_np.shape}, Labels shape: {y.shape}")
    assert n_channels_data == N_CHANNELS and n_samples_data == N_SAMPLES

    # 3. Calculate and Apply Scaler (Subject-Specific)
    # Use the same channel-wise time-point scaling as in CV
    print("Calculating subject-specific scaler...")
    scaler = StandardScaler()
    X_reshaped = X_np.reshape(n_epochs * n_channels_data, n_samples_data)
    scaler.fit(X_reshaped) # Fit scaler ONLY on this subject's training data
    print("Applying subject-specific standardisation...")
    X_scaled_reshaped = scaler.transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(n_epochs, n_channels_data, n_samples_data).astype(np.float32)

    # --- Save the Subject-Specific Scaler ---
    scaler_save_path = MODELS_PATH / f"scaler_subject{subject_id:02d}.npz"
    # Save scaler parameters (mean_ and scale_ == standard deviation)
    np.savez(scaler_save_path, mean=scaler.mean_, scale=scaler.scale_)
    print(f"Subject-specific scaler saved to {scaler_save_path}")

    # 4. Prepare PyTorch DataLoader
    train_tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    train_tensor_y = torch.tensor(y, dtype=torch.long) # Use 0-based labels
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    # Shuffle data for training
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Created DataLoader with {len(train_dataset)} samples.")

    # 5. Initialise Model, Loss, Optimiser
    print("Initialising MindReaderModel...")
    model = MindReaderModel(n_channels=N_CHANNELS, n_samples=N_SAMPLES, num_classes=N_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 6. Training Loop (on all subject data)
    print(f"Starting final training for {NUM_EPOCHS} epochs...")
    model.train()
    start_time = time.time()
    epoch_losses = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for i, (batch_epochs, batch_labels) in enumerate(train_loader):
            batch_epochs, batch_labels = batch_epochs.to(DEVICE), batch_labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_epochs) # Pass data directly
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += batch_labels.size(0)
            epoch_correct += (predicted == batch_labels).sum().item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = epoch_correct / epoch_total
        epoch_losses.append(avg_epoch_loss)
        print(f'====> Epoch: {epoch+1}/{NUM_EPOCHS} Average loss: {avg_epoch_loss:.6f}, Accuracy: {epoch_acc:.4f}')


    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # 7. Save the Final Trained Model Weights
    model_save_path = MODELS_PATH / f"{FINAL_MODEL_NAME}.pth"
    print(f"Saving final trained model state dictionary to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    print("Final model saved successfully.")

    # 8. Plot Final Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Subject {subject_id:02d} Final Model Training Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_filename = RESULTS_PATH / f"subject_{subject_id:02d}_final_training_loss.png"
    plt.savefig(loss_plot_filename)
    print(f"Saved final training loss plot to {loss_plot_filename}")
    plt.close()


# --- Run the final training ---
if __name__ == "__main__":
    train_final_subject_model()

