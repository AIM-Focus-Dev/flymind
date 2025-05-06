# File: training/scripts/MindReaderModel/train_autoencoder.py
# Updated to train the EEGNetInspiredConvAE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import mne
from pathlib import Path
import time

# --- Import Components ---
try:
    from .preprocess_data import load_and_preprocess_subject_data, N_SUBJECTS, DATA_PATH, MODELS_PATH
    # *** Import the EEGNet-Inspired AE and its specific LATENT_DIM ***
    from .supervised_models import EEGNetInspiredConvAE, AE_LATENT_DIM # Use AE_LATENT_DIM
    print(f"Imported paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}")
except ImportError:
    print("Could not import from siblings, importing directly.")
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
    except NameError:
        PROJECT_ROOT = Path('.').resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
    MODELS_PATH = PROJECT_ROOT / "training" / "models"
    N_SUBJECTS = 9
    N_CHANNELS = 22
    N_SAMPLES = 751
    from preprocess_data import load_and_preprocess_subject_data
    # *** Import the EEGNet-Inspired AE and its specific LATENT_DIM ***
    from supervised_models import EEGNetInspiredConvAE, AE_LATENT_DIM # Use AE_LATENT_DIM
    print(f"Imported paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}")
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    NUM_EPOCHS = 50 # Keep epochs same for now, can increase later

# --- Configuration ---
# *** Use the latent dimension from the imported model ***
CURRENT_LATENT_DIM = AE_LATENT_DIM

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

MODELS_PATH.mkdir(parents=True, exist_ok=True)
# *** Update save path name for EEGNet-Inspired AE ***
ENCODER_SAVE_PATH = MODELS_PATH / f"encoder_eegnet_ae_latent{CURRENT_LATENT_DIM}.pth"
SCALER_SAVE_PATH = MODELS_PATH / f"eegnet_ae_scaler_latent{CURRENT_LATENT_DIM}.npz"
# --------------------

def train_eegnet_inspired_autoencoder():
    """Loads data, calculates scaler, trains the EEGNetInspiredConvAE,
       and saves the encoder weights and scaler."""
    print(f"=== Starting EEGNet-Inspired ConvAE Training (Latent Dim: {CURRENT_LATENT_DIM}) ===")

    # 1. Load and Prepare Data (Same as before)
    print("Loading and concatenating training data for all subjects...")
    all_train_epochs_list = []
    for subject_id in range(1, N_SUBJECTS + 1):
        print(f"  Loading Subject {subject_id}T...")
        epochs_train = load_and_preprocess_subject_data(subject_id, session_type='T')
        if epochs_train is not None and len(epochs_train) > 0:
            epoch_data = epochs_train.get_data(copy=False)
            all_train_epochs_list.append(epoch_data)
        else:
            print(f"  Skipping Subject {subject_id}T (no data).")

    if not all_train_epochs_list:
        print("Error: No training data loaded. Exiting.")
        return

    X_train_all = np.concatenate(all_train_epochs_list, axis=0)
    n_total_epochs, n_channels_data, n_samples_data = X_train_all.shape
    print(f"Total training epochs concatenated shape: {X_train_all.shape}")
    assert n_channels_data == N_CHANNELS and n_samples_data == N_SAMPLES

    # --- SCALING (Using feature-wise on flattened data, then reshape) ---
    # (Consider channel-wise scaling later if results are poor)
    print("Calculating scaler on flattened training data...")
    X_train_flat = X_train_all.reshape(n_total_epochs, -1)
    X_train_flat_64 = X_train_flat.astype(np.float64)
    scaler_mean = np.mean(X_train_flat_64, axis=0)
    scaler_std = np.std(X_train_flat_64, axis=0)
    epsilon = 1e-8
    scaler_std[scaler_std < epsilon] = epsilon
    print("Applying standardization to flattened data...")
    X_train_scaled_flat = ((X_train_flat_64 - scaler_mean) / scaler_std).astype(np.float32)
    X_train_scaled = X_train_scaled_flat.reshape(n_total_epochs, n_channels_data, n_samples_data)
    print(f"Reshaped Scaled Data Shape: {X_train_scaled.shape}")
    # --- Save the scaler ---
    np.savez(SCALER_SAVE_PATH, mean=scaler_mean, std=scaler_std, type='featurewise')
    print(f"Scaler saved to {SCALER_SAVE_PATH}")
    # --- End Scaling ---

    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor, train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    print(f"Created DataLoader with batch size {BATCH_SIZE}")

    # 2. Initialize Model, Loss, Optimizer
    # *** Instantiate the EEGNetInspiredConvAE ***
    print(f"Initializing EEGNetInspiredConvAE (Latent: {CURRENT_LATENT_DIM})...")
    model = EEGNetInspiredConvAE(n_channels=N_CHANNELS, n_samples=N_SAMPLES, latent_dim=CURRENT_LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Start with default LR

    # 3. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    model.train()
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for i, (batch_data, _) in enumerate(train_loader):
            # *** batch_data shape is (batch_size, n_channels, n_samples) ***
            batch_data = batch_data.to(DEVICE)
            outputs = model(batch_data) # Model expects this shape now
            loss = criterion(outputs, batch_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'====> Epoch: {epoch+1}/{NUM_EPOCHS} Average loss: {avg_epoch_loss:.8f}')
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # 4. Save the Trained Encoder Weights
    print(f"Saving trained encoder state dictionary to: {ENCODER_SAVE_PATH}")
    # Save the state_dict of the encoder part (which is the whole model up to fc_encode)
    # We need to ensure the 'encoder' attribute exists or save specific layers
    # In the EEGNetInspiredConvAE, the encode method uses multiple attributes.
    # For simplicity, let's save the whole model and load the whole model in the hybrid script,
    # then just use its .encode() method. Alternatively, restructure EEGNetInspiredConvAE
    # to have a self.encoder = nn.Sequential(...) like the first ConvAE.
    # Saving the whole model is easier for now:
    WHOLE_MODEL_SAVE_PATH = MODELS_PATH / f"eegnet_ae_latent{CURRENT_LATENT_DIM}_full.pth"
    torch.save(model.state_dict(), WHOLE_MODEL_SAVE_PATH)
    print(f"Full AE model saved to {WHOLE_MODEL_SAVE_PATH}")
    # If you MUST save only the encoder part, you'd need to restructure the model definition
    # or manually save the state_dict of constituent layers:
    # encoder_state = {
    #     **model.conv1.state_dict(), **model.bn1_enc.state_dict(), ..., **model.fc_encode.state_dict()
    # }
    # torch.save(encoder_state, ENCODER_SAVE_PATH)


# --- Run the training ---
if __name__ == "__main__":
    train_eegnet_inspired_autoencoder()
