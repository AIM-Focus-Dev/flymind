# File: training/scripts/MindReaderModel/train_autoencoder.py
# Updated for ConvAutoencoder1D

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
    # Assuming execution from project root or configured PYTHONPATH
    from .preprocess_data import load_and_preprocess_subject_data, N_SUBJECTS, DATA_PATH, MODELS_PATH # Import dims too
    # *** Import the CORRECT model ***
    from .autoencoder_model import ConvAutoencoder1D, LATENT_DIM # Import ConvAE and Latent Dim
    print(f"Imported paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}")
except ImportError:
    # Fallback if running script directly from its own directory
    print("Could not import from siblings, importing directly.")
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
    except NameError:
        PROJECT_ROOT = Path('.').resolve().parents[2] # Adjust if running interactively from script dir
    DATA_PATH = PROJECT_ROOT / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
    MODELS_PATH = PROJECT_ROOT / "training" / "models"
    N_SUBJECTS = 9
    N_CHANNELS = 22 # Need these if running directly
    N_SAMPLES = 751
    from preprocess_data import load_and_preprocess_subject_data
    # *** Import the CORRECT model ***
    from autoencoder_model import ConvAutoencoder1D, LATENT_DIM
    print(f"Imported paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}")
    # Define default training params if config doesn't exist/isn't used yet
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64 # Keep consistent for now
    NUM_EPOCHS = 100# Keep consistent for now
    # LATENT_DIM = 128 # Ensure this matches the definition in autoencoder_model.py


# --- Configuration ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

MODELS_PATH.mkdir(parents=True, exist_ok=True)
# *** Update save path name ***
ENCODER_SAVE_PATH = MODELS_PATH / f"encoder_conv_ae_latent{LATENT_DIM}.pth"
SCALER_SAVE_PATH = MODELS_PATH / "conv_ae_scaler.npz" # Use a different scaler file potentially
# --------------------

def train_conv_autoencoder():
    """Loads data, calculates scaler, trains the ConvAutoencoder1D on scaled data,
       and saves the encoder weights and scaler."""
    print("=== Starting Convolutional Autoencoder Training ===")

    # 1. Load and Prepare Data
    print("Loading and concatenating training data for all subjects...")
    all_train_epochs_list = []
    for subject_id in range(1, N_SUBJECTS + 1):
        print(f"  Loading Subject {subject_id}T...")
        epochs_train = load_and_preprocess_subject_data(subject_id, session_type='T')
        if epochs_train is not None and len(epochs_train) > 0:
            epoch_data = epochs_train.get_data(copy=False) # Shape (n_epochs, n_channels, n_samples)
            all_train_epochs_list.append(epoch_data)
        else:
            print(f"  Skipping Subject {subject_id}T (no data).")

    if not all_train_epochs_list:
        print("Error: No training data loaded. Exiting.")
        return

    X_train_all = np.concatenate(all_train_epochs_list, axis=0)
    n_total_epochs, n_channels_data, n_samples_data = X_train_all.shape
    print(f"Total training epochs concatenated shape: {X_train_all.shape}")

    # Assert dimensions match config
    assert n_channels_data == N_CHANNELS, f"Data channel count ({n_channels_data}) doesn't match config ({N_CHANNELS})"
    assert n_samples_data == N_SAMPLES, f"Data sample count ({n_samples_data}) doesn't match config ({N_SAMPLES})"

    # --- SCALING ---
    # Option 1: Keep feature-wise scaling on flattened data (simpler for now)
    print("Calculating scaler on flattened training data...")
    X_train_flat = X_train_all.reshape(n_total_epochs, -1)
    X_train_flat_64 = X_train_flat.astype(np.float64)
    scaler_mean = np.mean(X_train_flat_64, axis=0)
    scaler_std = np.std(X_train_flat_64, axis=0)
    epsilon = 1e-8
    scaler_std[scaler_std < epsilon] = epsilon
    print("Applying standardization to flattened data...")
    X_train_scaled_flat = ((X_train_flat_64 - scaler_mean) / scaler_std).astype(np.float32)
    # *** Reshape back to (epochs, channels, samples) ***
    X_train_scaled = X_train_scaled_flat.reshape(n_total_epochs, n_channels_data, n_samples_data)
    print(f"Reshaped Scaled Data Shape: {X_train_scaled.shape}")

    # Option 2: Channel-wise scaling (potentially better for ConvNets)
    # print("Calculating channel-wise scaler...")
    # scaler_mean_ch = np.mean(X_train_all, axis=(0, 2), keepdims=True) # Mean per channel
    # scaler_std_ch = np.std(X_train_all, axis=(0, 2), keepdims=True) # Std per channel
    # epsilon = 1e-8
    # scaler_std_ch[scaler_std_ch < epsilon] = epsilon
    # print("Applying channel-wise standardization...")
    # X_train_scaled = ((X_train_all - scaler_mean_ch) / scaler_std_ch).astype(np.float32)
    # np.savez(SCALER_SAVE_PATH, mean=scaler_mean_ch, std=scaler_std_ch, type='channelwise') # Save channel-wise scaler

    # --- Save the chosen scaler ---
    # Saving the feature-wise scaler (from Option 1)
    np.savez(SCALER_SAVE_PATH, mean=scaler_mean, std=scaler_std, type='featurewise')
    print(f"Scaler saved to {SCALER_SAVE_PATH}")
    # --- End Scaling ---

    # Convert SCALED data (with correct shape) to PyTorch tensors
    # !!! Use the scaled data with shape (n_epochs, n_channels, n_samples) !!!
    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)

    # Create TensorDataset
    train_dataset = TensorDataset(train_tensor, train_tensor) # Input and target are the same

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    print(f"Created DataLoader with batch size {BATCH_SIZE}")

    # 2. Initialize Model, Loss, Optimizer
    # *** Instantiate the CORRECT model ***
    print(f"Initializing ConvAutoencoder1D (Latent: {LATENT_DIM})...")
    model = ConvAutoencoder1D(n_channels=N_CHANNELS, n_samples=N_SAMPLES, latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    model.train()
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for i, (batch_data, _) in enumerate(train_loader):
            # *** batch_data shape is now (batch_size, n_channels, n_samples) ***
            batch_data = batch_data.to(DEVICE)

            outputs = model(batch_data) # Model expects this shape now
            loss = criterion(outputs, batch_data) # Compare reconstruction

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'====> Epoch: {epoch+1}/{NUM_EPOCHS} Average loss: {avg_epoch_loss:.8f}')

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # 4. Save the Trained Encoder Weights
    # *** Update save path name ***
    print(f"Saving trained encoder state dictionary to: {ENCODER_SAVE_PATH}")
    torch.save(model.encoder.state_dict(), ENCODER_SAVE_PATH)
    print("Encoder saved successfully.")

# --- Run the training ---
if __name__ == "__main__":
    train_conv_autoencoder()
