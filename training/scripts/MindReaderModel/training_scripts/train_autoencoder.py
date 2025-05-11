# File: training/scripts/MindReaderModel/train_autoencoder.py
# Trains the ConvAutoencoder1D on all subjects' training data, saves encoder and scaler

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# Package imports
from ..preprocess_data import load_and_preprocess_subject_data
from ..configs.ae_config import N_SUBJECTS, N_CHANNELS, N_SAMPLES
from ..configs.config    import MODELS_PATH
from ..autoencoder_model import ConvAutoencoder1D, LATENT_DIM

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE    = 64
NUM_EPOCHS    = 150

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure model directory exists
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Paths for saving
ENCODER_SAVE_PATH = MODELS_PATH / f"encoder_conv_ae_latent{LATENT_DIM}.pth"
SCALER_SAVE_PATH  = MODELS_PATH / "conv_ae_scaler.npz"


def train_conv_autoencoder():
    """
    1) Loads and concatenates all subjects' training epochs
    2) Computes feature-wise scaler and standardizes data
    3) Trains ConvAutoencoder1D
    4) Saves encoder weights and scaler
    """
    print("=== Starting Convolutional Autoencoder Training ===")

    # 1. Load and concatenate
    epochs_list = []
    for sid in range(1, N_SUBJECTS + 1):
        epochs = load_and_preprocess_subject_data(sid, session_type='T')
        if epochs is not None and len(epochs) > 0:
            epochs_list.append(epochs.get_data(copy=False))
    if not epochs_list:
        print("Error: No training data loaded. Exiting.")
        return

    X = np.concatenate(epochs_list, axis=0)  # (n_epochs, n_channels, n_samples)
    n, c, t = X.shape
    assert c == N_CHANNELS, f"Channels mismatch: {c}!={N_CHANNELS}"
    assert t == N_SAMPLES,  f"Samples mismatch: {t}!={N_SAMPLES}"

    # 2. Feature-wise scaling
    X_flat = X.reshape(n, -1).astype(np.float64)
    mean = X_flat.mean(axis=0)
    std  = X_flat.std(axis=0)
    std[std < 1e-8] = 1e-8
    np.savez(SCALER_SAVE_PATH, mean=mean, std=std, type='featurewise')
    X_scaled = ((X_flat - mean) / std).astype(np.float32).reshape(n, c, t)
    print(f"Scaled data shape: {X_scaled.shape}")

    # 3. Prepare DataLoader
    tensor_data = torch.tensor(X_scaled)
    dataset = TensorDataset(tensor_data, tensor_data)  # autoencoder: input & target
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Model, loss, optimizer
    model     = ConvAutoencoder1D(n_channels=c, n_samples=t, latent_dim=LATENT_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training loop
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        model.train()
        for batch, _ in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}")
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # 6. Save encoder
    torch.save(model.encoder.state_dict(), ENCODER_SAVE_PATH)
    print(f"Saved encoder weights to {ENCODER_SAVE_PATH}")


if __name__ == "__main__":
    train_conv_autoencoder()
