import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from preprocess_data import load_and_preprocess_subject_data, N_SUBJECTS
from va_encoder import EEGVAE 

# --- Configuration ---
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 500
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
SAVE_PATH = Path(__file__).resolve().parents[1] / 'models'
SAVE_PATH.mkdir(exist_ok=True)
LATENT_DIM = 64  # Define latent dimension
# ---------------------

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.1 * kld  # lambda=0.1

def train_subject_encoder(subject_id):
    # Load training epochs
    epochs = load_and_preprocess_subject_data(subject_id, session_type='T')
    if epochs is None or len(epochs) == 0:
        print(f"Subject {subject_id}: No data, skipping.")
        return

    # Prepare data array and flatten
    X = epochs.get_data().astype(np.float32)  # (n_epochs, n_channels, n_times)
    X = X.reshape(len(epochs), -1)           # (n_epochs, INPUT_DIM)
    INPUT_DIM = X.shape[1]  # Compute input dimension

    # Create DataLoader
    tensor_x = torch.tensor(X)
    dataset = TensorDataset(tensor_x, tensor_x)  # VAE target = input
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model and optimizer
    model = EEGVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    print(f"Training VAE for Subject {subject_id} on {len(dataset)} samples.")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(batch_x)
            loss = vae_loss_function(recon, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Subject {subject_id} | Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save VAE model
    vae_path = SAVE_PATH / f"vae_subject{subject_id}.pth"
    torch.save(model.state_dict(), vae_path)
    print(f"Saved VAE for subject {subject_id} at {vae_path}")

if __name__ == '__main__':
    for sid in range(1, N_SUBJECTS + 1):
        train_subject_encoder(sid)
