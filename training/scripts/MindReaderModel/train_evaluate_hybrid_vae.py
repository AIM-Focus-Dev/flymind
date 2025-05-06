# File: training/scripts/MindReaderModel/train_evaluate_hybrid_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# --- Import your preprocessing and config ---
try:
    from .preprocess_data import load_and_preprocess_subject_data, N_SUBJECTS
    from .va_encoder import EEGVAE, EEGClassifier, vae_loss_function
    print("Imported preprocess_data and autoencoder_model from package.")
except ImportError:
    # fallback if you run this file directly
    from preprocess_data import load_and_preprocess_subject_data, N_SUBJECTS
    from va_encoder import EEGVAE, EEGClassifier, vae_loss_function
    print("Imported preprocess_data and autoencoder_model from local folder.")

# --- Hyperparameters ---
DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE    = 64
VAE_LR        = 1e-3
VAE_EPOCHS    = 20
CLS_LR        = 1e-3
CLS_EPOCHS    = 10
CV_FOLDS      = 5
LATENT_DIM = 64  # Define latent dimension
PRINT_EVERY   = True
# -----------------------

def train_vae(vae, loader, optimizer, device):
    vae.train()
    total_loss = 0.0
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, _ = vae(X_batch)
        loss = vae_loss_function(recon, X_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def extract_latent(vae, loader, device):
    vae.eval()
    feats = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            _, mu, _, _ = vae(X_batch)
            feats.append(mu.cpu().numpy())
    return np.vstack(feats)

def run_hybrid_for_subject(subject_id):
    # 1. Load epochs
    epochs = load_and_preprocess_subject_data(subject_id, session_type='T')
    if epochs is None or len(epochs) < CV_FOLDS:
        print(f"Subject {subject_id}: not enough data, skipping.")
        return None

    # 2. Get data & labels
    X = epochs.get_data().astype(np.float32)            # shape (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]                            # shape (n_epochs,)
    n_epochs = X.shape[0]

    # 3. Flatten and scale
    X = X.reshape(n_epochs, -1)                         # (n_epochs, INPUT_DIM)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 4. Prepare CV
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    accs = []

    # 5. CV loop
    for train_idx, test_idx in skf.split(Xs, y):
        # a) VAE training data
        X_train = Xs[train_idx]
        ds_train = TensorDataset(torch.tensor(X_train), torch.zeros(len(train_idx)))
        loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)

        # b) Build and train VAE
        vae = EEGVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        opt_vae = torch.optim.Adam(vae.parameters(), lr=VAE_LR)
        for ep in range(1, VAE_EPOCHS + 1):
            loss = train_vae(vae, loader_train, opt_vae, DEVICE)
            if PRINT_EVERY and ep % 5 == 0:
                print(f"Subject {subject_id} | VAE epoch {ep}/{VAE_EPOCHS} | loss {loss:.4f}")

        # c) Extract latent features
        Z_train = extract_latent(vae, loader_train, DEVICE)
        # and for test
        X_test = Xs[test_idx]
        ds_test = TensorDataset(torch.tensor(X_test), torch.zeros(len(test_idx)))
        loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
        Z_test = extract_latent(vae, loader_test, DEVICE)

        # d) Train classifier on Z_train
        clf = EEGClassifier(latent_dim=LATENT_DIM, num_classes=len(np.unique(y))).to(DEVICE)
        opt_clf = torch.optim.Adam(clf.parameters(), lr=CLS_LR)
        ds_ztrain = TensorDataset(torch.tensor(Z_train), torch.tensor(y[train_idx]))
        loader_ztrain = DataLoader(ds_ztrain, batch_size=BATCH_SIZE, shuffle=True)
        for ep in range(1, CLS_EPOCHS + 1):
            clf.train()
            for zb, yb in loader_ztrain:
                zb, yb = zb.to(DEVICE), yb.to(DEVICE)
                opt_clf.zero_grad()
                logits = clf(zb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt_clf.step()

        # e) Evaluate on Z_test
        clf.eval()
        preds = []
        with torch.no_grad():
            for zb, _ in DataLoader(TensorDataset(torch.tensor(Z_test), torch.zeros(len(Z_test))), batch_size=BATCH_SIZE):
                zb = zb.to(DEVICE)
                logits = clf(zb)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        accs.append(accuracy_score(y[test_idx], preds))

    mean_acc = np.mean(accs)
    print(f"Subject {subject_id} | Hybrid VAE accuracy: {mean_acc:.4f}")
    return mean_acc

def main():
    all_accs = []
    for sid in range(1, N_SUBJECTS+1):
        acc = run_hybrid_for_subject(sid)
        if acc is not None:
            all_accs.append(acc)
    if all_accs:
        print(f"\nOverall mean accuracy: {np.mean(all_accs):.4f} +/- {np.std(all_accs):.4f}")

if __name__ == '__main__':
    main()
