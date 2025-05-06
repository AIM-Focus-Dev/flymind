import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# ------------------ Model Definitions ------------------
class EEGVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, input_dim)
        )

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

class EEGClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ------------------ Helper Functions ------------------
def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.1 * kld  # lambda=0.1

# ------------------ Training & Evaluation ------------------
def train_vae(vae, data_loader, optimizer, device):
    vae.train()
    total_loss = 0
    for batch_x, _ in data_loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, _ = vae(batch_x)
        loss = vae_loss_function(recon, batch_x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader.dataset)


def extract_latent_features(vae, data_loader, device):
    vae.eval()
    features = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            _, mu, logvar, z = vae(batch_x)
            features.append(mu.cpu().numpy())
    return np.vstack(features)


def evaluate_hybrid(X, y, input_dim, latent_dim, n_splits=5, device='mps'):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        # Prepare data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Dataloaders
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        # Instantiate models
        vae = EEGVAE(input_dim, latent_dim).to(device)
        classifier = EEGClassifier(latent_dim, len(np.unique(y))).to(device)
        # Optimizers
        vae_opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
        cls_opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        # Train VAE
        for epoch in range(20):
            train_vae(vae, train_loader, vae_opt, device)
        # Extract features
        Z_train = extract_latent_features(vae, train_loader, device)
        Z_test = extract_latent_features(vae, test_loader, device)
        # Train classifier
        for epoch in range(10):
            classifier.train()
            for batch_z, batch_y in DataLoader(TensorDataset(torch.tensor(Z_train, dtype=torch.float32), torch.tensor(y_train)), batch_size=32, shuffle=True):
                batch_z, batch_y = batch_z.to(device), batch_y.to(device)
                cls_opt.zero_grad()
                logits = classifier(batch_z)
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                cls_opt.step()
        # Test
        classifier.eval()
        preds = []
        with torch.no_grad():
            for batch_z, _ in DataLoader(TensorDataset(torch.tensor(Z_test, dtype=torch.float32)), batch_size=32):
                logits = classifier(batch_z.to(device))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        accs.append(accuracy_score(y_test, preds))
    print(f"Hybrid VAE+Classifier mean accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

# ------------------ Example Usage ------------------
if __name__ == '__main__':
    # Assume X: (n_epochs, input_dim), y: labels loaded elsewhere
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    input_dim = X.shape[1]
    latent_dim = 64
    evaluate_hybrid(X, y, input_dim, latent_dim, n_splits=5, device=device)
