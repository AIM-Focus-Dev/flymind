# File: training/scripts/MindReaderModel/supervised_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from va_encoder import EEGVAE

class EEGHybridModel(nn.Module):
    """
    Hybrid model combining a pre-trained encoder (from Autoencoder)
    and a classifier head for EEG command classification.
    """
    def __init__(self, encoder_path: Path, num_classes: int, device: torch.device):
        super().__init__()
        # Load pretrained autoencoder encoder
        full_ae = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
        encoder = full_ae.encoder
        # load weights
        state = torch.load(str(encoder_path), map_location=device)
        encoder.load_state_dict(state)
        # freeze encoder parameters
        for param in encoder.parameters():
            param.requires_grad = False
        self.encoder = encoder.to(device)
        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, INPUT_DIM) flattened EEG epochs
        returns logits: (batch_size, num_classes)
        """
        # encode to latent
        z = self.encoder(x)
        # classify
        logits = self.classifier(z)
        return logits


def train_model(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """
    Train for one epoch. Returns average loss.
    """
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                   device: torch.device) -> float:
    """
    Evaluate model accuracy on a dataset.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds.cpu() == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total if total > 0 else 0.0

