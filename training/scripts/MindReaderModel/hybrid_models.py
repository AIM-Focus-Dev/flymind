# File: training/scripts/MindReaderModel/hybrid_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .configs.hy_config import LATENT_DIM, N_CLASSES, MLP_HIDDEN_1, MLP_HIDDEN_2, DROPOUT_RATE


class MLPClassifierPytorch(nn.Module):
    """
    MLP classifier implemented in PyTorch.
    Takes latent features (e.g., from an Autoencoder) as input
    and outputs class logits.
    """
    def __init__(self, input_dim=LATENT_DIM, num_classes=N_CLASSES,
                 hidden_1=MLP_HIDDEN_1, hidden_2=MLP_HIDDEN_2, dropout=DROPOUT_RATE):
        super(MLPClassifierPytorch, self).__init__()

        self.network = nn.Sequential(
            # Input Layer implicitly defined by first Linear layer
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_1), # Batch norm helps stabilize training
            nn.Dropout(dropout),

            # Hidden Layer 2
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_2),
            nn.Dropout(dropout),

            # Output Layer
            # Outputs raw scores (logits) for each class
            # nn.CrossEntropyLoss (used during training) will apply softmax internally
            nn.Linear(hidden_2, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        Input x: Batch of latent features (batch_size, input_dim)
        Output: Logits for each class (batch_size, num_classes)
        """
        return self.network(x)

# --- Test Block ---
if __name__ == '__main__':
    # Example Usage & Test
    print(f"Default Input Dim (Latent): {LATENT_DIM}, Default Num Classes: {N_CLASSES}")
    # Instantiate with default dimensions
    model = MLPClassifierPytorch()
    print("\nModel Architecture:")
    print(model)

    # Create a dummy batch of latent features (e.g., 16 samples)
    # Use the default LATENT_DIM for the test
    dummy_latent_features = torch.randn(16, LATENT_DIM)
    print(f"\nDummy Input Features Shape: {dummy_latent_features.shape}")

    # Test forward pass
    output_logits = model(dummy_latent_features)
    print(f"Output Logits Shape: {output_logits.shape}") # Should be (batch_size, num_classes)

    # Check number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {num_params:,}")

    # Example of instantiating with different dimensions
    print("\nExample with different dimensions (input=50, classes=2):")
    model_custom = MLPClassifierPytorch(input_dim=50, num_classes=2)
    print(model_custom)
    dummy_latent_features_custom = torch.randn(8, 50)
    output_logits_custom = model_custom(dummy_latent_features_custom)
    print(f"Custom Output Logits Shape: {output_logits_custom.shape}")

