# File: training/scripts/MindReaderModel/supervised_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS = 22
N_SAMPLES = 751 # From previous logs (288, 22, 751)
N_CLASSES = 4 # Number of motor imagery classes (LH, RH, F, T)

# EEGNet Hyperparameters (Based on original paper, may need tuning)
EEGNET_F1 = 8       # Number of temporal filters
EEGNET_D = 2       # Depth multiplier for spatial filters
EEGNET_F2 = EEGNET_F1 * EEGNET_D # Number of pointwise filters
EEGNET_KERNEL_T = 64 # Length of temporal convolution kernel (~0.25s at 250Hz)
# EEGNET_POOL_T = 4   # Temporal pooling size (adjust based on N_SAMPLES)
EEGNET_POOL_T = 8   # Increased pooling size for longer samples
EEGNET_KERNEL_S = N_CHANNELS # Spatial filter kernel size (learns across all channels)
EEGNET_DROPOUT = 0.25 # Dropout rate for EEGNet layers
# ----------------------------------------------------------

# --- EEGNet Implementation ---
# Based on the paper: "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
# https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/pdf

class EEGNet(nn.Module):
    """
    EEGNet model implementation for EEG classification.
    Input shape: (batch_size, 1, n_channels, n_samples)
    """
    def __init__(self, n_channels=N_CHANNELS, n_samples=N_SAMPLES, num_classes=N_CLASSES,
                 F1=EEGNET_F1, D=EEGNET_D, F2=EEGNET_F2, kernel_t=EEGNET_KERNEL_T,
                 pool_t=EEGNET_POOL_T, kernel_s=EEGNET_KERNEL_S, dropout=EEGNET_DROPOUT):
        super(EEGNet, self).__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.num_classes = num_classes

        # Block 1: Temporal Convolution + Depthwise Spatial Convolution
        # Input: (B, 1, C, T)
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_t), padding=(0, kernel_t // 2), bias=False)
        # Output: (B, F1, C, T)
        self.bn1 = nn.BatchNorm2d(F1)
        # Depthwise Convolution: Applies a separate filter per input channel (F1 groups)
        # Input: (B, F1, C, T)
        self.conv_depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        # Output: (B, F1*D, 1, T)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        # ELU activation is often used in EEGNet
        self.activation1 = nn.ELU()
        # Average pooling along the temporal dimension
        # Input: (B, F1*D, 1, T)
        self.pool1 = nn.AvgPool2d((1, pool_t))
        # Output: (B, F1*D, 1, T//pool_t)
        self.dropout1 = nn.Dropout(dropout)

        # Block 2: Separable Convolution (Depthwise Temporal + Pointwise)
        # Input: (B, F1*D, 1, T//pool_t)
        # Depthwise temporal conv
        self.conv_separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        # Output: (B, F1*D, 1, T//pool_t)
        # Pointwise conv (acts like a 1x1 conv combining features)
        self.conv_separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        # Output: (B, F2, 1, T//pool_t)
        self.bn3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        # Input: (B, F2, 1, T//pool_t)
        self.pool2 = nn.AvgPool2d((1, pool_t)) # Use same pool size or adjust
        # Output: (B, F2, 1, T//pool_t//pool_t)
        self.dropout2 = nn.Dropout(dropout)

        # Classifier
        # Calculate flattened size dynamically after convolutions and pooling
        self.flattened_size = self._get_flattened_size()
        self.fc_out = nn.Linear(self.flattened_size, num_classes)

    def _get_flattened_size(self):
        """ Helper function to calculate the output size after conv/pool layers """
        with torch.no_grad():
            # Dummy input with correct shape (batch=1, input_channels=1, eeg_channels, time_samples)
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            # Pass through the same layers as in forward() up to the flatten point
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.conv_depthwise(x)
            x = self.bn2(x)
            x = self.activation1(x)
            x = self.pool1(x)
            x = self.dropout1(x) # Dropout doesn't change shape

            x = self.conv_separable_depth(x)
            x = self.conv_separable_point(x)
            x = self.bn3(x)
            x = self.activation2(x)
            x = self.pool2(x)
            x = self.dropout2(x) # Dropout doesn't change shape

            # Calculate the number of elements in the resulting tensor
            return x.numel() # Total number of elements after flattening

    def forward(self, x):
        """
        Forward pass. Input x shape: (batch_size, n_channels, n_samples)
        or (batch_size, 1, n_channels, n_samples)
        """
        # Add channel dimension if not present (expecting B x C x T)
        if x.dim() == 3:
             x = x.unsqueeze(1) # Add the '1' channel dimension: B x 1 x C x T

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv_depthwise(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten and Classify
        x = x.view(x.size(0), -1) # Flatten: (Batch, Features)
        logits = self.fc_out(x)
        return logits
# ----------------------------------------------------------
# --- Configuration (Should match the output of the AE encoder) ---
# These values should ideally be passed during instantiation or read from config,
# but we define defaults here based on the ConvAE setup.
# Ensure LATENT_DIM matches the output dimension of your trained encoder.
LATENT_DIM = 128 # Default based on the ConvAE definition
N_CLASSES = 4 # Number of motor imagery classes (LH, RH, F, T)

# MLP Hyperparameters (can be tuned)
MLP_HIDDEN_1 = 128 # Neurons in first hidden layer
MLP_HIDDEN_2 = 64  # Neurons in second hidden layer
DROPOUT_RATE = 0.3 # Dropout probability for regularization
# ----------------------------------------------------------

class MLPClassifierPytorch(nn.Module):
    """
    A simple MLP classifier implemented in PyTorch.
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

