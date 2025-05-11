# File: training/scripts/MindReaderModel/supervised_models.py
# Contains the final MindReaderModel using the original EEGNet hyperparameters

import torch
import torch.nn as nn
import torch.nn.functional as F
from .configs.sm_config import N_CHANNELS, N_SAMPLES, N_CLASSES, EEGNET_F1, EEGNET_D, EEGNET_F2, EEGNET_KERNEL_T, EEGNET_POOL_T, EEGNET_KERNEL_S, EEGNET_DROPOUT 

class MindReaderModel(nn.Module):
    """
    EEG Classification Model based on the EEGNet architecture.
    Uses the original hyperparameter configuration.
    Input shape: (batch_size, 1, n_channels, n_samples)
    """
    def __init__(self, n_channels=N_CHANNELS, n_samples=N_SAMPLES, num_classes=N_CLASSES,
                 f1=EEGNET_F1, d=EEGNET_D, f2=EEGNET_F2, kernel_t=EEGNET_KERNEL_T,
                 pool_t=EEGNET_POOL_T, kernel_s=EEGNET_KERNEL_S, dropout=EEGNET_DROPOUT):
        super(MindReaderModel, self).__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.num_classes = num_classes

        # Block 1: Temporal Convolution + Depthwise Spatial Convolution
        self.conv1 = nn.Conv2d(1, f1, (1, kernel_t), padding=(0, kernel_t // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv_depthwise = nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * d)
        self.activation1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, pool_t))
        self.dropout1 = nn.Dropout(dropout)

        # Block 2: Separable Convolution (Depthwise Temporal + Pointwise)
        self.conv_separable_depth = nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False)
        self.conv_separable_point = nn.Conv2d(f1 * d, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, pool_t)) # Use same pool size
        self.dropout2 = nn.Dropout(dropout)

        # Classifier
        self.flattened_size = self._get_flattened_size()
        self.fc_out = nn.Linear(self.flattened_size, num_classes)

    def _get_flattened_size(self):
        """ Helper function to calculate the output size after conv/pool layers """
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            # Pass through the same layers as in forward() up to the flatten point
            x = self.dropout1(self.pool1(self.activation1(self.bn2(self.conv_depthwise(self.bn1(self.conv1(x)))))))
            x = self.dropout2(self.pool2(self.activation2(self.bn3(self.conv_separable_point(self.conv_separable_depth(x))))))
            return x.numel() # Total number of elements after flattening

    def forward(self, x):
        """ Forward pass. Input x shape: (batch_size, n_channels, n_samples) or (B, 1, C, T) """
        # Add channel dimension if not present (expecting B x C x T)
        if x.dim() == 3:
             x = x.unsqueeze(1) # Add the '1' channel dimension: B x 1 x C x T

        # Apply layers sequentially
        x = self.dropout1(self.pool1(self.activation1(self.bn2(self.conv_depthwise(self.bn1(self.conv1(x)))))))
        x = self.dropout2(self.pool2(self.activation2(self.bn3(self.conv_separable_point(self.conv_separable_depth(x))))))

        # Flatten and Classify
        x = x.view(x.size(0), -1) # Flatten: (Batch, Features)
        logits = self.fc_out(x)
        return logits

# --- Test Block ---
if __name__ == '__main__':
    print("--- Testing MindReaderModel (Original EEGNet Params) ---")
    print(f"Input: Channels={N_CHANNELS}, Samples={N_SAMPLES}, Classes={N_CLASSES}")
    model = MindReaderModel() # Uses original default params now
    print("MindReaderModel Architecture Defined.")

    dummy_eeg_epochs = torch.randn(4, N_CHANNELS, N_SAMPLES)
    print(f"\nDummy Input Epochs Shape: {dummy_eeg_epochs.shape}")

    output_logits = model(dummy_eeg_epochs)
    print(f"Output Logits Shape: {output_logits.shape}") 

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {num_params:,}")

