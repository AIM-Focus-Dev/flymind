# File: training/scripts/MindReaderModel/autoencoder_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .configs.ae_config import N_CHANNELS, N_SAMPLES, CONV_FILTERS_1, CONV_KERNEL_1, CONV_STRIDE_1, CONV_PADDING_1, CONV_FILTERS_2, CONV_KERNEL_2, CONV_STRIDE_2, CONV_PADDING_2, CONV_FILTERS_3, CONV_KERNEL_3, CONV_STRIDE_3, CONV_PADDING_3, LATENT_DIM


class ConvAutoencoder1D(nn.Module):
    """
    Convolutional Autoencoder using 1D convolutions along the time axis.
    Input shape: (batch_size, n_channels, n_samples)
    CORRECTED: Groups encoder/decoder layers into modules.
    """
    def __init__(self, n_channels=N_CHANNELS, n_samples=N_SAMPLES, latent_dim=LATENT_DIM):
        super(ConvAutoencoder1D, self).__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.latent_dim = latent_dim

        # --- Encoder Layers ---
        encoder_layers = [
            nn.Conv1d(n_channels, CONV_FILTERS_1, kernel_size=CONV_KERNEL_1, stride=CONV_STRIDE_1, padding=CONV_PADDING_1),
            nn.BatchNorm1d(CONV_FILTERS_1),
            nn.ReLU(True),
            nn.Conv1d(CONV_FILTERS_1, CONV_FILTERS_2, kernel_size=CONV_KERNEL_2, stride=CONV_STRIDE_2, padding=CONV_PADDING_2),
            nn.BatchNorm1d(CONV_FILTERS_2),
            nn.ReLU(True),
            nn.Conv1d(CONV_FILTERS_2, CONV_FILTERS_3, kernel_size=CONV_KERNEL_3, stride=CONV_STRIDE_3, padding=CONV_PADDING_3),
            nn.BatchNorm1d(CONV_FILTERS_3),
            nn.ReLU(True),
            nn.Flatten() # Flatten the output of conv layers
        ]
        # *** Group encoder layers into self.encoder ***
        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Calculate the size after convolutions dynamically
        self._calculate_conv_output_size()

        # Add the final linear layer to the encoder definition
        self.encoder_fc = nn.Linear(self.flattened_size, latent_dim)

        # *** Define the complete encoder module ***
        self.encoder = nn.Sequential(
            self.encoder_conv,
            self.encoder_fc
        )

        # --- Decoder Layers ---
        # Calculate output padding dynamically 
        # conv1_out_len, conv2_out_len, conv3_out_len from _calculate_conv_output_size
        op1 = self._calculate_output_padding(self.conv1_out_len, n_samples, CONV_KERNEL_1, CONV_STRIDE_1, CONV_PADDING_1)
        op2 = self._calculate_output_padding(self.conv2_out_len, self.conv1_out_len, CONV_KERNEL_2, CONV_STRIDE_2, CONV_PADDING_2)
        op3 = self._calculate_output_padding(self.conv3_out_len, self.conv2_out_len, CONV_KERNEL_3, CONV_STRIDE_3, CONV_PADDING_3)

        # *** Group decoder layers into self.decoder ***
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_size),
            # UnFlatten layer needs the shape before flattening in encoder
            nn.Unflatten(1, (CONV_FILTERS_3, self.conv3_out_len)),
            nn.ConvTranspose1d(CONV_FILTERS_3, CONV_FILTERS_2, kernel_size=CONV_KERNEL_3, stride=CONV_STRIDE_3, padding=CONV_PADDING_3, output_padding=op3),
            nn.BatchNorm1d(CONV_FILTERS_2),
            nn.ReLU(True),
            nn.ConvTranspose1d(CONV_FILTERS_2, CONV_FILTERS_1, kernel_size=CONV_KERNEL_2, stride=CONV_STRIDE_2, padding=CONV_PADDING_2, output_padding=op2),
            nn.BatchNorm1d(CONV_FILTERS_1),
            nn.ReLU(True),
            nn.ConvTranspose1d(CONV_FILTERS_1, n_channels, kernel_size=CONV_KERNEL_1, stride=CONV_STRIDE_1, padding=CONV_PADDING_1, output_padding=op1)
        )

    def _calculate_conv_output_size(self):
        """Helper function to calculate the output size after conv layers using a dummy input."""
        # Using a dummy tensor to pass through the convolutional part of the encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.n_channels, self.n_samples) # Batch size 1
            dummy_output = self.encoder_conv(dummy_input)
            self.flattened_size = dummy_output.shape[1] # Get the flattened size
            # Store intermediate lengths needed for decoder output padding calculation
            x = dummy_input
            x = self.encoder_conv[0](x) # conv1
            self.conv1_out_len = x.shape[2]
            x = self.encoder_conv[3](self.encoder_conv[2](self.encoder_conv[1](x))) # bn1, relu, conv2
            self.conv2_out_len = x.shape[2]
            x = self.encoder_conv[6](self.encoder_conv[5](self.encoder_conv[4](x))) # bn2, relu, conv3
            self.conv3_out_len = x.shape[2]

        print(f"Calculated flattened size after conv layers: {self.flattened_size}")
        print(f"Intermediate lengths: conv1={self.conv1_out_len}, conv2={self.conv2_out_len}, conv3={self.conv3_out_len}")


    def _calculate_output_padding(self, in_len, target_out_len, kernel, stride, padding):
        """ Calculates the required output_padding for ConvTranspose1d """
        calculated_out = (in_len - 1) * stride - 2 * padding + kernel
        output_padding = target_out_len - calculated_out
        if output_padding < 0:
             print(f"Warning: Negative output padding calculated ({output_padding}). Check layer parameters: in={in_len}, target={target_out_len}, k={kernel}, s={stride}, p={padding}")
             return 0
        return output_padding

    # --- encode/decode methods for clarity ---
    def encode(self, x):
        """Encodes the input into the latent space representation using self.encoder."""
        return self.encoder(x)

    def decode(self, z):
        """Decodes the latent space representation back using self.decoder."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass for training (reconstruction)."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

# --- Test Block ---
if __name__ == '__main__':
    print(f"Input: Channels={N_CHANNELS}, Samples={N_SAMPLES}")
    print(f"Target Latent Dim: {LATENT_DIM}")
    model = ConvAutoencoder1D(n_channels=N_CHANNELS, n_samples=N_SAMPLES, latent_dim=LATENT_DIM)
    print("\nModel Architecture Defined (encoder/decoder modules).")

    dummy_input_epochs = torch.randn(4, N_CHANNELS, N_SAMPLES)
    print(f"\nDummy Input Epochs Shape: {dummy_input_epochs.shape}")

    model.eval()
    with torch.no_grad():
        latent_vector = model.encode(dummy_input_epochs)
        print(f"Latent Vector Shape: {latent_vector.shape}")

        reconstruction = model.decode(latent_vector)
        print(f"Reconstruction Output Shape: {reconstruction.shape}")

        full_output = model(dummy_input_epochs)
        print(f"Full Forward Output Shape: {full_output.shape}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {num_params:,}")

