# Includes configuration/parameters for supervised model
# --- Configuration ---
N_CHANNELS = 22
N_SAMPLES = 751 # Based on BCI IV 2a with 3s epochs at 250Hz
N_CLASSES = 4   # LH, RH, F, T

# --- MindReaderModel ---
EEGNET_F1 = 8
EEGNET_D = 2
EEGNET_F2 = EEGNET_F1 * EEGNET_D # 16
EEGNET_KERNEL_T = 64 # ~250ms at 250Hz
EEGNET_POOL_T = 8    # pooling size 
EEGNET_KERNEL_S = N_CHANNELS
EEGNET_DROPOUT = 0.25
AE_LATENT_DIM = 128
