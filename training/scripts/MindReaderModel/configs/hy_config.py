# Includes configuration/parameters for hybrid model
N_SUBJECTS = 9
N_CHANNELS = 22
N_SAMPLES = 751
N_CLASSES = 4 # Number of motor imagery classes (LH, RH, F, T)
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 150 
LATENT_DIM = 128 # Default based on the ConvAE definition

# MLP Hyperparameters 
MLP_HIDDEN_1 = 128 # Neurons in first hidden layer
MLP_HIDDEN_2 = 64  # Neurons in second hidden layer
DROPOUT_RATE = 0.3 # Dropout probability for regularization
