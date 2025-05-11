# File: training/scripts/MindReaderModel/train_evaluate_hybrid.py
# Description: This script evaluates hybrid models for EEG classification.
# It uses a pre-trained Convolutional Autoencoder (ConvAE) to extract features
# from EEG data, which are then fed into either a PyTorch-based Multi-Layer Perceptron (MLP)
# or a Support Vector Machine (SVM) for classification.
# The evaluation is performed using subject-specific, within-session cross-validation.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import mne  # For EEG data handling.
from pathlib import Path
import time
from sklearn.pipeline import Pipeline # For creating scikit-learn pipelines (e.g., Scaler + SVM).
from sklearn.preprocessing import StandardScaler # For feature scaling.
from sklearn.svm import SVC # Support Vector Classifier.
from sklearn.model_selection import StratifiedKFold, cross_val_score # For cross-validation.
from sklearn.metrics import accuracy_score # For evaluating MLP performance per fold.

# --- Import Project-Specific Components ---
# This block attempts to import modules assuming a standard project structure
# where this script might be run from the project root or its PYTHONPATH is configured.
try:
    from ..preprocess_data import load_and_preprocess_subject_data
    from ..configs.config import DATA_PATH, MODELS_PATH, RESULTS_PATH
    from ..configs.hy_config import N_SUBJECTS, N_CHANNELS, N_SAMPLES, N_CLASSES
    from ..autoencoder_model import ConvAutoencoder1D, LATENT_DIM
    from ..hybrid_models import MLPClassifierPytorch # Custom PyTorch MLP implementation.
    print(f"Successfully imported project components. Paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}, RESULTS={RESULTS_PATH}")
except ImportError:
    # Fallback import mechanism: This is useful if the script is run directly
    # from its own directory (e.g., for debugging or isolated execution)
    # and the project root is not in PYTHONPATH.
    print("ImportError: Could not import from parent packages. Attempting fallback imports for direct script execution.")
    try:
        # Determine project root assuming a fixed directory structure.
        # __file__ might not be defined if running in certain interactive environments (e.g. some IDEs' consoles).
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
    except NameError: # Fallback if __file__ is not defined.
        PROJECT_ROOT = Path('.').resolve().parents[2] # Assumes script is in training/scripts/MindReaderModel
        print(f"Warning: '__file__' not defined. Assuming PROJECT_ROOT based on current directory: {PROJECT_ROOT}")

    # Define paths and constants manually if the primary import fails.
    DATA_PATH = PROJECT_ROOT / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
    MODELS_PATH = PROJECT_ROOT / "training" / "models"
    RESULTS_PATH = PROJECT_ROOT / "training" / "results"
    N_SUBJECTS = 9      # Number of subjects in the BCICIV 2a dataset.
    N_CHANNELS = 22     # Number of EEG channels.
    N_SAMPLES = 751     # Number of time samples per EEG epoch.
    N_CLASSES = 4       # Number of distinct mental imagery classes.
    
    # Attempt to import modules using sys.path manipulation if necessary,
    # or ensure they are directly accessible.
    # This part assumes that if the script is run directly, the necessary files (preprocess_data.py, autoencoder_model.py)
    # are either in the same directory or Python's path is already configured.
    from preprocess_data import load_and_preprocess_subject_data
    from autoencoder_model import ConvAutoencoder1D, LATENT_DIM # LATENT_DIM is imported from autoencoder_model
    print(f"Fallback import paths: DATA={DATA_PATH}, MODELS={MODELS_PATH}, RESULTS={RESULTS_PATH}")
    print(f"Fallback constants: N_SUBJECTS={N_SUBJECTS}, N_CHANNELS={N_CHANNELS}, N_SAMPLES={N_SAMPLES}, N_CLASSES={N_CLASSES}, LATENT_DIM={LATENT_DIM}")


# --- Configuration Parameters ---
# Path to the pre-trained encoder model and its associated scaler.
ENCODER_LOAD_PATH = MODELS_PATH / f"encoder_conv_ae_latent{LATENT_DIM}.pth"
SCALER_LOAD_PATH = MODELS_PATH / "conv_ae_scaler.npz" # Scaler used for the ConvAE input.

# Cross-validation parameters.
N_SPLITS = 5            # Number of folds for Stratified K-Fold cross-validation.
SHUFFLE_FOLDS = True    # Whether to shuffle data before splitting into folds.
RANDOM_STATE = 42       # Ensures reproducibility of shuffling and splits.
METRIC = 'accuracy'     # Primary metric for evaluation.

# DataLoader batch size for feature extraction (can be larger than training batch size).
BATCH_SIZE_FEATURE_EXTRACTION = 64

# PyTorch MLP Training parameters.
PT_LEARNING_RATE = 1e-4         # Learning rate for the AdamW optimiser.
PT_BATCH_SIZE_CLASSIFIER = 32   # Batch size for training the MLP classifier.
PT_NUM_EPOCHS = 100             # Maximum number of training epochs for the MLP.
PT_WEIGHT_DECAY = 1e-4          # Weight decay (L2 regularisation) for the AdamW optimiser.
PT_EARLY_STOPPING_PATIENCE = 10 # Number of epochs with no improvement on validation loss before stopping.

# SVM parameters (for scikit-learn's SVC).
SVM_C = 1.0             # Regularisation parameter.
SVM_KERNEL = 'rbf'      # Kernel type (Radial Basis Function).
SVM_GAMMA = 'scale'     # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 'scale' uses 1 / (n_features * X.var()).

# Device Setup: Selects CUDA or MPS (Apple Silicon GPU) if available, otherwise CPU.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Computation device selected: {DEVICE}")
# --------------------

def load_encoder(model_path=ENCODER_LOAD_PATH, n_channels=N_CHANNELS, n_samples=N_SAMPLES, latent_dim=LATENT_DIM, device=DEVICE):
    """
    Loads the pre-trained encoder part of the Convolutional Autoencoder (ConvAE) model.
    The full ConvAE model is instantiated, but only the encoder's state dictionary is loaded.

    Args:
        model_path (Path): Path to the saved state dictionary of the encoder.
        n_channels (int): Number of input channels for the encoder.
        n_samples (int): Number of input samples (time points) for the encoder.
        latent_dim (int): Dimensionality of the latent space.
        device (torch.device): The device to load the model onto (e.g., CPU, CUDA).

    Returns:
        torch.nn.Module or None: The loaded encoder model in evaluation mode, or None if loading fails.
    """
    full_ae_model = ConvAutoencoder1D(n_channels=n_channels, n_samples=n_samples, latent_dim=latent_dim)
    encoder = full_ae_model.encoder # Extract the encoder part.
    try:
        encoder.load_state_dict(torch.load(model_path, map_location=device))
        encoder.to(device)
        encoder.eval() # Set the encoder to evaluation mode (disables dropout, etc.).
        print(f"Encoder loaded successfully from {model_path}")
        return encoder
    except FileNotFoundError:
        print(f"Error: Encoder model file not found at {model_path}. This is critical for feature extraction.")
        return None
    except Exception as e:
        print(f"Error loading encoder model: {e}")
        return None

def extract_features(encoder, data_loader, device=DEVICE):
    """
    Extracts latent features from input data using the pre-trained encoder.

    Args:
        encoder (torch.nn.Module): The pre-trained encoder model.
        data_loader (DataLoader): PyTorch DataLoader providing batches of input data.
        device (torch.device): The device the encoder is on.

    Returns:
        np.ndarray or None: A NumPy array of extracted features, or None if extraction fails.
    """
    if encoder is None:
        print("Feature extraction skipped: Encoder is not available.")
        return None
        
    all_features = []
    print("Extracting features using the loaded encoder...")
    with torch.no_grad(): # Disable gradient calculations during inference.
        for batch_data, _ in data_loader: # Labels from data_loader are dummies here.
            batch_data = batch_data.to(device)
            latent_features = encoder(batch_data)
            all_features.append(latent_features.cpu().numpy()) # Collect features on CPU as NumPy arrays.
            
    if not all_features:
        print("Warning: No features were extracted. The data_loader might be empty.")
        return None
    return np.concatenate(all_features, axis=0)

def train_pytorch_mlp_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, input_dim, num_classes, device):
    """
    Trains the PyTorch MLP classifier for a single cross-validation fold.
    Includes training loop, validation, and early stopping.

    Args:
        X_train_fold (np.ndarray): Training features for the current fold.
        y_train_fold (np.ndarray): Training labels for the current fold.
        X_val_fold (np.ndarray): Validation features for the current fold.
        y_val_fold (np.ndarray): Validation labels for the current fold.
        input_dim (int): Dimensionality of the input features (should match LATENT_DIM).
        num_classes (int): Number of output classes.
        device (torch.device): Device for training (CPU/GPU).

    Returns:
        MLPClassifierPytorch: The trained MLP model for this fold.
    """
    # Instantiate the MLP model, loss function, and optimiser.
    model = MLPClassifierPytorch(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification.
    optimizer = optim.AdamW(model.parameters(), lr=PT_LEARNING_RATE, weight_decay=PT_WEIGHT_DECAY)

    # Create TensorDatasets and DataLoaders for training and validation.
    train_dataset = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32), torch.tensor(y_train_fold, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_fold, dtype=torch.float32), torch.tensor(y_val_fold, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=PT_BATCH_SIZE_CLASSIFIER, shuffle=True, num_workers=0)
    # Validation DataLoader can use a larger batch size and does not need shuffling.
    val_loader = DataLoader(val_dataset, batch_size=PT_BATCH_SIZE_CLASSIFIER * 2, shuffle=False, num_workers=0)

    print(f"    Training PyTorch MLP for up to {PT_NUM_EPOCHS} epochs (Input dim: {input_dim}, Classes: {num_classes})...")
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(PT_NUM_EPOCHS):
        model.train() # Set model to training mode.
        current_train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()   # Clear previous gradients.
            outputs = model(features) # Forward pass.
            loss = criterion(outputs, labels) # Calculate loss.
            loss.backward()         # Backward pass (compute gradients).
            optimizer.step()        # Update model parameters.
            current_train_loss += loss.item()

        model.eval() # Set model to evaluation mode.
        current_val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculations for validation.
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                current_val_loss += loss.item()
        
        average_epoch_val_loss = current_val_loss / len(val_loader)

        # Early stopping logic.
        if average_epoch_val_loss < best_val_loss:
            best_val_loss = average_epoch_val_loss
            epochs_without_improvement = 0
            # save the best model state 
            # torch.save(model.state_dict(), 'best_mlp_fold_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PT_EARLY_STOPPING_PATIENCE:
                print(f'        Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss.')
                break
        
        if (epoch + 1) % 20 == 0: # Log progress periodically.
             print(f"        Epoch [{epoch+1}/{PT_NUM_EPOCHS}], Train Loss: {current_train_loss/len(train_loader):.4f}, Val Loss: {average_epoch_val_loss:.4f}")


    print(f"    Finished training MLP for fold. Best validation loss achieved: {best_val_loss:.4f}")
    return model # Return the model trained up to the point of early stopping or max epochs.

# --- Main Evaluation Function ---
def run_hybrid_evaluation():
    """
    Orchestrates the evaluation of hybrid models (ConvAE features + classifier).
    This involves:
    1. Loading the pre-trained scaler (used for ConvAE input).
    2. Loading the pre-trained ConvAE encoder.
    3. For each subject:
        a. Loading their preprocessed training ('T') session EEG data.
        b. Scaling the EEG data using the loaded scaler.
        c. Extracting latent features using the encoder.
        d. Performing stratified K-fold cross-validation on these features with:
            i. A PyTorch MLP classifier.
            ii. An SVM classifier.
    4. Aggregating and saving the results.
    """
    print("\n=== Starting Hybrid Model Evaluation (ConvAE Features + PyTorch MLP/SVM) ===")

    # 1. Load Scaler (mean and std for normalising ConvAE input)
    try:
        scaler_params = np.load(SCALER_LOAD_PATH)
        # Ensure keys 'mean' and 'std' exist, matching how they were saved.
        if 'mean' not in scaler_params or 'std' not in scaler_params:
            raise KeyError("Scaler file is missing 'mean' or 'std' arrays.")
        scaler_mean = scaler_params['mean']
        scaler_std = scaler_params['std']
        print(f"Scaler parameters (mean/std) loaded successfully from {SCALER_LOAD_PATH}")
    except FileNotFoundError:
        print(f"CRITICAL Error: Scaler file not found at {SCALER_LOAD_PATH}. Cannot proceed with feature extraction.")
        return
    except KeyError as e:
        print(f"CRITICAL Error: Issue with scaler file keys: {e}. Cannot proceed.")
        return
    except Exception as e:
        print(f"CRITICAL Error loading scaler: {e}. Cannot proceed.")
        return

    # 2. Load Pre-trained Encoder Model
    encoder = load_encoder(n_channels=N_CHANNELS, n_samples=N_SAMPLES, latent_dim=LATENT_DIM, device=DEVICE)
    if encoder is None:
        print("CRITICAL Error: Could not load the pre-trained encoder model. Evaluation cannot continue.")
        return

    all_subject_results = [] # List to store dictionaries of results for each subject and pipeline.

    # Establish a consistent mapping from MNE event IDs to 0-indexed labels (0, 1, 2, 3).
    # This is done once using data from the first subject.
    print("Establishing MNE event ID to 0-indexed label mapping...")
    try:
        # Load data from subject 1, session 'T' just to get the event_id dictionary.
        temp_epochs_for_mapping = load_and_preprocess_subject_data(1, session_type='T', data_path=DATA_PATH)
        if temp_epochs_for_mapping is None:
            print("CRITICAL Error: Cannot determine event ID mapping. Failed to load data for subject 1.")
            return
        if not hasattr(temp_epochs_for_mapping, 'event_id') or not temp_epochs_for_mapping.event_id:
            print("CRITICAL Error: Loaded epochs object for mapping has no 'event_id' attribute or it's empty.")
            return
        
        # Create mapping: e.g., {769:0, 770:1, 771:2, 772:3} if sorted MNE IDs are 769,770,771,772
        mne_event_id_to_task_name = {v: k for k, v in temp_epochs_for_mapping.event_id.items()} # {769:'left_hand', ...}
        sorted_mne_event_ids = sorted(mne_event_id_to_task_name.keys()) # [769, 770, 771, 772]
        mne_id_to_zero_based_label = {mne_id: i for i, mne_id in enumerate(sorted_mne_event_ids)}
        print(f"Mapping MNE event IDs to 0-based labels established: {mne_id_to_zero_based_label}")
        del temp_epochs_for_mapping # Free up memory.
    except Exception as e:
        print(f"CRITICAL Error during event ID mapping setup: {e}")
        return

    # Iterate through each subject for evaluation.
    for subject_id in range(1, N_SUBJECTS + 1):
        print(f"\n--- Processing Subject {subject_id} ---")

        # 3. Load preprocessed TRAINING ('T') session data for the current subject.
        # Cross-validation will be performed within this training data.
        epochs_train_subject = load_and_preprocess_subject_data(subject_id, session_type='T', data_path=DATA_PATH)

        if epochs_train_subject is None or len(epochs_train_subject) == 0:
            print(f"Skipping Subject {subject_id} - No training epochs found or data processing failed.")
            continue

        # 4. Prepare Data for Feature Extraction
        # Extract raw EEG data and MNE event IDs.
        X_eeg_raw = epochs_train_subject.get_data(copy=False).astype(np.float64) # (n_epochs, n_channels, n_samples)
        y_mne_event_ids = epochs_train_subject.events[:, -1] # Extracts the event type/ID column.
        
        # Convert MNE event IDs to 0-indexed labels using the established map.
        try:
            y_labels_zero_based = np.array([mne_id_to_zero_based_label[mne_id] for mne_id in y_mne_event_ids])
        except KeyError as e:
            print(f"Error mapping event IDs for Subject {subject_id}. Unseen MNE event ID: {e}.")
            print(f"  Available MNE ID map: {mne_id_to_zero_based_label}")
            print(f"  Unique MNE IDs in this subject's data: {np.unique(y_mne_event_ids)}")
            print(f"Skipping Subject {subject_id} due to event ID mapping error.")
            continue

        n_epochs_subj, n_channels_data_subj, n_samples_data_subj = X_eeg_raw.shape
        print(f"Subject {subject_id}: Raw data shape: {X_eeg_raw.shape}, Labels shape: {y_labels_zero_based.shape}, Unique labels: {np.unique(y_labels_zero_based)}")

        # Check if there are enough epochs for cross-validation.
        if n_epochs_subj < N_SPLITS:
            print(f"Skipping Subject {subject_id} - Only {n_epochs_subj} epochs available, which is less than n_splits={N_SPLITS} for CV.")
            continue

        # Apply the pre-loaded scaler (used for ConvAE input).
        # The scaler expects data flattened per epoch: (n_epochs, n_channels * n_samples).
        print("Applying saved scaler to raw EEG data (Flatten -> Scale -> Reshape)...")
        X_eeg_flat = X_eeg_raw.reshape(n_epochs_subj, -1)
        
        # Dimensionality check for scaler compatibility.
        if X_eeg_flat.shape[1] != len(scaler_mean) or X_eeg_flat.shape[1] != len(scaler_std):
            print(f"CRITICAL Error for Subject {subject_id}: Scaler dimensions ({len(scaler_mean)}) do not match flattened EEG data dimensions ({X_eeg_flat.shape[1]}).")
            print(f"  Expected {N_CHANNELS*N_SAMPLES} features for scaler. Got {X_eeg_flat.shape[1]}. Check N_CHANNELS/N_SAMPLES consistency.")
            print(f"Skipping Subject {subject_id}.")
            continue
        
        X_scaled_flat = ((X_eeg_flat - scaler_mean) / scaler_std).astype(np.float32)
        # Reshape back to (n_epochs, n_channels, n_samples) for the ConvAE encoder.
        X_scaled_for_encoder = X_scaled_flat.reshape(n_epochs_subj, n_channels_data_subj, n_samples_data_subj)
        print(f"Subject {subject_id}: Scaled & Reshaped data for encoder: {X_scaled_for_encoder.shape}")

        # Create DataLoader for batch-wise feature extraction.
        X_tensor_for_encoder = torch.tensor(X_scaled_for_encoder, dtype=torch.float32)
        # Labels are not used by the encoder but are required by TensorDataset.
        dummy_targets_for_encoder = torch.zeros(n_epochs_subj, dtype=torch.long) 
        feature_extraction_dataset = TensorDataset(X_tensor_for_encoder, dummy_targets_for_encoder)
        feature_extraction_loader = DataLoader(feature_extraction_dataset, batch_size=BATCH_SIZE_FEATURE_EXTRACTION, shuffle=False, num_workers=0)

        # 5. Extract Latent Features using the loaded ConvAE encoder.
        X_latent_features = extract_features(encoder, feature_extraction_loader, device=DEVICE)

        if X_latent_features is None:
            print(f"Skipping Subject {subject_id} - Feature extraction failed or yielded no features.")
            continue
        print(f"Subject {subject_id}: Extracted latent features shape: {X_latent_features.shape}")
        
        # Verify extracted feature dimensionality.
        current_extracted_latent_dim = X_latent_features.shape[1]
        if current_extracted_latent_dim != LATENT_DIM:
            print(f"Warning for Subject {subject_id}: Extracted feature dimension ({current_extracted_latent_dim}) "
                  f"differs from configured LATENT_DIM ({LATENT_DIM}). "
                  f"The MLP input dimension will be set to {current_extracted_latent_dim}.")
        
        # 6. Define Cross-Validation Strategy for classifier evaluation.
        # StratifiedKFold ensures that each fold is a good representative of the whole dataset in terms of class proportions.
        cv_strategy = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE_FOLDS, random_state=RANDOM_STATE)

        # --- Evaluate ConvAE features + PyTorch MLP ---
        print(f"\nRunning {N_SPLITS}-fold Cross-Validation for ConvAE+PyTorchMLP (Subject {subject_id})...")
        fold_accuracies_mlp = []
        for fold_num, (train_indices, val_indices) in enumerate(cv_strategy.split(X_latent_features, y_labels_zero_based)):
            print(f"    MLP Fold {fold_num + 1}/{N_SPLITS}...")
            X_train_fold_features, X_val_fold_features = X_latent_features[train_indices], X_latent_features[val_indices]
            y_train_fold_labels, y_val_fold_labels = y_labels_zero_based[train_indices], y_labels_zero_based[val_indices]

            # This scaler is fit only on the training part of the current fold.
            feature_scaler_mlp = StandardScaler()
            X_train_fold_features_scaled = feature_scaler_mlp.fit_transform(X_train_fold_features)
            X_val_fold_features_scaled = feature_scaler_mlp.transform(X_val_fold_features)

            # Train the MLP for the current fold.
            trained_mlp_model = train_pytorch_mlp_fold(
                X_train_fold_features_scaled, y_train_fold_labels, 
                X_val_fold_features_scaled, y_val_fold_labels,
                input_dim=current_extracted_latent_dim, # Using the actual dimension of extracted features.
                num_classes=N_CLASSES, device=DEVICE
            )

            # Evaluate the trained MLP on the validation part of the fold.
            trained_mlp_model.eval()
            with torch.no_grad():
                X_val_tensor_scaled = torch.tensor(X_val_fold_features_scaled, dtype=torch.float32).to(DEVICE)
                outputs_val = trained_mlp_model(X_val_tensor_scaled)
                _, y_pred_fold_mlp = torch.max(outputs_val.data, 1) # Get predicted class indices.
                y_pred_fold_mlp = y_pred_fold_mlp.cpu().numpy()

            fold_accuracy = accuracy_score(y_val_fold_labels, y_pred_fold_mlp)
            fold_accuracies_mlp.append(fold_accuracy)
            print(f"    MLP Fold {fold_num + 1} Accuracy: {fold_accuracy:.4f}")

        mean_accuracy_mlp = np.mean(fold_accuracies_mlp)
        std_accuracy_mlp = np.std(fold_accuracies_mlp)
        print(f"    Overall Mean Accuracy (ConvAE+PyTorchMLP) for Subject {subject_id}: {mean_accuracy_mlp:.4f} (+/- {std_accuracy_mlp:.4f})")
        all_subject_results.append({
            'subject': f'S{subject_id:02d}', # Standardised subject naming.
            'pipeline': 'ConvAE+PyTorchMLP',
            f'mean_{METRIC}': mean_accuracy_mlp, 
            f'std_{METRIC}': std_accuracy_mlp,
            'n_total_epochs_data': n_epochs_subj # Number of epochs used from this subject's data.
        })

        # --- Evaluate ConvAE features + SVM ---
        print(f"\nRunning {N_SPLITS}-fold Cross-Validation for ConvAE+SVM (Subject {subject_id})...")
        try:
            # Define the SVM classifier and create a pipeline with scaling.
            # Scaling is crucial for SVM performance.
            svm_classifier = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=False, random_state=RANDOM_STATE)
            svm_pipeline = Pipeline([('scaler', StandardScaler()), ('SVM', svm_classifier)])
            
            # Perform cross-validation using scikit-learn's utility.
            # n_jobs=-1 uses all available CPU cores for parallelisation if the underlying model supports it (SVC does for some parts).
            accuracies_svm_cv = cross_val_score(svm_pipeline, X_latent_features, y_labels_zero_based, cv=cv_strategy, scoring=METRIC, n_jobs=-1)

            mean_accuracy_svm = np.mean(accuracies_svm_cv)
            std_accuracy_svm = np.std(accuracies_svm_cv)
            print(f"    Mean Accuracy (ConvAE+SVM) for Subject {subject_id}: {mean_accuracy_svm:.4f} (+/- {std_accuracy_svm:.4f})")
            print(f"    Individual fold scores for SVM: {accuracies_svm_cv}")
            all_subject_results.append({
                'subject': f'S{subject_id:02d}', 
                'pipeline': 'ConvAE+SVM',
                f'mean_{METRIC}': mean_accuracy_svm, 
                f'std_{METRIC}': std_accuracy_svm,
                'n_total_epochs_data': n_epochs_subj
            })
        except Exception as e:
            # Catch any unexpected errors during SVM cross-validation.
            print(f"    An unexpected error occurred during Cross-Validation for ConvAE+SVM on Subject {subject_id}: {e}")
            print("    Skipping SVM evaluation for this subject.")


    # --- Process and Save Final Results ---
    if not all_subject_results:
        print("\nNo results were generated for any subject or pipeline. Check logs for errors.")
        return

    results_dataframe = pd.DataFrame(all_subject_results)
    print("\n=== Hybrid ConvAE Model Evaluation Summary (Within-Session CV) ===")
    # .to_string() ensures the full DataFrame is printed, not truncated.
    print(results_dataframe.to_string())

    # Calculate average performance across all subjects for each pipeline.
    average_pipeline_performance = results_dataframe.groupby('pipeline')[f'mean_{METRIC}'].agg(['mean', 'std'])
    print("\n--- Average Performance Across All Subjects ---")
    print(average_pipeline_performance)

    # Ensure the results directory exists.
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Save detailed per-subject results to a CSV file.
    detailed_results_filename = RESULTS_PATH / f"hybrid_conv_ae_evaluation_{METRIC}_detailed.csv"
    results_dataframe.to_csv(detailed_results_filename, index=False, float_format='%.4f')
    print(f"\nDetailed per-subject results saved to: {detailed_results_filename}")

    # Save average cross-subject results to a CSV file.
    average_results_filename = RESULTS_PATH / f"hybrid_conv_ae_evaluation_{METRIC}_average_across_subjects.csv"
    average_pipeline_performance.to_csv(average_results_filename, float_format='%.4f')
    print(f"Average cross-subject results saved to: {average_results_filename}")

# --- Script Execution Entry Point ---
if __name__ == "__main__":
    # Record start time for overall script duration.
    script_start_time = time.time()
    
    # Ensure the directory for saving models and results exists.
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    run_hybrid_evaluation()
    
    script_duration = time.time() - script_start_time
    print(f"\nTotal script execution time: {script_duration:.2f} seconds ({script_duration/60:.2f} minutes).")
