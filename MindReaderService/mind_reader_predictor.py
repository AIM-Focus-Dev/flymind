import torch
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler

# --- Default Configuration Constants ---
# These constants define default parameters for the predictor.
# Instance-specific values can override these if provided during initialisation.
TARGET_SUBJECT_ID = 8   # Default subject identifier for model loading.
N_CHANNELS = 22         # Number of EEG channels, based on the dataset (e.g., BCICIV_2a).
N_SAMPLES = 751         # Number of time samples per EEG epoch, consistent with training (see supervised_models.py).
N_CLASSES = 4           # Number of distinct classes/commands the model predicts (see supervised_models.py).

class MindReaderPredictor:
    """
    Manages loading a pre-trained MindReader EEG classification model and its associated
    data scaler to predict commands from new EEG epochs. It handles path configurations,
    device selection (CPU/GPU), dynamic import of model components, and class mapping.
    """
    def __init__(self, subject_id=TARGET_SUBJECT_ID, project_root_path=None):
        """
        Initialises the MindReaderPredictor for a specific subject.

        Args:
            subject_id (int): The identifier of the subject whose trained model should be loaded.
            project_root_path (str, optional): Absolute path to the project root directory.
                                               If None, it attempts to auto-detect the root.
        """
        self.subject_id = subject_id
        self.is_ready = False  # Flag indicating if the predictor initialised successfully.

        # Determine and configure the project's root directory.
        if project_root_path:
            self.PROJECT_ROOT = Path(project_root_path)
        else:
            # Auto-detect project root assuming this script is within a known structure
            self.PROJECT_ROOT = Path(__file__).resolve().parent.parent

        # Add project root to sys.path to allow for absolute imports of project modules.
        if str(self.PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(self.PROJECT_ROOT))
        
        print(f"MindReaderPredictor: Project root set to: {self.PROJECT_ROOT}")

        # Define critical paths for data and models relative to the project root.
        self.DATA_PATH_PREDICT = self.PROJECT_ROOT / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
        self.MODELS_PATH_PREDICT = self.PROJECT_ROOT / "training" / "models"

        # Configure the computation device (MPS for Apple Silicon, CUDA for NVIDIA, else CPU).
        if torch.backends.mps.is_available():
            self.DEVICE = torch.device("mps")
        elif torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        else:
            self.DEVICE = torch.device("cpu")
        print(f"MindReaderPredictor: Using computation device: {self.DEVICE}")

        # Dynamically import core components from the 'training' module.
        # This approach can help manage dependencies or allow for more flexible deployment.
        try:
            # Import the neural network model architecture.
            from training.scripts.MindReaderModel.supervised_models import MindReaderModel
            self.MindReaderModel = MindReaderModel
            # Import the utility function for loading and preprocessing EEG data.
            from training.scripts.MindReaderModel.preprocess_data import load_and_preprocess_subject_data
            self.load_and_preprocess_subject_data_util = load_and_preprocess_subject_data
            print("MindReaderPredictor: Core model and data utilities imported successfully.")
        except ImportError as e:
            print(f"MindReaderPredictor Error: Critical dependency import failed: {e}")
            # Initialisation cannot proceed without these components.
            return # self.is_ready will remain False

        # Initialise essential components for prediction.
        # Use data from subject 1 for a stable class mapping, assuming consistent event IDs.
        self._get_class_names_and_mapping(subject_id_for_map_check=1)
        
        if not self._load_model_and_scaler():
            # Model or scaler loading failed; predictor is not ready.
            return # self.is_ready will remain False
        
        self.is_ready = True
        print(f"MindReaderPredictor for Subject {self.subject_id} initialised successfully.")

    def _get_class_names_and_mapping(self, subject_id_for_map_check=1):
        """
        Determines the mapping from MNE event IDs and internal model output indices
        to human-readable class names and target drone actions. This ensures consistent
        interpretation of model predictions. It uses a reference subject's training
        data to establish this mapping.

        Stores:
            self.INDEX_TO_CLASS_NAME_MAP (dict): Maps 0-indexed model output to class names (e.g., 'left_hand').
            self.MNE_ID_TO_INDEX_MAP (dict): Maps MNE software's event IDs to 0-indexed model outputs.
            self.EEG_CLASS_TO_DRONE_ACTION_MAP (dict): Maps 0-indexed model output to drone command strings.
        
        Args:
            subject_id_for_map_check (int): Subject ID used to load epoch data for event ID inspection.
                                            Chosen for stability of event_id dictionary.
        """
        print("MindReaderPredictor: Determining class name and action mapping...")
        try:
            # Load training session data for the reference subject to inspect event IDs.
            temp_epochs_train = self.load_and_preprocess_subject_data_util(
                subject_id_for_map_check,
                session_type='T', # 'T' for training session
                data_path=self.DATA_PATH_PREDICT
            )
            if temp_epochs_train and hasattr(temp_epochs_train, 'event_id') and temp_epochs_train.event_id:
                # MNE's event_id is a dictionary like {'left_hand': 769, 'right_hand': 770, ...}
                # We need to create a sorted, 0-indexed mapping for model output.
                mne_id_to_task_name = {v: k for k, v in temp_epochs_train.event_id.items()}
                sorted_mne_ids = sorted(mne_id_to_task_name.keys())
                
                self.MNE_ID_TO_INDEX_MAP = {mne_id: i for i, mne_id in enumerate(sorted_mne_ids)}
                ordered_class_names = [mne_id_to_task_name[mne_id] for mne_id in sorted_mne_ids]
                self.INDEX_TO_CLASS_NAME_MAP = {i: name for i, name in enumerate(ordered_class_names)}
                
                print(f"  Determined MNE event ID to 0-indexed map: {self.MNE_ID_TO_INDEX_MAP}")
                print(f"  Ordered class names (0 to {N_CLASSES-1}): {ordered_class_names}")

                # Define a default mapping from EEG class indices to drone actions.
                # The actual drone commands (e.g., "TURN_LEFT") are application-specific
                # and interpreted by the consuming system (e.g., a Pygame script).
                self.EEG_CLASS_TO_DRONE_ACTION_MAP = {}
                for i, name in enumerate(ordered_class_names):
                    if 'left_hand' in name.lower(): self.EEG_CLASS_TO_DRONE_ACTION_MAP[i] = "TURN_LEFT"
                    elif 'right_hand' in name.lower(): self.EEG_CLASS_TO_DRONE_ACTION_MAP[i] = "TURN_RIGHT"
                    elif 'feet' in name.lower(): self.EEG_CLASS_TO_DRONE_ACTION_MAP[i] = "FORWARD"
                    elif 'tongue' in name.lower(): self.EEG_CLASS_TO_DRONE_ACTION_MAP[i] = "UP"
                    else: self.EEG_CLASS_TO_DRONE_ACTION_MAP[i] = "HOVER" # Fallback for unrecognised classes
                
                print(f"  Internal EEG Class Index to Drone Action Map: {self.EEG_CLASS_TO_DRONE_ACTION_MAP}")

            else:
                raise RuntimeError("Could not determine class mapping from temp_epochs_train attributes (event_id missing or empty).")
        except Exception as e:
            print(f"MindReaderPredictor Error: Could not automatically determine class name mapping: {e}")
            # Fallback mapping if automatic determination fails. This is less reliable as it assumes
            # a fixed order of classes ('left_hand', 'right_hand', 'feet', 'tongue').
            self.INDEX_TO_CLASS_NAME_MAP = {0: 'left_hand', 1: 'right_hand', 2: 'feet', 3: 'tongue'}
            self.EEG_CLASS_TO_DRONE_ACTION_MAP = {0: "TURN_LEFT", 1: "TURN_RIGHT", 2: "FORWARD", 3: "UP"}
            self.MNE_ID_TO_INDEX_MAP = None # Cannot determine this reliably on fallback.
            print("  Warning: Using fallback class name and command mapping. Accuracy may be affected if class order differs.")


    def _load_model_and_scaler(self):
        """
        Loads the pre-trained MindReaderModel (PyTorch .pth file) and the corresponding
        StandardScaler object (from a .npz file) for the initialised subject.
        The scaler is essential for preprocessing input EEG data to match the training distribution.

        Returns:
            bool: True if both model and scaler were loaded successfully, False otherwise.
        """
        model_filename = f"mindreader_subject{self.subject_id:02d}_final.pth"
        scaler_filename = f"scaler_subject{self.subject_id:02d}.npz"

        model_path = self.MODELS_PATH_PREDICT / model_filename
        scaler_path = self.MODELS_PATH_PREDICT / scaler_filename

        if not model_path.exists():
            print(f"MindReaderPredictor Error: Model file not found: {model_path}")
            return False
        if not scaler_path.exists():
            print(f"MindReaderPredictor Error: Scaler file not found: {scaler_path}")
            return False

        try:
            # Load the MindReaderModel
            self.model = self.MindReaderModel(n_channels=N_CHANNELS, n_samples=N_SAMPLES, num_classes=N_CLASSES)
            print(f"  Loading model from {model_path}...")
            # `weights_only=True` is a security best practice for loading PyTorch models.
            self.model.load_state_dict(torch.load(model_path, map_location=self.DEVICE, weights_only=True))
            self.model.to(self.DEVICE) # Move model to the selected computation device.
            self.model.eval()         # Set model to evaluation mode (disables dropout, etc.).
            print(f"  Model loaded successfully.")

            # Load and reconstruct the StandardScaler from .npz file.
            # The .npz file should contain 'mean_' and 'scale_' arrays.
            print(f"  Loading scaler from .npz file: {scaler_path}...")
            scaler_data = np.load(scaler_path)
            self.scaler = StandardScaler()

            # Expected and optional keys for robust scaler reconstruction.
            # Handles variations in key names (e.g., 'mean' vs 'mean_').
            expected_keys = {'mean_': ['mean_', 'mean'], 'scale_': ['scale_', 'scale']}
            optional_keys = {'var_': ['var_', 'var'], 'n_samples_seen_': ['n_samples_seen_', 'n_samples_seen']}

            for attr_name, possible_keys in expected_keys.items():
                found_key = None
                for key in possible_keys:
                    if key in scaler_data:
                        setattr(self.scaler, attr_name, scaler_data[key])
                        found_key = key
                        break
                if not found_key:
                    raise KeyError(f"Required scaler attribute '{attr_name}' (tried keys: {possible_keys}) not found in {scaler_path}. Available keys: {list(scaler_data.keys())}")

            for attr_name, possible_keys in optional_keys.items():
                for key in possible_keys:
                    if key in scaler_data:
                        setattr(self.scaler, attr_name, scaler_data[key])
                        break
            
            # Verify essential attributes are present after loading.
            if not (hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_')):
                raise ValueError("Scaler reconstructed without essential 'mean_' or 'scale_' attributes.")
            
            print(f"  Scaler loaded and reconstructed successfully from {scaler_path}")
            scaler_data.close() # Close the .npz file.
            return True

        except Exception as e:
            print(f"MindReaderPredictor Error: Failed to load model or scaler: {e}")
            return False


    def predict_command(self, eeg_epoch_data_np):
        """
        Predicts a drone command from a single EEG epoch.

        The input EEG data undergoes scaling using the pre-loaded StandardScaler,
        then is fed into the neural network model for classification.

        Args:
            eeg_epoch_data_np (np.ndarray): A single epoch of EEG data, expected to have
                                            shape (N_CHANNELS, N_SAMPLES).

        Returns:
            tuple: (command_str, confidence_float)
                   - command_str (str): The predicted drone command (e.g., "FORWARD", "HOVER").
                                        Returns None if prediction fails.
                   - confidence_float (float): The model's confidence (softmax probability)
                                               in the predicted command. Returns 0.0 if prediction fails.
        """
        if not self.is_ready:
            print("MindReaderPredictor not ready. Cannot perform prediction.")
            return None, 0.0

        # Validate input data shape.
        if eeg_epoch_data_np.shape != (N_CHANNELS, N_SAMPLES):
            print(f"MindReaderPredictor Error: Input EEG epoch data shape mismatch. Expected ({N_CHANNELS}, {N_SAMPLES}), but got {eeg_epoch_data_np.shape}")
            return None, 0.0

        try:
            # Apply scaling: StandardScaler expects input of shape (n_samples_or_trials, n_features).
            # For EEG data shaped (channels, timepoints), we treat each channel as a "sample"
            # and its timepoints as "features" for the scaler. This means the scaler was fit
            # channel-wise during training (i.e., `scaler.fit(all_epochs_data.reshape(-1, N_SAMPLES))`).
            scaled_epoch_data = self.scaler.transform(eeg_epoch_data_np) # Shape: (N_CHANNELS, N_SAMPLES)
            
            # Reshape data for the model input: (batch_size, n_channels, n_samples).
            # Here, batch_size is 1 as we process a single epoch.
            # The MindReaderModel architecture might internally expect (Batch, 1, Channels, Timepoints)
            # and handle the addition of the singleton dimension for "height" itself if it's a 2D ConvNet.
            scaled_epoch_data_model_input = scaled_epoch_data.reshape(1, N_CHANNELS, N_SAMPLES)

            # Convert to PyTorch tensor and move to the configured device.
            input_tensor = torch.tensor(scaled_epoch_data_model_input, dtype=torch.float32).to(self.DEVICE)

            # Perform inference.
            with torch.no_grad(): # Disable gradient calculations for inference.
                outputs = self.model(input_tensor) # Model output shape: (1, N_CLASSES)
                
                # Apply softmax to convert logits to probabilities.
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get the predicted class index and its confidence.
                confidence_tensor, predicted_class_idx_tensor = torch.max(probabilities, 1)
                
                predicted_class_idx = predicted_class_idx_tensor.item() # Extract scalar value
                confidence = confidence_tensor.item()                   # Extract scalar value

            # Map the numeric predicted class index to a human-readable command string.
            command_str = self.EEG_CLASS_TO_DRONE_ACTION_MAP.get(predicted_class_idx, "HOVER") # Default to "HOVER" if index is somehow out of map.
            
            return command_str, confidence

        except Exception as e:
            print(f"MindReaderPredictor Error: Prediction failed due to an exception: {e}")
            return None, 0.0

# --- Test Execution Block ---
# This block allows for direct testing of the MindReaderPredictor class
# when the script is executed as the main program.
if __name__ == '__main__':
    print("Testing MindReaderPredictor Class...")
    # Note on PROJECT_ROOT for standalone testing:
    # If this script is moved, or if the current working directory is not
    # the parent of 'MindReaderService' (e.g., 'flymind/'), the auto-detection
    # of PROJECT_ROOT might fail. In such cases, pass `project_root_path` explicitly
    # to the MindReaderPredictor constructor during testing.
    # For example, if run from the 'flymind' directory as:
    # python MindReaderService/mind_reader_predictor.py
    # the default Path(__file__).resolve().parent.parent should correctly point to 'flymind'.
    
    predictor_test = MindReaderPredictor(subject_id=TARGET_SUBJECT_ID) # Uses default subject ID
    
    if predictor_test.is_ready:
        print(f"\nPredictor for Subject {TARGET_SUBJECT_ID} initialised and ready.")
        print(f"  Using device: {predictor_test.DEVICE}")
        print(f"  Class to action map being used: {predictor_test.EEG_CLASS_TO_DRONE_ACTION_MAP}")
        print(f"  Index to class name map: {predictor_test.INDEX_TO_CLASS_NAME_MAP}")

        # Create a dummy EEG epoch for testing the prediction pipeline.
        # Data is random noise, so the prediction itself will be arbitrary.
        dummy_epoch_data = np.random.randn(N_CHANNELS, N_SAMPLES).astype(np.float64)
        print(f"\nTesting prediction with a dummy EEG epoch of shape: {dummy_epoch_data.shape}")
        
        command, confidence = predictor_test.predict_command(dummy_epoch_data)
        
        if command is not None:
            predicted_class_name = "Unknown"
            # Find class name for better understanding of dummy prediction
            for idx, cmd_name in predictor_test.EEG_CLASS_TO_DRONE_ACTION_MAP.items():
                if cmd_name == command:
                    predicted_class_name = predictor_test.INDEX_TO_CLASS_NAME_MAP.get(idx, "Unknown")
                    break
            print(f"  Predicted Command: '{command}' (corresponds to class: '{predicted_class_name}'), Confidence: {confidence:.4f}")
        else:
            print("  Prediction test failed to produce a command.")
    else:
        print(f"Predictor for Subject {TARGET_SUBJECT_ID} failed to initialise. Check error messages above.")

