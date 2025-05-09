import torch
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler

# --- Constants to be accessible by Pygame if imported directly ---
# These are defaults; the class can also store its own instance-specific values.
TARGET_SUBJECT_ID = 8 # Default subject
N_CHANNELS = 22
N_SAMPLES = 751 # From supervised_models.py
N_CLASSES = 4   # From supervised_models.py

class MindReaderPredictor:
    def __init__(self, subject_id=TARGET_SUBJECT_ID, project_root_path=None):
        self.subject_id = subject_id
        self.is_ready = False

        if project_root_path:
            self.PROJECT_ROOT = Path(project_root_path)
        else:
            # Auto-detect project root (assuming this file is in MindReaderService, and flymind is parent)
            self.PROJECT_ROOT = Path(__file__).resolve().parent.parent

        if str(self.PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(self.PROJECT_ROOT))
        
        print(f"MindReaderPredictor using PROJECT_ROOT: {self.PROJECT_ROOT}")

        # Define paths based on the determined PROJECT_ROOT
        self.DATA_PATH_PREDICT = self.PROJECT_ROOT / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
        self.MODELS_PATH_PREDICT = self.PROJECT_ROOT / "training" / "models"

        # Device setup
        if torch.backends.mps.is_available():
            self.DEVICE = torch.device("mps")
        elif torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        else:
            self.DEVICE = torch.device("cpu")
        print(f"MindReaderPredictor using device: {self.DEVICE}")

        # Import necessary model class and data loading utility
        try:
            from training.scripts.MindReaderModel.supervised_models import MindReaderModel #
            self.MindReaderModel = MindReaderModel
            from training.scripts.MindReaderModel.preprocess_data import load_and_preprocess_subject_data
            self.load_and_preprocess_subject_data_util = load_and_preprocess_subject_data
            print("MindReaderPredictor: Core model and data utilities imported.")
        except ImportError as e:
            print(f"MindReaderPredictor Error: Could not import dependencies: {e}")
            return # is_ready remains False

        # Initialize components
        self._get_class_names_and_mapping(subject_id_for_map_check=1) # Use subject 1 for stable mapping
        if not self._load_model_and_scaler():
            return # is_ready remains False
        
        self.is_ready = True
        print(f"MindReaderPredictor for Subject {self.subject_id} initialized successfully.")

    def _get_class_names_and_mapping(self, subject_id_for_map_check=1):
        """
        Determines class name mapping from training data to ensure consistency.
        Stores: self.INDEX_TO_CLASS_NAME_MAP, self.MNE_ID_TO_INDEX_MAP, self.CLASS_TO_COMMAND_MAP
        """
        print("MindReaderPredictor: Determining class name mapping...")
        try:
            temp_epochs_train = self.load_and_preprocess_subject_data_util(
                subject_id_for_map_check,
                session_type='T',
                data_path=self.DATA_PATH_PREDICT
            )
            if temp_epochs_train and hasattr(temp_epochs_train, 'event_id') and temp_epochs_train.event_id:
                mne_id_to_task_name = {v: k for k, v in temp_epochs_train.event_id.items()}
                sorted_mne_ids = sorted(mne_id_to_task_name.keys())
                
                self.MNE_ID_TO_INDEX_MAP = {mne_id: i for i, mne_id in enumerate(sorted_mne_ids)}
                ordered_class_names = [mne_id_to_task_name[mne_id] for mne_id in sorted_mne_ids]
                self.INDEX_TO_CLASS_NAME_MAP = {i: name for i, name in enumerate(ordered_class_names)}
                
                print(f"  Determined MNE ID to 0-indexed map: {self.MNE_ID_TO_INDEX_MAP}")
                print(f"  Ordered class names (0 to {N_CLASSES-1}): {ordered_class_names}")

                # Define default command map based on typical BCI2000 class interpretations
                # User's Pygame script will define its own preferred final commands based on these names
                # For example: left_hand -> "TURN_LEFT", right_hand -> "TURN_RIGHT", feet -> "FORWARD", tongue -> "UP"
                self.EEG_CLASS_TO_DRONE_ACTION_MAP = {}
                for i, name in enumerate(ordered_class_names):
                    if 'left_hand' in name.lower(): self.EEG_CLASS_TO_DRONE_ACTION_MAP[i] = "TURN_LEFT"
                    elif 'right_hand' in name.lower(): self.EEG_CLASS_TO_DRONE_ACTION_MAP[i] = "TURN_RIGHT"
                    elif 'feet' in name.lower(): self.EEG_CLASS_TO_DRONE_ACTION_MAP[i] = "FORWARD"
                    elif 'tongue' in name.lower(): self.EEG_CLASS_TO_DRONE_ACTION_MAP[i] = "UP"
                    else: self.EEG_CLASS_TO_DRONE_ACTION_MAP[i] = "HOVER" # Fallback
                
                print(f"  Internal EEG Class Index to Drone Action Map: {self.EEG_CLASS_TO_DRONE_ACTION_MAP}")

            else:
                raise RuntimeError("Could not determine class mapping from temp_epochs_train attributes.")
        except Exception as e:
            print(f"MindReaderPredictor Error: Could not automatically determine class name mapping: {e}")
            # Fallback (less reliable, depends on consistent data processing)
            self.INDEX_TO_CLASS_NAME_MAP = {0: 'left_hand', 1: 'right_hand', 2: 'feet', 3: 'tongue'}
            self.EEG_CLASS_TO_DRONE_ACTION_MAP = {0: "TURN_LEFT", 1: "TURN_RIGHT", 2: "FORWARD", 3: "UP"}
            self.MNE_ID_TO_INDEX_MAP = None # Cannot determine this reliably on fallback
            print("  Using fallback class name and command mapping.")


    def _load_model_and_scaler(self):
        """Loads the trained MindReaderModel and the StandardScaler from .npz."""
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
            # Load model
            self.model = self.MindReaderModel(n_channels=N_CHANNELS, n_samples=N_SAMPLES, num_classes=N_CLASSES)
            print(f"  Loading model from {model_path}...")
            self.model.load_state_dict(torch.load(model_path, map_location=self.DEVICE, weights_only=True))
            self.model.to(self.DEVICE)
            self.model.eval()
            print(f"  Model loaded successfully.")

            # Load scaler from .npz file
            print(f"  Loading scaler from .npz file: {scaler_path}...")
            scaler_data = np.load(scaler_path)
            self.scaler = StandardScaler()

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
                    raise KeyError(f"Required scaler attribute '{attr_name}' (keys: {possible_keys}) not in {scaler_path}. Available: {list(scaler_data.keys())}")

            for attr_name, possible_keys in optional_keys.items():
                for key in possible_keys:
                    if key in scaler_data:
                        setattr(self.scaler, attr_name, scaler_data[key])
                        break
            
            if not (hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_')):
                 raise ValueError("Scaler reconstructed without 'mean_' or 'scale_'.")
            
            print(f"  Scaler loaded and reconstructed successfully from {scaler_path}")
            scaler_data.close()
            return True

        except Exception as e:
            print(f"MindReaderPredictor Error: Failed to load model or scaler: {e}")
            return False


    def predict_command(self, eeg_epoch_data_np):
        """
        Predicts a drone command from a single EEG epoch (NumPy array).
        Args:
            eeg_epoch_data_np (np.ndarray): Single epoch data, shape (N_CHANNELS, N_SAMPLES).
        Returns:
            tuple: (command_str, confidence_float) or (None, 0.0) if prediction fails.
        """
        if not self.is_ready:
            print("MindReaderPredictor not ready. Cannot predict.")
            return None, 0.0

        if eeg_epoch_data_np.shape != (N_CHANNELS, N_SAMPLES):
            print(f"MindReaderPredictor Error: Input epoch data shape mismatch. Expected ({N_CHANNELS}, {N_SAMPLES}), got {eeg_epoch_data_np.shape}")
            return None, 0.0

        try:
            # Scaling: StandardScaler expects (n_samples_or_trials, n_features).
            # For EEG (channels, timepoints), each channel is a "sample" and timepoints are "features".
            scaled_epoch_data = self.scaler.transform(eeg_epoch_data_np)
            
            # Reshape for model: (1, n_channels, n_samples) for batch_size=1
            # The MindReaderModel handles internal unsqueezing if it expects (B, 1, C, T)
            scaled_epoch_data_model_input = scaled_epoch_data.reshape(1, N_CHANNELS, N_SAMPLES)

            input_tensor = torch.tensor(scaled_epoch_data_model_input, dtype=torch.float32).to(self.DEVICE)

            with torch.no_grad():
                outputs = self.model(input_tensor) # Shape: (1, num_classes)
                probabilities = torch.softmax(outputs, dim=1)
                confidence_tensor, predicted_class_idx_tensor = torch.max(probabilities, 1)
                
                predicted_class_idx = predicted_class_idx_tensor.item()
                confidence = confidence_tensor.item()

            # Map predicted class index to command string
            command_str = self.EEG_CLASS_TO_DRONE_ACTION_MAP.get(predicted_class_idx, "HOVER") # Default to HOVER
            
            # For debugging in Pygame: get class name
            # class_name = self.INDEX_TO_CLASS_NAME_MAP.get(predicted_class_idx, "Unknown")
            # return command_str, confidence, class_name
            return command_str, confidence

        except Exception as e:
            print(f"MindReaderPredictor Error: Failed during prediction: {e}")
            return None, 0.0

# This file is now intended to be imported as a module.
# The old main execution block is removed.
# If you want to test this class independently:
if __name__ == '__main__':
    print("Testing MindReaderPredictor Class...")
    # Note: PROJECT_ROOT might need to be passed if this file is moved relative to 'flymind'
    # Or ensure this test is run when cwd is 'flymind/MindReaderService' or 'flymind/'
    # and adjust __file__ logic or pass project_root_path explicitly for testing.
    
    # Assuming the test is run from `flymind` directory:
    # python MindReaderService/mind_reader_predictor.py
    # The Path(__file__).resolve().parent.parent should correctly point to flymind
    
    predictor_test = MindReaderPredictor(subject_id=TARGET_SUBJECT_ID)
    if predictor_test.is_ready:
        print(f"\nPredictor for Subject {TARGET_SUBJECT_ID} is ready.")
        print(f"  Device: {predictor_test.DEVICE}")
        print(f"  Class to action map: {predictor_test.EEG_CLASS_TO_DRONE_ACTION_MAP}")

        # Create a dummy epoch for testing prediction
        dummy_epoch = np.random.randn(N_CHANNELS, N_SAMPLES).astype(np.float64)
        print(f"\nTesting prediction with dummy epoch of shape: {dummy_epoch.shape}")
        command, conf = predictor_test.predict_command(dummy_epoch)
        if command:
            print(f"  Predicted Command: {command}, Confidence: {conf:.4f}")
        else:
            print("  Prediction test failed.")
    else:
        print(f"Predictor for Subject {TARGET_SUBJECT_ID} failed to initialize.")
