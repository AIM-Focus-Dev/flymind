# Includes config for hybrid model training script
from pathlib import Path  # For handling paths reliably

try:
    #  configs → MindReaderModel → scripts → training → (repo) flymind
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
except NameError:
    # if running interactively from project root, just use cwd
    PROJECT_ROOT = Path('.').resolve()

DATA_PATH    = PROJECT_ROOT / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
RESULTS_PATH = PROJECT_ROOT / "training" / "results"  # Added for potential saving later
MODELS_PATH  = PROJECT_ROOT / "training" / "models"   # Added for saving models later

# make sure directories exist
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

FS        = 250    # Sampling frequency of the dataset (Hz)
LOW_FREQ  = 8.0    # Lower cut-off frequency for band-pass filter (Hz)
HIGH_FREQ = 35.0   # Upper cut-off frequency for band-pass filter (Hz)
TMIN      = 0.5    # Start epochs 0.5 seconds after the cue
TMAX      = 3.5    # End epochs 3.5 seconds after the cue (3-second epoch length)
N_SUBJECTS = 9     # Total number of subjects in dataset

# Descriptions mapping to task names (keys are strings matching potential annotations)
EVENT_DESC_MAP = {
    '769': 'left_hand',
    '770': 'right_hand',
    '771': 'feet',
    '772': 'tongue'
}
