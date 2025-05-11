# Flymind

Flymind software for mind-controlled drones.

## Overview

Flymind is a project focused on controlling drones using brain-computer interface (BCI) technology, specifically through electroencephalography (EEG) signals. The project includes data preprocessing, several machine learning model architectures for EEG signal classification, and a Pygame-based simulation environment to visualize the drone's response to predicted commands.

The EEG data is preprocessed using `training/scripts/MindReaderModel/preprocess_data.py`, which handles loading, filtering, and epoching of the raw GDF data files.

## Dataset

The dataset used in this project, is the BCI competition 2a dataset which can be found [here](https://bbci.de/competition/iv/download/)

## Models

This project explores three main types of models for classifying motor imagery tasks from EEG data:

1. **CSP + Classifier (Baseline)**

   - Traditional approach using Common Spatial Patterns (CSP) for feature extraction.
   - Features are classified using either Linear Discriminant Analysis (LDA) or Support Vector Machine (SVM).
   - Training and evaluation scripts: `train_evaluate_baseline.py`

2. **Hybrid Model (Unsupervised Pre-training + Supervised Classifier)**

   - **Unsupervised Feature Learning**: A 1D Convolutional Autoencoder (`autoencoder_model.py`) is first trained on all subjects' data to learn a compressed representation (latent features) of the EEG signals. This is handled by `train_autoencoder.py`.
   - **Supervised Classification**: Latent features are then used to train a classifier:

     - A Multi-Layer Perceptron (MLP) implemented in PyTorch (`hybrid_models.py`).
     - Or a Support Vector Machine (SVM).

   - Training and evaluation script: `train_evaluate_hybrid.py`

3. **MindReaderModel (Supervised End-to-End Deep Learning)**

   - Supervised, end-to-end model based on the EEGNet architecture (`supervised_models.py`).
   - Learns features directly from preprocessed EEG data and classifies into one of four motor imagery tasks (left hand, right hand, feet, tongue).
   - Cross-validation script: `train_evaluate_mindreader.py`
   - Final training script for deployment: `train_final_model.py`

## Getting Started

### 1. Setup

Navigate to repository:

```bash
cd flymind
```

Install requirements:

```bash
pip install -r requirements.txt
```

or for conda:

```bash
conda env create -f flymind.yml
```

### 2. Running the Pygame Simulation

The Pygame simulation visualizes a drone's movement based on EEG command predictions or ground truth data.

**Example:** Run simulation for Subject 9, Training session:

```bash
python pygame_simulation.py --subject_id 9 --session_type T
```

**General Usage:**

```bash
python pygame_simulation.py --subject_id <ID> --session_type <TYPE>

```

- `<ID>`: Subject ID (e.g., 1, 2, ..., 9).
- `<TYPE>`: Session type (`T` for training, `E` for evaluation).

The simulation uses `MindReaderService/mind_reader_predictor.py` to load a pre-trained model (default: Subject 8) and predict commands in real time from EEG epochs. If no predictor is available or disabled, ground truth commands are used.

### 3. Training Models

Training and evaluation scripts are located in `flymind/training/scripts/MindReaderModel/training_scripts/`. Run using the `-m` flag from the root directory to resolve module imports correctly.

- **Train/Evaluate Final MindReaderModel:**

  ```bash
  python -m training.scripts.MindReaderModel.training_scripts.train_final_model
  ```

  Saves a `.pth` model file and a scaler (`.npz`) in `training/models/`.

- **Train/Evaluate Hybrid Models (ConvAE + MLP/SVM):**

  ```bash
  python -m training.scripts.MindReaderModel.training_scripts.train_evaluate_hybrid
  ```

- **Train Convolutional Autoencoder:**

  ```bash
  python -m training.scripts.MindReaderModel.training_scripts.train_autoencoder
  ```

- **Cross-Validation for Baseline CSP Models:**

  ```bash
  python -m training.scripts.MindReaderModel.training_scripts.train_evaluate_baseline
  ```

- **Cross-Validation for End-to-End MindReaderModel:**

  ```bash
  python -m training.scripts.MindReaderModel.training_scripts.train_evaluate_mindreader
  ```

Configuration files for hyperparameters are in `flymind/training/scripts/MindReaderModel/configs/`. Evaluation results (accuracy scores, plots) are saved in `flymind/training/results/`.

> **Note:** ROS 2 and Gazebo integration details will be added later.
