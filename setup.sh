# Create directories (using -p ensures parent directories are created if needed)
mkdir -p training/models
mkdir -p training/results                                   # Added a place to store results
mkdir -p training/scripts/MindReaderModel/FlightSafetyModel # Keep structure consistent
mkdir -p training/scripts/MindReaderModel                   # Already created by above, but safe

# Create empty files within training/scripts/MindReaderModel/
touch training/scripts/MindReaderModel/__init__.py
touch training/scripts/MindReaderModel/config.py
touch training/scripts/MindReaderModel/preprocess_data.py
touch training/scripts/MindReaderModel/autoencoder_model.py
touch training/scripts/MindReaderModel/train_autoencoder.py
touch training/scripts/MindReaderModel/supervised_models.py
touch training/scripts/MindReaderModel/train_evaluate_hybrid.py
touch training/scripts/MindReaderModel/train_evaluate_baseline.py
touch training/scripts/MindReaderModel/utils.py

# Optional: Create empty files for the FlightSafetyModel part if needed now
touch training/scripts/FlightSafetyModel/__init__.py
touch training/scripts/FlightSafetyModel/cv_model.py
touch training/scripts/FlightSafetyModel/train_cv_model.py
touch training/scripts/FlightSafetyModel/evaluate_cv_model.py
