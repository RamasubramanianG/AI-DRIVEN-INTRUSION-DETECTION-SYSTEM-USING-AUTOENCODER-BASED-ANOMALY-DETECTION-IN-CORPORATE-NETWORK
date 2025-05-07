from pathlib import Path
from src.config import *
import joblib
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
import os

# Directory and artifact utilities
def create_directories():
    """Create necessary directories"""
    for directory in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def load_preprocessing_artifacts():
    """Load encoders and scaler"""
    return (
        joblib.load(ENCODERS_PATH),
        joblib.load(SCALER_PATH),
        joblib.load(FEATURE_NAMES_PATH)
    )

# Model utilities
def save_keras_model(model, path):
    """Robust model saving with proper serialization
    Args:
        model: Keras model to save
        path: Path to save the model
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_model(
            model,
            path,
            overwrite=True,
            include_optimizer=True,
            save_format='tf'  # Using TensorFlow format for better compatibility
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {str(e)}")

def load_keras_model(path):
    """Robust model loading with custom objects
    Args:
        path: Path to saved model
    Returns:
        Loaded Keras model
    """
    try:
        return load_model(
            path,
            custom_objects={
                'mse': MeanSquaredError(),
                'mean_squared_error': MeanSquaredError(),
                'root_mean_squared_error': RootMeanSquaredError()
            },
            compile=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def model_exists(path=MODEL_PATH):
    """Check if model file exists and is valid
    Args:
        path: Path to model file
    Returns:
        bool: True if model exists and is valid
    """
    return os.path.exists(path) and os.path.getsize(path) > 0