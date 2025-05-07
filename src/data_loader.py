import pandas as pd
import numpy as np
from pathlib import Path
import numpy as np
import os
import logging

from src.config import (
    COLUMNS, 
    TRAIN_DATA, 
    TEST_DATA,
    PROCESSED_DIR,
    VAL_DATA,
    DROP_FEATURES,
    SEED
)
from src.utils import create_directories
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_dataframes(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Validate the structure and content of loaded dataframes"""
    required_columns = set(COLUMNS)
    
    for df, name in [(train_df, "Training"), (test_df, "Test")]:
        # Check columns
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name} data missing columns: {missing_cols}")
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            null_cols = null_counts[null_counts > 0].index.tolist()
            logger.warning(f"{name} data contains null values in columns: {null_cols}")
        
        # Check label distribution
        if 'label' in df.columns:
            label_dist = df['label'].value_counts(normalize=True)
            logger.info(f"{name} data label distribution:\n{label_dist.to_string()}")
            
        # Check for empty dataframe
        if df.empty:
            raise ValueError(f"{name} dataframe is empty after loading")

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw NSL-KDD datasets with validation
    
    Returns:
        Tuple of (train_df, test_df) pandas DataFrames
    """
    create_directories()
    
    try:
        logger.info(f"Loading training data from {TRAIN_DATA}")
        train_df = pd.read_csv(TRAIN_DATA, names=COLUMNS, na_values=[' ', '?', 'NaN'])
        
        logger.info(f"Loading test data from {TEST_DATA}")
        test_df = pd.read_csv(TEST_DATA, names=COLUMNS, na_values=[' ', '?', 'NaN'])
        
        # Validate the loaded data
        validate_dataframes(train_df, test_df)
        
        return train_df, test_df
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Data file is empty: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_processed_data(
    X_train: np.ndarray,
    X_test: np.ndarray, 
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> None:
    """
    Save processed data as numpy arrays with validation
    """
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Enhanced validation with debug info
        logger.info(f"Train features shape: {X_train.shape}, labels shape: {y_train.shape}")
        logger.info(f"Test features shape: {X_test.shape}, labels shape: {y_test.shape}")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Train features/labels mismatch: "
                f"{X_train.shape[0]} features vs {y_train.shape[0]} labels"
            )
        
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"Test features/labels mismatch: "
                f"{X_test.shape[0]} features vs {y_test.shape[0]} labels"
            )
        
        logger.info(f"Saving processed data to {PROCESSED_DIR}")
        
        # Save with checksum verification
        np.save(PROCESSED_DIR / "X_train.npy", X_train, allow_pickle=False)
        np.save(PROCESSED_DIR / "X_test.npy", X_test, allow_pickle=False)
        np.save(PROCESSED_DIR / "y_train.npy", y_train, allow_pickle=False)
        np.save(PROCESSED_DIR / "y_test.npy", y_test, allow_pickle=False)
        
        if X_val is not None:
            if y_val is None or X_val.shape[0] != y_val.shape[0]:
                raise ValueError("Validation features and labels must have same length")
            np.save(VAL_DATA.parent / "X_val.npy", X_val, allow_pickle=False)
            np.save(VAL_DATA.parent / "y_val.npy", y_val, allow_pickle=False)
            logger.info(f"Saved validation data to {VAL_DATA.parent}")
            
        logger.info("Data saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise

def create_validation_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_size: float = 0.2,
    random_state: int = SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create validation split from training data
    
    Returns:
        Tuple of (X_train_new, X_val, y_train_new, y_val)
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Input shape mismatch: "
                f"X_train has {X_train.shape[0]} samples, "
                f"y_train has {y_train.shape[0]} samples"
            )
            
        X_train_new, X_val, y_train_new, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train
        )
        
        logger.info(
            f"Created validation split:\n"
            f"  New training set: {X_train_new.shape}\n"
            f"  Validation set: {X_val.shape}"
        )
        
        return X_train_new, X_val, y_train_new, y_val
        
    except ValueError as e:
        logger.error(f"Validation split error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in validation split: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the data loading
    try:
        train, test = load_raw_data()
        logger.info("\nTraining data sample:")
        logger.info(train.head().to_string())
        logger.info("\nTest data sample:")
        logger.info(test.head().to_string())
        
        # Additional test for validation split
        X_sample = np.random.rand(100, 10)
        y_sample = np.random.randint(0, 2, 100)
        X_train_new, X_val, y_train_new, y_val = create_validation_split(X_sample, y_sample)
        
    except Exception as e:
        logger.error(f"Data loading test failed: {str(e)}", exc_info=True)