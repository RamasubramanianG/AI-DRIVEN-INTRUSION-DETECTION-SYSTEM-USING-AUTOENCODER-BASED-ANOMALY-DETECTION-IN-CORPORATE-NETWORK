import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import time

from src.model import build_autoencoder
from src.preprocess import preprocess_data
from src.data_loader import save_processed_data
from src.config import *
from src.utils import save_keras_model, create_directories

def train_model():
    """Train the autoencoder model with enhanced features"""
    
    # Create directories if they don't exist
    create_directories()
    
    # Preprocess data
    print("⏳ Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Split training data into train and validation sets (features AND labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, 
        y_train,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        shuffle=True
    )
    
    # Save processed data (including validation data)
    print("💾 Saving processed data...")
    save_processed_data(X_train, X_test, y_train, y_test, X_val, y_val)
    
    # Build model
    print("🛠️ Building autoencoder model...")
    autoencoder = build_autoencoder()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            patience=10,
            monitor='val_loss',
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]
    
    # Start training
    start_time = datetime.now()
    print(f"\n🚀 Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    history = autoencoder.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # End of training
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"\n🏁 Training completed in {training_duration}")
    
    # Save the final model
    print("\n💾 Saving final model...")
    save_keras_model(autoencoder, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    
    # Save training history
    history_path = os.path.join(OUTPUTS_DIR, 'training_history.npy')
    np.save(history_path, history.history)
    print(f"✅ Training history saved to {history_path}")
    
    return history, autoencoder

if __name__ == "__main__":
    train_model()
