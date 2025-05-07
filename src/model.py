from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.config import (
    INPUT_DIM,
    ENCODING_DIM,
    HIDDEN_DIMS,
    LEARNING_RATE,
    DROPOUT_RATE,
    L1_REG,
    L2_REG,
    MODELS_DIR,
    SEED
)
import os
import numpy as np
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

def build_autoencoder():
    """
    Build and compile a robust autoencoder model with enhanced architecture
    
    Returns:
        A compiled Keras Model instance with encoder/decoder architecture
    """
    # ======================
    #  Encoder Architecture
    # ======================
    input_layer = Input(shape=(INPUT_DIM,), name='input')
    
    # Build encoder layers dynamically from config
    x = input_layer
    for i, units in enumerate(HIDDEN_DIMS):
        x = Dense(
            units,
            activation='relu',
            kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG),
            name=f'encoder_dense_{i}'
        )(x)
        x = BatchNormalization(name=f'encoder_bn_{i}')(x)
        x = Dropout(DROPOUT_RATE, seed=SEED, name=f'encoder_dropout_{i}')(x)
    
    # Bottleneck layer
    encoder_output = Dense(
        ENCODING_DIM,
        activation='relu',
        kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG),
        name='bottleneck'
    )(x)
    
    # ======================
    #  Decoder Architecture 
    # ======================
    x = encoder_output
    for i, units in enumerate(reversed(HIDDEN_DIMS)):
        x = Dense(
            units,
            activation='relu',
            kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG),
            name=f'decoder_dense_{i}'
        )(x)
        x = BatchNormalization(name=f'decoder_bn_{i}')(x)
        x = Dropout(DROPOUT_RATE, seed=SEED, name=f'decoder_dropout_{i}')(x)
    
    # Output layer
    decoder_output = Dense(
        INPUT_DIM,
        activation='sigmoid',
        name='output'
    )(x)
    
    # ======================
    #  Model Definition
    # ======================
    autoencoder = Model(
        inputs=input_layer,
        outputs=decoder_output,
        name='autoencoder'
    )
    
    # Custom optimizer configuration
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False
    )
    
    # Compile model
    autoencoder.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Print model summary
    autoencoder.summary()
    
    return autoencoder

def get_callbacks():
    """Create list of callbacks for model training"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

if __name__ == "__main__":
    # Test model building
    model = build_autoencoder()
    print("\n✅ Autoencoder built successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")