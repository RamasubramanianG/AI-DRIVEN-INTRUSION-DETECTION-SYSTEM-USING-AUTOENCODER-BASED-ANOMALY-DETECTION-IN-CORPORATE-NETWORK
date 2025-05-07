from pathlib import Path
import os

# ====================== #
#   Directory Structure  #
# ====================== #
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "NSL_KDD"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ====================== #
#       Data Files       #
# ====================== #
TRAIN_DATA = DATA_DIR / "KDDTrain+.txt"
TEST_DATA = DATA_DIR / "KDDTest+.txt"
VAL_DATA = PROCESSED_DIR / "X_val.npy"  # Validation data path

# ====================== #
#    Model Artifacts     #
# ====================== #
MODEL_PATH = MODELS_DIR / "autoencoder_model.h5"
BEST_MODEL_PATH = MODELS_DIR / "best_model.h5"  # For model checkpointing
HISTORY_PATH = OUTPUTS_DIR / "training_history.npy"

# Preprocessing artifacts
ENCODERS_PATH = MODELS_DIR / "encoders.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"

# ====================== #
#    Dataset Columns     #
# ====================== #
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# ====================== #
#  Feature Configuration #
# ====================== #
# Categorical features
CATEGORICAL_COLUMNS = [
    'protocol_type',  # tcp, udp, icmp
    'service',       # http, ftp, smtp, etc.
    'flag',          # SF, S0, S1, etc.
    'land',          # Binary (0/1 but categorical)
    'logged_in',     # Binary (0/1 but categorical)
    'is_host_login', # Binary (0/1 but categorical)
    'is_guest_login' # Binary (0/1 but categorical)
]

# Features to potentially drop (adjust based on your needs)
DROP_FEATURES = [
    'num_outbound_cmds'  # Typically has zero variance
]

# ====================== #
#    Label Configuration #
# ====================== #
NORMAL_LABEL = 0
ATTACK_LABEL = 1

# ====================== #
#    Model Configuration #
# ====================== #
# Autoencoder architecture
INPUT_DIM = 41  # Number of features after preprocessing
ENCODING_DIM = 14  # Size of encoded representation
HIDDEN_DIMS = [64, 32]  # Hidden layer dimensions

# Training parameters
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
PATIENCE = 10  # For early stopping

# Optimization
LEARNING_RATE = 0.001
LR_PATIENCE = 3  # For ReduceLROnPlateau
LR_FACTOR = 0.5  # For ReduceLROnPlateau

# Anomaly detection
ANOMALY_THRESHOLD_PERCENTILE = 95  # For anomaly detection
RECONSTRUCTION_ERROR_METRIC = 'mse'  # 'mse' or 'mae'

# ====================== #
#    Runtime Settings    #
# ====================== #
SEED = 42  # For reproducibility
VERBOSE = 1  # 0 = silent, 1 = progress bar, 2 = one line per epoch
# Add these to your existing config
DROPOUT_RATE = 0.2  # Dropout rate for regularization
L1_REG = 0.001  # L1 regularization strength
L2_REG = 0.001  # L2 regularization strength
HIDDEN_DIMS = [64, 32]  # Hidden layer dimensions