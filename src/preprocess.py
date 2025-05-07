import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

class RobustLabelEncoder(BaseEstimator, TransformerMixin):
    """Custom label encoder that handles unseen categories"""
    def __init__(self):
        self.classes_ = None
        self.unseen_label = -1

    def fit(self, y):
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        return np.where(np.isin(y, self.classes_), 
                       np.searchsorted(self.classes_, y), 
                       self.unseen_label)

def debug_data_info(df, name=""):
    """Print detailed dataframe information"""
    print(f"\n=== {name} Data Info ===")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)

    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    if cat_cols:
        print("\nCategorical Columns:")
        for col in cat_cols:
            uniq = df[col].unique()
            print(f"- {col}: {len(uniq)} unique values")
            if len(uniq) <= 10:
                print(f"  Values: {uniq}")

def validate_dataframes(df, required_columns):
    """Validate dataframe structure"""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Data missing columns: {missing}")

def preprocess_data():
    """Complete preprocessing pipeline for NSL-KDD dataset"""
    # Configuration (replace with your actual paths/constants)
    CONFIG = {
        'TRAIN_DATA': 'train.txt',
        'TEST_DATA': 'test.txt',
        'COLUMNS': ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                   'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                   'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                   'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                   'num_access_files', 'num_outbound_cmds', 'is_host_login',
                   'is_guest_login', 'count', 'srv_count', 'serror_rate',
                   'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                   'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                   'dst_host_srv_count', 'dst_host_same_srv_rate',
                   'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                   'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                   'dst_host_srv_rerror_rate', 'label'],
        'CATEGORICAL_COLUMNS': ['protocol_type', 'service', 'flag'],
        'MODELS_DIR': Path('models'),
        'NORMAL_LABEL': 0,
        'ATTACK_LABEL': 1
    }

    # Load raw data
    df_train = pd.read_csv(CONFIG['TRAIN_DATA'], names=CONFIG['COLUMNS'], header=None)
    df_test = pd.read_csv(CONFIG['TEST_DATA'], names=CONFIG['COLUMNS'], header=None)
    df = pd.concat([df_train, df_test], ignore_index=True)

    debug_data_info(df, "Combined Raw")
    validate_dataframes(df, CONFIG['COLUMNS'])

    # Binary labels conversion
    df["label"] = df["label"].apply(
        lambda x: CONFIG['NORMAL_LABEL'] if x.strip() == "normal" else CONFIG['ATTACK_LABEL']
    )

    # Check label distribution
    label_counts = df["label"].value_counts()
    print("\n=== Label Distribution in Full Dataset ===")
    print(label_counts)

    if set(label_counts.index) != {CONFIG['NORMAL_LABEL'], CONFIG['ATTACK_LABEL']}:
        raise ValueError(
            f"Dataset missing required classes. Expected [0, 1], got {label_counts.index.tolist()}"
        )

    # Separate features and labels
    y = df["label"]
    X = df.drop(columns=["label"])

    # Convert numeric columns
    numeric_cols = [col for col in X.columns if col not in CONFIG['CATEGORICAL_COLUMNS']]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # Encode categorical columns
    encoders = {}
    actual_categoricals = [col for col in X.columns if X[col].dtype == 'object']
    all_categoricals = list(set(CONFIG['CATEGORICAL_COLUMNS']) | set(actual_categoricals))

    print("\n=== Categorical Encoding ===")
    for col in all_categoricals:
        if col not in X.columns:
            continue
        encoder = RobustLabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        encoders[col] = encoder
        print(f"Encoded {col} ({len(encoder.classes_)} categories)")

    # Convert all to float
    X = X.astype(float)

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save artifacts
    CONFIG['MODELS_DIR'].mkdir(exist_ok=True)
    joblib.dump(encoders, CONFIG['MODELS_DIR']/'encoders.joblib')
    joblib.dump(scaler, CONFIG['MODELS_DIR']/'scaler.joblib')
    joblib.dump(list(X.columns), CONFIG['MODELS_DIR']/'feature_names.joblib')

    print("\n=== Preprocessing Complete ===")
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print("Test label distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values