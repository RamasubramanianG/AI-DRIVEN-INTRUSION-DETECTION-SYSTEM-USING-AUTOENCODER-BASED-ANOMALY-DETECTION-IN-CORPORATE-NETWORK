import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from src.config import *
from src.utils import load_keras_model, create_directories
import os

def evaluate_model():
    """Evaluate model performance with comprehensive metrics"""
    try:
        create_directories()
        
        # Load test data
        X_test = np.load(PROCESSED_DIR / "X_test.npy")
        y_test = np.load(PROCESSED_DIR / "y_test.npy")
        
        # Load model
        autoencoder = load_keras_model(MODEL_PATH)
        print("\n✅ Model loaded successfully")
        
        # Get reconstruction errors
        print("⏳ Computing reconstructions...")
        reconstructions = autoencoder.predict(X_test, verbose=1)
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
        
        # Use separate validation set for threshold
        X_val = np.load(PROCESSED_DIR / "X_val.npy")
        val_reconstructions = autoencoder.predict(X_val, verbose=0)
        val_mse = np.mean(np.power(X_val - val_reconstructions, 2), axis=1)
        threshold = np.percentile(val_mse, ANOMALY_THRESHOLD_PERCENTILE)
        
        # Adjust predictions based on threshold
        y_pred = (mse > threshold).astype(int)
        
        # Only proceed if we have both classes
        if len(np.unique(y_test)) > 1:
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Print comprehensive metrics
            print("\n📊 Performance Metrics:")
            print(f"🔵 Accuracy: {accuracy:.4f}")
            print(f"🔵 Precision: {precision:.4f}")
            print(f"🔵 Recall: {recall:.4f}")
            print(f"🔵 F1 Score: {f1:.4f}")
            
            # Detailed classification report
            print("\n📊 Detailed Classification Report:")
            print(classification_report(y_test, y_pred, 
                                      target_names=["Normal", "Attack"]))
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, mse)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall
            precision_pr, recall_pr, _ = precision_recall_curve(y_test, mse)
            avg_precision = average_precision_score(y_test, mse)
            
            print(f"\n🔵 ROC AUC: {roc_auc:.4f}")
            print(f"🔵 Average Precision: {avg_precision:.4f}")
            
            # Plot ROC Curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(OUTPUTS_DIR / 'roc_curve.png')
            plt.close()
            
            # Plot Precision-Recall Curve
            plt.figure(figsize=(8, 6))
            plt.plot(recall_pr, precision_pr, color='blue', lw=2, 
                    label=f'Precision-Recall (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="upper right")
            plt.savefig(OUTPUTS_DIR / 'precision_recall_curve.png')
            plt.close()
            
        else:
            print("\n⚠️ Only one class present in y_test - cannot compute classification metrics")
        
        print(f"\n🔵 Anomaly Threshold (percentile {ANOMALY_THRESHOLD_PERCENTILE}): {threshold:.6f}")
        print(f"🔵 MSE Range: [{mse.min():.6f}, {mse.max():.6f}]")
        
        # Plot histogram of MSE scores
        plt.figure(figsize=(10, 6))
        plt.hist(mse[y_test == 0], bins=50, alpha=0.5, label='Normal')
        plt.hist(mse[y_test == 1], bins=50, alpha=0.5, label='Attack')
        plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label='Threshold')
        plt.xlabel('Reconstruction Error (MSE)')
        plt.ylabel('Count')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.savefig(OUTPUTS_DIR / 'error_distribution.png')
        plt.close()

        # Confusion Matrix with better visualization
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=["Normal", "Attack"], 
                   yticklabels=["Normal", "Attack"],
                   annot_kws={"size": 16})
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('True', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)
        plt.savefig(OUTPUTS_DIR / 'confusion_matrix.png')
        plt.close()

        # Return all important metrics and data
        return {
            'threshold': threshold,
            'mse_scores': mse,
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': accuracy if len(np.unique(y_test)) > 1 else None,
            'precision': precision if len(np.unique(y_test)) > 1 else None,
            'recall': recall if len(np.unique(y_test)) > 1 else None,
            'f1_score': f1 if len(np.unique(y_test)) > 1 else None,
            'roc_auc': roc_auc if len(np.unique(y_test)) > 1 else None,
            'avg_precision': avg_precision if len(np.unique(y_test)) > 1 else None
        }
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        raise