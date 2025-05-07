import argparse
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="NSL-KDD Anomaly Detection")
    parser.add_argument('--mode', choices=['train', 'evaluate', 'full'], 
                       default='full', help="Run mode")
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'full']:
        print("🚀 Training model...")
        train_model()
    
    if args.mode in ['evaluate', 'full']:
        print("🧪 Evaluating model...")
        evaluate_model()

if __name__ == "__main__":
    main()