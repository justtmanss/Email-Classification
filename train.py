# train.py
import argparse
import os
from models import train_from_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train email classification model")
    parser.add_argument("--data", required=True, help="Path to CSV file with 'email' and 'type' columns")
    parser.add_argument("--model-type", default="svm", choices=["svm", "rf"], help="Model type (svm or rf)")
    parser.add_argument("--output", default="trained_model.pkl", help="Path to save trained model")
    
    args = parser.parse_args()
    
    print(f"Training model with {args.model_type} using data from {args.data}")
    metrics = train_from_csv(args.data, args.model_type, args.output)
    
    print(f"\nTraining completed! Model saved to {args.output}")
    print(f"Best accuracy: {metrics['accuracy']:.4f}")