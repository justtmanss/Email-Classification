import argparse
import pickle
import pandas as pd
import sys
import os
from sklearn.pipeline import Pipeline

def load_model(model_path):
    """Load the trained model from a pickle file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_single(model, text):
    """Predict the category for a single email text."""
    result = model.predict([text])[0]

    # Fallback for models like LinearSVC that don't support predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        classes = model.classes_
        probs = {cls: prob for cls, prob in zip(classes, proba)}
    else:
        probs = {result: 1.0}  # 100% confidence in predicted class

    return result, probs

def predict_from_file(model, file_path, output_path=None):
    """Predict categories for emails in a CSV file."""
    df = pd.read_csv(file_path)
    
    if 'emal' not in df.columns:
        raise ValueError("CSV file must contain an 'email' column with email content")
    
    df['predicted_category'] = model.predict(df['email'])
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:
        print(df[['email', 'predicted_category']])

def main():
    parser = argparse.ArgumentParser(description='Predict email categories using trained model')
    parser.add_argument('--model', default='trained_model.pkl', help='Path to trained model pickle file')
    
    subparsers = parser.add_subparsers(dest='command', help='Prediction mode')
    
    # Parser for single text prediction
    text_parser = subparsers.add_parser('text', help='Predict category for a single text')
    text_parser.add_argument('text', help='Email text to classify')
    
    # Parser for file-based prediction
    file_parser = subparsers.add_parser('file', help='Predict categories for emails in a file')
    file_parser.add_argument('input', help='Input CSV file with email texts')
    file_parser.add_argument('--output', help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Load the model
    try:
        model = load_model(args.model)
        print(f"Model loaded from {args.model}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    # Process based on command
    if args.command == 'text':
        result, probs = predict_single(model, args.text)
        print(f"Predicted Category: {result}")
        print("Class Probabilities:")
        for cls, prob in probs.items():
            print(f"  {cls}: {prob:.4f}")
    elif args.command == 'file':
        predict_from_file(model, args.input, args.output)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
