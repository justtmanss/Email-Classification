# models.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, Tuple, Any, Optional

def create_and_train_model(df: pd.DataFrame, model_type: str = 'svm') -> Tuple[Pipeline, Dict[str, Any]]:
    """Create and train a text classification model.
    
    Args:
        df: DataFrame with 'cleaned_email' and 'type' columns
        model_type: Type of model to use ('svm' or 'rf')
        
    Returns:
        Tuple of (trained_pipeline, metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_email'], df['type'], test_size=0.2, random_state=42, stratify=df['type']
    )
    
    # Create pipeline with TF-IDF and classifier
    if model_type.lower() == 'svm':
        classifier = LinearSVC(random_state=42)
        param_grid = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.1, 1.0, 5.0]
        }
    else:  # Random Forest
        classifier = RandomForestClassifier(n_jobs=-1, random_state=42)
        param_grid = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20]
        }
    
    # Create the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True)),
        ('classifier', classifier)
    ])
    
    # Use Grid Search for hyperparameter tuning
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, n_jobs=-1, verbose=1
    )
    
    # Train the model
    print("Training the model...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    metrics = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'best_params': grid_search.best_params_,
        'accuracy': grid_search.best_score_
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_model, metrics

def save_model(model: Pipeline, file_path: str) -> None:
    """Save the trained model to disk."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def load_model(file_path: str) -> Optional[Pipeline]:
    """Load a trained model from disk."""
    if not os.path.exists(file_path):
        print(f"Warning: Model file not found at {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def train_from_csv(csv_path: str, model_type: str = 'svm', output_path: str = 'trained_model.pkl') -> Dict[str, Any]:
    """Train model from CSV file and save it.
    
    Args:
        csv_path: Path to CSV file with 'email' and 'type' columns
        model_type: Type of model ('svm' or 'rf')
        output_path: Path to save the trained model
        
    Returns:
        Dictionary with training metrics
    """
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded: {df.shape[0]} rows")
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV file: {str(e)}")
    
    # Check required columns
    if 'email' not in df.columns or 'type' not in df.columns:
        raise ValueError("CSV must contain 'email' and 'type' columns")
    
    # Preprocess data
    from utils import preprocess_data
    df_processed = preprocess_data(df)
    print(f"Data preprocessed: {df_processed.shape[0]} rows")
    
    # Print class distribution
    print("Class distribution:")
    print(df_processed['type'].value_counts())
    
    # Train model
    model, metrics = create_and_train_model(df_processed, model_type)
    
    # Save model
    save_model(model, output_path)
    
    return metrics