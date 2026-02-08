import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'text_col': 'dialogue',
    'target_col': 'genres',
    'tfidf_params': {
        'max_features': 25000,     # Increased vocabulary size slightly
        'stop_words': 'english',   # Remove common English stop words
        'ngram_range': (1, 2),     # Use unigrams and bigrams
        'min_df': 2                # Ignore extremely rare words
    },
    'model_params': {
        'solver': 'liblinear',     # Good for sparse datasets
        'C': 1.0,                  # Regularization strength
        'random_state': 42,
        'class_weight': 'balanced' # Important: helps with rare genres
    }
}

def parse_genres(x):
    """
    Parses a string like "[u'drama', u'romance']" into a python list ['drama', 'romance'].
    """
    try:
        # ast.literal_eval safely evaluates a string containing a Python literal
        return ast.literal_eval(x)
    except:
        # Fallback in case of parsing error
        return []

def run_pipeline():
    print(">>> [1/5] Loading data...")
    try:
        train_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return

    # --- PREPROCESSING ---
    print(">>> [2/5] Processing Labels (Correct Parsing)...")
    
    # 1. Parse genres correctly using AST
    # Fill NaNs with empty list string "[]" before parsing
    train_df[CONFIG['target_col']] = train_df[CONFIG['target_col']].fillna("[]").apply(parse_genres)
    
    # 2. Binarize Labels (One-Hot for Multi-Label)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(train_df[CONFIG['target_col']])
    
    classes = mlb.classes_
    print(f"   Unique Genres ({len(classes)}): {classes}")
    # Expected: 20 genres like 'action', 'adventure', 'comedy', etc.

    # --- TEXT VECTORIZATION ---
    print(">>> [3/5] Vectorizing Text (TF-IDF)...")
    
    tfidf = TfidfVectorizer(**CONFIG['tfidf_params'])
    # Fill NaNs in text column to avoid errors
    X = tfidf.fit_transform(train_df[CONFIG['text_col']].fillna(""))
    
    print(f"   Vocabulary Size: {X.shape[1]}")
    print(f"   Training Samples: {X.shape[0]}")

    # --- ROBUST VALIDATION (HOLDOUT) ---
    print("\n>>> [4/5] Data Split (Holdout)...")
    
    # Random split is standard for multi-label baselines
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG['seed']
    )
    
    print(f"   Train Rows: {X_train.shape[0]}")
    print(f"   Val Rows:   {X_val.shape[0]}")

    # --- TRAINING ---
    print("\n>>> [5/5] Training One-vs-Rest Logistic Regression...")
    
    # We train 20 independent binary classifiers (one per genre)
    clf = OneVsRestClassifier(LogisticRegression(**CONFIG['model_params']))
    clf.fit(X_train, y_train)
    
    # --- EVALUATION ---
    print("\n" + "="*40)
    print("FINAL HOLDOUT EVALUATION")
    print("="*40)
    
    # Predict binary labels (0 or 1)
    val_preds = clf.predict(X_val)
    
    # Metric F1-Score (average='samples' is required by the competition)
    # This calculates F1 for each instance and averages them
    score = f1_score(y_val, val_preds, average='samples')
    
    print(f"Validation Mean F1-Score: {score:.4f}")
    
    print("\n--- Example Predictions ---")
    # Inverse transform to get human-readable labels
    pred_labels = mlb.inverse_transform(val_preds[:3])
    true_labels = mlb.inverse_transform(y_val[:3])
    
    for i in range(3):
        print(f"True: {true_labels[i]}")
        print(f"Pred: {pred_labels[i]}")
        print("-")

    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()