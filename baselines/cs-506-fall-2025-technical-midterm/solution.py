import pandas as pd
import numpy as np
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'holdout_size': 0.2,       # 20% for Holdout Validation
    'n_folds': 5,              # 5 Folds for stability check
    'target_col': 'Score',
    'id_col': 'id',
    'text_cols': ['summary', 'reviewText'],
    # Logistic Regression Parameters
    'model_params': {
        'solver': 'saga',          # Efficient for large datasets
        'C': 1.0,                  # Regularization strength
        'max_iter': 1000,
        'class_weight': 'balanced',# Essential for F1-Macro optimization
        'n_jobs': -1,
        'random_state': 42
    },
    'tfidf_params': {
        'max_features': 50000,     # Limit vocabulary size
        'ngram_range': (1, 2),     # Unigrams + Bigrams
        'stop_words': 'english',
        'sublinear_tf': True
    }
}

def feature_engineering(df):
    """
    Classical feature engineering.
    """
    df = df.copy()
    
    # 1. Text Combination
    for col in CONFIG['text_cols']:
        df[col] = df[col].fillna('')
        
    df['combined_text'] = df['summary'] + " " + df['reviewText']
    
    # 2. Helpfulness Ratio
    df['helpfulness_ratio'] = df['VotedHelpful'] / (df['TotalVotes'] + 1)
    
    # 3. Temporal Features
    df['date'] = pd.to_datetime(df['unixReviewTime'], unit='s')
    df['review_year'] = df['date'].dt.year.astype(float).fillna(0)
    
    return df

def run_pipeline():
    print(">>> [1/6] Loading data...")
    try:
        raw_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return

    # --- PREPROCESSING & SEPARATION ---
    print(">>> [2/6] Processing & Separating Labeled/Unlabeled...")
    
    processed_df = feature_engineering(raw_df)
    
    # Separate Labeled (Train) and Unlabeled (Competition Test)
    df_labeled = processed_df[processed_df[CONFIG['target_col']].notna()].copy()
    df_unlabeled = processed_df[processed_df[CONFIG['target_col']].isna()].copy()
    
    # Ensure Target is Integer
    df_labeled[CONFIG['target_col']] = df_labeled[CONFIG['target_col']].astype(int)
    
    print(f"   Labeled Data:   {len(df_labeled)}")
    print(f"   Unlabeled Data: {len(df_unlabeled)}")

    # --- PIPELINE SETUP ---
    # ColumnTransformer: Text -> TF-IDF, Numerics -> Scaler
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(**CONFIG['tfidf_params']), 'combined_text'),
            ('num', StandardScaler(), ['helpfulness_ratio', 'review_year'])
        ]
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(**CONFIG['model_params']))
    ])

    # --- SPLIT FOR INTERNAL VALIDATION ---
    print(f"\n>>> [3/6] Splitting Labeled Data (Holdout {CONFIG['holdout_size']*100}%)...")
    
    X = df_labeled
    y = df_labeled[CONFIG['target_col']]
    
    # Split Labeled Data into Train_Split and Holdout_Split
    X_train_split, X_holdout_split, y_train_split, y_holdout_split = train_test_split(
        X, y, test_size=CONFIG['holdout_size'], stratify=y, random_state=CONFIG['seed']
    )
    
    print(f"   Internal Train:   {len(X_train_split)}")
    print(f"   Internal Holdout: {len(X_holdout_split)}")

    # --- FOLD CROSS-VALIDATION (ON TRAIN SPLIT) ---
    print("\n>>> [4/6] Running CV on Internal Train (Stability Check)...")
    
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    
    # Evaluate metric on Train Split using CV
    cv_scores = cross_val_score(
        pipeline, X_train_split, y_train_split, 
        cv=skf, scoring='f1_macro', n_jobs=-1
    )
    
    print(f"   CV F1-Macro Scores: {cv_scores}")
    print(f"   Average CV F1:      {np.mean(cv_scores):.5f}")

    # --- HOLDOUT EVALUATION ---
    print("\n>>> [5/6] Training on Internal Train -> Evaluating on Holdout...")
    
    # Fit on Train Split
    pipeline.fit(X_train_split, y_train_split)
    
    # Predict on Holdout Split
    holdout_preds = pipeline.predict(X_holdout_split)
    
    # Calculate Score
    holdout_f1 = f1_score(y_holdout_split, holdout_preds, average='macro')
    
    print("="*40)
    print(f"HOLDOUT F1-MACRO: {holdout_f1:.5f}")
    print("="*40)

    # --- FINAL RETRAINING & PREDICTION ---
    print("\n>>> [6/6] Retraining on ALL Labeled Data -> Predicting on Unlabeled...")
    
    # Fit on ALL Labeled Data (df_labeled)
    pipeline.fit(X, y)
    
    # Predict on Unlabeled Data
    final_preds = pipeline.predict(df_unlabeled)
    
    # Create Submission
    submission = pd.DataFrame({
        CONFIG['id_col']: df_unlabeled[CONFIG['id_col']],
        CONFIG['target_col']: final_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("   'submission.csv' saved successfully.")
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()