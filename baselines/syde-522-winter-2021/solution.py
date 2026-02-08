import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'n_folds': 5,
    'target_col': 'label',
    'lgb_params': {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
}

def run_pipeline():
    print(">>> [1/4] Loading data...")
    try:
        train_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return

    # --- PREPROCESSING ---
    print(">>> [2/4] Preprocessing...")

    X = train_df.drop(columns=[CONFIG['target_col']])
    y_raw = train_df[CONFIG['target_col']]

    # Normalize (0-255 -> 0-1)
    X = X / 255.0

    # Label Encoding (Fixes the 1-9 vs 0-8 issue)
    # y becomes a numpy array here!
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    num_classes = len(le.classes_)
    CONFIG['lgb_params']['num_class'] = num_classes

    print(f"   Features: {X.shape[1]}")
    print(f"   Classes:  {num_classes}")

    # --- ROBUST VALIDATION (HOLDOUT) ---
    print("\n>>> [3/4] Data Split (Holdout)...")
    
    # X is a DataFrame, y is a Numpy Array
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG['seed'], stratify=y
    )
    
    print(f"   Train (Dev): {len(X_dev)} rows")
    print(f"   Holdout:     {len(X_holdout)} rows")

    # --- TRAINING (K-FOLD) ---
    print(f"\n>>> [4/4] Training LightGBM ({CONFIG['n_folds']} Folds)...")
    
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
        # FIX IS HERE:
        # X_dev is a DataFrame -> use .iloc[idx]
        # y_dev is a Numpy Array -> use [idx] (no .iloc)
        X_tr, y_tr = X_dev.iloc[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev.iloc[val_idx], y_dev[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            CONFIG['lgb_params'],
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0)
            ]
        )
        models.append(model)
        
        # Check Fold Accuracy
        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        val_preds = np.argmax(val_probs, axis=1)
        acc = accuracy_score(y_val, val_preds)
        print(f"   Fold {fold+1}: Accuracy = {acc:.4f}")

    # --- FINAL HOLDOUT EVALUATION ---
    print("\n" + "="*40)
    print("FINAL HOLDOUT EVALUATION")
    print("="*40)
    
    holdout_probs = np.zeros((len(X_holdout), num_classes))
    
    for model in models:
        holdout_probs += model.predict(X_holdout, num_iteration=model.best_iteration)
    
    holdout_probs /= CONFIG['n_folds']
    holdout_preds_idx = np.argmax(holdout_probs, axis=1)
    
    final_acc = accuracy_score(y_holdout, holdout_preds_idx)
    
    print(f"Ensemble Accuracy on Holdout: {final_acc:.4f}")
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()