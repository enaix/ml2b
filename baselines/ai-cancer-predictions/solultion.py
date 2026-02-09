import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'n_folds': 5,
    'target_col': 'diagnosis',
    # The dataset is small, so we use conservative model parameters
    # to prevent overfitting
    'lgb_params': {
        'objective': 'binary',
        'metric': 'binary_logloss', # Optimize logloss
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 15,           # Low number of leaves for small dataset
        'max_depth': 4,             # Limit depth
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 10,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
}

def run_pipeline():
    print(">>> [1/4] Loading and cleaning data...")
    
    # Load training data only
    try:
        train = pd.read_csv('2021.AI.cancer-train.csv')
    except FileNotFoundError:
        # Fallback to standard name
        train = pd.read_csv('train.csv')

    # --- PREPROCESSING ---
    
    # 1. Remove 'Unnamed: 32' column if present (common artifact in this dataset)
    if 'Unnamed: 32' in train.columns:
        train = train.drop(columns=['Unnamed: 32'])

    # 2. Prepare X and y
    # Drop ID and target from features
    drop_cols = ['id', CONFIG['target_col']]
    features = [c for c in train.columns if c not in drop_cols]
    
    X = train[features]
    y = train[CONFIG['target_col']]

    print(f"   Train Size: {X.shape}")

    # --- ROBUST VALIDATION (HOLDOUT) ---
    print("\n>>> [2/4] Data Split (Holdout)...")
    # Reserve 20% for final check to detect overfitting
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG['seed'], stratify=y
    )
    
    print(f"   Train (Dev): {len(X_dev)} rows")
    print(f"   Holdout:     {len(X_holdout)} rows")

    # --- TRAINING (K-FOLD) ---
    print(f"\n>>> [3/4] Training LightGBM ({CONFIG['n_folds']} Folds)...")
    
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
        X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_val, y_val = X_dev.iloc[val_idx], y_dev.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            CONFIG['lgb_params'],
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0) # Silent mode
            ]
        )
        models.append(model)
        
        # Predict on validation
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Calculate fold metrics
        acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
        auc = roc_auc_score(y_val, val_preds)
        print(f"   Fold {fold+1}: Accuracy={acc:.4f}, AUC={auc:.4f}")

    # --- CHECK ON HOLDOUT ---
    print("\n>>> [4/4] Final check on Holdout...")
    
    holdout_preds = np.zeros(len(X_holdout))
    for model in models:
        holdout_preds += model.predict(X_holdout, num_iteration=model.best_iteration)
    holdout_preds /= CONFIG['n_folds']
    
    # Holdout Metrics
    final_acc = accuracy_score(y_holdout, (holdout_preds > 0.5).astype(int))
    final_auc = roc_auc_score(y_holdout, holdout_preds)
    
    print("="*40)
    print(f"HOLDOUT ACCURACY: {final_acc:.4f}")
    print(f"HOLDOUT AUC:      {final_auc:.4f}")
    print("="*40)
    print("✓ Pipeline finished. No submission generated.")

if __name__ == "__main__":
    run_pipeline()