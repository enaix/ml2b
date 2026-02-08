import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'n_folds': 5,
    'target_col': 'y',          # Porto Seguro target is 'y'
    'id_col': 'id',             # ID column
    'lgb_params': {
        'objective': 'binary',
        'metric': 'auc',        # Optimize AUC during training
        'is_unbalance': True,   # Critical for rare events (marketing conversion)
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

def optimize_threshold(y_true, y_probs):
    """
    Finds the best threshold to maximize F1-Score.
    For imbalanced data, the optimal threshold is rarely 0.5.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    
    # Check thresholds from 0.1 to 0.9
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = t
            
    return best_threshold, best_f1

def run_pipeline():
    print(">>> [1/5] Loading data...")
    try:
        train_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return

    # --- PREPROCESSING ---
    print(">>> [2/5] Preprocessing...")
    
    # 1. Identify Categorical Columns
    # In this dataset, columns are anonymized (var1, var2...). 
    # We treat 'object' columns as categorical.
    cat_cols = []
    for col in train_df.columns:
        if col not in [CONFIG['id_col'], CONFIG['target_col']]:
            if train_df[col].dtype == 'object':
                train_df[col] = train_df[col].astype('category')
                cat_cols.append(col)
    
    print(f"   Categorical Features identified: {len(cat_cols)}")

    # Prepare X and y
    X = train_df.drop(columns=[CONFIG['id_col'], CONFIG['target_col']])
    y = train_df[CONFIG['target_col']]

    print(f"   Features: {X.shape[1]}")
    print(f"   Class Balance: {y.value_counts(normalize=True).to_dict()}")

    # --- ROBUST VALIDATION (Stratified K-Fold) ---
    print("\n>>> [3/5] Data Split (Stratified K-Fold)...")
    
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    
    # Store Out-Of-Fold probabilities to find global best threshold
    oof_probs = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)
        
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
        
        # Predict Probabilities
        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        oof_probs[val_idx] = val_probs
        
        print(f"   Fold {fold+1} processed.")

    # --- THRESHOLD OPTIMIZATION ---
    print("\n>>> [4/5] Optimizing F1 Threshold...")
    
    # Find the probability threshold that gives the highest F1
    best_t, best_f1 = optimize_threshold(y, oof_probs)
    
    print(f"   Best Threshold: {best_t:.2f}")
    print(f"   Best F1-Score:  {best_f1:.5f}")

    # --- FINAL METRICS ---
    print("\n" + "="*40)
    print("FINAL EVALUATION (OOF)")
    print("="*40)
    
    # Convert probabilities to classes using the optimal threshold
    final_preds = (oof_probs >= best_t).astype(int)
    
    precision = precision_score(y, final_preds)
    recall = recall_score(y, final_preds)
    
    print(f"F1-Score:  {best_f1:.5f}")
    print(f"Recall:    {recall:.5f}")
    print(f"Precision: {precision:.5f}")
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()