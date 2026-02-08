import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'n_folds': 5,
    'target_col': 'loss',
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
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
    
    # Drop ID column if present (it carries no predictive info)
    if 'id' in train_df.columns:
        train_df = train_df.drop(columns=['id'])

    # Separate Features and Target
    X = train_df.drop(columns=[CONFIG['target_col']])
    y = train_df[CONFIG['target_col']]

    print(f"   Features: {X.shape[1]}")
    print(f"   Total Samples: {len(X)}")

    # --- ROBUST VALIDATION (HOLDOUT) ---
    print("\n>>> [3/4] Data Split (Holdout)...")
    
    # Standard Shuffle Split is fine here (not time-series)
    # 80% for Training/CV, 20% for Final Holdout Check
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG['seed'], shuffle=True
    )
    
    print(f"   Train (Dev): {len(X_dev)} rows")
    print(f"   Holdout:     {len(X_holdout)} rows")

    # --- TRAINING (K-FOLD on Dev set) ---
    print(f"\n>>> [4/4] Training LightGBM ({CONFIG['n_folds']} Folds)...")
    
    kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev, y_dev)):
        # Split Dev into Train/Val for this fold
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
        
        # Check Fold Metric (RMSE)
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Manual RMSE calculation (compatible with all sklearn versions)
        mse = mean_squared_error(y_val, val_preds)
        rmse = np.sqrt(mse)
        
        print(f"   Fold {fold+1}: RMSE = {rmse:.5f}")

    # --- FINAL HOLDOUT EVALUATION ---
    print("\n" + "="*40)
    print("FINAL HOLDOUT EVALUATION")
    print("="*40)
    
    # Ensemble prediction on Holdout
    holdout_preds = np.zeros(len(X_holdout))
    
    for model in models:
        holdout_preds += model.predict(X_holdout, num_iteration=model.best_iteration)
    
    holdout_preds /= CONFIG['n_folds']
    
    # Final Metric
    final_mse = mean_squared_error(y_holdout, holdout_preds)
    final_rmse = np.sqrt(final_mse)
    
    print(f"Ensemble RMSE on Holdout: {final_rmse:.5f}")
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()