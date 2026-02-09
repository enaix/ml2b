import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'target_col': 'col_5',
    'lgb_params': {
        'objective': 'regression',
        'metric': 'mse',  # Mean Squared Error as requested
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
    
    # Drop 'id' as it is just an identifier
    if 'id' in train_df.columns:
        train_df = train_df.drop(columns=['id'])

    # Prepare Features and Target
    X = train_df.drop(columns=[CONFIG['target_col']])
    y = train_df[CONFIG['target_col']]

    print(f"   Features: {X.shape[1]}")
    print(f"   Total Samples: {len(X)}")

    # --- ROBUST VALIDATION (HOLDOUT) ---
    print("\n>>> [3/4] Data Split (Holdout)...")
    
    # Simple Shuffle Split (80% Train, 20% Holdout)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG['seed'], shuffle=True
    )
    
    print(f"   Train Rows: {len(X_train)}")
    print(f"   Holdout Rows: {len(X_holdout)}")

    # --- TRAINING ---
    print("\n>>> [4/4] Training LightGBM...")
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_holdout, label=y_holdout, reference=dtrain)
    
    model = lgb.train(
        CONFIG['lgb_params'],
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(0) # Silent mode
        ]
    )
    
    # --- EVALUATION ---
    print("\n" + "="*40)
    print("FINAL HOLDOUT EVALUATION")
    print("="*40)
    
    # Predict on Holdout
    holdout_preds = model.predict(X_holdout, num_iteration=model.best_iteration)
    
    # Calculate MSE
    mse = mean_squared_error(y_holdout, holdout_preds)
    
    print(f"Validation MSE: {mse:.5f}")
    
    # Feature Importance
    print("\nTop Features:")
    lgb.plot_importance(model, max_num_features=5, importance_type='gain', figsize=(10, 4))
    
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()