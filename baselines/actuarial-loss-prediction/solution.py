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
    'n_folds': 5,
    'target_col': 'UltimateIncurredClaimCost',
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
    print(">>> [2/4] Preprocessing & Feature Engineering...")
    
    # 1. Date Features & Reporting Delay
    # Actuarial Insight: Late reporting is a huge red flag for cost escalation.
    train_df['DateTimeOfAccident'] = pd.to_datetime(train_df['DateTimeOfAccident'])
    train_df['DateReported'] = pd.to_datetime(train_df['DateReported'])
    
    # Calculate delay in days
    train_df['ReportDelay'] = (train_df['DateReported'] - train_df['DateTimeOfAccident']).dt.days
    
    # Extract cyclical date features
    train_df['AccidentYear'] = train_df['DateTimeOfAccident'].dt.year
    train_df['AccidentMonth'] = train_df['DateTimeOfAccident'].dt.month
    train_df['AccidentHour'] = train_df['DateTimeOfAccident'].dt.hour
    
    # Drop original date columns
    train_df = train_df.drop(columns=['DateTimeOfAccident', 'DateReported'])

    # 2. Text Features (ClaimDescription)
    # Simple feature: Length of the description (More complex claims -> longer text?)
    train_df['DescriptionLength'] = train_df['ClaimDescription'].astype(str).apply(len)
    train_df = train_df.drop(columns=['ClaimDescription']) # Drop raw text for baseline

    # 3. Clean categorical columns (Gender, MaritalStatus, PartTimeFullTime)
    # Convert them to 'category' type for LightGBM
    cat_cols = ['Gender', 'MaritalStatus', 'PartTimeFullTime']
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')

    # 4. Handle ID column
    if 'ClaimNumber' in train_df.columns:
        train_df = train_df.drop(columns=['ClaimNumber'])

    # 5. Prepare X and y
    # Log-transform the Target because insurance claims are heavily right-skewed
    X = train_df.drop(columns=[CONFIG['target_col']])
    y_raw = train_df[CONFIG['target_col']]
    y_log = np.log1p(y_raw)
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Total Claims: {len(X)}")

    # --- ROBUST VALIDATION (HOLDOUT) ---
    print("\n>>> [3/4] Data Split (Holdout)...")
    
    # Random split is acceptable here as these are independent policies
    # Stratify is hard with regression, so we use shuffle
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y_log, test_size=0.2, random_state=CONFIG['seed'], shuffle=True
    )
    
    print(f"   Train (Dev): {len(X_dev)} rows")
    print(f"   Holdout:     {len(X_holdout)} rows")

    # --- TRAINING ---
    print(f"\n>>> [4/4] Training LightGBM on Log-Target...")
    
    # Note: We are training on log-transformed target
    dtrain = lgb.Dataset(X_dev, label=y_dev, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_holdout, label=y_holdout, categorical_feature=cat_cols, reference=dtrain)
    
    model = lgb.train(
        CONFIG['lgb_params'],
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=0) # Silent mode
        ]
    )
    
    # --- EVALUATION ---
    print("\n" + "="*40)
    print("FINAL HOLDOUT EVALUATION")
    print("="*40)
    
    # 1. Predict (Log scale)
    log_preds = model.predict(X_holdout, num_iteration=model.best_iteration)
    
    # 2. Inverse Transform (Exponential) to get actual currency amounts
    # expm1 is the inverse of log1p
    final_preds = np.expm1(log_preds)
    
    # 3. Get original ground truth
    y_holdout_original = np.expm1(y_holdout)
    
    # 4. Calculate RMSE
    mse = mean_squared_error(y_holdout_original, final_preds)
    rmse = np.sqrt(mse)
    
    print(f"Validation RMSE: {rmse:.5f}")
    
    # Feature Importance: See what drives the cost
    print("\nTop 5 Important Drivers of Cost:")
    lgb.plot_importance(model, max_num_features=5, importance_type='gain', figsize=(10, 4))
    
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()