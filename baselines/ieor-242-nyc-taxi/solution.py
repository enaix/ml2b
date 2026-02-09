import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'n_folds': 5,
    'target_col': 'duration',
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
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
    
    # 1. Cleaning Outliers (Crucial for Taxi Data)
    # Filter extremely short (< 1 min) or extremely long (> 3 hours) trips for training
    # This prevents the model from learning noise.
    original_len = len(train_df)
    train_df = train_df[
        (train_df['duration'] > 60) & 
        (train_df['duration'] < 3 * 3600)
    ]
    print(f"   Removed {original_len - len(train_df)} outlier rows.")
    
    # 2. Date-Time Features
    train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
    train_df['hour'] = train_df['pickup_datetime'].dt.hour
    train_df['day_of_week'] = train_df['pickup_datetime'].dt.dayofweek
    train_df['month'] = train_df['pickup_datetime'].dt.month
    
    # 3. Handle Categorical Features
    # 'VendorID', 'pickup_borough', 'dropoff_borough' etc. are strings.
    # We convert them to 'category' for LightGBM.
    cat_cols = ['VendorID', 'pickup_borough', 'dropoff_borough', 
                'pickup_zone', 'dropoff_zone']
    
    # Ensure they are strings first (sometimes IDs are read as ints)
    for col in cat_cols:
        train_df[col] = train_df[col].astype(str).astype('category')
        
    # 4. Prepare X and y
    # Log-transform target: Data is skewed, log makes it normal-like.
    X = train_df.drop(columns=['row_id', 'duration', 'pickup_datetime'])
    y_log = np.log1p(train_df['duration'])
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Total Samples: {len(X)}")

    # --- ROBUST VALIDATION (HOLDOUT) ---
    print("\n>>> [3/4] Data Split (Holdout)...")
    
    # Standard Shuffle Split (80/20)
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y_log, test_size=0.2, random_state=CONFIG['seed'], shuffle=True
    )
    
    print(f"   Train (Dev): {len(X_dev)} rows")
    print(f"   Holdout:     {len(X_holdout)} rows")

    # --- TRAINING ---
    print(f"\n>>> [4/4] Training LightGBM (Log-Target)...")
    
    dtrain = lgb.Dataset(X_dev, label=y_dev, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_holdout, label=y_holdout, categorical_feature=cat_cols, reference=dtrain)
    
    model = lgb.train(
        CONFIG['lgb_params'],
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(0) # Silent mode
        ]
    )
    
    # --- EVALUATION ---
    print("\n" + "="*40)
    print("FINAL HOLDOUT EVALUATION")
    print("="*40)
    
    # 1. Predict (Log scale)
    log_preds = model.predict(X_holdout, num_iteration=model.best_iteration)
    
    # 2. Inverse Transform (exp) to get seconds
    final_preds = np.expm1(log_preds)
    
    # 3. Get original ground truth (seconds)
    y_holdout_original = np.expm1(y_holdout)
    
    # 4. Calculate RMSE
    mse = mean_squared_error(y_holdout_original, final_preds)
    rmse = np.sqrt(mse)
    
    print(f"Validation RMSE: {rmse:.5f} seconds")
    
    # Feature Importance
    print("\nTop 5 Drivers of Duration:")
    lgb.plot_importance(model, max_num_features=5, importance_type='gain', figsize=(10, 4))
    
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()