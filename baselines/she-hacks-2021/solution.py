import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'target_col': 'Unique Headcount',
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,      # Lower LR for better convergence
        'num_leaves': 128,          # Increased leaves
        'max_depth': 12,            # Allow deeper trees
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'lambda_l1': 1.0,           # Regularization
        'lambda_l2': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
}

def run_pipeline():
    print(">>> [1/4] Loading data...")
    try:
        train_df = pd.read_csv('wells_fargo_train.csv')
    except FileNotFoundError:
        print("Error: 'wells_fargo_train.csv' not found.")
        return

    # --- PREPROCESSING ---
    print(">>> [2/4] Preprocessing & Feature Engineering...")
    
    # 1. Clean Columns (Remove trailing spaces)
    train_df.columns = train_df.columns.str.strip()
    
    # 2. Log-Transform Target (Key improvement for Count Data)
    # We teach the model to predict log(y + 1)
    train_df['target_log'] = np.log1p(train_df[CONFIG['target_col']])
    
    X = train_df.drop(columns=[CONFIG['target_col'], 'target_log'])
    y = train_df['target_log']

    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])

    # 3. Feature Engineering: Interactions
    # Create combinations of features as trends often depend on pairs.
    # Fixed Column Names based on your error log:
    # 'Program Grouping' exists.
    # 'Faculty Group' exists (was previously 'Faculty Grouping').
    
    if 'Gender' in X.columns and 'Program Grouping' in X.columns:
        X['Gender_Program'] = X['Gender'] + "_" + X['Program Grouping']
        
    if 'Gender' in X.columns and 'Faculty Group' in X.columns:
        X['Gender_Faculty'] = X['Gender'] + "_" + X['Faculty Group']
    
    # 4. Native Categorical Handling
    # LightGBM works better with pandas category type than LabelEncoder
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"   Categorical Columns: {len(cat_cols)}")
    
    for col in cat_cols:
        X[col] = X[col].astype('category')

    # 5. Prepare Time Split
    # The column is named 'Fiscal Year' (e.g., '2008-09')
    if 'Fiscal Year' in train_df.columns:
        # Extract first 4 digits: '2008-09' -> 2008
        years = train_df['Fiscal Year'].astype(str).str[:4].astype(int)
    else:
        print("Warning: 'Fiscal Year' not found. Using simple index split.")
        years = pd.Series(np.zeros(len(X))) # Fallback

    # --- TIME-BASED SPLIT ---
    print("\n>>> [3/4] Data Split (Time-Based)...")
    
    # We validate on the LAST available year to simulate predicting the future
    last_year = years.max()
    print(f"   Splitting by last year found: {last_year}")
    
    # If years are all 0 (column not found), this splits everything into val, 
    # so we add a safeguard for fallback
    if last_year == 0:
        # Fallback: simple 80/20 split
        split_idx = int(len(X) * 0.8)
        val_mask = X.index >= split_idx
    else:
        val_mask = years == last_year
        
    train_mask = ~val_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    print(f"   Train Rows: {len(X_train)}")
    print(f"   Val Rows:   {len(X_val)}")

    # --- TRAINING ---
    print("\n>>> [4/4] Training LightGBM (Log-Target)...")
    
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature='auto')
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature='auto', reference=dtrain)
    
    model = lgb.train(
        CONFIG['lgb_params'],
        dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(500)
        ]
    )
    
    # --- EVALUATION ---
    print("\n" + "="*40)
    print("FINAL VALIDATION EVALUATION")
    print("="*40)
    
    # Predict log values
    log_preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Inverse transform: exp(x) - 1
    # Also clip negative values and round to nearest integer
    final_preds = np.expm1(log_preds)
    final_preds = np.maximum(final_preds, 0)
    final_preds = np.round(final_preds) 
    
    # Calculate RMSE on original scale (we must exponentiate y_val too)
    y_val_original = np.expm1(y_val)
    
    mse = mean_squared_error(y_val_original, final_preds)
    rmse = np.sqrt(mse)
    
    print(f"Validation RMSE: {rmse:.5f}")
    
    # Feature Importance
    print("\nTop 5 Important Features:")
    lgb.plot_importance(model, max_num_features=5, importance_type='gain', figsize=(10, 4))
    
    print("="*40)
    print("✓ Improved Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()