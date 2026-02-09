import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'n_folds': 5,
    'targets': ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides'],
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
    
    train_df['date_time'] = pd.to_datetime(train_df['date_time'])
    
    # Extract Date Features
    train_df['hour'] = train_df['date_time'].dt.hour
    train_df['day_of_week'] = train_df['date_time'].dt.dayofweek
    train_df['month'] = train_df['date_time'].dt.month
    
    train_df = train_df.drop(columns=['date_time'])
    
    # Log-Transform Targets (RMSLE -> RMSE optimization)
    for target in CONFIG['targets']:
        train_df[target] = np.log1p(train_df[target])

    # --- ROBUST VALIDATION (TIME-BASED HOLDOUT) ---
    print("\n>>> [3/4] Data Split (Time-Based Holdout)...")
    
    train_data, holdout_data = train_test_split(
        train_df, test_size=0.2, shuffle=False
    )
    
    print(f"   Train (Past):    {len(train_data)} rows")
    print(f"   Holdout (Future): {len(holdout_data)} rows")

    # --- TRAINING LOOP ---
    print(f"\n>>> [4/4] Training Models (TimeSeriesSplit: {CONFIG['n_folds']} Folds)...")
    
    tscv = TimeSeriesSplit(n_splits=CONFIG['n_folds'])
    final_scores = {}
    
    for target in CONFIG['targets']:
        print(f"\n   --- Training for: {target} ---")
        
        features = [c for c in train_data.columns if c not in CONFIG['targets']]
        X_dev = train_data[features]
        y_dev = train_data[target]
        
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_dev)):
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
                    lgb.log_evaluation(0)
                ]
            )
            models.append(model)
            
            # --- FIX: Manual Square Root Calculation ---
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            mse = mean_squared_error(y_val, val_preds) # Calculate MSE
            rmse = np.sqrt(mse)                        # Calculate Square Root manually
            print(f"      Fold {fold+1}: Log-RMSE = {rmse:.5f}")

        # --- EVALUATE ON HOLDOUT ---
        X_holdout = holdout_data[features]
        y_holdout_log = holdout_data[target]
        
        holdout_preds_log = np.zeros(len(X_holdout))
        for model in models:
            holdout_preds_log += model.predict(X_holdout, num_iteration=model.best_iteration)
        holdout_preds_log /= len(models)
        
        # --- FIX: Manual Square Root Calculation ---
        final_mse = mean_squared_error(y_holdout_log, holdout_preds_log)
        final_rmsle = np.sqrt(final_mse)
        
        final_scores[target] = final_rmsle
        print(f"   >>> Holdout RMSLE for {target}: {final_rmsle:.5f}")

    # --- FINAL OVERALL SCORE ---
    print("\n" + "="*40)
    print("FINAL HOLDOUT EVALUATION (Mean RMSLE)")
    print("="*40)
    
    overall_score = np.mean(list(final_scores.values()))
    
    for t, s in final_scores.items():
        print(f"{t}: {s:.5f}")
        
    print("-" * 40)
    print(f"OVERALL MEAN RMSLE: {overall_score:.5f}")
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()