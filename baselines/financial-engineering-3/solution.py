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
    'n_folds_oof': 5,  # Folds for generating col_3 OOF predictions
    'n_folds_main': 5, # Folds for main validation
    'targets': ['col_5', 'col_8'],
    'aux_target': 'col_3',
    'lgb_params_reg': {
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

def get_oof_predictions(df, features, target, model_params):
    """
    Generates Out-of-Fold predictions for a specific target.
    Used to create the 'pred_col_3' feature without data leakage.
    """
    kf = KFold(n_splits=CONFIG['n_folds_oof'], shuffle=True, random_state=CONFIG['seed'])
    oof_preds = np.zeros(len(df))
    models = []
    
    # Store RMSE scores to check quality of auxiliary model
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx][target]
        X_val, y_val = df.iloc[val_idx][features], df.iloc[val_idx][target]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            model_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0)
            ]
        )
        
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        scores.append(rmse)
    
    print(f"   Auxiliary Model ({target}) CV RMSE: {np.mean(scores):.5f}")
    return oof_preds

def run_pipeline():
    print(">>> [1/5] Loading data...")
    try:
        train_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return

    # --- PREPROCESSING ---
    print(">>> [2/5] Preprocessing...")
    
    # Drop ID as it's usually not predictive (or handled via index if time-series)
    # Correlation analysis showed ID is highly correlated with col_1/col_2.
    if 'id' in train_df.columns:
        train_df = train_df.drop(columns=['id'])

    # Define Feature Sets
    # Features available in TEST set (col_3 is missing there)
    # Based on file inspection: col_1, col_2, col_4, col_6, col_7 are likely safe.
    # col_5 and col_8 are targets.
    base_features = ['col_1', 'col_2', 'col_4', 'col_6', 'col_7']
    
    # --- STAGE 1: IMPUTE COL_3 ---
    print(f"\n>>> [3/5] Generating OOF predictions for auxiliary target '{CONFIG['aux_target']}'...")
    
    # We predict col_3 using base_features. 
    # We use K-Fold OOF to ensure the training data for Stage 2 
    # has 'pred_col_3' values similar to what a model would produce on unseen test data.
    train_df['pred_col_3'] = get_oof_predictions(
        train_df, 
        base_features, 
        CONFIG['aux_target'], 
        CONFIG['lgb_params_reg']
    )
    
    # Now we have a new strong feature!
    augmented_features = base_features + ['pred_col_3']

    # --- STAGE 2: MAIN TRAINING ---
    print("\n>>> [4/5] Training Main Models (Split Holdout)...")
    
    # Standard Holdout Split for final evaluation
    X_dev, X_holdout = train_test_split(
        train_df, test_size=0.2, random_state=CONFIG['seed'], shuffle=True
    )
    
    results = {}
    
    for target in CONFIG['targets']:
        print(f"\n   --- Predicting Target: {target} ---")
        
        # Select Features & Target
        X_tr = X_dev[augmented_features]
        y_tr = X_dev[target]
        X_val = X_holdout[augmented_features]
        y_val = X_holdout[target]
        
        # Train Model
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        # Note: Even for col_8 (discrete/binary), we use 'regression' (RMSE) 
        # because the evaluation metric is RMSE. Predicting probabilities 
        # (via regression objective on 0/1) minimizes RMSE better than hard classes.
        model = lgb.train(
            CONFIG['lgb_params_reg'],
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=0) # Silent
            ]
        )
        
        # Evaluate
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        results[target] = rmse
        print(f"      Holdout RMSE for {target}: {rmse:.5f}")
        
        # Optional: Feature Importance
        # lgb.plot_importance(model, max_num_features=5, title=f'Importance for {target}')

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    print("FINAL HOLDOUT EVALUATION")
    print("="*40)
    
    for target, score in results.items():
        print(f"{target}: RMSE = {score:.5f}")
        
    avg_rmse = np.mean(list(results.values()))
    print("-" * 40)
    print(f"AVERAGE RMSE: {avg_rmse:.5f}")
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()