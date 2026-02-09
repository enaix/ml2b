import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'holdout_size': 0.2,       # 20% of Breaths for Holdout
    'n_folds': 5,              # 5 Folds for internal CV
    'target_col': 'pressure',
    'group_col': 'breath_id',
    'id_col': 'id',
    'lgb_params': {
        'objective': 'regression_l1', # MAE optimization
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 100,         # Complex time-series need deeper trees
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
}

def load_data():
    print(">>> [1/5] Loading data...")
    try:
        df = pd.read_csv('train.csv')
        if CONFIG['target_col'] not in df.columns:
            print(f"[ERROR] Column '{CONFIG['target_col']}' not found!")
            print(f"Columns: {df.columns.tolist()}")
            return None
        return df
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return None

def engineer_features(df):
    """
    Time-series feature engineering for Ventilator data.
    """
    print("    Generatings Lag and Diff features (this may take a moment)...")
    df = df.copy()
    
    # 1. Basic Engineering
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    
    # 2. Lag Features (Previous time steps)
    # We shift within each breath_id group
    for lag in [1, 2, 3]:
        df[f'u_in_lag{lag}'] = df.groupby('breath_id')['u_in'].shift(lag).fillna(0)
        
    # 3. Difference Features (Rate of change)
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    
    # 4. R & C Interactions (Lung attributes)
    # FIX: Do NOT overwrite R and C with strings. Just use them to create the interaction.
    df['R_C'] = df['R'].astype(str) + "_" + df['C'].astype(str)
    df['R_C'] = df['R_C'].astype('category')
    
    # Ensure R and C are integers (LightGBM likes numeric or category)
    df['R'] = df['R'].astype(int)
    df['C'] = df['C'].astype(int)
    
    # Drop standard ID (keep breath_id for grouping)
    if CONFIG['id_col'] in df.columns:
        df = df.drop(columns=[CONFIG['id_col']])
        
    return df

def run_pipeline():
    df = load_data()
    if df is None: return

    # --- PREPROCESSING ---
    print(">>> [2/5] Preprocessing & Feature Engineering...")
    
    df = engineer_features(df)
    
    # Prepare X, y, and Groups
    X = df.drop(columns=[CONFIG['target_col']])
    y = df[CONFIG['target_col']]
    groups = df[CONFIG['group_col']] # Vital for splitting
    
    # Drop group col from features (it's an ID, not a feature)
    X = X.drop(columns=[CONFIG['group_col']])

    print(f"   Total Samples: {len(X)}")
    print(f"   Total Breaths: {groups.nunique()}")
    print(f"   Feature Count: {X.shape[1]}")

    # --- SPLIT TRAIN / HOLDOUT (GROUP BASED) ---
    print(f"\n>>> [3/5] Group-Splitting Data (Holdout {CONFIG['holdout_size']*100}%)...")
    
    # We use GroupShuffleSplit to ensure NO leakage of breaths between sets
    gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG['holdout_size'], random_state=CONFIG['seed'])
    train_idx, holdout_idx = next(gss.split(X, y, groups))
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_holdout, y_holdout = X.iloc[holdout_idx], y.iloc[holdout_idx]
    
    # We also need groups for Internal CV
    groups_train = groups.iloc[train_idx]
    
    print(f"   Train Shape:   {X_train.shape}")
    print(f"   Holdout Shape: {X_holdout.shape}")

    # --- INTERNAL CV ON TRAIN SPLIT (GROUP K-FOLD) ---
    print("\n>>> [4/5] Running Internal Group-CV (Stability Check)...")
    
    gkf = GroupKFold(n_splits=CONFIG['n_folds'])
    scores = []
    
    # Important: R_C is the only explicit categorical feature now
    cat_feats = ['R_C']
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_feats)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_feats, reference=dtrain)
        
        model = lgb.train(
            CONFIG['lgb_params'],
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0) # Silent
            ]
        )
        
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        mae = mean_absolute_error(y_val, preds)
        scores.append(mae)
        
        print(f"   Fold {fold+1}: MAE = {mae:.4f}")
        
    print(f"   Average Internal CV MAE: {np.mean(scores):.4f}")

    # --- FINAL EVALUATION ON HOLDOUT ---
    print("\n>>> [5/5] Final Training & Holdout Evaluation...")
    
    # Train on Full 80% Train Split
    dtrain_full = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats)
    dholdout = lgb.Dataset(X_holdout, label=y_holdout, categorical_feature=cat_feats, reference=dtrain_full)
    
    final_model = lgb.train(
        CONFIG['lgb_params'],
        dtrain_full,
        num_boost_round=2000,
        valid_sets=[dholdout],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100)
        ]
    )
    
    # Final Prediction
    holdout_preds = final_model.predict(X_holdout, num_iteration=final_model.best_iteration)
    final_mae = mean_absolute_error(y_holdout, holdout_preds)
    
    print("\n" + "="*40)
    print("FINAL HOLDOUT EVALUATION")
    print("="*40)
    print(f"Holdout MAE: {final_mae:.4f}")
    
    print("\nTop Features:")
    lgb.plot_importance(final_model, max_num_features=5, importance_type='gain', figsize=(10, 4))
    
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()