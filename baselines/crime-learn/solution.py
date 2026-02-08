import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'holdout_size': 0.2,       # 20% for Deferred Test (Holdout)
    'n_folds': 5,              # 5 Folds for internal metric estimation
    'target_col': 'ViolentCrimesPerPop',
    'id_col': 'ID',
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 10,
        'max_depth': 3,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'lambda_l1': 2.0,
        'lambda_l2': 2.0,
        'min_child_samples': 20,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
}

def engineer_features(df):
    """
    Feature engineering: Rates, Interactions, and Wealth Disparity.
    """
    df = df.copy()
    
    # 1. Convert "Num" (Count) columns to Per Capita (Rates)
    num_cols = [c for c in df.columns if c.startswith('Num') and c != 'numbUrban']
    for col in num_cols:
        df[f'Rate_{col}'] = df[col] / (df['population'] + 1.0)
    
    # 2. Socio-Economic Interactions
    if 'PctPopUnderPov' in df.columns and 'PctNotHSGrad' in df.columns:
        df['Poverty_x_NoHS'] = df['PctPopUnderPov'] * df['PctNotHSGrad']
    
    if 'PctUnemployed' in df.columns and 'agePct16t24' in df.columns:
        df['Unemployed_Youth'] = df['PctUnemployed'] * df['agePct16t24']
        
    if 'PctPersDenseHous' in df.columns and 'PctPopUnderPov' in df.columns:
        df['Dense_Housing_Poverty'] = df['PctPersDenseHous'] * df['PctPopUnderPov']

    if 'PctKids2Par' in df.columns:
        df['PctKidsSinglePar'] = 100 - df['PctKids2Par']
        
    if 'perCapInc' in df.columns and 'medIncome' in df.columns:
        df['Income_Disparity'] = df['perCapInc'] / (df['medIncome'] + 1)
        
    return df

def run_pipeline():
    print(">>> [1/5] Loading data...")
    try:
        train_df = pd.read_csv('crime_train.csv')
    except FileNotFoundError:
        print("Error: 'crime_train.csv' not found.")
        return

    # --- PREPROCESSING ---
    print(">>> [2/5] Feature Engineering...")
    
    train_df = engineer_features(train_df)
    
    if CONFIG['id_col'] in train_df.columns:
        train_df = train_df.drop(columns=[CONFIG['id_col']])

    X = train_df.drop(columns=[CONFIG['target_col']])
    y_raw = train_df[CONFIG['target_col']]
    
    # Log Transform Target
    y_log = np.log1p(y_raw)

    print(f"   Total Features: {X.shape[1]}")
    print(f"   Total Samples: {len(X)}")

    # --- SPLIT TRAIN / HOLDOUT ---
    print(f"\n>>> [3/5] Splitting Data (Holdout {CONFIG['holdout_size']*100}%)...")
    
    # Split into 80% Train (for CV/Training) and 20% Holdout (Deferred Test)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y_log, test_size=CONFIG['holdout_size'], random_state=CONFIG['seed'], shuffle=True
    )
    
    print(f"   Train Shape:   {X_train.shape}")
    print(f"   Holdout Shape: {X_holdout.shape}")

    # --- INTERNAL CV ON TRAIN SPLIT ---
    print("\n>>> [4/5] Running Internal CV (Ridge + LGBM)...")
    
    kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Ridge (RobustScaler is key for outliers)
        ridge = make_pipeline(RobustScaler(), Ridge(alpha=5.0))
        ridge.fit(X_tr, y_tr)
        p_ridge = ridge.predict(X_val)
        
        # LGBM
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        lgb_m = lgb.train(CONFIG['lgb_params'], dtrain, num_boost_round=1000)
        p_lgb = lgb_m.predict(X_val)
        
        # Blend 50/50
        p_blend = 0.5 * p_ridge + 0.5 * p_lgb
        
        # Metric (Original Scale)
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(p_blend)))
        scores.append(rmse)
        
    print(f"   Average Internal CV RMSE: {np.mean(scores):.2f}")

    # --- FINAL EVALUATION ON HOLDOUT ---
    print("\n>>> [5/5] Final Training & Holdout Evaluation...")
    
    # Train on full X_train
    # 1. Ridge
    final_ridge = make_pipeline(RobustScaler(), Ridge(alpha=5.0))
    final_ridge.fit(X_train, y_train)
    
    # 2. LGBM (Train for fixed rounds or use internal validation logic if desired)
    dtrain_full = lgb.Dataset(X_train, label=y_train)
    final_lgb = lgb.train(CONFIG['lgb_params'], dtrain_full, num_boost_round=1200)
    
    # Predict on Holdout
    h_ridge = final_ridge.predict(X_holdout)
    h_lgb = final_lgb.predict(X_holdout)
    h_blend = 0.5 * h_ridge + 0.5 * h_lgb
    
    final_rmse = np.sqrt(mean_squared_error(np.expm1(y_holdout), np.expm1(h_blend)))
    
    print("\n" + "="*40)
    print("FINAL HOLDOUT RESULTS")
    print("="*40)
    print(f"Holdout RMSE: {final_rmse:.2f}")
    
    print("\nTop Features (LGBM):")
    lgb.plot_importance(final_lgb, max_num_features=5, importance_type='gain', figsize=(10, 4))
    
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()