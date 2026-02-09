import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'holdout_size': 0.2,       # 20% for Deferred Test (Holdout)
    'n_folds': 5,              # 5 Folds for metric estimation
    'target_col': 'price',
    'id_col': 'id',
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,  # Conservative learning rate
        'num_leaves': 50,       # Increased complexity slightly
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
}

def extract_features(df):
    """
    Advanced feature engineering to extract details from complex string columns.
    """
    df = df.copy()
    
    # --- 1. Engine Parsing ---
    # Extract Horsepower (e.g., "172.0HP")
    df['hp'] = df['engine'].apply(lambda x: float(re.search(r'(\d+\.?\d*)HP', str(x)).group(1)) 
                                  if re.search(r'(\d+\.?\d*)HP', str(x)) else np.nan)
    
    # Extract Displacement (e.g., "1.6L")
    df['liters'] = df['engine'].apply(lambda x: float(re.search(r'(\d+\.?\d*)L', str(x)).group(1)) 
                                      if re.search(r'(\d+\.?\d*)L', str(x)) else np.nan)
    
    # Extract Cylinder Count (e.g., "4 Cylinder" or "V6")
    def get_cylinders(x):
        s = str(x)
        m = re.search(r'(\d+)\s*Cylinder', s)
        if m: return int(m.group(1))
        if 'V6' in s: return 6
        if 'V8' in s: return 8
        if 'V10' in s: return 10
        if 'V12' in s: return 12
        return np.nan
    df['cylinders'] = df['engine'].apply(get_cylinders)
    
    # --- 2. Transmission Parsing ---
    # Extract Speed (e.g., "8-Speed")
    df['gears'] = df['transmission'].apply(lambda x: float(re.search(r'(\d+)-Speed', str(x)).group(1)) 
                                           if re.search(r'(\d+)-Speed', str(x)) else np.nan)
    
    # --- 3. Categorical Cleaning ---
    # Accident: Map to ordinal (0 = None, 1 = Accident)
    # NaNs in accident are treated as 1 (Unknown/Risk) or 0. Mode imputation (0) is safer here.
    df['accident_clean'] = df['accident'].fillna('None reported').map({
        'None reported': 0,
        'At least 1 accident or damage reported': 1
    })
    
    # Clean Title: "Yes" or NaN. Map to 1 (Yes) and 0 (Missing/No)
    df['clean_title_clean'] = df['clean_title'].map({'Yes': 1}).fillna(0)
    
    # Car Age
    # Use max year from data + 1 as current reference
    current_year = df['model_year'].max() + 1
    df['car_age'] = current_year - df['model_year']
    
    # Mileage: Log transform to handle skew
    df['log_mileage'] = np.log1p(df['milage'])
    
    return df

def run_pipeline():
    print(">>> [1/5] Loading data...")
    try:
        raw_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return

    # --- PREPROCESSING ---
    print(">>> [2/5] Feature Engineering...")
    
    df_processed = extract_features(raw_df)
    
    # Identify Categorical Columns for Encoding
    # We exclude the original complex string columns we parsed
    drop_cols = ['engine', 'transmission', 'accident', 'clean_title', CONFIG['id_col'], CONFIG['target_col']]
    cat_cols = [c for c in df_processed.columns if df_processed[c].dtype == 'object' and c not in drop_cols]
    
    print(f"   Extracted Features: HP, Liters, Cylinders, Gears")
    print(f"   Categorical Features to Encode: {cat_cols}")
    
    # Label Encoding
    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col] = df_processed[col].astype(str)
        df_processed[col] = le.fit_transform(df_processed[col])
        df_processed[col] = df_processed[col].astype('category')

    # --- SPLITTING ---
    print(f"\n>>> [3/5] Splitting Data (Holdout {CONFIG['holdout_size']*100}%)...")
    
    X = df_processed.drop(columns=drop_cols + [CONFIG['target_col']], errors='ignore')
    # Log-transform target for better regression performance
    y = np.log1p(df_processed[CONFIG['target_col']])
    
    # Split: X_train (for CV + Training) vs X_holdout (Deferred Test)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=CONFIG['holdout_size'], random_state=CONFIG['seed'], shuffle=True
    )
    
    print(f"   Train Shape:   {X_train.shape}")
    print(f"   Holdout Shape: {X_holdout.shape}")

    # --- CROSS-VALIDATION (METRIC ONLY) ---
    print("\n>>> [4/5] Running Cross-Validation on Train Split...")
    
    kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    cv_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)
        
        model = lgb.train(
            CONFIG['lgb_params'],
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        preds_log = model.predict(X_val, num_iteration=model.best_iteration)
        # Calculate RMSE on Real Prices (Inverse Log)
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(preds_log)))
        cv_scores.append(rmse)
        
        print(f"   Fold {fold+1} RMSE: {rmse:.2f}")
        
    print(f"   Average CV RMSE: {np.mean(cv_scores):.2f}")

    # --- FINAL TRAINING & PREDICTION ---
    print("\n>>> [5/5] Final Training on Full Train Split & Predict on Holdout...")
    
    # Train on ALL of X_train (no splitting)
    dtrain_full = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    
    # Note: We don't have a validation set here because we are training on the full X_train.
    # We rely on the number of rounds found roughly during CV or a fixed high number.
    # Using a conservative fixed number or re-using X_holdout for early stopping is debatable.
    # Given the prompt "Predict on deferred test", strict holdout implies we shouldn't peek at it during training.
    # So we train for a fixed optimal number (e.g., from CV average) or just a safe high number.
    
    final_model = lgb.train(
        CONFIG['lgb_params'],
        dtrain_full,
        num_boost_round=1200 # Roughly based on typical convergence
    )
    
    # Predict on Holdout (Deferred Test)
    holdout_preds_log = final_model.predict(X_holdout, num_iteration=final_model.best_iteration)
    holdout_preds = np.expm1(holdout_preds_log)
    y_holdout_real = np.expm1(y_holdout)
    
    final_rmse = np.sqrt(mean_squared_error(y_holdout_real, holdout_preds))
    
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Holdout (Deferred Test) RMSE: {final_rmse:.2f}")
    
    print("\nTop Important Features:")
    lgb.plot_importance(final_model, max_num_features=5, importance_type='gain', figsize=(10, 4))
    
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()