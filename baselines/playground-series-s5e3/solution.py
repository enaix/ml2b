import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'holdout_size': 0.2,       # 20% for Holdout (Deferred Test)
    'n_folds': 10,             # 10 folds for internal CV stability check
    'target_col': 'rainfall',  # Target column name
    'id_col': 'id',
    'lgb_params': {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,  # Slower learning rate
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
}

def load_data():
    print(">>> [1/5] Loading data...")
    try:
        train_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return None
    return train_df

def feature_engineering(df):
    """
    Creates interaction features relevant to weather.
    """
    df = df.copy()
    
    # 1. Temperature Range
    if 'min_temp' in df.columns and 'max_temp' in df.columns:
        df['temp_range'] = df['max_temp'] - df['min_temp']
        
    # 2. Dew Point Proxy (Temp * Humidity)
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interact'] = df['temperature'] * df['humidity']
        
    # 3. Wind Interaction
    if 'wind_speed' in df.columns and 'pressure' in df.columns:
        df['wind_pressure_ratio'] = df['wind_speed'] / (df['pressure'] + 1)
        
    return df

def run_pipeline():
    train_df = load_data()
    if train_df is None: return

    # --- PREPROCESSING ---
    print(">>> [2/5] Preprocessing & Feature Engineering...")
    
    train_df = feature_engineering(train_df)
    
    # Identify Categorical Features
    cat_cols = [c for c in train_df.columns if train_df[c].dtype == 'object' and c != CONFIG['target_col']]
    print(f"   Categorical Features: {cat_cols}")
    
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        
    # Drop ID
    if CONFIG['id_col'] in train_df.columns:
        train_df = train_df.drop(columns=[CONFIG['id_col']])
        
    X = train_df.drop(columns=[CONFIG['target_col']])
    y = train_df[CONFIG['target_col']]

    # Encode Target if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"   Target encoded. Positive class: {le.classes_[1]}")

    print(f"   Features: {X.shape[1]}")
    print(f"   Positive Class Rate: {y.mean():.2%}")

    # --- SPLIT TRAIN / HOLDOUT ---
    print(f"\n>>> [3/5] Splitting Data (Holdout {CONFIG['holdout_size']*100}%)...")
    
    # Stratified split to maintain class balance in Holdout
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=CONFIG['holdout_size'], stratify=y, random_state=CONFIG['seed']
    )
    
    print(f"   Train Shape:   {X_train.shape}")
    print(f"   Holdout Shape: {X_holdout.shape}")

    # --- INTERNAL CV ON TRAIN SPLIT ---
    print("\n>>> [4/5] Running Internal CV (Stability Check)...")
    
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)
        
        model = lgb.train(
            CONFIG['lgb_params'],
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=150),
                lgb.log_evaluation(0) # Silent
            ]
        )
        
        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, val_probs)
        scores.append(auc)
        
        print(f"   Fold {fold+1}: AUC = {auc:.5f}")
        
    print(f"   Average Internal CV AUC: {np.mean(scores):.5f}")

    # --- FINAL EVALUATION ON HOLDOUT ---
    print("\n>>> [5/5] Final Training & Holdout Evaluation...")
    
    # Train on Full 80% Train Split
    # We use X_holdout as valid_set just for early stopping here to get the best iteration.
    dtrain_full = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    dholdout = lgb.Dataset(X_holdout, label=y_holdout, categorical_feature=cat_cols, reference=dtrain_full)
    
    final_model = lgb.train(
        CONFIG['lgb_params'],
        dtrain_full,
        num_boost_round=3000,
        valid_sets=[dholdout],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150),
            lgb.log_evaluation(100)
        ]
    )
    
    # Predict on Holdout
    holdout_probs = final_model.predict(X_holdout, num_iteration=final_model.best_iteration)
    final_auc = roc_auc_score(y_holdout, holdout_probs)
    
    print("\n" + "="*40)
    print("FINAL HOLDOUT EVALUATION")
    print("="*40)
    print(f"Holdout ROC AUC: {final_auc:.5f}")
    
    print("\nTop 5 Important Features:")
    lgb.plot_importance(final_model, max_num_features=5, importance_type='gain', figsize=(10, 4))
    
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()