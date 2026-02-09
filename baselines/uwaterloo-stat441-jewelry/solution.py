import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'target_col': 'Revenue',
    'lgb_params': {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True,       # Crucial for imbalanced datasets!
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
        # Trying the name from description first
        train_df = pd.read_csv('online_jewellery_shop_train.csv')
    except FileNotFoundError:
        print("Warning: Specific filename not found, trying 'train.csv'...")
        try:
            train_df = pd.read_csv('train.csv')
        except FileNotFoundError:
            print("Error: Training file not found.")
            return

    # --- PREPROCESSING ---
    print(">>> [2/4] Preprocessing & Feature Engineering...")
    
    # 1. Drop ID (Not predictive)
    if 'ID' in train_df.columns:
        train_df = train_df.drop(columns=['ID'])

    # 2. Handle Target
    # Ensure target is 0/1 integer (sometimes it loads as bool)
    train_df[CONFIG['target_col']] = train_df[CONFIG['target_col']].astype(int)
    
    X = train_df.drop(columns=[CONFIG['target_col']])
    y = train_df[CONFIG['target_col']]

    # 3. Handle Categorical Features
    # Based on dataset description, these are the categorical ones:
    # Month, VisitorType, Weekend (Bool), OperatingSystems, Browser, Region, TrafficType
    
    # Identify object/bool columns automatically
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    # Add numeric columns that are actually categorical codes
    known_cats = ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Month', 'VisitorType', 'Weekend']
    for c in known_cats:
        if c in X.columns and c not in cat_cols:
            cat_cols.append(c)
            
    # Remove duplicates
    cat_cols = list(set(cat_cols))
    print(f"   Categorical features identified: {cat_cols}")

    # Convert to standard format for LightGBM
    for col in cat_cols:
        # Convert to string first to handle mixed types (like booleans or numbers)
        X[col] = X[col].astype(str)
        
        # Label Encode to 0..N-1
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        
        # Convert to pandas 'category' type
        X[col] = X[col].astype('category')

    print(f"   Features: {X.shape[1]}")
    print(f"   Class Balance: {y.value_counts(normalize=True).to_dict()}")

    # --- ROBUST VALIDATION (HOLDOUT) ---
    print("\n>>> [3/4] Data Split (Stratified Holdout)...")
    
    # Stratify is VERY important here because the positive class is rare
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG['seed'], stratify=y
    )
    
    print(f"   Train (Dev): {len(X_dev)} rows")
    print(f"   Holdout:     {len(X_holdout)} rows")

    # --- TRAINING ---
    print("\n>>> [4/4] Training LightGBM...")
    
    dtrain = lgb.Dataset(X_dev, label=y_dev, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_holdout, label=y_holdout, categorical_feature=cat_cols, reference=dtrain)
    
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
    
    # Predict Probabilities (not classes!)
    val_probs = model.predict(X_holdout, num_iteration=model.best_iteration)
    
    auc = roc_auc_score(y_holdout, val_probs)
    
    print(f"Validation ROC AUC: {auc:.5f}")
    
    # Feature Importance
    print("\nTop 5 Features impacting Revenue:")
    lgb.plot_importance(model, max_num_features=5, importance_type='gain', figsize=(10, 4))
    
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()