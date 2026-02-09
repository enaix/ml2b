import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'holdout_size': 0.2,       # 20% for Holdout (Deferred Test)
    'n_folds': 5,              # 5 folds for internal CV
    'target_col': 'Fertilizer Name',
    'id_col': 'id',
    'lgb_params': {
        'objective': 'multiclass',  # Multi-class classification
        'metric': 'multi_logloss',  # Logloss optimizes probabilities well for MAP@N
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

def load_data():
    print(">>> [1/5] Loading data...")
    try:
        train_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return None
    return train_df

def map3_score(y_true, y_pred_probs):
    """
    Approximation of MAP@3 validation score.
    y_true: actual encoded labels (indices)
    y_pred_probs: probability matrix
    """
    # Get top 3 indices
    top3_idx = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]
    
    score = 0
    n = len(y_true)
    
    for i in range(n):
        actual = y_true[i]
        predicted = top3_idx[i]
        
        # Check rank 1
        if actual == predicted[0]:
            score += 1.0
        # Check rank 2
        elif actual == predicted[1]:
            score += 1/2
        # Check rank 3
        elif actual == predicted[2]:
            score += 1/3
            
    return score / n

def run_pipeline():
    train_df = load_data()
    if train_df is None: return

    # --- PREPROCESSING ---
    print(">>> [2/5] Preprocessing...")
    
    # 1. Encode Target
    le = LabelEncoder()
    train_df['target_encoded'] = le.fit_transform(train_df[CONFIG['target_col']])
    num_classes = len(le.classes_)
    CONFIG['lgb_params']['num_class'] = num_classes
    
    print(f"   Classes found: {num_classes}")
    print(f"   Class Mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # 2. Handle Categorical Features
    # Identify object columns (excluding target)
    features = [c for c in train_df.columns if c not in [CONFIG['id_col'], CONFIG['target_col'], 'target_encoded']]
    cat_cols = [c for c in features if train_df[c].dtype == 'object']
    
    # Encode categorical features for LGBM
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        
    print(f"   Categorical Features: {cat_cols}")

    # Prepare X and y
    X = train_df[features]
    y = train_df['target_encoded']

    # --- SPLIT TRAIN / HOLDOUT ---
    print(f"\n>>> [3/5] Splitting Data (Holdout {CONFIG['holdout_size']*100}%)...")
    
    # Stratified split to maintain class balance
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
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0) # Silent
            ]
        )
        
        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        fold_score = map3_score(y_val.values, val_probs)
        scores.append(fold_score)
        
        print(f"   Fold {fold+1}: MAP@3 = {fold_score:.5f}")

    print(f"   Average Internal CV MAP@3: {np.mean(scores):.5f}")

    # --- FINAL EVALUATION ON HOLDOUT ---
    print("\n>>> [5/5] Final Training & Holdout Evaluation...")
    
    # Train on Full 80% Train Split
    # We use X_holdout as valid_set for early stopping (proxy for Test)
    dtrain_full = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    dholdout = lgb.Dataset(X_holdout, label=y_holdout, categorical_feature=cat_cols, reference=dtrain_full)
    
    final_model = lgb.train(
        CONFIG['lgb_params'],
        dtrain_full,
        num_boost_round=1000,
        valid_sets=[dholdout],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
        ]
    )
    
    # Predict on Holdout
    holdout_probs = final_model.predict(X_holdout, num_iteration=final_model.best_iteration)
    final_map3 = map3_score(y_holdout.values, holdout_probs)
    
    print("\n" + "="*40)
    print("FINAL HOLDOUT EVALUATION")
    print("="*40)
    print(f"Holdout MAP@3: {final_map3:.5f}")
    
    print("\nTop 5 Important Features:")
    lgb.plot_importance(final_model, max_num_features=5, importance_type='gain', figsize=(10, 4))
    
    print("="*40)
    print("✓ Pipeline finished.")
    
    # In a real scenario with test.csv, you would now train on ALL data (X, y)
    # and predict on test.csv using the best params found.

if __name__ == "__main__":
    run_pipeline()