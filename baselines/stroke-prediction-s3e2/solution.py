import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'n_folds': 10,              # More folds for stable validation on imbalanced data
    'target_col': 'stroke',
    'id_col': 'id',
    'lgb_params': {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'is_unbalance': True,   # vital for 4% positive class
        'learning_rate': 0.01,  # Slower learning for better generalization
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
}

def feature_engineering(df):
    """
    Creates medical risk interaction features.
    """
    df = df.copy()
    
    # 1. Interaction: Age is the biggest driver. Combine with other risks.
    # Older people with hypertension are at higher risk.
    df['Age_Hypertension'] = df['age'] * (df['hypertension'] + 1)
    df['Age_HeartDisease'] = df['age'] * (df['heart_disease'] + 1)
    
    # 2. Glucose & BMI Interaction (Metabolic health)
    df['Glucose_BMI'] = df['avg_glucose_level'] * df['bmi']
    
    # 3. Risk Score (Simple count of risk factors)
    # Hypertension(1) + HeartDis(1) + HighGlucose(>140 -> 1) + HighBMI(>30 -> 1)
    # Note: This is a rough heuristic feature
    high_glucose = (df['avg_glucose_level'] > 140).astype(int)
    obesity = (df['bmi'] > 30).astype(int)
    df['Risk_Factor_Count'] = df['hypertension'] + df['heart_disease'] + high_glucose + obesity
    
    return df

def run_pipeline():
    print(">>> [1/4] Loading data...")
    try:
        train_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return

    # --- PREPROCESSING ---
    print(">>> [2/4] Preprocessing & Feature Engineering...")
    
    # 1. Feature Engineering
    train_df = feature_engineering(train_df)
    
    # 2. Handle Categorical Columns
    # LightGBM prefers 'category' dtype over One-Hot Encoding
    cat_cols = [c for c in train_df.columns if train_df[c].dtype == 'object']
    print(f"   Categorical features: {cat_cols}")
    
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        
    # 3. Prepare X and y
    if CONFIG['id_col'] in train_df.columns:
        train_df = train_df.drop(columns=[CONFIG['id_col']])
        
    X = train_df.drop(columns=[CONFIG['target_col']])
    y = train_df[CONFIG['target_col']]

    print(f"   Features: {X.shape[1]}")
    print(f"   Positive Class Rate: {y.mean():.2%}")

    # --- CROSS-VALIDATION ---
    print("\n>>> [3/4] Training LightGBM (Stratified K-Fold)...")
    
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    
    oof_preds = np.zeros(len(X))
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)
        
        model = lgb.train(
            CONFIG['lgb_params'],
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(0) # Silent
            ]
        )
        
        # Predict Probabilities (Target is 1 for stroke)
        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_probs
        
        auc = roc_auc_score(y_val, val_probs)
        scores.append(auc)
        
        print(f"   Fold {fold+1}: AUC = {auc:.5f}")

    # --- FINAL EVALUATION ---
    print("\n" + "="*40)
    print("FINAL EVALUATION")
    print("="*40)
    
    avg_auc = np.mean(scores)
    print(f"Average ROC AUC: {avg_auc:.5f}")
    
    # Calculate Overall AUC on OOF (usually similar to average)
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"OOF ROC AUC:     {oof_auc:.5f}")
    
    # Feature Importance
    print("\nTop 5 Important Features:")
    lgb.plot_importance(model, max_num_features=5, importance_type='gain', figsize=(10, 4))
    
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()