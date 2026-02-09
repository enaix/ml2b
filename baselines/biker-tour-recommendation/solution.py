import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import gc

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION AND UTILS
# ==========================================
CONFIG = {
    'seed': 42,
    'n_folds': 5,
    'input_path': '.', # Path to folder with csv files
    'target_col': 'target'
}

def apk(actual, predicted, k=100):
    """
    Computes the average precision at k.
    actual : list of items that are relevant (ground truth)
    predicted : ordered list of predicted items
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=100):
    """
    Computes the mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def process_dates(df, col_name, prefix):
    """Converts string dates into features"""
    if col_name in df.columns:
        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
        df[f'{prefix}_year'] = df[col_name].dt.year
        df[f'{prefix}_month'] = df[col_name].dt.month
        df[f'{prefix}_day'] = df[col_name].dt.day
        # Convert date to timestamp
        df[f'{prefix}_timestamp'] = df[col_name].astype('int64') // 10**9
    return df

# ==========================================
# 2. DATA PREPARATION
# ==========================================
def load_and_preprocess():
    print(">>> [1/4] Loading data...")
    try:
        train = pd.read_csv(f"{CONFIG['input_path']}/train.csv")
        bikers = pd.read_csv(f"{CONFIG['input_path']}/bikers.csv")
        tours = pd.read_csv(f"{CONFIG['input_path']}/tours.csv")
        network = pd.read_csv(f"{CONFIG['input_path']}/bikers_network.csv")
    except FileNotFoundError:
        print("Error: One or more data files not found.")
        return None, None

    # --- Create Target ---
    # Task: Predict interest. Interest is like=1.
    # If dislike=1 or (0,0) - we consider it class 0.
    train['target'] = train['like'].fillna(0).astype(int)
    
    # Remove unnecessary columns from train (dislike is not a feature, it's part of target)
    # 'invited' and 'timestamp' are kept as features
    df = train.drop(columns=['like', 'dislike'])
    
    # --- Feature Engineering: Bikers ---
    print(">>> [2/4] Feature Engineering: Bikers & Friends...")
    
    # Process friends (count them)
    network['friend_count'] = network['friends'].astype(str).apply(lambda x: len(x.split()))
    bikers = bikers.merge(network[['biker_id', 'friend_count']], on='biker_id', how='left')
    bikers['friend_count'] = bikers['friend_count'].fillna(0)
    
    # Process biker dates
    bikers = process_dates(bikers, 'member_since', 'member')
    bikers['bornIn'] = pd.to_numeric(bikers['bornIn'], errors='coerce')
    bikers['age'] = 2021 - bikers['bornIn'] # Approximate age
    
    # Encode biker categories
    biker_cat_cols = ['language_id', 'location_id', 'gender', 'area', 'time_zone']
    for col in biker_cat_cols:
        bikers[col] = bikers[col].astype(str).fillna('unknown')
        le = LabelEncoder()
        bikers[col] = le.fit_transform(bikers[col])
        
    df = df.merge(bikers, on='biker_id', how='left')
    
    # --- Feature Engineering: Tours ---
    print(">>> [3/4] Feature Engineering: Tours...")
    
    # Process tour dates
    tours = process_dates(tours, 'tour_date', 'tour')
    
    # Encode tour categories
    # biker_id in tours table is the ORGANIZER
    tours = tours.rename(columns={'biker_id': 'organizer_id'})
    
    # Label Encoding for tours
    for col in ['city', 'state', 'pincode', 'country', 'organizer_id']:
        tours[col] = tours[col].astype(str).fillna('unknown')
        le = LabelEncoder()
        tours[col] = le.fit_transform(tours[col])
        
    df = df.merge(tours, on='tour_id', how='left')
    
    # Drop raw date columns
    drop_cols = ['timestamp', 'member_since', 'tour_date', 'friends', 'bornIn']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Convert categorical features to 'category' type for LGBM
    cat_cols = biker_cat_cols + ['city', 'state', 'pincode', 'country', 'organizer_id']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    return df, cat_cols

# ==========================================
# 3. TRAINING AND VALIDATION
# ==========================================
def run_training():
    df, cat_cols = load_and_preprocess()
    
    if df is None:
        return

    X = df.drop(columns=['target', 'biker_id', 'tour_id'])
    y = df['target']
    
    # --- Holdout Split (Robust Validation) ---
    print("\n>>> [4/4] Data Split (Holdout) & Training...")
    # Reserve 20% of data for the final MAP@K calculation
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        df, y, test_size=0.2, random_state=CONFIG['seed'], stratify=y
    )
    
    print(f"   Train (Dev): {len(X_dev)} rows")
    print(f"   Holdout:     {len(X_holdout)} rows")
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc', # AUC correlates well with ranking quality
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': CONFIG['seed']
    }
    
    # Train K-Fold on Dev set
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    models = []
    
    # Prepare data for loop
    # We need to drop ID columns for training, but keep them in X_holdout for evaluation grouping
    X_dev_feats = X_dev.drop(columns=['target', 'biker_id', 'tour_id'])
    y_dev_vals = X_dev['target']
    
    X_holdout_feats = X_holdout.drop(columns=['target', 'biker_id', 'tour_id'])

    print(f"   Training Ensemble ({CONFIG['n_folds']} Folds)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev_feats, y_dev_vals)):
        X_tr = X_dev_feats.iloc[train_idx]
        y_tr = y_dev_vals.iloc[train_idx]
        X_val = X_dev_feats.iloc[val_idx]
        y_val = y_dev_vals.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)
        
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0) # Silent mode
            ]
        )
        models.append(model)
        
        # Optional: Print fold AUC
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        # We don't print here to keep output clean, focusing on final MAP
        # print(f"   Fold {fold+1} done.")

    # --- CALCULATE MAP@K ON HOLDOUT ---
    print("\n>>> Calculating MAP@K on Holdout set...")
    
    # 1. Predict probabilities using ensemble
    holdout_preds = np.zeros(len(X_holdout))
    for model in models:
        holdout_preds += model.predict(X_holdout_feats, num_iteration=model.best_iteration)
    holdout_preds /= len(models)
    
    # 2. Add predictions to dataframe
    holdout_eval_df = X_holdout[['biker_id', 'tour_id']].copy()
    holdout_eval_df['pred_score'] = holdout_preds
    holdout_eval_df['actual_relevance'] = y_holdout
    
    # 3. Group by user to calculate MAP
    map_scores = []
    
    grouped = holdout_eval_df.groupby('biker_id')
    
    for biker_id, group in grouped:
        # Ground Truth: tour_ids where target=1
        actual = group[group['actual_relevance'] == 1]['tour_id'].tolist()
        
        # Predicted: tour_ids sorted by predicted score
        predicted = group.sort_values('pred_score', ascending=False)['tour_id'].tolist()
        
        # Calculate APK
        map_scores.append(apk(actual, predicted, k=100))
        
    final_map = np.mean(map_scores)

    print("="*40)
    print(f"HOLDOUT MAP SCORE: {final_map:.5f}")
    print("="*40)
    print("✓ Pipeline finished. No submission generated.")

if __name__ == "__main__":
    run_training()