import os
import gc
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'img_size': 224,               # Image size for the Neural Network
    'batch_size': 32,              # Batch size
    'model_name': 'tf_efficientnet_b0_ns', # Efficient model for feature extraction
    'n_folds': 5,                  # Number of folds
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'input_path': '.',             # Path to data folder
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])

# ==========================================
# 1. DATA PREPARATION & DATASET
# ==========================================

class SheepDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get filename from dataframe
        file_name = self.df.iloc[idx]['filename']
        file_path = os.path.join(self.img_dir, file_name)
        
        try:
            image = Image.open(file_path).convert('RGB')
        except Exception as e:
            # Handle broken images by creating a black square
            print(f"Warning: Error reading {file_path}: {e}")
            image = Image.new('RGB', (CONFIG['img_size'], CONFIG['img_size']))

        if self.transform:
            image = self.transform(image)
            
        return image

def get_transforms():
    # Standard normalization for ImageNet pre-trained models
    return transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def extract_features(df, img_dir):
    """
    Passes images through the CNN and returns a feature table (embeddings).
    """
    print(f">>> Loading model {CONFIG['model_name']} for feature extraction...")
    # Load pre-trained model, remove the classifier layer (num_classes=0)
    model = timm.create_model(CONFIG['model_name'], pretrained=True, num_classes=0)
    model = model.to(CONFIG['device'])
    model.eval()
    
    dataset = SheepDataset(df, img_dir, transform=get_transforms())
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    embeddings = []
    
    print(f">>> Extracting features from {len(df)} images...")
    with torch.no_grad():
        for images in tqdm(loader):
            images = images.to(CONFIG['device'])
            # Get feature vector
            features = model(images)
            embeddings.append(features.cpu().numpy())
            
    # Concatenate all batches
    return np.concatenate(embeddings)

# ==========================================
# MAIN PIPELINE
# ==========================================
def run_pipeline():
    # --- LOADING ---
    print(">>> [1/5] Loading metadata...")
    try:
        train_df = pd.read_csv(os.path.join(CONFIG['input_path'], 'train_labels.csv'))
    except FileNotFoundError:
        print("Error: 'train_labels.csv' not found.")
        return

    # Encode labels (Breed String -> Int)
    le = LabelEncoder()
    train_df['label_code'] = le.fit_transform(train_df['label'])
    NUM_CLASSES = len(le.classes_)
    print(f"Classes ({NUM_CLASSES}): {le.classes_}")

    # --- FEATURE EXTRACTION ---
    # This is the most computationally expensive part
    print("\n>>> [2/5] Generating embeddings for Train...")
    train_imgs_dir = os.path.join(CONFIG['input_path'], 'train')
    
    # Check if images directory exists
    if not os.path.exists(train_imgs_dir):
        print(f"Error: Directory {train_imgs_dir} not found.")
        return

    X_features = extract_features(train_df, train_imgs_dir)
    y = train_df['label_code'].values

    # Clean up torch memory
    gc.collect()
    torch.cuda.empty_cache()

    # --- HOLDOUT SPLIT ---
    print("\n>>> [3/5] Data Split (Holdout)...")
    # Reserve 20% for final validation (stratified by sheep breed)
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X_features, y, test_size=0.2, random_state=CONFIG['seed'], stratify=y
    )
    
    print(f"   Train (Dev): {len(X_dev)} samples")
    print(f"   Holdout:     {len(X_holdout)} samples")
    
    # --- LGBM TRAINING (CV) ---
    print(f"\n>>> [4/5] Training LightGBM (Stratified K-Fold: {CONFIG['n_folds']})...")
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': NUM_CLASSES,
        'metric': 'multi_logloss', # Optimizing logloss, but monitoring F1
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': CONFIG['seed']
    }

    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    
    models = []
    oof_preds = np.zeros((len(X_dev), NUM_CLASSES))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
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
        
        # Predict on validation (probabilities)
        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_probs
        
        # Calculate current F1 Macro
        val_labels = np.argmax(val_probs, axis=1)
        score = f1_score(y_val, val_labels, average='macro')
        print(f"   Fold {fold+1}: F1 Macro = {score:.4f}")

    # Score on entire Dev set
    dev_labels = np.argmax(oof_preds, axis=1)
    dev_score = f1_score(y_dev, dev_labels, average='macro')
    print(f"   >>> CV Average F1 Macro: {dev_score:.4f}")

    # --- FINAL CHECK ON HOLDOUT ---
    print(f"\n>>> [5/5] Final Validation on Holdout (Unseen Data)...")
    
    # Ensemble: Average probabilities from 5 models
    holdout_probs = np.zeros((len(X_holdout), NUM_CLASSES))
    for model in models:
        holdout_probs += model.predict(X_holdout, num_iteration=model.best_iteration)
    holdout_probs /= CONFIG['n_folds']
    
    holdout_labels = np.argmax(holdout_probs, axis=1)
    holdout_score = f1_score(y_holdout, holdout_labels, average='macro')
    
    print("="*40)
    print(f"REAL VALIDATION SCORE (F1 Macro): {holdout_score:.4f}")
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()