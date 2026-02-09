import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'img_size': 224,           # Standard for ResNet
    'batch_size': 64,          # Adjust based on GPU memory
    'epochs': 5,               # Number of passes through data
    'learning_rate': 0.001,
    'model_name': 'resnet18',  # Light and effective backbone
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 176         # From dataset description
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
# 1. DATASET CLASS
# ==========================================
class LeafDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Path is usually 'images/123.jpg' inside the CSV
        # We assume the script is running where 'images' folder exists
        img_path = self.df.iloc[idx]['image']
        label = self.df.iloc[idx]['label_code']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (CONFIG['img_size'], CONFIG['img_size']))

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

# ==========================================
# MAIN PIPELINE
# ==========================================
def run_pipeline():
    print(">>> [1/5] Loading data...")
    try:
        train_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return

    # --- PREPROCESSING ---
    print(">>> [2/5] Encoding Labels...")
    le = LabelEncoder()
    train_df['label_code'] = le.fit_transform(train_df['label'])
    
    # Verify classes
    num_classes = len(le.classes_)
    print(f"   Total Images: {len(train_df)}")
    print(f"   Unique Classes: {num_classes} (Expected: 176)")
    
    # --- HOLDOUT SPLIT ---
    print("\n>>> [3/5] Data Split (Holdout)...")
    # 80% Train, 20% Holdout Validation
    # Stratify is crucial here because we have 176 classes!
    train_split, holdout_split = train_test_split(
        train_df, test_size=0.2, random_state=CONFIG['seed'], stratify=train_df['label_code']
    )
    
    print(f"   Train Set:   {len(train_split)} images")
    print(f"   Holdout Set: {len(holdout_split)} images")

    # Create DataLoaders
    train_ds = LeafDataset(train_split, transform=get_transforms('train'))
    valid_ds = LeafDataset(holdout_split, transform=get_transforms('valid'))
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    # --- MODEL SETUP ---
    print(f"\n>>> [4/5] Initializing ResNet18 on {CONFIG['device']}...")
    model = models.resnet18(pretrained=True)
    
    # Modify the final Fully Connected layer to match our number of classes (176)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    model = model.to(CONFIG['device'])
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # --- TRAINING LOOP ---
    print(f"\n>>> [5/5] Training for {CONFIG['epochs']} Epochs...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training Phase
        loop = tqdm(train_loader, leave=False)
        for images, labels in loop:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            loop.set_description(f"Epoch [{epoch+1}/{CONFIG['epochs']}]")
            loop.set_postfix(loss=loss.item())
            
        train_acc = correct_train / total_train
        
        # Validation Phase (Holdout)
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f} | Holdout Acc={val_acc:.4f}")

    print("\n" + "="*40)
    print(f"FINAL HOLDOUT ACCURACY: {val_acc:.4f}")
    print("="*40)
    print("✓ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()