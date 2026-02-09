import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from xgboost import XGBClassifier

# 1. Load data
train = pd.read_csv("train.csv")

# Target variable
target_col = "Status"   # C / CL / D
y = train[target_col].copy()
X = train.drop(columns=[target_col])

# 2. Handle missing values (simple baseline strategy)
# Numerical columns — fill with median
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
# Categorical columns — fill with mode
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# 3. LabelEncoding for categorical features
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# 4. LabelEncoding for target variable
y_le = LabelEncoder()
y_enc = y_le.fit_transform(y)

# Class list in correct order
classes_ = list(y_le.classes_)   # e.g. ['C', 'CL', 'D']

# 5. Train/Validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 6. XGBoost model (baseline hyperparameters)
model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(classes_),
    eval_metric="mlogloss",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)

model.fit(X_train, y_train)

# 7. Validation score (LogLoss)
probs_valid = model.predict_proba(X_valid)
val_logloss = log_loss(y_valid, probs_valid)
print(f"Validation LogLoss: {val_logloss:.5f}")

# 8. Final metric calculation
#    Cross-Validation LogLoss (5-fold, for more robust evaluation)
from sklearn.model_selection import StratifiedKFold

cv_scores = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y_enc)):
    X_fold_train, X_fold_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_fold_train, y_fold_valid = y_enc[train_idx], y_enc[valid_idx]

    model_cv = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes_),
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42
    )
    model_cv.fit(X_fold_train, y_fold_train)

    probs_cv = model_cv.predict_proba(X_fold_valid)
    score = log_loss(y_fold_valid, probs_cv)
    cv_scores.append(score)

print(f"CV LogLoss: {np.mean(cv_scores):.5f}")