import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

# Load data
df = pd.read_csv('train.csv')

# Assume columns: text content in unnamed column, label in last column
# From snippets, format like "564,1,htmlhead..." so first col ID, second Popularity (1/-1), third text
df.columns = ['id', 'Popularity', 'text']
df['Popularity'] = df['Popularity'].map({1:1, -1:0})  # Convert to binary 0/1 [file:1]

# Simple preprocessing - clean text
df['text'] = df['text'].str.replace(r'<[^>]+>', '', regex=True).str.lower()

# Feature engineering: TF-IDF on text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['Popularity']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Baseline model: Logistic Regression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print(f'Baseline AUC: {auc:.5f}')