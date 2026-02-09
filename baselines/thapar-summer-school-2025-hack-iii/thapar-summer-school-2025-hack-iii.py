import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# 1. Load data
df = pd.read_csv('train.csv')

target_col = 'output'
X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Train/validation split (optional)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = X.columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', model)
])

# 3. Train on split and evaluate
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_valid)
split_r2 = r2_score(y_valid, y_pred)

# 4. Final metric: 5-fold CV on full data
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()
print("R2 score:", round(cv_mean, 5))
