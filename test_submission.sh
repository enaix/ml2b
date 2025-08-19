#!/bin/bash
set -e

cat > ./python/submission/code.py << EOL
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
def train_and_predict(X_train, y_train, X_test):
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled = scaler.transform(X_test[numeric_cols])
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model.predict_proba(X_test_scaled)[:, 1]
EOL

echo "" > ./python/submission/__init__.py

export COMPETITION_ID="wids-datathon-2020"
export BENCH_LANG="English"

# Execute
docker compose run bench_python

cat ./python/submission/results.json
