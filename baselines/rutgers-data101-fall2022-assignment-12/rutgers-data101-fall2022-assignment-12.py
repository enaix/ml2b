import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('train.csv')

# Time features extraction
df['starttime'] = pd.to_datetime(df['starttime'])
df['hour'] = df['starttime'].dt.hour
df['dayofweek'] = df['starttime'].dt.dayofweek
df['age'] = 2025 - df['birth.year']  # current year 2025 [file:1]

# Distance features (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c * 1000  # in meters

df['distance'] = haversine(df['start.station.latitude'], df['start.station.longitude'],
                           df['end.station.latitude'], df['end.station.longitude'])

# Select features
features = ['start.station.id', 'end.station.id', 'hour', 'dayofweek', 'usertype', 'gender', 'age', 'distance']
df['usertype'] = df['usertype'].map({'Subscriber': 0, 'Customer': 1})

X = df[features]
y = np.log1p(df['tripduration'])  # log transform for RMSE

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions on log scale
y_pred_log = model.predict(X_test)

# Final metric calculation (original scale RMSE)
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred_log)
rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
print(f'RMSE: {rmse_orig:.5f}')
