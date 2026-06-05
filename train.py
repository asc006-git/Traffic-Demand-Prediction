import pandas as pd, numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

def engineer(df):
    df = df.copy()
    p = df['timestamp'].str.split(':', expand=True)
    df['hour'] = p[0].astype(int)
    df['minute'] = p[1].astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['gh3'] = df['geohash'].str[:3]
    df['gh4'] = df['geohash'].str[:4]
    return df

train = engineer(train)
test = engineer(test)

num_cols = ['hour', 'minute', 'hour_sin', 'hour_cos', 'NumberofLanes', 'Temperature']
cat_cols = ['geohash', 'gh3', 'gh4', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']

for c in num_cols:
    m = train[c].median()
    train[c] = train[c].fillna(m)
    test[c] = test[c].fillna(m)

for c in cat_cols:
    m = train[c].mode()[0]
    train[c] = train[c].fillna(m)
    test[c] = test[c].fillna(m)

print("Data preprocessing complete.")
print(f"Train: {train.shape}, Test: {test.shape}")

features = num_cols + cat_cols
X_train = train[features]
X_test = test[features]
cat_indices = [i for i, c in enumerate(features) if c in cat_cols]
