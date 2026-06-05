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

    bins = [0, 6, 10, 14, 18, 22, 24]
    labels = ['late_night', 'morning', 'midday', 'afternoon', 'evening', 'night']
    df['time_period'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False).astype(str)

    df['gh3'] = df['geohash'].str[:3]
    df['gh4'] = df['geohash'].str[:4]

    df['RoadType_enc'] = df['RoadType'].map({'Residential':0,'Main Road':1,'Highway':2})
    df['LargeVeh'] = (df['LargeVehicles']=='Allowed').astype(int)
    df['Landmark'] = (df['Landmarks']=='Yes').astype(int)
    weather_map = {'Sunny':0,'Cloudy':1,'Rainy':2,'Snowy':3}
    df['Weather_enc'] = df['Weather'].map(weather_map)

    df['veh_lanes'] = df['LargeVeh'] * df['NumberofLanes']
    df['road_lanes'] = df['RoadType_enc'].fillna(0) * df['NumberofLanes']
    df['land_veh'] = df['Landmark'] * df['LargeVeh']
    df['weather_temp'] = df['Weather_enc'].fillna(1) * df['Temperature'].fillna(df['Temperature'].median())
    return df

train = engineer(train)
test = engineer(test)

num_cols = ['hour', 'hour_sin', 'hour_cos', 'NumberofLanes', 'Temperature',
            'LargeVeh', 'Landmark', 'RoadType_enc', 'Weather_enc',
            'veh_lanes', 'road_lanes', 'land_veh', 'weather_temp']
cat_cols = ['geohash', 'gh3', 'gh4', 'time_period', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']

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
