import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings, time
warnings.filterwarnings('ignore')

t0 = time.time()
print("Loading data...")
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
sub = pd.read_csv('dataset/sample_submission.csv')

# FEATURE ENGINEERING

def add_features(df):
    df = df.copy()
    parts = df['timestamp'].str.split(':', expand=True)
    df['hour'] = parts[0].astype(int)
    df['minute'] = parts[1].astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    bins = [0, 6, 10, 14, 18, 22, 24]
    labels = ['late_night', 'morning', 'midday', 'afternoon', 'evening', 'night']
    df['time_period'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False).astype(str)

    # Geohash prefixes
    df['gh3'] = df['geohash'].str[:3]
    df['gh4'] = df['geohash'].str[:4]

    # Encodings
    df['RoadType_enc'] = df['RoadType'].map({'Residential':0,'Main Road':1,'Highway':2})
    df['LargeVeh'] = (df['LargeVehicles']=='Allowed').astype(int)
    df['Landmark'] = (df['Landmarks']=='Yes').astype(int)
    weather_map = {'Sunny':0,'Cloudy':1,'Rainy':2,'Snowy':3}
    df['Weather_enc'] = df['Weather'].map(weather_map)

    # Interactions
    df['veh_lanes'] = df['LargeVeh'] * df['NumberofLanes']
    df['road_lanes'] = df['RoadType_enc'].fillna(0) * df['NumberofLanes']
    df['land_veh'] = df['Landmark'] * df['LargeVeh']
    df['weather_temp'] = df['Weather_enc'].fillna(1) * df['Temperature'].fillna(df['Temperature'].median())
    return df

print("Feature engineering...")
train = add_features(train)
test = add_features(test)


# FILL MISSING

cat_cols = ['geohash','gh3','gh4','time_period','RoadType','Weather']
num_cols = ['hour','hour_sin','hour_cos','NumberofLanes','Temperature',
            'LargeVeh','Landmark','RoadType_enc','Weather_enc',
            'veh_lanes','road_lanes','land_veh','weather_temp']

for c in num_cols:
    if train[c].isnull().sum() > 0:
        v = train[c].median()
        train[c] = train[c].fillna(v)
        test[c] = test[c].fillna(v)

for c in cat_cols:
    if train[c].isnull().sum() > 0:
        v = train[c].mode()[0]
        train[c] = train[c].fillna(v)
        test[c] = test[c].fillna(v)

 
# TARGET
 
y = np.log1p(train['demand'].values)
print(f"Target skew: {train['demand'].skew():.4f} -> log-skew: {pd.Series(y).skew():.4f}")

 
# FEATURES
 
feature_cols = num_cols + cat_cols
X_train = train[feature_cols]
X_test = test[feature_cols]

cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_cols]
print(f"Features: {len(feature_cols)}, Cat indices: {cat_indices}")

 
# CROSS-VALIDATION (3 folds, faster)
 
print("\n=== CatBoost 3-Fold CV ===")
n_folds = 3
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
cv_scores, models = [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\nFold {fold+1}/{n_folds}")
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.07,
        depth=8,
        l2_leaf_reg=5,
        border_count=128,
        cat_features=cat_indices,
        loss_function='RMSE',
        random_seed=42+fold,
        verbose=200,
        early_stopping_rounds=80,
        thread_count=-1,
        subsample=0.8,
        min_data_in_leaf=5,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose_eval=200, plot=False)

    preds = np.expm1(model.predict(X_val))
    actuals = np.expm1(y_val)
    score = r2_score(actuals, preds)
    cv_scores.append(score)
    models.append(model)
    print(f"Fold {fold+1} R²: {score:.6f}")

print(f"\n=== CV Results ===")
print(f"Mean R²: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")

 
# ENSEMBLE PREDICT
 
print("\nEnsembling fold models on test...")
ensemble_preds = np.zeros(len(X_test))
for model in models:
    ensemble_preds += np.expm1(model.predict(X_test))
ensemble_preds = np.maximum(
    ensemble_preds / len(models),
    0
)

submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': ensemble_preds
})

submission.to_csv('submission.csv', index=False)

print("submission.csv generated successfully")
print(submission.head())

print(f"Submission saved: min={ensemble_preds.min():.6f}, max={ensemble_preds.max():.6f}, mean={ensemble_preds.mean():.6f}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")

# Feature importance
fi = pd.DataFrame({'feature':feature_cols,'importance':models[0].feature_importances_}).sort_values('importance',ascending=False)
print("\nTop 15 features:")
print(fi.head(15).to_string())

print("\nDone!")
