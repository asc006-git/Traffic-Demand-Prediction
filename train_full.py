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

y = np.log1p(train['demand'].values)
features = num_cols + cat_cols
X_train, X_test = train[features], test[features]
cat_indices = [i for i, c in enumerate(features) if c in cat_cols]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = []

for fold, (tr, va) in enumerate(kf.split(X_train)):
    m = CatBoostRegressor(
        iterations=1500, learning_rate=0.07, depth=8, l2_leaf_reg=5,
        cat_features=cat_indices, random_seed=42 + fold,
        verbose=0, early_stopping_rounds=80, thread_count=-1, subsample=0.8
    )
    m.fit(X_train.iloc[tr], y[tr], eval_set=(X_train.iloc[va], y[va]), use_best_model=True)
    models.append(m)
    r2 = r2_score(np.expm1(y[va]), np.expm1(m.predict(X_train.iloc[va])))
    print(f'Fold {fold+1} R2: {r2:.6f}')

preds = np.clip(np.expm1(np.mean([m.predict(X_test) for m in models], axis=0)), 0, 1)
pd.DataFrame({'Index': test['Index'], 'demand': preds}).to_csv('submission.csv', index=False)
print(f'CV mean R2: {np.mean([r2_score(np.expm1(y[va]), np.expm1(models[i].predict(X_train.iloc[va]))) for i, (tr, va) in enumerate(kf.split(X_train))]):.6f}')
