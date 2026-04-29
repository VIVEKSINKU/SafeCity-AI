import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import joblib
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = os.path.dirname(__file__)
DATA_PATH    = os.path.join(BASE, '..', 'data', 'unified_crime_data.csv')
XGB_PATH     = os.path.join(BASE, 'xgb_model.pkl')
RF_PATH      = os.path.join(BASE, 'rf_model.pkl')
ENCODER_PATH = os.path.join(BASE, 'label_encoder.pkl')

CITY_LAT, CITY_LNG = 31.3260, 75.5762

FEATURES = [
    'Latitude', 'Longitude',
    'Hour', 'DayOfWeek', 'Month',
    'Is_Weekend', 'Is_Night', 'Is_Evening', 'Is_Morning',
    'Hour_sin', 'Hour_cos',
    'Month_sin', 'Month_cos',
    'Day_sin', 'Day_cos',
    'Dist_Centre',
    'Lat_Zone', 'Lng_Zone',
]

def engineer_features(df):
    df = df.copy()
    df['Is_Weekend']  = df['DayOfWeek'].apply(lambda d: 1 if d >= 5 else 0)
    df['Is_Night']    = df['Hour'].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)
    df['Is_Evening']  = df['Hour'].apply(lambda h: 1 if (18 <= h < 22) else 0)
    df['Is_Morning']  = df['Hour'].apply(lambda h: 1 if (6 <= h < 12) else 0)
    df['Hour_sin']    = np.sin(2 * np.pi * df['Hour']       / 24)
    df['Hour_cos']    = np.cos(2 * np.pi * df['Hour']       / 24)
    df['Month_sin']   = np.sin(2 * np.pi * df['Month']      / 12)
    df['Month_cos']   = np.cos(2 * np.pi * df['Month']      / 12)
    df['Day_sin']     = np.sin(2 * np.pi * df['DayOfWeek']  / 7)
    df['Day_cos']     = np.cos(2 * np.pi * df['DayOfWeek']  / 7)
    df['Dist_Centre'] = np.sqrt((df['Latitude'] - CITY_LAT)**2 + (df['Longitude'] - CITY_LNG)**2)
    df['Lat_Zone']    = pd.cut(df['Latitude'],  bins=6, labels=False).fillna(0).astype(int)
    df['Lng_Zone']    = pd.cut(df['Longitude'], bins=6, labels=False).fillna(0).astype(int)
    return df


def augment_data(df, target_per_class=200):
    """
    Augment sparse classes by adding small Gaussian noise to existing samples.
    This is domain-safe: a crime 10m away at the same hour is statistically similar.
    """
    augmented = [df]
    for crime_type, group in df.groupby('Crime_Type'):
        needed = max(0, target_per_class - len(group))
        if needed == 0:
            continue
        rng = np.random.RandomState(42)
        samples = group.sample(needed, replace=True, random_state=42).copy()
        samples['Latitude']  += rng.normal(0, 0.003, needed)   # ~300m noise
        samples['Longitude'] += rng.normal(0, 0.003, needed)
        samples['Hour']      = (samples['Hour'] + rng.randint(-1, 2, needed)) % 24
        augmented.append(samples)
    result = pd.concat(augmented, ignore_index=True)
    print(f"  Dataset after augmentation: {len(result)} records")
    return result


def train():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records | {df['Crime_Type'].nunique()} crime categories")
    print(f"Class distribution:\n{df['Crime_Type'].value_counts().to_string()}\n")

    # Step 1: augment rare classes
    print("Augmenting sparse classes to >= 200 samples each ...")
    df = augment_data(df, target_per_class=200)

    # Step 2: feature engineering
    df = engineer_features(df)

    X = df[FEATURES].fillna(0)
    y = df['Crime_Type']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # SMOTE inside cross-val via imblearn Pipeline (no leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── XGBoost ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Tuning XGBoost with Optuna (25 trials) ...")

    def xgb_objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'max_depth':        trial.suggest_int('max_depth', 3, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'gamma':            trial.suggest_float('gamma', 0, 3),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1, 4),
        }
        min_cnt = pd.Series(y_train).value_counts().min()
        k = max(1, min(5, min_cnt - 1))
        pipe = ImbPipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=k)),
            ('clf',   XGBClassifier(**params, eval_metric='mlogloss',
                                    random_state=42, n_jobs=-1)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring='accuracy', n_jobs=1)
        return scores.mean()

    xgb_study = optuna.create_study(direction='maximize')
    xgb_study.optimize(xgb_objective, n_trials=25)
    print(f"  Best XGB CV accuracy : {xgb_study.best_value:.2%}")

    # Train final XGB on full training set (with SMOTE)
    min_cnt = pd.Series(y_train).value_counts().min()
    k = max(1, min(5, min_cnt - 1))
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    best_xgb = XGBClassifier(**xgb_study.best_params, eval_metric='mlogloss',
                              random_state=42, n_jobs=-1)
    best_xgb.fit(X_res, y_res)
    xgb_acc = accuracy_score(y_test, best_xgb.predict(X_test))
    print(f"  XGBoost Test Accuracy: {xgb_acc:.2%}")
    print(classification_report(y_test, best_xgb.predict(X_test),
                                target_names=le.classes_, zero_division=0))

    # ── Random Forest ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Tuning Random Forest with Optuna (20 trials) ...")

    def rf_objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 200, 600),
            'max_depth':         trial.suggest_int('max_depth', 10, 40),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 4),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }
        min_cnt = pd.Series(y_train).value_counts().min()
        k = max(1, min(5, min_cnt - 1))
        pipe = ImbPipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=k)),
            ('clf',   RandomForestClassifier(**params, class_weight='balanced_subsample',
                                             random_state=42, n_jobs=-1)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring='accuracy', n_jobs=1)
        return scores.mean()

    rf_study = optuna.create_study(direction='maximize')
    rf_study.optimize(rf_objective, n_trials=20)
    print(f"  Best RF CV accuracy  : {rf_study.best_value:.2%}")

    best_rf = RandomForestClassifier(**rf_study.best_params,
                                     class_weight='balanced_subsample',
                                     random_state=42, n_jobs=-1)
    best_rf.fit(X_res, y_res)
    rf_acc = accuracy_score(y_test, best_rf.predict(X_test))
    print(f"  RF Test Accuracy     : {rf_acc:.2%}")
    print(classification_report(y_test, best_rf.predict(X_test),
                                target_names=le.classes_, zero_division=0))

    # ── Save both models ───────────────────────────────────────────────────────
    print("=" * 60)
    print(f"XGBoost  test accuracy : {xgb_acc:.2%}")
    print(f"RandomForest test acc  : {rf_acc:.2%}")

    joblib.dump(best_xgb, XGB_PATH)
    joblib.dump(best_rf,  RF_PATH)
    joblib.dump(le,       ENCODER_PATH)

    winner = "XGBoost" if xgb_acc >= rf_acc else "Random Forest"
    print(f"\nWINNER: {winner}")
    print(f"  xgb_model.pkl  saved")
    print(f"  rf_model.pkl   saved")
    print(f"  label_encoder  saved")

    print("\nXGBoost Feature Importances (top 10):")
    fi = sorted(zip(FEATURES, best_xgb.feature_importances_),
                key=lambda x: x[1], reverse=True)[:10]
    for name, imp in fi:
        print(f"  {name:20s}: {imp:.4f}")


if __name__ == '__main__':
    train()
