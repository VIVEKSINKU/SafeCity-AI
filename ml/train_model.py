import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load data
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'unified_crime_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records with {df['Crime_Type'].nunique()} crime categories")
    print(f"Categories: {df['Crime_Type'].unique().tolist()}")
    
    # Features — only use columns that carry real signal
    # Latitude & Longitude are strong spatial features
    # Hour provides temporal pattern (night crimes vs day crimes)
    FEATURES = ['Latitude', 'Longitude', 'Hour']
    
    X = df[FEATURES]
    y = df['Crime_Type']
    
    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Use more trees + balanced class weighting for small, imbalanced datasets
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.2%}")
    print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))
    
    # Feature importance
    print("Feature Importances:")
    for name, imp in zip(FEATURES, rf.feature_importances_):
        print(f"  {name:15s}: {imp:.3f}")
    
    # Save Model and Encoder
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Encoder saved to {ENCODER_PATH}")

if __name__ == "__main__":
    train()
