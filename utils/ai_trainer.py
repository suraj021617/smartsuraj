import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

DATA_FILE = "data/4d_training_ready.csv"
MODEL_FILE = "models/xgb_model.pkl"
LABEL_ENCODER_FILE = "models/label_encoder.pkl"

def train_ai():
    print("🚀 Training AI with XGBoost...")

    if not os.path.exists(DATA_FILE):
        print(f"❌ File not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    if 'grid' not in df.columns or 'next_win' not in df.columns:
        print("❌ CSV missing required 'grid' or 'next_win' columns.")
        return

    # Debug info
    print(f"📊 Loaded {len(df)} rows from CSV")

    # Convert grid string to list of integers
    def decode_grid(s):
        try:
            values = [int(x.strip()) for x in s.strip("[]").split(",") if x.strip().isdigit()]
            return values if len(values) == 16 else None
        except:
            return None

    df['grid'] = df['grid'].apply(decode_grid)
    df = df[df['grid'].notnull()]
    df = df[df['next_win'].notnull()]

    print(f"✅ {len(df)} valid rows after cleaning")

    if len(df) < 10:
        print("❌ Not enough valid samples to train.")
        return

    # Features and label
    X = np.array(df['grid'].tolist())
    y_raw = df['next_win'].astype(str)

    # Encode target labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"🎯 Model Accuracy: {accuracy:.4f}")

    # Save model and label encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoder, LABEL_ENCODER_FILE)
    print("💾 Model and label encoder saved to models/")

if __name__ == "__main__":
    train_ai()
