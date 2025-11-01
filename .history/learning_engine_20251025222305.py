"""
Learning Engine - Feedback Loop & Model Training
Learns from actual results and retrains model
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
import xgboost as xgb
from datetime import datetime
from collections import defaultdict, Counter

MODEL_FILE = 'master_model.pkl'
SCALER_FILE = 'master_scaler.pkl'
CSV_FILE = 'master_predictions.csv'

def add_actual_result(date, actual_numbers):
    """Add actual winning numbers"""
    if not os.path.exists(CSV_FILE):
        print("❌ No predictions file found")
        return
    
    df = pd.read_csv(CSV_FILE)
    mask = df['date'] == date
    
    if not mask.any():
        print(f"❌ No prediction found for {date}")
        return
    
    df.loc[mask, 'actual_numbers'] = ','.join(actual_numbers)
    df.to_csv(CSV_FILE, index=False)
    print(f"✅ Actual results added for {date}: {actual_numbers}")
    
    # Auto-learn
    learn_from_results()

def learn_from_results():
    """Calculate match counts and accuracy"""
    if not os.path.exists(CSV_FILE):
        return
    
    df = pd.read_csv(CSV_FILE)
    
    for idx, row in df.iterrows():
        if row['actual_numbers'] and not row['learned']:
            predicted = set(str(row['predicted_numbers']).split(','))
            actual = set(str(row['actual_numbers']).split(','))
            matches = len(predicted & actual)
            
            df.at[idx, 'match_count'] = matches
            df.at[idx, 'learned'] = True
            
            print(f"📊 Learned: {row['date']} - {matches} matches")
    
    df.to_csv(CSV_FILE, index=False)

def extract_ml_features(row, df=None):
    """Extract comprehensive features for ML - enhanced version"""
    features = []

    # Basic features
    features.append(float(row.get('hot_cold_score', 0)))
    features.append(float(row.get('confidence_score', 0)))

    # Gap pattern features
    gap_str = str(row.get('gap_pattern', '+0,+0'))
    gaps = [int(g.replace('+', '')) for g in gap_str.split(',') if g]
    features.extend(gaps[:2] + [0] * (2 - len(gaps)))

    # Pattern source features
    sources = str(row.get('pattern_source', '')).split('+')
    features.append(len(sources))

    # Digit frequency in predicted numbers
    predicted = str(row.get('predicted_numbers', '')).replace(',', '')
    for digit in range(10):
        features.append(predicted.count(str(digit)))

    # Advanced features from ml_predictor.py
    if df is not None and not df.empty:
        # Position-wise digit frequencies
        recent_df = df.tail(100)
        all_numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            nums = recent_df[col].astype(str).str.zfill(4).tolist()
            all_numbers.extend(nums)

        pos_freq = [{}, {}, {}, {}]
        for num in all_numbers:
            if len(num) == 4:
                for i, digit in enumerate(num):
                    pos_freq[i][digit] = pos_freq[i].get(digit, 0) + 1

        # Add position frequencies for predicted number
        if len(predicted) == 4:
            for pos in range(4):
                digit = predicted[pos]
                features.append(pos_freq[pos].get(digit, 0))

        # Transition probabilities
        transitions = defaultdict(int)
        for i in range(len(all_numbers) - 1):
            current = all_numbers[i]
            next_num = all_numbers[i + 1]
            if len(current) == 4 and len(next_num) == 4:
                for j in range(4):
                    trans_key = f"{current[j]}->{next_num[j]}"
                    transitions[trans_key] += 1

        trans_prob = 0
        if len(predicted) == 4:
            for pos in range(4):
                trans_key = f"{predicted[pos]}->{predicted[pos]}"
                trans_prob += transitions.get(trans_key, 0)
        features.append(trans_prob)

        # Day of week patterns
        dow_patterns = defaultdict(lambda: defaultdict(int))
        for _, r in recent_df.iterrows():
            dow = pd.to_datetime(r['date_parsed']).weekday()
            for col in ['1st_real', '2nd_real', '3rd_real']:
                num = str(r[col]).zfill(4)
                dow_patterns[dow][num] += 1

        current_dow = datetime.now().weekday()
        features.append(dow_patterns[current_dow].get(predicted, 0))

        # Statistical features
        if len(predicted) == 4:
            digits = [int(d) for d in predicted]
            features.extend([
                sum(digits),  # sum
                np.mean(digits),  # mean
                np.std(digits),  # std
                len(set(digits)),  # unique digits
                1 if len(set(digits)) < 4 else 0,  # has repeats
                1 if predicted == predicted[::-1] else 0,  # palindrome
                1 if all(digits[i] + 1 == digits[i+1] for i in range(3)) else 0  # sequential
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])  # padding

    return features

def train_model():
    """Train ML model from learned data"""
    if not os.path.exists(CSV_FILE):
        print("❌ No predictions file found")
        return None, None
    
    df = pd.read_csv(CSV_FILE)
    learned = df[df['learned'] == True]
    
    if len(learned) < 10:
        print(f"⚠️ Need at least 10 learned predictions (have {len(learned)})")
        return None, None
    
    print(f"🧠 Training model with {len(learned)} samples...")
    
    X = []
    y = []
    
    for idx, row in learned.iterrows():
        features = extract_ml_features(row)
        X.append(features)
        # Label: 1 if match_count >= 2, else 0
        y.append(1 if row['match_count'] >= 2 else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Calculate accuracy
    accuracy = model.score(X_scaled, y)
    print(f"✅ Model trained - Accuracy: {accuracy:.2%}")
    
    return model, scaler

def predict_with_model(features):
    """Use trained model to predict"""
    if not os.path.exists(MODEL_FILE):
        return None
    
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    
    features_scaled = scaler.transform([features])
    confidence = model.predict_proba(features_scaled)[0][1]
    
    return confidence

def view_stats():
    """View learning statistics"""
    if not os.path.exists(CSV_FILE):
        print("❌ No predictions yet")
        return
    
    df = pd.read_csv(CSV_FILE)
    learned = df[df['learned'] == True]
    
    if len(learned) == 0:
        print("❌ No learned predictions yet")
        return
    
    print("\n" + "="*60)
    print("📈 LEARNING STATISTICS")
    print("="*60)
    print(f"Total Predictions: {len(df)}")
    print(f"Learned: {len(learned)}")
    print(f"Pending: {len(df) - len(learned)}")
    print(f"\nAverage Matches: {learned['match_count'].mean():.2f}")
    print(f"Best Match: {learned['match_count'].max()}")
    print(f"Total Matches: {learned['match_count'].sum()}")
    
    # Pattern source analysis
    print("\n📊 Pattern Source Performance:")
    for source in learned['pattern_source'].unique():
        source_data = learned[learned['pattern_source'] == source]
        avg_matches = source_data['match_count'].mean()
        print(f"  {source}: {avg_matches:.2f} avg matches")
    
    # Recent performance
    print("\n📅 Recent Predictions:")
    recent = learned[['date', 'match_count', 'confidence_score', 'pattern_source']].tail(10)
    print(recent.to_string(index=False))
    
    print("="*60)

def retrain_model():
    """Retrain model with latest data"""
    print("🔄 Retraining model...")
    learn_from_results()
    model, scaler = train_model()
    if model:
        print("✅ Model retrained successfully")
    else:
        print("⚠️ Not enough data to retrain")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "add_result":
            if len(sys.argv) < 4:
                print("Usage: python learning_engine.py add_result YYYY-MM-DD 1234,5678,9012")
            else:
                date = sys.argv[2]
                nums = sys.argv[3].split(',')
                add_actual_result(date, nums)
        
        elif command == "train":
            train_model()
        
        elif command == "stats":
            view_stats()
        
        elif command == "retrain":
            retrain_model()
        
        else:
            print("Commands: add_result, train, stats, retrain")
    else:
        view_stats()
