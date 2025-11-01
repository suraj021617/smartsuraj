"""
Auto-Learning Prediction System
Predicts tomorrow's numbers, saves them, learns from results, and improves daily
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

# File paths
PREDICTIONS_CSV = 'daily_predictions.csv'
MODEL_FILE = 'learning_model.pkl'
SCALER_FILE = 'scaler.pkl'

def initialize_prediction_csv():
    """Create prediction CSV if it doesn't exist"""
    if not os.path.exists(PREDICTIONS_CSV):
        df = pd.DataFrame(columns=[
            'prediction_date', 'draw_date', 'predicted_numbers', 
            'actual_numbers', 'matches', 'accuracy', 'learned',
            'hot_cold_score', 'gap_pattern', 'position_pattern'
        ])
        df.to_csv(PREDICTIONS_CSV, index=False)

def load_historical_data():
    """Load historical lottery data"""
    df = pd.read_csv('4d_results_history.csv')
    df['date_parsed'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.dropna(subset=['date_parsed']).sort_values('date_parsed')
    return df

def extract_features(df, lookback=30):
    """Extract advanced features from recent draws for ML"""
    recent = df.tail(lookback)
    
    features = []
    # Feature 1-10: Digit frequency (0-9)
    all_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_nums.extend([n for n in recent[col].astype(str) if n.isdigit() and len(n) == 4])
    
    digit_freq = [sum(1 for num in all_nums for d in num if d == str(i)) for i in range(10)]
    features.extend(digit_freq)
    
    # Feature 11-13: Average gaps
    if len(all_nums) >= 3:
        last_3 = [int(n) for n in all_nums[-3:]]
        gaps = [last_3[i+1] - last_3[i] for i in range(len(last_3)-1)]
        features.extend(gaps + [0] * (3 - len(gaps)))
    else:
        features.extend([0, 0, 0])
    
    # Feature 14: Day of week
    features.append(df.iloc[-1]['date_parsed'].dayofweek)
    
    # Feature 15: Hot/Cold score
    from collections import Counter
    freq = Counter(all_nums)
    hot_cold_score = len([n for n, c in freq.most_common(10)]) / len(freq) if freq else 0
    features.append(hot_cold_score)
    
    # Feature 16-17: Position patterns (Box1, Box2 most common)
    box1_nums = [n for n in recent['1st_real'].astype(str) if n.isdigit() and len(n) == 4]
    box2_nums = [n for n in recent['2nd_real'].astype(str) if n.isdigit() and len(n) == 4]
    box1_freq = Counter(box1_nums).most_common(1)[0][1] if box1_nums else 0
    box2_freq = Counter(box2_nums).most_common(1)[0][1] if box2_nums else 0
    features.extend([box1_freq, box2_freq])
    
    return features

def predict_tomorrow_numbers(df, model=None, scaler=None):
    """Generate predictions for tomorrow"""
    if model and scaler:
        # Use trained model
        features = extract_features(df)
        features_scaled = scaler.transform([features])
        predictions = model.predict_proba(features_scaled)[0]
        top_indices = predictions.argsort()[-6:][::-1]
        predicted_numbers = [f"{i:04d}" for i in top_indices]
    else:
        # Use simple frequency-based prediction
        all_nums = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            all_nums.extend([n for n in df[col].astype(str).tail(50) if n.isdigit() and len(n) == 4])
        
        from collections import Counter
        freq = Counter(all_nums)
        predicted_numbers = [num for num, _ in freq.most_common(6)]
    
    return predicted_numbers[:6]

def calculate_hot_cold_score(df):
    """Calculate hot/cold score for current state"""
    from collections import Counter
    all_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_nums.extend([n for n in df[col].astype(str).tail(30) if n.isdigit() and len(n) == 4])
    freq = Counter(all_nums)
    return len([n for n, c in freq.most_common(10)]) / len(freq) if freq else 0

def calculate_gap_pattern(predicted_numbers):
    """Calculate gap pattern from predicted numbers"""
    nums = sorted([int(n) for n in predicted_numbers[:3]])
    gaps = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
    return ','.join([f"+{g}" for g in gaps])

def calculate_position_pattern(predicted_numbers):
    """Calculate position pattern"""
    return f"Box1={predicted_numbers[0]},Box2={predicted_numbers[1]}"

def save_prediction(predicted_numbers, hot_cold_score, gap_pattern, position_pattern):
    """Save today's prediction with advanced features"""
    initialize_prediction_csv()
    df = pd.read_csv(PREDICTIONS_CSV)
    
    new_row = {
        'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'draw_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'predicted_numbers': ','.join(predicted_numbers),
        'actual_numbers': '',
        'matches': 0,
        'accuracy': 0.0,
        'learned': False,
        'hot_cold_score': round(hot_cold_score, 4),
        'gap_pattern': gap_pattern,
        'position_pattern': position_pattern
    }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"✅ Prediction saved for {new_row['draw_date']}: {predicted_numbers}")
    print(f"   Hot/Cold Score: {hot_cold_score:.4f}")
    print(f"   Gap Pattern: {gap_pattern}")
    print(f"   Position Pattern: {position_pattern}")

def add_actual_results(draw_date, actual_numbers):
    """Add actual winning numbers after draw"""
    df = pd.read_csv(PREDICTIONS_CSV)
    
    mask = df['draw_date'] == draw_date
    if mask.any():
        df.loc[mask, 'actual_numbers'] = ','.join(actual_numbers)
        df.to_csv(PREDICTIONS_CSV, index=False)
        print(f"✅ Actual results added for {draw_date}: {actual_numbers}")
    else:
        print(f"❌ No prediction found for {draw_date}")

def learn_from_results():
    """Compare predictions vs actuals and calculate accuracy"""
    df = pd.read_csv(PREDICTIONS_CSV)
    
    for idx, row in df.iterrows():
        if row['actual_numbers'] and not row['learned']:
            predicted = set(row['predicted_numbers'].split(','))
            actual = set(row['actual_numbers'].split(','))
            matches = len(predicted & actual)
            accuracy = (matches / len(predicted)) * 100
            
            df.at[idx, 'matches'] = matches
            df.at[idx, 'accuracy'] = round(accuracy, 2)
            df.at[idx, 'learned'] = True
            
            print(f"📊 Learned: {row['draw_date']} - {matches} matches ({accuracy:.1f}% accuracy)")
    
    df.to_csv(PREDICTIONS_CSV, index=False)

def train_ml_model():
    """Train ML model from learned data"""
    pred_df = pd.read_csv(PREDICTIONS_CSV)
    hist_df = load_historical_data()
    
    learned = pred_df[pred_df['learned'] == True]
    if len(learned) < 10:
        print("⚠️ Need at least 10 learned predictions to train model")
        return None, None
    
    X, y = [], []
    for idx, row in learned.iterrows():
        draw_date = pd.to_datetime(row['draw_date'])
        hist_before = hist_df[hist_df['date_parsed'] < draw_date]
        
        if len(hist_before) < 30:
            continue
        
        features = extract_features(hist_before)
        actual_nums = row['actual_numbers'].split(',')
        
        for num in actual_nums:
            if num.isdigit():
                X.append(features)
                y.append(int(num))
    
    if len(X) < 10:
        print("⚠️ Insufficient training data")
        return None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Model trained with {len(X)} samples")
    return model, scaler

def load_trained_model():
    """Load existing trained model"""
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None

def auto_predict_daily():
    """Main function: Predict tomorrow's numbers automatically"""
    print("🚀 Starting Auto-Prediction System...")
    
    # Step 1: Load data
    df = load_historical_data()
    print(f"📊 Loaded {len(df)} historical draws")
    
    # Step 2: Learn from previous predictions
    learn_from_results()
    
    # Step 3: Train/load model
    model, scaler = load_trained_model()
    if not model:
        print("🧠 Training new model...")
        model, scaler = train_ml_model()
    
    # Step 4: Predict tomorrow
    predicted = predict_tomorrow_numbers(df, model, scaler)
    
    # Step 5: Calculate advanced features
    hot_cold_score = calculate_hot_cold_score(df)
    gap_pattern = calculate_gap_pattern(predicted)
    position_pattern = calculate_position_pattern(predicted)
    
    # Step 6: Save prediction with features
    save_prediction(predicted, hot_cold_score, gap_pattern, position_pattern)
    
    print(f"\n🎯 Tomorrow's Predictions: {predicted}")
    print(f"💾 Saved to {PREDICTIONS_CSV}")

def view_performance():
    """View prediction performance stats with advanced metrics"""
    if not os.path.exists(PREDICTIONS_CSV):
        print("❌ No predictions yet")
        return
    
    df = pd.read_csv(PREDICTIONS_CSV)
    learned = df[df['learned'] == True]
    
    if len(learned) == 0:
        print("❌ No learned predictions yet")
        return
    
    print("\n📈 PREDICTION PERFORMANCE")
    print("=" * 50)
    print(f"Total Predictions: {len(df)}")
    print(f"Learned: {len(learned)}")
    print(f"Average Accuracy: {learned['accuracy'].mean():.2f}%")
    print(f"Best Match: {learned['matches'].max()} numbers")
    print(f"Total Matches: {learned['matches'].sum()}")
    
    if 'hot_cold_score' in learned.columns:
        print(f"\nAverage Hot/Cold Score: {learned['hot_cold_score'].mean():.4f}")
        high_score_wins = learned[learned['hot_cold_score'] > 0.5]
        if len(high_score_wins) > 0:
            print(f"High Score Win Rate: {(high_score_wins['matches'] >= 2).sum() / len(high_score_wins) * 100:.1f}%")
    
    print("\nRecent Predictions:")
    cols = ['draw_date', 'matches', 'accuracy']
    if 'hot_cold_score' in learned.columns:
        cols.append('hot_cold_score')
    print(learned[cols].tail(10).to_string(index=False))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "predict":
            auto_predict_daily()
        
        elif command == "add_result":
            if len(sys.argv) < 4:
                print("Usage: python auto_predictor.py add_result YYYY-MM-DD 1234,5678,9012")
            else:
                draw_date = sys.argv[2]
                actual_nums = sys.argv[3].split(',')
                add_actual_results(draw_date, actual_nums)
                learn_from_results()
        
        elif command == "train":
            train_ml_model()
        
        elif command == "stats":
            view_performance()
        
        elif command == "retrain":
            print("🔄 Retraining model with latest data...")
            learn_from_results()
            train_ml_model()
            print("✅ Model retrained successfully")
        
        else:
            print("Commands: predict, add_result, train, stats, retrain")
    else:
        # Default: auto predict
        auto_predict_daily()
