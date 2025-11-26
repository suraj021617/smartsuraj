"""
Machine Learning-based 4D Prediction Engine
Uses XGBoost and statistical analysis for better predictions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    def __init__(self, model_path='models/4d_xgboost_model.joblib', encoder_path='models/label_encoder.pkl'):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.encoder = None
        self.load_model()

    def load_model(self):
        """Load pre-trained model and encoder"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            if os.path.exists(self.encoder_path):
                self.encoder = joblib.load(self.encoder_path)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.model = None
            self.encoder = None

    def extract_features(self, df, current_number=None):
        """Extract comprehensive features from historical data"""
        features = []

        if df.empty:
            return pd.DataFrame()

        # Basic statistics
        recent_df = df.tail(100)  # Last 100 draws

        # Frequency analysis
        all_numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            nums = recent_df[col].astype(str).str.zfill(4).tolist()
            all_numbers.extend(nums)

        freq_counter = Counter(all_numbers)

        # Position-wise digit frequencies
        pos_freq = [{}, {}, {}, {}]
        for num in all_numbers:
            if len(num) == 4:
                for i, digit in enumerate(num):
                    pos_freq[i][digit] = pos_freq[i].get(digit, 0) + 1

        # Recent trends (last 10 draws)
        last_10 = df.tail(10)
        recent_numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            nums = last_10[col].astype(str).str.zfill(4).tolist()
            recent_numbers.extend(nums)

        # Calculate transitions between consecutive draws
        transitions = defaultdict(int)
        for i in range(len(recent_numbers) - 1):
            current = recent_numbers[i]
            next_num = recent_numbers[i + 1]
            if len(current) == 4 and len(next_num) == 4:
                for j in range(4):
                    trans_key = f"{current[j]}->{next_num[j]}"
                    transitions[trans_key] += 1

        # Day of week patterns
        dow_patterns = defaultdict(lambda: defaultdict(int))
        for _, row in recent_df.iterrows():
            dow = pd.to_datetime(row['date_parsed']).weekday()
            for col in ['1st_real', '2nd_real', '3rd_real']:
                num = str(row[col]).zfill(4)
                dow_patterns[dow][num] += 1

        # Generate candidate features
        candidates = set()

        # 1. Most frequent numbers
        candidates.update([num for num, _ in freq_counter.most_common(20)])

        # 2. Recent numbers (last 5 draws)
        candidates.update(recent_numbers[-15:])  # Last 5 draws * 3 prizes

        # 3. Numbers similar to recent ones (digit transitions)
        for recent_num in recent_numbers[-3:]:  # Last 3 numbers
            if len(recent_num) == 4:
                # Generate variations
                for pos in range(4):
                    for digit in '0123456789':
                        if digit != recent_num[pos]:
                            variation = list(recent_num)
                            variation[pos] = digit
                            candidates.add(''.join(variation))

        # 4. Hot digits per position
        for pos in range(4):
            hot_digits = sorted(pos_freq[pos].items(), key=lambda x: x[1], reverse=True)[:3]
            for digit, _ in hot_digits:
                # Create numbers with this hot digit in this position
                for base in recent_numbers[-5:]:
                    if len(base) == 4:
                        variation = list(base)
                        variation[pos] = digit
                        candidates.add(''.join(variation))

        # Convert to feature vectors
        feature_list = []
        for candidate in list(candidates)[:200]:  # Limit to top candidates
            if len(candidate) != 4 or not candidate.isdigit():
                continue

            features_dict = {
                'candidate': candidate,
                'freq_score': freq_counter.get(candidate, 0),
                'is_recent': 1 if candidate in recent_numbers[-10:] else 0,
                'recency': max([i for i, num in enumerate(recent_numbers) if num == candidate] or [-100]),
            }

            # Position digit frequencies
            for pos in range(4):
                digit = candidate[pos]
                features_dict[f'pos_{pos}_freq'] = pos_freq[pos].get(digit, 0)

            # Transition probabilities
            trans_prob = 0
            if current_number and len(current_number) == 4:
                for pos in range(4):
                    trans_key = f"{current_number[pos]}->{candidate[pos]}"
                    trans_prob += transitions.get(trans_key, 0)
            features_dict['transition_prob'] = trans_prob

            # Day of week score
            current_dow = datetime.now().weekday()
            features_dict['dow_score'] = dow_patterns[current_dow].get(candidate, 0)

            # Statistical features
            digits = [int(d) for d in candidate]
            features_dict.update({
                'sum_digits': sum(digits),
                'mean_digit': np.mean(digits),
                'std_digit': np.std(digits),
                'unique_digits': len(set(digits)),
                'has_repeats': 1 if len(set(digits)) < 4 else 0,
                'is_palindrome': 1 if candidate == candidate[::-1] else 0,
                'is_sequential': 1 if all(digits[i] + 1 == digits[i+1] for i in range(3)) else 0,
            })

            feature_list.append(features_dict)

        return pd.DataFrame(feature_list)

    def predict_top_numbers(self, df, top_n=6):
        """Predict top N numbers using ML model"""
        if self.model is None:
            # Fallback to frequency-based prediction
            return self.fallback_prediction(df, top_n)

        try:
            # Extract features
            features_df = self.extract_features(df)

            if features_df.empty:
                return self.fallback_prediction(df, top_n)

            # Prepare features for prediction
            feature_cols = [col for col in features_df.columns if col not in ['candidate']]
            X = features_df[feature_cols]

