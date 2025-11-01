"""
POWER PREDICTOR - Enhanced ML System
Combines multiple advanced algorithms with confidence scoring
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class PowerPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def extract_features(self, number):
        """Extract 30+ features from a 4D number"""
        if not isinstance(number, str) or len(number) != 4:
            return None
            
        digits = [int(d) for d in number]
        
        features = {
            # Basic features
            'digit_sum': sum(digits),
            'digit_product': np.prod(digits),
            'digit_mean': np.mean(digits),
            'digit_std': np.std(digits),
            
            # Individual digits
            'd1': digits[0], 'd2': digits[1], 'd3': digits[2], 'd4': digits[3],
            
            # Pairs
            'pair_12': digits[0] * 10 + digits[1],
            'pair_23': digits[1] * 10 + digits[2],
            'pair_34': digits[2] * 10 + digits[3],
            
            # Odd/Even
            'odd_count': sum(1 for d in digits if d % 2 == 1),
            'even_count': sum(1 for d in digits if d % 2 == 0),
            
            # High/Low (5-9 is high, 0-4 is low)
            'high_count': sum(1 for d in digits if d >= 5),
            'low_count': sum(1 for d in digits if d < 5),
            
            # Patterns
            'is_ascending': int(all(digits[i] <= digits[i+1] for i in range(3))),
            'is_descending': int(all(digits[i] >= digits[i+1] for i in range(3))),
            'has_repeat': int(len(set(digits)) < 4),
            'all_unique': int(len(set(digits)) == 4),
            
            # Divisibility
            'div_by_2': int(int(number) % 2 == 0),
            'div_by_3': int(int(number) % 3 == 0),
            'div_by_5': int(int(number) % 5 == 0),
            'div_by_7': int(int(number) % 7 == 0),
            
            # Ranges
            'in_range_0_2500': int(int(number) < 2500),
            'in_range_2500_5000': int(2500 <= int(number) < 5000),
            'in_range_5000_7500': int(5000 <= int(number) < 7500),
            'in_range_7500_10000': int(int(number) >= 7500),
            
            # Digit differences
            'diff_12': abs(digits[0] - digits[1]),
            'diff_23': abs(digits[1] - digits[2]),
            'diff_34': abs(digits[2] - digits[3]),
            'max_diff': max(abs(digits[i] - digits[i+1]) for i in range(3)),
        }
        
        return list(features.values())
    
    def train_models(self, historical_numbers, lookback=500):
        """Train multiple ML models on historical data"""
        if len(historical_numbers) < 50:
            return False
            
        # Prepare training data
        X, y = [], []
        for i in range(len(historical_numbers) - 1):
            features = self.extract_features(historical_numbers[i])
            if features:
                X.append(features)
                # Target: will this number appear in next 10 draws?
                next_10 = historical_numbers[i+1:i+11]
                y.append(1 if historical_numbers[i] in next_10 else 0)
        
        if len(X) < 20:
            return False
            
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.models['random_forest'].fit(X_scaled, y)
        
        # Train Gradient Boosting
        self.models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.models['gradient_boost'].fit(X_scaled, y)
        
        return True
    
    def predict_with_confidence(self, candidate_numbers, historical_numbers):
        """
        Predict with confidence scores
        Returns: [(number, confidence, method), ...]
        """
        if not self.models:
            self.train_models(historical_numbers)
        
        predictions = []
        
        for number in candidate_numbers:
            features = self.extract_features(number)
            if not features:
                continue
                
            features_scaled = self.scaler.transform([features])
            
            # Get predictions from all models
            confidences = []
            
            if 'random_forest' in self.models:
                rf_prob = self.models['random_forest'].predict_proba(features_scaled)[0][1]
                confidences.append(rf_prob)
            
            if 'gradient_boost' in self.models:
                gb_prob = self.models['gradient_boost'].predict_proba(features_scaled)[0][1]
                confidences.append(gb_prob)
            
            # Average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.5
            
            # Boost confidence based on frequency
            freq_boost = historical_numbers[-100:].count(number) / 100
            final_confidence = min(avg_confidence + freq_boost * 0.2, 1.0)
            
            predictions.append((number, final_confidence, 'PowerML'))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:20]
    
    def get_feature_importance(self):
        """Get which features are most important"""
        if 'random_forest' not in self.models:
            return {}
            
        importance = self.models['random_forest'].feature_importances_
        feature_names = [
            'sum', 'product', 'mean', 'std', 'd1', 'd2', 'd3', 'd4',
            'pair_12', 'pair_23', 'pair_34', 'odd_count', 'even_count',
            'high_count', 'low_count', 'ascending', 'descending',
            'has_repeat', 'all_unique', 'div_2', 'div_3', 'div_5', 'div_7',
            'range_0_2500', 'range_2500_5000', 'range_5000_7500', 'range_7500_10000',
            'diff_12', 'diff_23', 'diff_34', 'max_diff'
        ]
        
        return dict(zip(feature_names, importance))


def enhanced_predictor(df, provider='all', lookback=300):
    """
    Enhanced predictor using PowerPredictor
    """
    # Get historical numbers
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        if col in df.columns:
            nums = [n for n in df[col].astype(str) if len(n) == 4 and n.isdigit()]
            all_numbers.extend(nums)
    
    if len(all_numbers) < 50:
        return []
    
    # Use last 'lookback' numbers
    recent_numbers = all_numbers[-lookback:]
    
    # Get candidate pool (unique numbers from recent history)
    candidate_pool = list(set(recent_numbers[-200:]))
    
    # Initialize PowerPredictor
    predictor = PowerPredictor()
    
    # Get predictions with confidence
    predictions = predictor.predict_with_confidence(candidate_pool, recent_numbers)
    
    return predictions[:10]
