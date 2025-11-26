"""Unified Predictor - Combines all 4 AI methods for all providers"""
from collections import Counter
from utils.frequency_analyzer import analyze_frequency
from utils.day_to_day_learner import learn_day_to_day_patterns, predict_tomorrow
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from itertools import product
import numpy as np

def get_all_predictions(df):
    """Get predictions from all 4 methods for all providers"""
    providers = ['all'] + sorted(df['provider'].dropna().unique().tolist())
    all_predictions = {}
    
    for provider in providers:
        provider_df = df[df['provider'] == provider] if provider != 'all' else df
        all_predictions[provider] = {
            'frequency': get_frequency_predictions(provider_df),
            'day_to_day': get_day_to_day_predictions(provider_df),
            'auto_weight': get_auto_weight_predictions(provider_df),
            'ml': get_ml_predictions(provider_df)
        }
    
    return all_predictions

def get_frequency_predictions(df):
    """Frequency analyzer - REAL hot numbers from CSV"""
    all_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums = df[col].astype(str).tolist()
        all_nums.extend([n for n in nums if len(n) == 4 and n.isdigit()])
    if not all_nums:
        return []
    freq = Counter(all_nums)
    return [num for num, _ in freq.most_common(5)]

def get_day_to_day_predictions(df):
    """Day-to-day pattern - learns consecutive day transitions (includes special + consolation)"""
    if len(df) < 2:
        return get_frequency_predictions(df)
    patterns = learn_day_to_day_patterns(df)
    latest_date = df['date_parsed'].max()
    today_data = df[df['date_parsed'] == latest_date]
    today_numbers = [str(n) for col in ['1st_real', '2nd_real', '3rd_real'] for n in today_data[col].tolist() if len(str(n)) == 4 and str(n).isdigit()]
    for _, row in today_data.iterrows():
        if 'special' in row and row['special']:
            for num in str(row['special']).split():
                if len(num) == 4 and num.isdigit():
                    today_numbers.append(num)
        if 'consolation' in row and row['consolation']:
            for num in str(row['consolation']).split():
                if len(num) == 4 and num.isdigit():
                    today_numbers.append(num)
    recent_nums = [str(n) for col in ['1st_real', '2nd_real', '3rd_real'] for n in df[col].tail(50).tolist() if len(str(n)) == 4 and str(n).isdigit()]
    if not today_numbers or not recent_nums:
        return get_frequency_predictions(df)
    predictions = predict_tomorrow(today_numbers, patterns, recent_nums)
    return [num for num, _, _ in predictions[:5]]

def get_auto_weight_predictions(df):
    """Smart auto weight - optimizes hot digit vs pair pattern weights"""
    recent_nums = [str(n) for col in ['1st_real', '2nd_real', '3rd_real'] for n in df[col].tail(100).tolist() if len(str(n)) == 4 and str(n).isdigit()]
    if len(recent_nums) < 10:
        return get_frequency_predictions(df)
    digit_freq = Counter(''.join(recent_nums))
    pair_freq = Counter([n[i:i+2] for n in recent_nums for i in range(3)])
    combos = [(h, p) for h, p in product([0.3, 0.4, 0.5, 0.6, 0.7], repeat=2) if abs((h+p)-1) < 0.05]
    best_score = -1
    best_weights = (0.5, 0.5)
    for hot_w, pair_w in combos:
        score = sum(hot_w * sum(digit_freq.get(d, 0) for d in num) + pair_w * sum(pair_freq.get(num[i:i+2], 0) for i in range(3)) for num in recent_nums[-30:])
        if score > best_score:
            best_score = score
            best_weights = (hot_w, pair_w)
    hot_w, pair_w = best_weights
    all_nums = set(recent_nums)
    scores = {}
    for num in all_nums:
        scores[num] = hot_w * sum(digit_freq.get(d, 0) for d in num) + pair_w * sum(pair_freq.get(num[i:i+2], 0) for i in range(3))
    for num in [f"{i:04d}" for i in range(10000)]:
        if num not in scores:
            scores[num] = hot_w * sum(digit_freq.get(d, 0) for d in num) + pair_w * sum(pair_freq.get(num[i:i+2], 0) for i in range(3))
    return [num for num, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]]

def get_ml_predictions(df):
    """ML predictor - uses Linear Regression on historical sequences"""
    nums = [n for col in ['1st_real', '2nd_real', '3rd_real'] for n in df[col].astype(str) if len(n) == 4 and n.isdigit()]
    if len(nums) < 20:
        return get_frequency_predictions(df)
    try:
        X = np.array([[int(d) for d in nums[i]] for i in range(len(nums) - 1)])
        y = np.array([int(nums[i + 1]) for i in range(len(nums) - 1)])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        recent = nums[-5:]
        candidates = set(nums[-50:])
        for n in recent:
            for d in range(10):
                for pos in range(4):
                    new = list(n)
                    new[pos] = str(d)
                    candidates.add(''.join(new))
        scored = []
        for num in candidates:
            try:
                pred = float(model.predict(scaler.transform([[int(d) for d in num]]))[0])
                actual = int(num)
                diff = abs(pred - actual)
                scored.append((num, -diff))
            except:
                pass
        scored.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in scored[:5]]
    except:
        return get_frequency_predictions(df)

def check_matches(prediction, actual):
    """Check if prediction matches actual (4, 3, 2 digits or near)"""
    if prediction == actual:
        return {'type': '4D_EXACT', 'match': 4}
    common = len(set(prediction) & set(actual))
    if common >= 3:
        return {'type': '3D_MATCH', 'match': 3}
    if common >= 2:
        return {'type': '2D_MATCH', 'match': 2}
    # Check "near" (difference <= 100)
    try:
        diff = abs(int(prediction) - int(actual))
        if diff <= 100:
            return {'type': 'NEAR', 'match': 1, 'diff': diff}
    except:
        pass
    return {'type': 'NO_MATCH', 'match': 0}
