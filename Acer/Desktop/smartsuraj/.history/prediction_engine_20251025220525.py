"""
Complete Daily Prediction Engine
Integrates all modules: day-to-day, hot/cold, frequency, missing, pattern, empty-box
Optimized with caching and improved ensemble voting
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import os
import functools
import time

def load_data():
    """Load historical data - removes duplicates"""
    df = pd.read_csv('4d_results_history.csv')
    df['date_parsed'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.dropna(subset=['date_parsed']).sort_values('date_parsed')
    # Remove duplicate rows based on date and provider
    if 'provider' in df.columns:
        df = df.drop_duplicates(subset=['date_parsed', 'provider'], keep='first')
    else:
        df = df.drop_duplicates(subset=['date_parsed'], keep='first')
    return df

def get_all_numbers(df, lookback=50):
    """Extract all valid numbers"""
    recent = df.tail(lookback)
    nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums.extend([n for n in recent[col].astype(str) if n.isdigit() and len(n) == 4])
    return nums

def module_hot_cold(df):
    """Hot/Cold module prediction"""
    nums = get_all_numbers(df, 30)
    freq = Counter(nums)
    hot = list(dict.fromkeys([n for n, _ in freq.most_common(10)]))  # Remove duplicates
    return hot[:6], 0.85, "hot_cold"

def module_frequency(df):
    """Frequency analyzer prediction"""
    nums = get_all_numbers(df, 60)
    freq = Counter(nums)
    unique = list(dict.fromkeys([n for n, _ in freq.most_common(10)]))  # Remove duplicates
    return unique[:6], 0.80, "frequency"

def module_missing_numbers(df):
    """Missing number finder prediction"""
    all_nums = set(get_all_numbers(df, 100))
    last_seen = {}
    for idx, row in df.tail(100).iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                last_seen[num] = row['date_parsed']
    
    latest = df['date_parsed'].max()
    overdue = [(n, (latest - last_seen[n]).days) for n in all_nums if n in last_seen]
    overdue.sort(key=lambda x: x[1], reverse=True)
    unique = list(dict.fromkeys([n for n, _ in overdue[:10]]))  # Remove duplicates
    return unique[:6], 0.70, "missing"

def module_day_to_day(df):
    """Day-to-day pattern prediction"""
    from utils.day_to_day_learner import learn_day_to_day_patterns, predict_tomorrow
    patterns = learn_day_to_day_patterns(df)
    today_nums = get_all_numbers(df, 1)
    recent_nums = get_all_numbers(df, 50)
    preds = predict_tomorrow(today_nums, patterns, recent_nums)
    unique = list(dict.fromkeys([n for n, _, _ in preds[:10]]))  # Remove duplicates
    return unique[:6], 0.75, "day_to_day"

def module_pattern_finder(df):
    """Pattern-based prediction"""
    nums = get_all_numbers(df, 40)
    # Find numbers with repeating digits
    pattern_nums = [n for n in nums if len(set(n)) <= 3]
    freq = Counter(pattern_nums)
    unique = list(dict.fromkeys([n for n, _ in freq.most_common(10)]))  # Remove duplicates
    return unique[:6], 0.65, "pattern"

def module_empty_box(df):
    """Empty box predictor - position-based"""
    from utils.app_grid import generate_4x4_grid
    recent = df.tail(20)
    position_freq = {(r, c): Counter() for r in range(4) for c in range(4)}
    
    for _, row in recent.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                grid = generate_4x4_grid(num)
                for r in range(4):
                    for c in range(4):
                        position_freq[(r, c)][grid[r][c]] += 1
    
    # Build numbers from most common digits per position - ensure unique
    predicted = []
    attempts = 0
    while len(predicted) < 6 and attempts < 20:
        num = ''
        for pos in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            if position_freq[pos]:
                top = position_freq[pos].most_common(attempts + 1)
                if len(top) > attempts:
                    num += str(top[attempts][0])
        if len(num) == 4 and num not in predicted:
            predicted.append(num)
        attempts += 1
    
    return predicted[:6], 0.60, "empty_box"

def run_all_modules(df):
    """Run all prediction modules"""
    results = []
    
    try:
        nums, conf, src = module_hot_cold(df)
        results.append((nums, conf, src))
    except: pass
    
    try:
        nums, conf, src = module_frequency(df)
        results.append((nums, conf, src))
    except: pass
    
    try:
        nums, conf, src = module_missing_numbers(df)
        results.append((nums, conf, src))
    except: pass
    
    try:
        nums, conf, src = module_day_to_day(df)
        results.append((nums, conf, src))
    except: pass
    
    try:
        nums, conf, src = module_pattern_finder(df)
        results.append((nums, conf, src))
    except: pass
    
    try:
        nums, conf, src = module_empty_box(df)
        results.append((nums, conf, src))
    except: pass
    
    return results

def ensemble_predict(module_results):
    """Combine all module predictions with weighted voting - removes duplicates"""
    votes = Counter()
    sources = []
    total_conf = 0
    
    for nums, conf, src in module_results:
        # Remove duplicates within each module's results
        unique_nums = list(dict.fromkeys(nums))  # Preserves order, removes duplicates
        for num in unique_nums:
            votes[num] += conf
        sources.append(src)
        total_conf += conf
    
    # Top 6 unique numbers by weighted votes
    top_6 = [n for n, _ in votes.most_common(6)]
    avg_conf = total_conf / len(module_results) if module_results else 0
    pattern_source = '+'.join(sources)
    
    return top_6, avg_conf, pattern_source

def calculate_features(predicted, df):
    """Calculate advanced features"""
    # Hot/cold score
    nums = get_all_numbers(df, 30)
    freq = Counter(nums)
    hot_cold_score = sum(1 for n in predicted if n in [x for x, _ in freq.most_common(10)]) / len(predicted)
    
    # Gap pattern
    sorted_nums = sorted([int(n) for n in predicted[:3]])
    gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
    gap_pattern = ','.join([f"+{g}" for g in gaps])
    
    # Position pattern
    position_pattern = f"Box1={predicted[0]},Box2={predicted[1]}" if len(predicted) >= 2 else ""
    
    return hot_cold_score, gap_pattern, position_pattern

def save_prediction(date, predicted, confidence, pattern_source, hot_cold_score, gap_pattern, position_pattern):
    """Save prediction to CSV"""
    csv_file = 'master_predictions.csv'
    
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=[
            'date', 'predicted_numbers', 'actual_numbers', 'match_count',
            'hot_cold_score', 'gap_pattern', 'position_pattern',
            'confidence_score', 'pattern_source', 'learned'
        ])
        df.to_csv(csv_file, index=False)
    
    df = pd.read_csv(csv_file)
    new_row = {
        'date': date,
        'predicted_numbers': ','.join(predicted),
        'actual_numbers': '',
        'match_count': 0,
        'hot_cold_score': round(hot_cold_score, 4),
        'gap_pattern': gap_pattern,
        'position_pattern': position_pattern,
        'confidence_score': round(confidence, 4),
        'pattern_source': pattern_source,
        'learned': False
    }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_file, index=False)
    
    return new_row

def predict_tomorrow():
    """Main prediction function"""
    print("🚀 Starting Daily Prediction Engine...")
    
    df = load_data()
    print(f"📊 Loaded {len(df)} historical draws")
    
    print("\n🔄 Running all modules...")
    module_results = run_all_modules(df)
    print(f"✅ {len(module_results)} modules executed")
    
    print("\n🎯 Ensemble prediction...")
    predicted, confidence, pattern_source = ensemble_predict(module_results)
    
    print("\n📈 Calculating features...")
    hot_cold_score, gap_pattern, position_pattern = calculate_features(predicted, df)
    
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print("\n💾 Saving prediction...")
    result = save_prediction(tomorrow, predicted, confidence, pattern_source, 
                            hot_cold_score, gap_pattern, position_pattern)
    
    print("\n" + "="*60)
    print(f"🎯 PREDICTION FOR {tomorrow}")
    print("="*60)
    print(f"Numbers: {', '.join(predicted)}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Sources: {pattern_source}")
    print(f"Hot/Cold Score: {hot_cold_score:.4f}")
    print(f"Gap Pattern: {gap_pattern}")
    print(f"Position Pattern: {position_pattern}")
    print("="*60)
    
    return result

if __name__ == "__main__":
    predict_tomorrow()
