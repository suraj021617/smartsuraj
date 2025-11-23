import pandas as pd
import pickle
from collections import Counter
from datetime import datetime

def find_patterns(numbers):
    """Find patterns from numbers"""
    return {
        'frequency': Counter(numbers),
        'digit_freq': Counter(''.join(numbers)),
        'pairs': Counter([numbers[i:i+2] for i in range(len(numbers)-1)]),
        'hot_numbers': Counter(numbers).most_common(20),
        'transitions': {numbers[i]: numbers[i+1] for i in range(len(numbers)-1)}
    }

def auto_learn():
    """Auto-learn: Load patterns, apply to new data, learn, predict"""
    
    # Step 1: Load existing patterns
    try:
        with open('data/patterns.pkl', 'rb') as f:
            patterns = pickle.load(f)
        print(f"✅ Loaded patterns from {patterns['total_draws']} draws")
    except:
        patterns = None
        print("⚠️ No patterns found, starting fresh...")
    
    # Step 2: Load CSV
    df = pd.read_csv('4d_results_history.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    
    # Step 3: Get new data only
    if patterns:
        last_date = pd.to_datetime(patterns['last_date'])
        new_df = df[df['date'] > last_date]
        if len(new_df) == 0:
            print("✅ No new data")
            return patterns
        print(f"📊 Found {len(new_df)} new draws")
    else:
        new_df = df.tail(3000)
        print(f"🚀 Training on {len(new_df)} draws")
    
    # Step 4: Extract numbers
    new_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        if col in new_df.columns:
            new_numbers.extend([n for n in new_df[col].astype(str) if len(n) == 4 and n.isdigit()])
    
    # Step 5: Apply patterns + Learn new patterns
    if patterns:
        # Merge old + new
        all_numbers = patterns['all_numbers'] + new_numbers
        new_patterns = find_patterns(all_numbers)
        patterns.update(new_patterns)
        patterns['total_draws'] += len(new_df)
        patterns['all_numbers'] = all_numbers[-3000:]  # Keep last 3000
    else:
        # First time
        patterns = find_patterns(new_numbers)
        patterns['all_numbers'] = new_numbers
        patterns['total_draws'] = len(new_df)
    
    patterns['last_date'] = new_df['date'].max().isoformat()
    patterns['last_updated'] = datetime.now().isoformat()
    
    # Step 6: Predict
    predictions = []
    for num, count in patterns['hot_numbers'][:5]:
        score = count / len(patterns['all_numbers'])
        predictions.append((num, score, 'Pattern'))
    
    patterns['predictions'] = predictions
    
    # Step 7: Save
    with open('data/patterns.pkl', 'wb') as f:
        pickle.dump(patterns, f)
    
    print(f"✅ Learned {len(new_numbers)} new numbers")
    print(f"📊 Total: {len(patterns['all_numbers'])} numbers, {patterns['total_draws']} draws")
    print(f"🎯 Top 5 predictions:")
    for num, score, reason in predictions:
        print(f"   {num} - {score*100:.1f}%")
    
    return patterns

if __name__ == '__main__':
    auto_learn()
