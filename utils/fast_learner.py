import pandas as pd
import re
from collections import Counter

def fast_learn_patterns(csv_path='4d_results_history.csv', limit=500):
    """Ultra-fast pattern learning - processes only recent data"""
    
    # Read CSV with error handling
    df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8', low_memory=False)
    df = df.tail(limit)
    
    # Extract all 4-digit numbers
    all_nums = []
    for _, row in df.iterrows():
        text = str(row.values)
        nums = re.findall(r'\b\d{4}\b', text)
        all_nums.extend(nums)
    
    # Quick digit position analysis
    pos_freq = [Counter(), Counter(), Counter(), Counter()]
    for num in all_nums:
        if len(num) == 4:
            for i, d in enumerate(num):
                pos_freq[i][d] += 1
    
    # Simple transition patterns (last 100 only)
    transitions = {}
    for i in range(len(all_nums) - 1):
        if len(all_nums[i]) == 4 and len(all_nums[i+1]) == 4:
            transitions[all_nums[i]] = all_nums[i+1]
    
    return {
        'position_freq': pos_freq,
        'transitions': transitions,
        'recent_numbers': all_nums[-50:]
    }

def fast_predict(patterns):
    """Generate predictions instantly"""
    predictions = []
    
    # Method 1: Most frequent digits by position
    pred1 = ''.join(patterns['position_freq'][i].most_common(1)[0][0] for i in range(4))
    predictions.append((pred1, 0.7, 'freq'))
    
    # Method 2: Recent transitions
    if patterns['recent_numbers']:
        last = patterns['recent_numbers'][-1]
        if last in patterns['transitions']:
            predictions.append((patterns['transitions'][last], 0.8, 'transition'))
    
    # Method 3: Variations of recent
    recent = patterns['recent_numbers'][-5:]
    for num in recent:
        try:
            predictions.append((str(int(num) + 1).zfill(4), 0.5, 'plus1'))
            predictions.append((num[::-1], 0.4, 'reverse'))
        except:
            pass
    
    # Remove duplicates
    seen = set()
    unique = []
    for num, conf, method in predictions:
        if num not in seen and len(num) == 4:
            seen.add(num)
            unique.append((num, conf, method))
    
    return unique[:10]
