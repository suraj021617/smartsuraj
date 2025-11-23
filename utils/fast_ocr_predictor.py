"""
⚡ FAST OCR PREDICTOR - Uses only last 100 rows for INSTANT predictions
No caching needed - just fast!
"""
import pandas as pd
import re
from collections import Counter

def fast_ocr_predict(df, provider='all', lookback=100):
    """⚡ INSTANT: Predict using only recent data"""
    if df.empty:
        return [], {}, []
    
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    if df.empty:
        return [], {}, []
    
    # Use ONLY last 100 rows - FAST!
    recent = df.tail(lookback)
    
    # Extract numbers (vectorized)
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums = recent[col].astype(str)
        all_numbers.extend(nums[(nums.str.len() == 4) & (nums.str.isdigit())].tolist())
    
    if not all_numbers:
        return [], {}, []
    
    # Build OCR table
    ocr_table = {d: [0]*4 for d in range(10)}
    for num in all_numbers:
        for pos in range(4):
            ocr_table[int(num[pos])][pos] += 1
    
    # Get hot digits
    hot_per_pos = []
    for pos in range(4):
        sorted_d = sorted(range(10), key=lambda d: ocr_table[d][pos], reverse=True)
        hot_per_pos.append(sorted_d[:5])
    
    current_pattern = ''.join([str(hot_per_pos[i][0]) for i in range(4)])
    
    # Generate predictions
    predictions = []
    
    # 1. Current hot pattern
    predictions.append({
        'number': current_pattern,
        'confidence': 85,
        'strategy': 'Hot Pattern',
        'position_analysis': {
            f'pos{i+1}': {'digit': current_pattern[i], 'frequency': ocr_table[int(current_pattern[i])][i]}
            for i in range(4)
        }
    })
    
    # 2. Most frequent numbers
    freq = Counter(all_numbers)
    for num, count in freq.most_common(5):
        if num not in [p['number'] for p in predictions]:
            conf = min(95, int((count / len(all_numbers)) * 200))
            predictions.append({
                'number': num,
                'confidence': conf,
                'strategy': f'Frequent ({count}x)',
                'position_analysis': {
                    f'pos{i+1}': {'digit': num[i], 'frequency': ocr_table[int(num[i])][i]}
                    for i in range(4)
                }
            })
    
    # 3. Variations
    for pos in range(4):
        if len(predictions) >= 10:
            break
        alt = list(current_pattern)
        alt[pos] = str(hot_per_pos[pos][1])
        alt_num = ''.join(alt)
        if alt_num not in [p['number'] for p in predictions]:
            predictions.append({
                'number': alt_num,
                'confidence': 70 - len(predictions) * 3,
                'strategy': f'Variation Pos{pos+1}',
                'position_analysis': {
                    f'pos{i+1}': {'digit': alt_num[i], 'frequency': ocr_table[int(alt_num[i])][i]}
                    for i in range(4)
                }
            })
    
    return predictions[:10], ocr_table, hot_per_pos
