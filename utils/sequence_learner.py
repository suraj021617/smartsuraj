"""
Sequence Pattern Learner - Find repeating sequences
Example: 1234 → 5678 → 9012 (ascending pattern)
"""
from collections import Counter

def detect_digit_patterns(numbers):
    """Learn digit-level patterns"""
    patterns = {
        'ascending': [],
        'descending': [],
        'repeating': [],
        'alternating': []
    }
    
    for num in numbers:
        if len(num) != 4:
            continue
        
        digits = [int(d) for d in num]
        
        # Ascending: 1234, 2345
        if all(digits[i] <= digits[i+1] for i in range(3)):
            patterns['ascending'].append(num)
        
        # Descending: 4321, 5432
        elif all(digits[i] >= digits[i+1] for i in range(3)):
            patterns['descending'].append(num)
        
        # Repeating: 1111, 2222
        elif len(set(digits)) <= 2:
            patterns['repeating'].append(num)
        
        # Alternating: 1212, 3434
        elif digits[0] == digits[2] and digits[1] == digits[3]:
            patterns['alternating'].append(num)
    
    return patterns

def find_sum_patterns(numbers):
    """Learn sum patterns (digit sum trends)"""
    sums = []
    for num in numbers:
        if len(num) == 4 and num.isdigit():
            digit_sum = sum(int(d) for d in num)
            sums.append((num, digit_sum))
    
    # Find most common sum ranges
    sum_ranges = Counter()
    for num, s in sums:
        if s <= 10:
            sum_ranges['low'] += 1
        elif s <= 20:
            sum_ranges['mid'] += 1
        else:
            sum_ranges['high'] += 1
    
    return sum_ranges

def predict_by_patterns(numbers, top_n=5):
    """Predict based on learned patterns"""
    patterns = detect_digit_patterns(numbers[-100:])
    sum_ranges = find_sum_patterns(numbers[-100:])
    
    predictions = []
    
    # Most common pattern type
    pattern_counts = {k: len(v) for k, v in patterns.items()}
    dominant_pattern = max(pattern_counts, key=pattern_counts.get)
    
    # Get candidates from dominant pattern
    candidates = patterns[dominant_pattern]
    if candidates:
        freq = Counter(candidates)
        for num, count in freq.most_common(top_n):
            confidence = (count / len(candidates)) * 100
            predictions.append((num, confidence, f'{dominant_pattern} pattern'))
    
    return predictions

def sequence_predictor(df, provider='all', lookback=300):
    """Main sequence learning predictor"""
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    all_nums = []
    for _, row in df.tail(lookback).iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if len(num) == 4 and num.isdigit():
                all_nums.append(num)
    
    if not all_nums:
        return []
    
    predictions = predict_by_patterns(all_nums, top_n=10)
    
    # Fallback to frequency
    if not predictions:
        freq = Counter(all_nums[-50:])
        predictions = [(num, 50, 'frequency') for num, _ in freq.most_common(10)]
    
    return predictions[:5]
