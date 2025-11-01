"""
Sequential Pattern Learning - Learn from historical number transitions
"""
from collections import defaultdict, Counter
import pandas as pd

def learn_sequences(df, provider='all', lookback=1):
    """Learn what numbers follow other numbers"""
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    # Collect all winning numbers in sequence
    sequence = []
    for _, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if len(num) == 4 and num.isdigit():
                sequence.append(num)
    
    # Learn transitions: number -> next numbers
    transitions = defaultdict(list)
    for i in range(len(sequence) - lookback):
        current = sequence[i]
        next_num = sequence[i + lookback]
        transitions[current].append(next_num)
    
    # Calculate probabilities
    patterns = {}
    for num, nexts in transitions.items():
        freq = Counter(nexts)
        total = len(nexts)
        patterns[num] = [(n, count/total, count) for n, count in freq.most_common(10)]
    
    return patterns, sequence

def learn_digit_transitions(df, provider='all'):
    """Learn how digits transition position by position"""
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    # Track transitions for each position (0-3)
    position_transitions = [defaultdict(list) for _ in range(4)]
    
    prev_num = None
    for _, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if len(num) == 4 and num.isdigit():
                if prev_num:
                    for pos in range(4):
                        position_transitions[pos][prev_num[pos]].append(num[pos])
                prev_num = num
    
    # Calculate probabilities per position
    position_patterns = []
    for pos in range(4):
        patterns = {}
        for digit, nexts in position_transitions[pos].items():
            freq = Counter(nexts)
            total = len(nexts)
            patterns[digit] = [(d, count/total, count) for d, count in freq.most_common()]
        position_patterns.append(patterns)
    
    return position_patterns

def predict_next_numbers(recent_numbers, patterns, top_n=10):
    """Predict next numbers based on recent history"""
    predictions = Counter()
    
    for num in recent_numbers[-5:]:  # Use last 5 numbers
        if num in patterns:
            for next_num, prob, count in patterns[num]:
                predictions[next_num] += prob * (count / 10)  # Weight by frequency
    
    results = []
    for num, score in predictions.most_common(top_n):
        confidence = min(95, score * 100)
        results.append({
            'number': num,
            'confidence': round(confidence, 1),
            'score': round(score, 3),
            'reason': f'Follows recent pattern'
        })
    
    return results

def predict_by_position(recent_numbers, position_patterns):
    """Predict by analyzing each digit position"""
    if not recent_numbers:
        return []
    
    last_num = recent_numbers[-1]
    if len(last_num) != 4:
        return []
    
    # Get most likely digit for each position
    predicted_digits = []
    for pos in range(4):
        current_digit = last_num[pos]
        if current_digit in position_patterns[pos]:
            top_next = position_patterns[pos][current_digit][0]  # Most likely
            predicted_digits.append((top_next[0], top_next[1]))
        else:
            predicted_digits.append(('0', 0.1))
    
    # Generate predictions
    predictions = []
    
    # Best prediction: most likely digit per position
    best = ''.join([d[0] for d in predicted_digits])
    avg_conf = sum([d[1] for d in predicted_digits]) / 4 * 100
    predictions.append({
        'number': best,
        'confidence': round(min(95, avg_conf), 1),
        'reason': 'Position-based transition'
    })
    
    return predictions

def analyze_cycles(df, provider='all'):
    """Find repeating cycles in the data"""
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    # Collect sequence
    sequence = []
    for _, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if len(num) == 4 and num.isdigit():
                sequence.append(num)
    
    # Find gaps between same number appearances
    gaps = defaultdict(list)
    last_seen = {}
    
    for idx, num in enumerate(sequence):
        if num in last_seen:
            gap = idx - last_seen[num]
            gaps[num].append(gap)
        last_seen[num] = idx
    
    # Calculate average gaps
    cycle_patterns = {}
    for num, gap_list in gaps.items():
        if len(gap_list) >= 2:
            avg_gap = sum(gap_list) / len(gap_list)
            cycle_patterns[num] = {
                'avg_gap': round(avg_gap, 1),
                'min_gap': min(gap_list),
                'max_gap': max(gap_list),
                'appearances': len(gap_list) + 1,
                'last_seen': last_seen[num],
                'overdue': len(sequence) - last_seen[num]
            }
    
    return cycle_patterns, len(sequence)

def get_overdue_numbers(cycle_patterns, total_draws, threshold=1.5):
    """Find numbers that are overdue based on their cycle"""
    overdue = []
    
    for num, data in cycle_patterns.items():
        expected_gap = data['avg_gap']
        actual_gap = data['overdue']
        
        if actual_gap > expected_gap * threshold:
            overdue_factor = actual_gap / expected_gap
            confidence = min(95, 50 + (overdue_factor * 10))
            overdue.append({
                'number': num,
                'confidence': round(confidence, 1),
                'overdue_by': round(actual_gap - expected_gap, 1),
                'expected_gap': round(expected_gap, 1),
                'actual_gap': actual_gap,
                'reason': f'Overdue by {round(overdue_factor, 1)}x normal cycle'
            })
    
    return sorted(overdue, key=lambda x: x['overdue_by'], reverse=True)
