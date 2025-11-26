"""
Complete Advanced Pattern Analysis Engine
All features in one place
"""
from collections import defaultdict, Counter
import pandas as pd

# ============ 1. HOT & COLD NUMBERS ============
def analyze_hot_cold(df, provider='all', days=30):
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    periods = {'7d': 7, '30d': 30, '90d': 90}
    results = {}
    
    for period_name, period_days in periods.items():
        recent = df.tail(period_days * 3)
        nums = [str(row[col]) for _, row in recent.iterrows() 
                for col in ['1st_real', '2nd_real', '3rd_real'] 
                if len(str(row[col])) == 4]
        freq = Counter(nums)
        results[period_name] = {
            'hot': freq.most_common(10),
            'cold': [(n, c) for n, c in sorted(freq.items(), key=lambda x: x[1])[:10]]
        }
    
    return results

# ============ 2. NUMBER PAIR ANALYSIS ============
def analyze_pairs(df, provider='all'):
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    pairs = []
    for _, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if len(num) == 4:
                pairs.extend([num[i:i+2] for i in range(3)])
    
    pair_freq = Counter(pairs)
    
    # Consecutive pairs (numbers appearing together in same draw)
    consecutive = []
    for _, row in df.iterrows():
        nums = [str(row[col]) for col in ['1st_real', '2nd_real', '3rd_real'] if len(str(row[col])) == 4]
        for i in range(len(nums)-1):
            consecutive.append((nums[i], nums[i+1]))
    
    consecutive_freq = Counter(consecutive)
    
    return {
        'digit_pairs': pair_freq.most_common(20),
        'consecutive_pairs': consecutive_freq.most_common(20)
    }

# ============ 3. SUM ANALYSIS ============
def analyze_sums(df, provider='all'):
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    sums = []
    for _, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if len(num) == 4 and num.isdigit():
                digit_sum = sum(int(d) for d in num)
                sums.append((num, digit_sum))
    
    sum_ranges = {'0-9': 0, '10-19': 0, '20-29': 0, '30-36': 0}
    for _, s in sums:
        if s <= 9: sum_ranges['0-9'] += 1
        elif s <= 19: sum_ranges['10-19'] += 1
        elif s <= 29: sum_ranges['20-29'] += 1
        else: sum_ranges['30-36'] += 1
    
    total = len(sums)
    sum_percentages = {k: round(v/total*100, 1) for k, v in sum_ranges.items()}
    
    # Most common sums
    sum_freq = Counter([s for _, s in sums])
    
    return {
        'ranges': sum_percentages,
        'most_common': sum_freq.most_common(10),
        'predictions': [num for num, _ in sorted(sums, key=lambda x: sum_freq[x[1]], reverse=True)[:10]]
    }

# ============ 4. POSITION ANALYSIS ============
def analyze_positions(df, provider='all'):
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    positions = [Counter() for _ in range(4)]
    
    for _, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if len(num) == 4:
                for pos in range(4):
                    positions[pos][num[pos]] += 1
    
    # Heat map data
    heatmap = [[positions[pos].get(str(d), 0) for d in range(10)] for pos in range(4)]
    
    # Top digits per position
    top_per_position = [pos.most_common(5) for pos in positions]
    
    # Generate prediction from strongest digits
    prediction = ''.join([pos.most_common(1)[0][0] for pos in positions])
    
    return {
        'heatmap': heatmap,
        'top_per_position': top_per_position,
        'prediction': prediction
    }

# ============ 5. GAP ANALYSIS ============
def analyze_gaps(df, provider='all'):
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    last_seen = {}
    gaps = defaultdict(list)
    
    for idx, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if len(num) == 4:
                if num in last_seen:
                    gaps[num].append(idx - last_seen[num])
                last_seen[num] = idx
    
    # Calculate overdue
    total_draws = len(df)
    overdue = []
    for num, gap_list in gaps.items():
        if len(gap_list) >= 2:
            avg_gap = sum(gap_list) / len(gap_list)
            current_gap = total_draws - last_seen[num]
            if current_gap > avg_gap * 1.5:
                overdue.append({
                    'number': num,
                    'avg_gap': round(avg_gap, 1),
                    'current_gap': current_gap,
                    'overdue_by': round(current_gap - avg_gap, 1)
                })
    
    return sorted(overdue, key=lambda x: x['overdue_by'], reverse=True)[:20]

# ============ 6. ODD/EVEN PATTERNS ============
def analyze_odd_even(df, provider='all'):
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    patterns = []
    for _, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if len(num) == 4 and num.isdigit():
                odd_count = sum(1 for d in num if int(d) % 2 == 1)
                patterns.append(f"{odd_count}O{4-odd_count}E")
    
    pattern_freq = Counter(patterns)
    
    # Last 10 patterns
    recent_patterns = patterns[-10:]
    
    return {
        'distribution': pattern_freq.most_common(),
        'recent': recent_patterns,
        'recommendation': pattern_freq.most_common(1)[0][0] if pattern_freq else '2O2E'
    }

# ============ 7. PROVIDER COMPARISON ============
def compare_providers(df):
    providers = df['provider'].unique()
    comparison = {}
    
    for provider in providers:
        prov_df = df[df['provider'] == provider]
        nums = [str(row[col]) for _, row in prov_df.iterrows() 
                for col in ['1st_real', '2nd_real', '3rd_real'] 
                if len(str(row[col])) == 4]
        
        freq = Counter(nums)
        digit_freq = Counter(''.join(nums))
        
        comparison[provider] = {
            'total_draws': len(prov_df),
            'top_5': freq.most_common(5),
            'favorite_digit': digit_freq.most_common(1)[0] if digit_freq else ('0', 0),
            'unique_numbers': len(freq)
        }
    
    return comparison

# ============ 8. SEQUENCE LEARNING ============
def learn_sequences(df, provider='all'):
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    sequence = []
    for _, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if len(num) == 4 and num.isdigit():
                sequence.append(num)
    
    transitions = defaultdict(list)
    for i in range(len(sequence) - 1):
        transitions[sequence[i]].append(sequence[i + 1])
    
    patterns = {}
    for num, nexts in transitions.items():
        freq = Counter(nexts)
        total = len(nexts)
        patterns[num] = [(n, round(count/total*100, 1), count) for n, count in freq.most_common(5)]
    
    return patterns, sequence

# ============ 9. MULTI-STEP PREDICTIONS ============
def predict_multi_step(sequence, patterns, steps=5):
    if not sequence:
        return []
    
    predictions = []
    current = sequence[-1]
    
    for step in range(steps):
        if current in patterns and patterns[current]:
            next_num = patterns[current][0][0]  # Most likely
            prob = patterns[current][0][1]
            predictions.append({
                'step': step + 1,
                'number': next_num,
                'probability': prob,
                'from': current
            })
            current = next_num
        else:
            break
    
    return predictions

# ============ 10. TIME-BASED PATTERNS ============
def analyze_time_patterns(df, provider='all'):
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    df['day_of_week'] = df['date_parsed'].dt.day_name()
    df['month'] = df['date_parsed'].dt.month
    
    # Day of week patterns
    day_patterns = {}
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        day_df = df[df['day_of_week'] == day]
        nums = [str(row[col]) for _, row in day_df.iterrows() 
                for col in ['1st_real', '2nd_real', '3rd_real'] 
                if len(str(row[col])) == 4]
        freq = Counter(nums)
        day_patterns[day] = freq.most_common(5)
    
    # Month patterns
    month_patterns = {}
    for month in range(1, 13):
        month_df = df[df['month'] == month]
        nums = [str(row[col]) for _, row in month_df.iterrows() 
                for col in ['1st_real', '2nd_real', '3rd_real'] 
                if len(str(row[col])) == 4]
        freq = Counter(nums)
        month_patterns[month] = freq.most_common(5)
    
    return {
        'day_patterns': day_patterns,
        'month_patterns': month_patterns
    }

# ============ 11. NUMBER REPEATER ANALYSIS ============
def analyze_repeaters(df, provider='all', days=7):
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    df = df.sort_values('date_parsed', ascending=False).head(days * 3)
    
    nums = []
    for _, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if len(num) == 4 and num.isdigit():
                nums.append(num)
    
    freq = Counter(nums)
    repeaters = [(num, count) for num, count in freq.items() if count >= 2]
    repeaters.sort(key=lambda x: x[1], reverse=True)
    
    hot_streaks = []
    for num, count in repeaters[:20]:
        if count >= 3:
            confidence = min(95, 60 + (count * 10))
            hot_streaks.append({
                'number': num,
                'count': count,
                'confidence': round(confidence, 1),
                'status': 'HOT STREAK' if count >= 4 else 'REPEATING'
            })
    
    return hot_streaks

# ============ 12. DIGIT FREQUENCY HEATMAP ============
def analyze_digit_frequency(df, provider='all'):
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    digit_counts = Counter()
    total_digits = 0
    
    for _, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if len(num) == 4 and num.isdigit():
                for digit in num:
                    digit_counts[digit] += 1
                    total_digits += 1
    
    digit_percentages = {d: round(digit_counts.get(d, 0) / total_digits * 100, 2) for d in '0123456789'}
    
    # Generate predictions based on hot digits
    hot_digits = [d for d, _ in sorted(digit_percentages.items(), key=lambda x: x[1], reverse=True)[:4]]
    
    predictions = []
    from itertools import permutations
    for perm in list(permutations(hot_digits, 4))[:10]:
        num = ''.join(perm)
        avg_pct = sum(digit_percentages[d] for d in num) / 4
        confidence = min(95, avg_pct * 8)
        predictions.append({
            'number': num,
            'confidence': round(confidence, 1),
            'reason': f'Hot digits: {avg_pct:.1f}% avg'
        })
    
    return {
        'percentages': digit_percentages,
        'predictions': predictions,
        'hot_digits': hot_digits
    }

# ============ 13. PREDICTION ACCURACY TRACKER ============
def track_accuracy(df, provider='all', lookback=30):
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    methods = {
        'hot_numbers': [],
        'position_based': [],
        'sequence_based': [],
        'overdue': [],
        'sum_based': []
    }
    
    for i in range(len(df) - lookback - 1, len(df) - 1):
        if i < 10:
            continue
        
        historical = df.iloc[:i]
        actual_row = df.iloc[i + 1]
        actual = [str(actual_row[col]) for col in ['1st_real', '2nd_real', '3rd_real'] if len(str(actual_row[col])) == 4]
        
        # Method 1: Hot numbers
        recent_nums = [str(row[col]) for _, row in historical.tail(30).iterrows() 
                      for col in ['1st_real', '2nd_real', '3rd_real'] if len(str(row[col])) == 4]
        freq = Counter(recent_nums)
        hot_pred = [num for num, _ in freq.most_common(5)]
        methods['hot_numbers'].append(any(p in actual for p in hot_pred))
        
        # Method 2: Position-based
        pos_data = analyze_positions(historical, provider)
        methods['position_based'].append(pos_data['prediction'] in actual)
        
        # Method 3: Sequence
        patterns, seq = learn_sequences(historical, provider)
        if seq and seq[-1] in patterns:
            seq_pred = [n for n, _, _ in patterns[seq[-1]][:5]]
            methods['sequence_based'].append(any(p in actual for p in seq_pred))
        
        # Method 4: Overdue
        overdue = analyze_gaps(historical, provider)
        overdue_pred = [item['number'] for item in overdue[:5]]
        methods['overdue'].append(any(p in actual for p in overdue_pred))
        
        # Method 5: Sum-based
        sum_data = analyze_sums(historical, provider)
        methods['sum_based'].append(any(p in actual for p in sum_data['predictions'][:5]))
    
    accuracy = {}
    for method, results in methods.items():
        if results:
            accuracy[method] = {
                'rate': round(sum(results) / len(results) * 100, 1),
                'wins': sum(results),
                'total': len(results)
            }
    
    # Rank methods
    ranked = sorted(accuracy.items(), key=lambda x: x[1]['rate'], reverse=True)
    
    return {
        'accuracy': accuracy,
        'ranked': ranked,
        'best_method': ranked[0][0] if ranked else None
    }

# ============ MASTER PREDICTION FUNCTION ============
def get_master_predictions(df, provider='all', top_n=20):
    """Combine all analysis methods for ultimate predictions"""
    
    # Run all analyses
    hot_cold = analyze_hot_cold(df, provider)
    pairs = analyze_pairs(df, provider)
    sums = analyze_sums(df, provider)
    positions = analyze_positions(df, provider)
    gaps = analyze_gaps(df, provider)
    odd_even = analyze_odd_even(df, provider)
    patterns, sequence = learn_sequences(df, provider)
    
    # Score each number
    scores = defaultdict(float)
    reasons = defaultdict(list)
    
    # Hot numbers (30d) - 25% weight
    for num, count in hot_cold['30d']['hot']:
        scores[num] += count * 0.25
        reasons[num].append(f"Hot: {count}x in 30d")
    
    # Position-based - 20% weight
    pred_num = positions['prediction']
    scores[pred_num] += 20
    reasons[pred_num].append("Position-optimal")
    
    # Overdue - 20% weight
    for item in gaps[:10]:
        scores[item['number']] += item['overdue_by'] * 0.2
        reasons[item['number']].append(f"Overdue: +{item['overdue_by']}")
    
    # Sequence patterns - 20% weight
    if sequence:
        last_num = sequence[-1]
        if last_num in patterns:
            for next_num, prob, count in patterns[last_num][:5]:
                scores[next_num] += prob * 0.2
                reasons[next_num].append(f"Follows {last_num}: {prob}%")
    
    # Sum analysis - 15% weight
    for num in sums['predictions'][:10]:
        scores[num] += 1.5
        reasons[num].append("Optimal sum range")
    
    # Compile results
    results = []
    for num, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        confidence = min(95, (score / max(scores.values())) * 100)
        results.append({
            'number': num,
            'confidence': round(confidence, 1),
            'score': round(score, 2),
            'reasons': reasons[num][:3]
        })
    
    return results
