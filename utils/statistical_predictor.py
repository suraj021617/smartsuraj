from collections import Counter, defaultdict
import pandas as pd

def get_provider_statistics(df, provider):
    """Get comprehensive statistics for a specific provider"""
    
    # Filter by provider
    if provider and provider != 'all':
        df = df[df['provider'] == provider]
    
    # Extract all winning numbers
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums = df[col].astype(str).tolist()
        all_numbers.extend([n for n in nums if len(n) == 4 and n.isdigit()])
    
    if not all_numbers:
        return None
    
    # Number frequency (how many times each number appeared)
    number_freq = Counter(all_numbers)
    
    # Digit frequency (how often each digit 0-9 appears)
    all_digits = ''.join(all_numbers)
    digit_freq = Counter(all_digits)
    
    # Position-based digit frequency
    position_freq = [Counter(), Counter(), Counter(), Counter()]
    for num in all_numbers:
        for i, digit in enumerate(num):
            position_freq[i][digit] += 1
    
    # Consecutive pair frequency
    pair_freq = Counter()
    for num in all_numbers:
        for i in range(3):
            pair_freq[num[i:i+2]] += 1
    
    # Recent hot numbers (last 20 draws)
    recent_numbers = all_numbers[-60:] if len(all_numbers) > 60 else all_numbers
    recent_freq = Counter(recent_numbers)
    
    # Cold numbers (haven't appeared recently)
    all_unique = set(all_numbers)
    recent_unique = set(recent_numbers)
    cold_numbers = all_unique - recent_unique
    
    return {
        'total_draws': len(all_numbers),
        'number_freq': number_freq,
        'digit_freq': digit_freq,
        'position_freq': position_freq,
        'pair_freq': pair_freq,
        'recent_freq': recent_freq,
        'cold_numbers': cold_numbers,
        'hot_numbers': [n for n, _ in recent_freq.most_common(20)]
    }

def generate_smart_predictions(df, provider, top_n=5):
    """Generate predictions based on statistical analysis"""
    
    stats = get_provider_statistics(df, provider)
    if not stats:
        return []
    
    candidates = {}
    
    # Strategy 1: Most frequent numbers (30% weight)
    for num, count in stats['number_freq'].most_common(50):
        score = (count / stats['total_draws']) * 30
        candidates[num] = candidates.get(num, 0) + score
    
    # Strategy 2: Hot numbers from recent draws (25% weight)
    for num in stats['hot_numbers'][:20]:
        score = 25 / len(stats['hot_numbers'][:20])
        candidates[num] = candidates.get(num, 0) + score
    
    # Strategy 3: Build from hot digit positions (20% weight)
    for d1, _ in stats['position_freq'][0].most_common(3):
        for d2, _ in stats['position_freq'][1].most_common(3):
            for d3, _ in stats['position_freq'][2].most_common(3):
                for d4, _ in stats['position_freq'][3].most_common(3):
                    num = d1 + d2 + d3 + d4
                    candidates[num] = candidates.get(num, 0) + 20 / 81
    
    # Strategy 4: Hot pairs combination (15% weight)
    hot_pairs = [p for p, _ in stats['pair_freq'].most_common(15)]
    for i, p1 in enumerate(hot_pairs[:10]):
        for p2 in hot_pairs[:10]:
            if p1[1] == p2[0]:  # Overlapping
                num = p1 + p2[1]
                candidates[num] = candidates.get(num, 0) + 15 / 100
    
    # Strategy 5: Recent number variations (10% weight)
    recent_top = [n for n, _ in stats['recent_freq'].most_common(10)]
    for num in recent_top:
        # Reverse
        rev = num[::-1]
        candidates[rev] = candidates.get(rev, 0) + 5
        # Rotate
        for i in range(1, 4):
            rot = num[i:] + num[:i]
            candidates[rot] = candidates.get(rot, 0) + 5 / 3
    
    # Sort by score
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    # Format results with detailed reasons
    results = []
    for num, score in sorted_candidates[:top_n]:
        freq_count = stats['number_freq'].get(num, 0)
        recent_count = stats['recent_freq'].get(num, 0)
        
        reasons = []
        if freq_count > 0:
            reasons.append(f"appeared {freq_count}x")
        if recent_count > 0:
            reasons.append(f"recent {recent_count}x")
        if num in stats['hot_numbers']:
            reasons.append("hot")
        
        reason = ", ".join(reasons) if reasons else "pattern-based"
        normalized_score = score / sorted_candidates[0][1] if sorted_candidates else 0
        
        results.append((num, normalized_score, reason))
    
    return results

def get_prediction_accuracy(df, provider):
    """Calculate how accurate predictions would have been historically"""
    
    if provider and provider != 'all':
        df = df[df['provider'] == provider]
    
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    hits = 0
    total = 0
    
    for i in range(len(df) - 1):
        # Use data up to this point to predict
        train_df = df.iloc[:i+1]
        predictions = generate_smart_predictions(train_df, provider, top_n=5)
        pred_numbers = [p[0] for p in predictions]
        
        # Check against next draw
        next_row = df.iloc[i + 1]
        actual = [str(next_row.get(col, '')) for col in ['1st_real', '2nd_real', '3rd_real'] 
                  if len(str(next_row.get(col, ''))) == 4]
        
        if any(p in actual for p in pred_numbers):
            hits += 1
        total += 1
    
    accuracy = (hits / total * 100) if total > 0 else 0
    return accuracy, hits, total
