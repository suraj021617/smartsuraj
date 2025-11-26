from collections import Counter
import re

def analyze_historical_data(df):
    """Analyze CSV for frequencies and patterns"""
    
    # Extract all 4-digit numbers
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums = df[col].astype(str).tolist()
        all_numbers.extend([n for n in nums if len(n) == 4 and n.isdigit()])
    
    # Digit frequency
    digit_freq = Counter(''.join(all_numbers))
    
    # Number frequency
    number_freq = Counter(all_numbers)
    
    # Pair frequency (consecutive digits)
    pair_freq = Counter()
    for num in all_numbers:
        for i in range(3):
            pair_freq[num[i:i+2]] += 1
    
    # Triple frequency
    triple_freq = Counter()
    for num in all_numbers:
        for i in range(2):
            triple_freq[num[i:i+3]] += 1
    
    return {
        'digit_freq': digit_freq,
        'number_freq': number_freq,
        'pair_freq': pair_freq,
        'triple_freq': triple_freq,
        'total_draws': len(all_numbers)
    }

def predict_next_numbers(df, top_n=5, provider=None):
    """Generate predictions based on frequency analysis"""
    
    # Filter by provider if specified
    if provider and provider != 'all':
        df = df[df['provider'] == provider]
    
    analysis = analyze_historical_data(df)
    
    # Get most common digits
    hot_digits = [d for d, _ in analysis['digit_freq'].most_common(6)]
    
    # Get most common pairs
    hot_pairs = [p for p, _ in analysis['pair_freq'].most_common(20)]
    
    # Score all possible 4-digit combinations
    candidates = {}
    
    # Method 1: Use hot pairs to build numbers
    for p1 in hot_pairs[:10]:
        for p2 in hot_pairs[:10]:
            if p1[1] == p2[0]:  # Overlapping pairs
                num = p1 + p2[1]
                score = analysis['pair_freq'][p1] + analysis['pair_freq'][p2]
                candidates[num] = candidates.get(num, 0) + score
    
    # Method 2: Recent numbers with modifications
    recent = df.tail(20)
    for col in ['1st_real', '2nd_real', '3rd_real']:
        for num in recent[col].astype(str):
            if len(num) == 4 and num.isdigit():
                # Original number
                candidates[num] = candidates.get(num, 0) + 50
                
                # Reverse
                rev = num[::-1]
                candidates[rev] = candidates.get(rev, 0) + 30
                
                # Rotate digits
                for i in range(4):
                    rotated = num[i:] + num[:i]
                    candidates[rotated] = candidates.get(rotated, 0) + 20
    
    # Method 3: Hot digit combinations
    for d1 in hot_digits[:4]:
        for d2 in hot_digits[:4]:
            for d3 in hot_digits[:4]:
                for d4 in hot_digits[:4]:
                    num = d1 + d2 + d3 + d4
                    score = (analysis['digit_freq'][d1] + 
                            analysis['digit_freq'][d2] + 
                            analysis['digit_freq'][d3] + 
                            analysis['digit_freq'][d4])
                    candidates[num] = candidates.get(num, 0) + score * 0.5
    
    # Sort by score
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N with normalized scores
    max_score = sorted_candidates[0][1] if sorted_candidates else 1
    results = []
    for num, score in sorted_candidates[:top_n]:
        normalized_score = score / max_score
        reason = f"freq={analysis['number_freq'].get(num, 0)}"
        results.append((num, normalized_score, reason))
    
    return results
