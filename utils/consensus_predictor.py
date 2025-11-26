from collections import Counter

def get_consensus_predictions(df, provider='all', top_n=10):
    """
    Combines ALL prediction methods and finds numbers that appear most frequently.
    Higher consensus = more reliable prediction.
    """
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    all_predictions = []
    
    # Method 1: Hot numbers (last 20 draws)
    recent_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums = df[col].tail(20).tolist()
        recent_nums.extend([str(n) for n in nums if len(str(n)) == 4])
    
    freq = Counter(recent_nums)
    for num, count in freq.most_common(15):
        all_predictions.append(num)
    
    # Method 2: Trending (last 10 vs previous 10)
    last_10 = []
    prev_10 = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        last_10.extend([str(n) for n in df[col].tail(10).tolist() if len(str(n)) == 4])
        prev_10.extend([str(n) for n in df[col].iloc[-20:-10].tolist() if len(str(n)) == 4])
    
    last_freq = Counter(last_10)
    prev_freq = Counter(prev_10)
    
    for num in last_freq:
        if last_freq[num] > prev_freq.get(num, 0):
            all_predictions.append(num)
    
    # Method 3: Digit frequency
    all_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums = df[col].tail(30).tolist()
        all_nums.extend([str(n) for n in nums if len(str(n)) == 4])
    
    digit_freq = Counter(''.join(all_nums))
    hot_digits = [d for d, _ in digit_freq.most_common(6)]
    
    # Generate numbers with hot digits
    for num in all_nums[-50:]:
        hot_count = sum(1 for d in num if d in hot_digits)
        if hot_count >= 3:
            all_predictions.append(num)
    
    # Method 4: Pairs
    pair_freq = Counter([n[i:i+2] for n in all_nums for i in range(3)])
    hot_pairs = [p for p, _ in pair_freq.most_common(10)]
    
    for num in all_nums[-50:]:
        pair_count = sum(1 for i in range(3) if num[i:i+2] in hot_pairs)
        if pair_count >= 2:
            all_predictions.append(num)
    
    # Count consensus
    consensus = Counter(all_predictions)
    
    # Calculate confidence based on how many methods picked it
    results = []
    for num, count in consensus.most_common(top_n * 2):
        confidence = min(95, (count / len(all_predictions) * 100) * 20)
        methods = []
        
        if num in [n for n, _ in freq.most_common(15)]:
            methods.append('Hot')
        if num in last_freq and last_freq[num] > prev_freq.get(num, 0):
            methods.append('Trending')
        if sum(1 for d in num if d in hot_digits) >= 3:
            methods.append('Hot Digits')
        if sum(1 for i in range(3) if num[i:i+2] in hot_pairs) >= 2:
            methods.append('Hot Pairs')
        
        results.append({
            'number': num,
            'consensus': count,
            'confidence': round(confidence, 1),
            'methods': methods,
            'reason': f"Picked by {len(methods)} methods ({', '.join(methods)})"
        })
    
    return sorted(results, key=lambda x: (x['consensus'], x['confidence']), reverse=True)[:top_n]
