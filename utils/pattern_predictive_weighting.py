"""
Pattern Predictive Weighting
Score patterns based on their predictive value
"""
from collections import defaultdict, Counter
import numpy as np

def weight_patterns_predictively(df):
    """
    Assign weights to patterns based on predictive success
    """
    df_sorted = df.sort_values('date_parsed').reset_index(drop=True)
    
    pattern_success = defaultdict(lambda: {'hits': 0, 'attempts': 0, 'recency_score': 0})
    
    for idx in range(len(df_sorted) - 1):
        curr_row = df_sorted.iloc[idx]
        next_row = df_sorted.iloc[idx + 1]
        
        # Current draw patterns
        curr_numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(curr_row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                curr_numbers.append(int(num))
        
        # Next draw numbers
        next_numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(next_row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                next_numbers.append(int(num))
        
        if len(curr_numbers) < 2 or len(next_numbers) < 1:
            continue
        
        # Pattern: gap structure
        sorted_curr = sorted(curr_numbers)
        gaps = tuple(sorted_curr[i+1] - sorted_curr[i] for i in range(len(sorted_curr)-1))
        
        pattern_success[gaps]['attempts'] += 1
        
        # Check if pattern "predicted" next draw (any overlap)
        if any(num in next_numbers for num in curr_numbers):
            pattern_success[gaps]['hits'] += 1
        
        # Recency score (more recent = higher score)
        recency = 1.0 / (len(df_sorted) - idx)
        pattern_success[gaps]['recency_score'] += recency
    
    # Calculate weighted scores
    weighted_patterns = []
    for pattern, stats in pattern_success.items():
        if stats['attempts'] == 0:
            continue
        
        success_rate = stats['hits'] / stats['attempts']
        recency_weight = stats['recency_score']
        
        # Combined confidence score
        confidence = (success_rate * 0.6) + (recency_weight * 0.4)
        
        weighted_patterns.append({
            'pattern': str(pattern),
            'success_rate': round(success_rate * 100, 1),
            'hits': stats['hits'],
            'attempts': stats['attempts'],
            'confidence': round(confidence * 100, 1),
            'recency_score': round(recency_weight, 4)
        })
    
    weighted_patterns.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Generate suggested combinations from top patterns
    top_patterns = weighted_patterns[:10]
    suggestions = []
    for p in top_patterns:
        try:
            gaps = eval(p['pattern'])
            if isinstance(gaps, tuple) and len(gaps) > 0:
                # Generate sample numbers from gap pattern
                base = 1000
                numbers = [base]
                for gap in gaps:
                    numbers.append(numbers[-1] + gap)
                suggestions.append({
                    'pattern': p['pattern'],
                    'sample_numbers': numbers,
                    'confidence': p['confidence']
                })
        except:
            pass
    
    return {
        'weighted_leaderboard': weighted_patterns[:30],
        'suggested_combinations': suggestions[:10]
    }
