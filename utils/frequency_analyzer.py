"""
Frequency Analysis Module - Hot/Cold Numbers, Odd/Even Analysis
"""
from collections import Counter

def analyze_frequency(df, provider='all'):
    """Full frequency analysis from REAL CSV data"""
    if provider != 'all':
        df = df[df['provider'] == provider]
    all_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums = df[col].astype(str).tolist()
        all_nums.extend([n for n in nums if len(n) == 4 and n.isdigit()])
    
    if not all_nums:
        return {'hot_numbers': [], 'cold_numbers': [], 'odd_pct': 0, 'even_pct': 0, 'chart_data': [], 'top_4_prediction': [], 'total_draws': 0}
    
    freq = Counter(all_nums)
    hot_numbers = freq.most_common(10)
    cold_numbers = sorted(freq.items(), key=lambda x: x[1])[:10]
    
    # Odd/Even analysis
    odd_count = sum(1 for n in all_nums if int(n) % 2 == 1)
    even_count = len(all_nums) - odd_count
    total = len(all_nums)
    odd_pct = (odd_count / total * 100) if total > 0 else 0
    even_pct = (even_count / total * 100) if total > 0 else 0
    
    # Chart data (all numbers with frequency)
    chart_data = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:50]
    
    # AI Prediction: Top 4 hottest numbers
    top_4_prediction = [num for num, _ in hot_numbers[:4]]
    top_4_prediction.sort()
    
    return {
        'hot_numbers': hot_numbers,
        'cold_numbers': cold_numbers,
        'odd_pct': round(odd_pct, 2),
        'even_pct': round(even_pct, 2),
        'chart_data': chart_data,
        'top_4_prediction': top_4_prediction,
        'total_draws': total
    }
