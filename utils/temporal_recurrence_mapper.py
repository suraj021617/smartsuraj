"""
Temporal Recurrence Mapping
Track when specific patterns tend to recur
"""
from collections import defaultdict, Counter
import numpy as np

def map_temporal_recurrence(df):
    """
    Track pattern recurrence intervals and detect cyclical behavior
    """
    df_sorted = df.sort_values('date_parsed').reset_index(drop=True)
    
    pattern_timeline = defaultdict(list)  # {pattern: [draw_indices]}
    
    for idx, row in df_sorted.iterrows():
        date = row['date_parsed']
        draw_id = f"{date.date()}_{row.get('provider', 'unknown')}"
        
        # Extract numbers and create patterns
        numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                numbers.append(int(num))
        
        if len(numbers) < 2:
            continue
        
        # Pattern: sorted gaps
        sorted_nums = sorted(numbers)
        gaps = tuple(sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1))
        pattern_timeline[gaps].append((idx, date.date(), draw_id))
        
        # Pattern: digit sum
        digit_sum = sum(int(d) for num in numbers for d in str(num))
        pattern_timeline[f"sum_{digit_sum}"].append((idx, date.date(), draw_id))
    
    # Calculate recurrence intervals
    recurrence_stats = []
    for pattern, occurrences in pattern_timeline.items():
        if len(occurrences) < 2:
            continue
        
        indices = [occ[0] for occ in occurrences]
        intervals = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        
        if not intervals:
            continue
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals) if len(intervals) > 1 else 0
        
        # Predict next recurrence
        last_idx = indices[-1]
        next_expected = int(last_idx + avg_interval)
        
        recurrence_stats.append({
            'pattern': str(pattern),
            'frequency': len(occurrences),
            'avg_interval': round(avg_interval, 1),
            'std_interval': round(std_interval, 1),
            'last_seen': occurrences[-1][1],
            'next_expected_draw': next_expected,
            'draw_ids': [occ[2] for occ in occurrences[-5:]]
        })
    
    recurrence_stats.sort(key=lambda x: x['frequency'], reverse=True)
    return recurrence_stats[:50]
