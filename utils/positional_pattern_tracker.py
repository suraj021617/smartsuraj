"""
Positional Pattern Tracker
Analyzes number positions (Box 1, Box 2, etc.) across draws
"""
from collections import defaultdict, Counter

def analyze_positional_patterns(df):
    """
    Analyze patterns in specific positions across draws
    Returns:
    - position_heatmap: Frequency of each number per box
    - repeating_positions: Numbers that repeat in same positions
    - positional_swaps: Common position swaps
    - placeholder_patterns: Patterns like "----" in specific positions
    """
    df_sorted = df.sort_values('date_parsed').reset_index(drop=True)
    
    # Track which numbers appear in which positions
    position_numbers = defaultdict(Counter)  # {position: {number: count}}
    position_history = defaultdict(list)  # {position: [(date, number, draw_id)]}
    swap_patterns = Counter()  # Track position swaps
    placeholder_count = defaultdict(int)  # Track placeholders per position
    
    for idx, row in df_sorted.iterrows():
        date = row['date_parsed']
        draw_id = f"{date.date()}_{row.get('provider', 'unknown')}"
        
        # Analyze all prize positions
        positions = []
        for pos_idx, col in enumerate(['1st_real', '2nd_real', '3rd_real'], 1):
            num = str(row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                position_numbers[f"Box{pos_idx}"][num] += 1
                position_history[f"Box{pos_idx}"].append((date, num, draw_id))
                positions.append((pos_idx, num))
            elif num in ['----', '', 'nan', 'None']:
                placeholder_count[f"Box{pos_idx}"] += 1
        
        # Detect swaps between consecutive draws
        if idx > 0:
            prev_row = df_sorted.iloc[idx - 1]
            prev_positions = {}
            for pos_idx, col in enumerate(['1st_real', '2nd_real', '3rd_real'], 1):
                num = str(prev_row.get(col, ''))
                if num and num.isdigit() and len(num) == 4:
                    prev_positions[num] = pos_idx
            
            for curr_pos, curr_num in positions:
                if curr_num in prev_positions:
                    prev_pos = prev_positions[curr_num]
                    if prev_pos != curr_pos:
                        swap_patterns[f"Box{prev_pos}↔Box{curr_pos}"] += 1
    
    # Build heatmap: frequency of each number per box
    position_heatmap = {}
    for position, counter in position_numbers.items():
        position_heatmap[position] = [
            {'number': num, 'count': count, 'percentage': round((count / sum(counter.values())) * 100, 2)}
            for num, count in counter.most_common(10)
        ]
    
    # Find repeating numbers in same positions
    repeating_positions = []
    for position, counter in position_numbers.items():
        for num, count in counter.most_common(20):
            if count >= 2:
                history = [h for h in position_history[position] if h[1] == num]
                last_seen = history[-1][0].date() if history else None
                repeating_positions.append({
                    'pattern_type': f'Repeating in {position}',
                    'number': num,
                    'frequency': count,
                    'last_seen': last_seen,
                    'draw_ids': [h[2] for h in history[-5:]]
                })
    
    # Common positional swaps
    positional_swaps = []
    for swap, count in swap_patterns.most_common(20):
        positional_swaps.append({
            'pattern_type': 'Positional Swap',
            'swap': swap,
            'frequency': count,
            'last_seen': None
        })
    
    # Detect placeholder patterns (missing or invalid data)
    placeholder_patterns = []
    for position, count in placeholder_count.items():
        if count > 0:
            placeholder_patterns.append({
                'pattern_type': f'Placeholder in {position}',
                'frequency': count,
                'last_seen': None
            })
    
    return {
        'position_heatmap': position_heatmap,
        'repeating_positions': repeating_positions,
        'positional_swaps': positional_swaps,
        'placeholder_patterns': placeholder_patterns
    }
