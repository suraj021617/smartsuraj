"""
Arithmetic & Gap Pattern Analyzer
Identifies draws with consistent gaps and arithmetic progressions
"""
from collections import Counter

def analyze_arithmetic_gaps(df):
    """
    Analyze arithmetic progressions and gap patterns in draws
    Returns:
    - gap_frequencies: Most frequent gap sizes
    - arithmetic_progressions: Draws with perfect/near-perfect progressions
    - mirrored_gaps: Symmetric gap patterns
    """
    df_sorted = df.sort_values('date_parsed').reset_index(drop=True)
    
    gap_frequencies = Counter()
    arithmetic_progressions = []
    mirrored_gaps = []
    all_gaps = []
    
    for idx, row in df_sorted.iterrows():
        date = row['date_parsed']
        draw_id = f"{date.date()}_{row.get('provider', 'unknown')}"
        
        # Get all prize numbers for this draw
        numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                numbers.append(int(num))
        
        if len(numbers) < 2:
            continue
        
        # Sort numbers to calculate gaps
        sorted_nums = sorted(numbers)
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
        
        # Track gap frequencies
        for gap in gaps:
            gap_frequencies[gap] += 1
            all_gaps.append((gap, date.date(), draw_id))
        
        # Check for arithmetic progression (constant gap)
        if len(set(gaps)) == 1:
            arithmetic_progressions.append({
                'pattern_type': 'Perfect Arithmetic Progression',
                'numbers': sorted_nums,
                'gap': gaps[0],
                'frequency': 1,
                'last_seen': date.date(),
                'draw_ids': [draw_id]
            })
        elif len(gaps) >= 2 and abs(gaps[0] - gaps[1]) <= 5:
            arithmetic_progressions.append({
                'pattern_type': 'Near-Perfect Progression',
                'numbers': sorted_nums,
                'gaps': gaps,
                'frequency': 1,
                'last_seen': date.date(),
                'draw_ids': [draw_id]
            })
        
        # Check for mirrored gaps (e.g., [+10, +20, +10])
        if len(gaps) >= 2 and gaps[0] == gaps[-1]:
            mirrored_gaps.append({
                'pattern_type': 'Mirrored Gap Pattern',
                'numbers': sorted_nums,
                'gaps': gaps,
                'frequency': 1,
                'last_seen': date.date(),
                'draw_ids': [draw_id]
            })
    
    # Aggregate gap frequencies
    gap_freq_list = []
    for gap, count in gap_frequencies.most_common(20):
        matching_gaps = [g for g in all_gaps if g[0] == gap]
        last_seen = matching_gaps[-1][1] if matching_gaps else None
        draw_ids = [g[2] for g in matching_gaps[-5:]]
        gap_freq_list.append({
            'pattern_type': f'Gap of {gap}',
            'gap_size': gap,
            'frequency': count,
            'last_seen': last_seen,
            'draw_ids': draw_ids
        })
    
    # Consolidate arithmetic progressions by gap size
    prog_by_gap = {}
    for prog in arithmetic_progressions:
        if prog['pattern_type'] == 'Perfect Arithmetic Progression':
            gap = prog['gap']
            if gap not in prog_by_gap:
                prog_by_gap[gap] = {
                    'pattern_type': f'Arithmetic Progression (gap={gap})',
                    'gap': gap,
                    'frequency': 0,
                    'last_seen': None,
                    'draw_ids': []
                }
            prog_by_gap[gap]['frequency'] += 1
            prog_by_gap[gap]['last_seen'] = prog['last_seen']
            prog_by_gap[gap]['draw_ids'].extend(prog['draw_ids'])
    
    consolidated_progressions = list(prog_by_gap.values())
    
    return {
        'gap_frequencies': gap_freq_list,
        'arithmetic_progressions': consolidated_progressions,
        'mirrored_gaps': mirrored_gaps[:20]
    }

def find_high_prize_precursors(df):
    """
    Find patterns that tend to precede high-prize (1st prize) draws
    """
    df_sorted = df.sort_values('date_parsed').reset_index(drop=True)
    
    precursor_patterns = Counter()
    
    for idx in range(len(df_sorted) - 1):
        curr_row = df_sorted.iloc[idx]
        next_row = df_sorted.iloc[idx + 1]
        
        # Get current draw gaps
        curr_numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(curr_row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                curr_numbers.append(int(num))
        
        if len(curr_numbers) < 2:
            continue
        
        sorted_nums = sorted(curr_numbers)
        gaps = tuple(sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1))
        
        # Check if next draw has 1st prize
        next_1st = str(next_row.get('1st_real', ''))
        if next_1st and next_1st.isdigit() and len(next_1st) == 4:
            precursor_patterns[gaps] += 1
    
    precursor_list = []
    for gaps, count in precursor_patterns.most_common(10):
        precursor_list.append({
            'pattern_type': 'High-Prize Precursor',
            'gap_pattern': list(gaps),
            'frequency': count
        })
    
    return precursor_list
