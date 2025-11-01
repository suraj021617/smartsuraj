from collections import Counter, defaultdict
from utils.app_grid import generate_4x4_grid, generate_reverse_grid
from utils.pattern_finder import find_all_4digit_patterns

def learn_from_grids(df, provider=None):
    """Learn patterns from 4x4 grids of winning numbers"""
    
    if provider and provider != 'all':
        df = df[df['provider'] == provider]
    
    # Extract all winning numbers
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums = df[col].astype(str).tolist()
        all_numbers.extend([n for n in nums if len(n) == 4 and n.isdigit()])
    
    if not all_numbers:
        return None
    
    # Learn from grids
    pattern_to_next = defaultdict(list)  # What patterns lead to what numbers
    grid_cell_freq = [[Counter() for _ in range(4)] for _ in range(4)]  # Frequency per cell
    successful_patterns = Counter()  # Patterns that appeared in next draw
    
    for i in range(len(all_numbers) - 1):
        current_num = all_numbers[i]
        next_num = all_numbers[i + 1]
        
        # Generate grids
        grid = generate_4x4_grid(current_num)
        reverse_grid = generate_reverse_grid(current_num)
        
        # Learn cell frequencies
        for row_idx, row in enumerate(grid):
            for col_idx, cell in enumerate(row):
                grid_cell_freq[row_idx][col_idx][cell] += 1
        
        # Extract patterns from grid
        patterns = find_all_4digit_patterns(grid)
        reverse_patterns = find_all_4digit_patterns(reverse_grid)
        
        # Record what patterns led to next number
        for kind, idx, pattern, coords in patterns:
            pattern_to_next[pattern].append(next_num)
            if pattern == next_num:
                successful_patterns[pattern] += 1
        
        for kind, idx, pattern, coords in reverse_patterns:
            pattern_to_next[f"rev_{pattern}"].append(next_num)
            if pattern == next_num:
                successful_patterns[f"rev_{pattern}"] += 1
    
    return {
        'pattern_to_next': pattern_to_next,
        'grid_cell_freq': grid_cell_freq,
        'successful_patterns': successful_patterns,
        'total_samples': len(all_numbers)
    }

def predict_from_learned_patterns(df, provider=None, top_n=5):
    """Generate predictions based on learned grid patterns"""
    
    learned = learn_from_grids(df, provider)
    if not learned:
        return []
    
    # Get the most recent number to analyze
    if provider and provider != 'all':
        recent_df = df[df['provider'] == provider]
    else:
        recent_df = df
    
    recent_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums = recent_df[col].astype(str).tail(10).tolist()
        recent_numbers.extend([n for n in nums if len(n) == 4 and n.isdigit()])
    
    if not recent_numbers:
        return []
    
    last_number = recent_numbers[-1]
    
    # Generate grid from last number
    grid = generate_4x4_grid(last_number)
    reverse_grid = generate_reverse_grid(last_number)
    
    # Extract patterns
    patterns = find_all_4digit_patterns(grid)
    reverse_patterns = find_all_4digit_patterns(reverse_grid)
    
    # Score candidates based on learned patterns
    candidates = Counter()
    
    # Strategy 1: What did these patterns lead to historically?
    for kind, idx, pattern, coords in patterns:
        if pattern in learned['pattern_to_next']:
            next_nums = learned['pattern_to_next'][pattern]
            for num in next_nums:
                candidates[num] += 10
    
    for kind, idx, pattern, coords in reverse_patterns:
        rev_key = f"rev_{pattern}"
        if rev_key in learned['pattern_to_next']:
            next_nums = learned['pattern_to_next'][rev_key]
            for num in next_nums:
                candidates[num] += 8
    
    # Strategy 2: Build from most frequent grid cells
    cell_predictions = []
    for row_idx in range(4):
        for col_idx in range(4):
            most_common = learned['grid_cell_freq'][row_idx][col_idx].most_common(3)
            cell_predictions.extend([digit for digit, _ in most_common])
    
    # Build numbers from hot cells
    if len(cell_predictions) >= 4:
        for i in range(0, len(cell_predictions)-3, 4):
            num = ''.join(cell_predictions[i:i+4])
            if len(num) == 4 and num.isdigit():
                candidates[num] += 5
    
    # Strategy 3: Successful patterns that repeated
    for pattern, count in learned['successful_patterns'].most_common(20):
        pattern_clean = pattern.replace('rev_', '')
        if len(pattern_clean) == 4 and pattern_clean.isdigit():
            candidates[pattern_clean] += count * 2
    
    # Strategy 4: Patterns from current grid
    for kind, idx, pattern, coords in patterns:
        if len(pattern) == 4 and pattern.isdigit():
            candidates[pattern] += 15
    
    for kind, idx, pattern, coords in reverse_patterns:
        if len(pattern) == 4 and pattern.isdigit():
            candidates[pattern] += 12
    
    # Sort and format results
    sorted_candidates = candidates.most_common(top_n)
    
    if not sorted_candidates:
        return []
    
    max_score = sorted_candidates[0][1]
    results = []
    
    for num, score in sorted_candidates:
        normalized_score = score / max_score if max_score > 0 else 0
        
        reasons = []
        if num in [p for _, _, p, _ in patterns]:
            reasons.append("grid-pattern")
        if num in [p for _, _, p, _ in reverse_patterns]:
            reasons.append("reverse-pattern")
        if learned['successful_patterns'].get(num, 0) > 0:
            reasons.append(f"repeated {learned['successful_patterns'][num]}x")
        
        reason = ", ".join(reasons) if reasons else "learned-pattern"
        results.append((num, normalized_score, reason))
    
    return results
