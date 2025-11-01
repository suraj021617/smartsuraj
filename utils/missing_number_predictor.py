"""
Missing Number Predictor - Finds and predicts empty boxes in Special/Consolation
"""
from collections import Counter
import re

def detect_missing_positions(special_text, consolation_text):
    """
    Detect which positions are empty in special and consolation
    Returns: {position: 'A', 'B', 'C'...}
    """
    missing = []
    
    # Parse special numbers
    special_nums = []
    if special_text:
        parts = str(special_text).split()
        for i, part in enumerate(parts):
            if part in ['----', '---', '--', '']:
                missing.append({
                    'section': 'Special',
                    'position': i + 1,
                    'label': chr(65 + len(missing))  # A, B, C...
                })
            elif len(part) == 4 and part.isdigit():
                special_nums.append(part)
    
    # Parse consolation numbers
    consolation_nums = []
    if consolation_text:
        parts = str(consolation_text).split()
        for i, part in enumerate(parts):
            if part in ['----', '---', '--', '']:
                missing.append({
                    'section': 'Consolation',
                    'position': i + 1,
                    'label': chr(65 + len(missing))
                })
            elif len(part) == 4 and part.isdigit():
                consolation_nums.append(part)
    
    return missing, special_nums, consolation_nums

def predict_missing_numbers(df, missing_positions, current_prizes):
    """
    Predict what numbers should fill the missing positions
    Based on:
    1. Frequency analysis of special/consolation numbers
    2. Pattern matching with current prizes
    3. Historical position analysis
    """
    predictions = []
    
    # Collect all historical special and consolation numbers
    all_special = []
    all_consolation = []
    
    for _, row in df.iterrows():
        if 'special' in row and row['special']:
            for num in str(row['special']).split():
                if len(num) == 4 and num.isdigit():
                    all_special.append(num)
        
        if 'consolation' in row and row['consolation']:
            for num in str(row['consolation']).split():
                if len(num) == 4 and num.isdigit():
                    all_consolation.append(num)
    
    # Frequency analysis
    special_freq = Counter(all_special)
    consolation_freq = Counter(all_consolation)
    
    # Get current prize digits for pattern matching
    prize_digits = set()
    for prize in current_prizes:
        prize_digits.update(prize)
    
    for missing in missing_positions:
        candidates = {}
        
        # Choose frequency source based on section
        if missing['section'] == 'Special':
            freq_source = special_freq
        else:
            freq_source = consolation_freq
        
        # Score candidates
        for num, count in freq_source.most_common(100):
            score = count  # Base score from frequency
            
            # Bonus: shares digits with current prizes
            shared_digits = len(set(num) & prize_digits)
            score += shared_digits * 5
            
            # Bonus: digit sum patterns
            digit_sum = sum(int(d) for d in num)
            if 15 <= digit_sum <= 25:  # Common range
                score += 3
            
            candidates[num] = score
        
        # Get top 5 predictions
        top_5 = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:5]
        
        predictions.append({
            'label': missing['label'],
            'section': missing['section'],
            'position': missing['position'],
            'predictions': [{'number': num, 'score': score, 'confidence': min(95, 50 + score)} 
                          for num, score in top_5]
        })
    
    return predictions

def analyze_position_patterns(df, position, section='special'):
    """
    Analyze what numbers typically appear at a specific position
    """
    position_numbers = []
    
    for _, row in df.iterrows():
        col = 'special' if section == 'special' else 'consolation'
        if col in row and row[col]:
            parts = str(row[col]).split()
            if position < len(parts):
                num = parts[position]
                if len(num) == 4 and num.isdigit():
                    position_numbers.append(num)
    
    return Counter(position_numbers).most_common(10)
