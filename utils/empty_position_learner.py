"""
Empty Position Learner - Learns which positions are empty and predicts next empty positions
"""
from collections import Counter, defaultdict

def parse_special_consolation_grid(text):
    """Parse special/consolation text into grid positions with empty detection"""
    if not text:
        return []
    
    parts = str(text).split()
    positions = []
    
    for i, part in enumerate(parts):
        if part in ['----', '---', '--', '']:
            positions.append({'pos': i, 'value': '----', 'empty': True})
        elif len(part) == 4 and part.isdigit():
            positions.append({'pos': i, 'value': part, 'empty': False})
    
    return positions

def learn_empty_patterns(df):
    """
    Learn patterns of empty positions across all historical data
    Returns: which positions tend to be empty and what numbers fill them
    """
    empty_position_freq = Counter()  # Which positions are empty most often
    position_numbers = defaultdict(Counter)  # What numbers appear at each position
    empty_sequences = []  # Sequences of empty positions
    
    df_sorted = df.sort_values('date_parsed').reset_index(drop=True)
    
    for _, row in df_sorted.iterrows():
        special_grid = parse_special_consolation_grid(row.get('special', ''))
        consolation_grid = parse_special_consolation_grid(row.get('consolation', ''))
        
        # Track empty positions in special
        empty_in_special = []
        for item in special_grid:
            if item['empty']:
                empty_position_freq[('special', item['pos'])] += 1
                empty_in_special.append(item['pos'])
            else:
                position_numbers[('special', item['pos'])][item['value']] += 1
        
        # Track empty positions in consolation
        empty_in_consolation = []
        for item in consolation_grid:
            if item['empty']:
                empty_position_freq[('consolation', item['pos'])] += 1
                empty_in_consolation.append(item['pos'])
            else:
                position_numbers[('consolation', item['pos'])][item['value']] += 1
        
        if empty_in_special or empty_in_consolation:
            empty_sequences.append({
                'date': row['date_parsed'],
                'special_empty': empty_in_special,
                'consolation_empty': empty_in_consolation
            })
    
    return {
        'empty_position_freq': empty_position_freq,
        'position_numbers': position_numbers,
        'empty_sequences': empty_sequences
    }

def predict_next_empty_positions(patterns):
    """Predict which positions will be empty next"""
    empty_freq = patterns['empty_position_freq']
    
    # Get most common empty positions
    predictions = []
    label_idx = 0
    
    for (section, pos), count in empty_freq.most_common(10):
        predictions.append({
            'label': chr(65 + label_idx),  # A, B, C...
            'section': section.capitalize(),
            'position': pos + 1,
            'frequency': count,
            'probability': min(95, count * 5)
        })
        label_idx += 1
    
    return predictions

def predict_numbers_for_position(patterns, section, position):
    """Predict what numbers will appear at a specific position"""
    position_nums = patterns['position_numbers'].get((section, position), Counter())
    
    predictions = []
    for num, count in position_nums.most_common(5):
        predictions.append({
            'number': num,
            'frequency': count,
            'confidence': min(95, 50 + count * 2)
        })
    
    return predictions
