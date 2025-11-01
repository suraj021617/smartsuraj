"""
Advanced Empty Box Analyzer - Learns from ALL historical data
"""
from collections import Counter, defaultdict

def parse_grid_data(text):
    """Parse special/consolation text into grid with empty detection"""
    parts = [p.strip() for p in str(text).split() if p.strip()]
    grid = []
    for i in range(0, min(len(parts), 10), 5):
        row = parts[i:i+5]
        while len(row) < 5:
            row.append('')
        grid.append(row)
    return grid

def find_empty_boxes(grid, section_name):
    """Find all empty boxes in a grid"""
    empty = []
    for row_idx, row in enumerate(grid):
        for col_idx, val in enumerate(row):
            if val in ['----', '---', '--', ''] or (val and not val.isdigit()):
                empty.append({
                    'section': section_name,
                    'row': row_idx + 1,
                    'col': col_idx + 1,
                    'pos': row_idx * 5 + col_idx + 1
                })
    return empty

def analyze_empty_patterns(df, provider='all'):
    """Analyze which positions are empty most often"""
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    position_freq = Counter()
    section_freq = Counter()
    total_draws = 0
    
    for _, row in df.iterrows():
        special_grid = parse_grid_data(row.get('special', ''))
        consolation_grid = parse_grid_data(row.get('consolation', ''))
        
        special_empty = find_empty_boxes(special_grid, 'Special')
        consolation_empty = find_empty_boxes(consolation_grid, 'Consolation')
        
        for emp in special_empty + consolation_empty:
            key = f"{emp['section']}-Pos{emp['pos']}"
            position_freq[key] += 1
            section_freq[emp['section']] += 1
        
        total_draws += 1
    
    return {
        'position_freq': position_freq,
        'section_freq': section_freq,
        'total_draws': total_draws
    }

def predict_empty_positions(analysis, top_n=10):
    """Predict which positions will be empty next"""
    predictions = []
    total = analysis['total_draws']
    
    for pos_label, count in analysis['position_freq'].most_common(top_n):
        section, pos = pos_label.split('-Pos')
        probability = (count / total) * 100
        confidence = min(95, probability * 3)
        
        predictions.append({
            'section': section,
            'position': int(pos),
            'row': ((int(pos) - 1) // 5) + 1,
            'col': ((int(pos) - 1) % 5) + 1,
            'frequency': count,
            'probability': round(probability, 1),
            'confidence': round(confidence, 1)
        })
    
    return predictions

def get_numbers_for_position(df, section, position, provider='all'):
    """Get numbers that appeared in specific position historically"""
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    numbers = []
    col_name = 'special' if section == 'Special' else 'consolation'
    
    for _, row in df.iterrows():
        grid = parse_grid_data(row.get(col_name, ''))
        row_idx = (position - 1) // 5
        col_idx = (position - 1) % 5
        
        if row_idx < len(grid) and col_idx < len(grid[row_idx]):
            val = grid[row_idx][col_idx]
            if val and val.isdigit() and len(val) == 4:
                numbers.append(val)
    
    freq = Counter(numbers)
    return freq.most_common(10)
