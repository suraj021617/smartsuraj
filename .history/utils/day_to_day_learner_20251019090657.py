"""
Day-to-Day Pattern Learning Module
"""
from collections import defaultdict, Counter

def learn_day_to_day_patterns(draws):
    """
    Learn day-to-day patterns from historical draws
    """
    if not draws or len(draws) < 2:
        return {}
    
    patterns = {
        'digit_transitions': defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        'sequence_patterns': defaultdict(lambda: defaultdict(int))
    }
    
    for i in range(len(draws) - 1):
        current_num = str(draws[i].get('number', ''))
        next_num = str(draws[i + 1].get('number', ''))
        
        if len(current_num) == 4 and len(next_num) == 4:
            # Learn digit transitions
            for pos in range(4):
                current_digit = current_num[pos]
                next_digit = next_num[pos]
                patterns['digit_transitions'][current_digit][pos][next_digit] += 1
            
            # Learn sequence patterns
            patterns['sequence_patterns'][current_num][next_num] += 1
    
    return patterns