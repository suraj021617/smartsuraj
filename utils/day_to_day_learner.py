"""
Day-to-Day Pattern Learning Module
"""
from collections import defaultdict, Counter

def learn_day_to_day_patterns(draws):
    """
    FAST OPTIMIZED - Learn patterns from recent draws only
    """
    if not draws or len(draws) < 2:
        return {}

    # Use only last 500 for speed
    draws = draws[-500:] if len(draws) > 500 else draws

    patterns = {
        'digit_transitions': defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        'sequence_patterns': {}
    }

    for i in range(len(draws) - 1):
        current_num = str(draws[i].get('number', ''))
        next_num = str(draws[i + 1].get('number', ''))

        if len(current_num) == 4 and len(next_num) == 4:
            # Digit transitions only
            for pos in range(4):
                patterns['digit_transitions'][current_num[pos]][pos][next_num[pos]] += 1
            
            # Limited sequence storage
            if len(patterns['sequence_patterns']) < 1000:
                if current_num not in patterns['sequence_patterns']:
                    patterns['sequence_patterns'][current_num] = {}
                patterns['sequence_patterns'][current_num][next_num] = \
                    patterns['sequence_patterns'][current_num].get(next_num, 0) + 1

    return patterns

def predict_tomorrow(today_nums, patterns, recent_nums):
    """
    Predict tomorrow's numbers based on learned patterns - FIXED for diversity
    """
    if not patterns or not today_nums:
        return []

    predictions = defaultdict(lambda: {'score': 0, 'reasons': []})
    
    # Use ALL today's numbers, not just the last one
    for today_num in set(str(n) for n in today_nums[-10:]):
        if len(today_num) != 4:
            continue

        # Method 1: Direct sequence patterns (HIGHEST PRIORITY)
        if today_num in patterns['sequence_patterns']:
            next_candidates = patterns['sequence_patterns'][today_num]
            total_occurrences = sum(next_candidates.values())

            for next_num, count in next_candidates.items():
                confidence = count / total_occurrences
                predictions[next_num]['score'] += confidence * 3.0  # High weight
                predictions[next_num]['reasons'].append(f"Follows {today_num}")

    # Method 2: Build diverse candidates from digit transitions
    digit_transitions = patterns.get('digit_transitions', {})
    
    # Get most likely digit for each position
    position_digits = []
    for pos in range(4):
        digit_scores = defaultdict(float)
        for today_num in set(str(n) for n in today_nums[-5:]):
            if len(str(today_num)) == 4:
                current_digit = str(today_num)[pos]
                if current_digit in digit_transitions and pos in digit_transitions[current_digit]:
                    for next_digit, count in digit_transitions[current_digit][pos].items():
                        digit_scores[next_digit] += count
        
        # Get top 3 digits for this position
        top_digits = sorted(digit_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        position_digits.append([d for d, s in top_digits] if top_digits else ['0', '1', '2'])
    
    # Generate diverse combinations
    import itertools
    for combo in itertools.product(*position_digits):
        candidate = ''.join(combo)
        if candidate not in predictions:
            predictions[candidate]['score'] += 0.5
            predictions[candidate]['reasons'].append('digit_transition')
    
    # Method 3: Use recent hot numbers
    recent_freq = Counter(str(n) for n in recent_nums[-30:])
    for num, count in recent_freq.most_common(15):
        if len(num) == 4 and num.isdigit():
            predictions[num]['score'] += (count / 30) * 0.8
            predictions[num]['reasons'].append('hot_number')
    
    # Convert to list format
    result = []
    for num, data in predictions.items():
        if len(num) == 4 and num.isdigit():
            reason = '+'.join(data['reasons'][:2])
            result.append((num, data['score'], reason))
    
    # Sort by score and return diverse top predictions
    result.sort(key=lambda x: x[1], reverse=True)
    
    # Ensure diversity - no similar numbers
    diverse_results = []
    for num, score, reason in result:
        # Check if too similar to existing predictions
        is_diverse = True
        for existing, _, _ in diverse_results:
            # Count matching digits in same positions
            matches = sum(1 for i in range(4) if num[i] == existing[i])
            if matches >= 3:  # Too similar
                is_diverse = False
                break
        
        if is_diverse:
            diverse_results.append((num, score, reason))
        
        if len(diverse_results) >= 10:
            break
    
    return diverse_results[:10]
