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
    Predict tomorrow's numbers based on learned patterns
    """
    if not patterns or not today_nums:
        return []

    predictions = []
    confidence_scores = []

    # Get the most recent number as reference
    today_num = str(today_nums[-1]) if today_nums else ""
    if len(today_num) != 4:
        return []

    # Method 1: Direct sequence patterns
    if today_num in patterns['sequence_patterns']:
        next_candidates = patterns['sequence_patterns'][today_num]
        total_occurrences = sum(next_candidates.values())

        for next_num, count in next_candidates.items():
            confidence = count / total_occurrences
            predictions.append((next_num, confidence, "sequence_pattern"))

    # Method 2: Digit transition patterns
    transition_predictions = defaultdict(float)

    for pos in range(4):
        current_digit = today_num[pos]
        if current_digit in patterns['digit_transitions'] and pos in patterns['digit_transitions'][current_digit]:
            transitions = patterns['digit_transitions'][current_digit][pos]
            total_transitions = sum(transitions.values())

            for next_digit, count in transitions.items():
                transition_confidence = count / total_transitions

                # Build candidate numbers by replacing digit at position
                for base_num in [today_num] + recent_nums[-5:]:  # Use today + last 5 as base
                    if len(str(base_num)) == 4:
                        candidate = list(str(base_num))
                        candidate[pos] = next_digit
                        candidate_num = ''.join(candidate)

                        transition_predictions[candidate_num] += transition_confidence

    # Convert transition predictions to list
    for candidate, confidence in transition_predictions.items():
        predictions.append((candidate, confidence * 0.8, "digit_transition"))  # Slightly lower confidence

    # Method 3: Pattern-based variations (SIMPLIFIED)
    pattern_predictions = []
    recent_set = set(str(x) for x in recent_nums[-20:])

    # Quick transformations
    try:
        candidates = [
            (today_num, 0.3, "original"),
            (str(int(today_num) + 1).zfill(4), 0.25, "plus_one"),
            (str(int(today_num) - 1).zfill(4), 0.25, "minus_one"),
            (today_num[::-1], 0.2, "reverse")
        ]
        
        for cand, conf, reason in candidates:
            if len(cand) == 4 and cand.isdigit() and cand in recent_set:
                pattern_predictions.append((cand, conf, reason))
    except:
        pass

    predictions.extend(pattern_predictions)

    # Remove duplicates and sort by confidence
    seen = set()
    unique_predictions = []
    for num, conf, reason in predictions:
        if num not in seen:
            seen.add(num)
            unique_predictions.append((num, conf, reason))

    # Sort by confidence and return top predictions
    unique_predictions.sort(key=lambda x: x[1], reverse=True)
    return unique_predictions[:10]  # Return top 10
