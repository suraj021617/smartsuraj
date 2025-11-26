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

    # Method 3: Pattern-based variations
    pattern_predictions = []

    # Common transformations
    transformations = [
        lambda x: x,  # Original
        lambda x: str(int(x) + 1).zfill(4),  # +1
        lambda x: str(int(x) - 1).zfill(4),  # -1
        lambda x: x[::-1],  # Reverse
        lambda x: x[1:] + x[0],  # Rotate left
        lambda x: x[-1] + x[:-1],  # Rotate right
    ]

    for transform in transformations:
        try:
            candidate = transform(today_num)
            if len(candidate) == 4 and candidate.isdigit():
                # Check if this transformation occurred in history
                transform_freq = 0
                for recent in recent_nums[-20:]:  # Check last 20 numbers
                    try:
                        if transform(str(recent)) == candidate:
                            transform_freq += 1
                    except:
                        continue

                if transform_freq > 0:
                    confidence = min(transform_freq / 10, 0.6)  # Cap at 60%
                    pattern_predictions.append((candidate, confidence, "pattern_transform"))
        except:
            continue
