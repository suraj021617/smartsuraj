from utils.app_grid import generate_4x4_grid
from utils.pattern_detector import detect_patterns
from collections import Counter

def convert_to_grid(number):
    """Flatten 4x4 grid into 1D list of digits"""
    return [digit for row in generate_4x4_grid(number) for digit in row]

def find_missing_digits(grid):
    """Digits 0–9 not present in the grid"""
    return [d for d in range(10) if d not in grid]

def predict_from_today_grid(today_number, transitions):
    """
    Generate candidate numbers based on today's grid patterns and missing digits.
    Returns a ranked list of candidate numbers.
    """
    grid = generate_4x4_grid(today_number)
    patterns = detect_patterns(grid)
    missing = find_missing_digits([digit for row in grid for digit in row])

    candidates = []

    # Add candidates from pattern transitions
    for pattern in patterns:
        candidates += transitions.get(pattern, [])

    # Add candidates from missing digit transitions
    for digit in missing:
        key = f"missing_{digit}"
        candidates += transitions.get(key, [])

    # Rank by frequency
    ranked = Counter(candidates).most_common()
    return [num for num, _ in ranked]