from utils.app_grid import generate_4x4_grid

def convert_to_grid(number):
    return [digit for row in generate_4x4_grid(number) for digit in row]

def detect_patterns(grid):
    # Basic placeholder: returns unique digits as "patterns"
    return list(set(grid))

def find_missing_digits(grid):
    return [d for d in range(10) if d not in grid]

def learn_pattern_transitions(past_draws):
    transitions = {}

    for i in range(len(past_draws) - 1):
        today = past_draws[i]
        tomorrow = past_draws[i + 1]

        today_grid = convert_to_grid(today["number"])
        today_patterns = detect_patterns(today_grid)
        today_missing = find_missing_digits(today_grid)

        for pattern in today_patterns:
            transitions.setdefault(pattern, []).append(tomorrow["number"])

        for digit in today_missing:
            key = f"missing_{digit}"
            transitions.setdefault(key, []).append(tomorrow["number"])

    return transitions