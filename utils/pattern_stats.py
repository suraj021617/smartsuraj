from collections import defaultdict

def compute_pattern_frequencies(draws):
    pattern_counts = defaultdict(int)
    for draw in draws:
        for kind, idx, value, coords in draw.get("patterns", []):
            pattern_counts[kind] += 1
    return sorted(pattern_counts.items(), key=lambda x: -x[1])

def compute_cell_heatmap(draws):
    cell_counts = [[0] * 4 for _ in range(4)]
    for draw in draws:
        for patterns in [draw.get('highlight', []), draw.get('reverse_highlight', [])]:
            for coords in patterns:
                if isinstance(coords, (tuple, list)) and len(coords) == 2:
                    i, j = coords
                    if 0 <= i < 4 and 0 <= j < 4:
                        cell_counts[i][j] += 1
    return cell_counts
