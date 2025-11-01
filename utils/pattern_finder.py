# utils/pattern_finder.py

from utils.app_grid import generate_4x4_grid, generate_reverse_grid


def find_all_4digit_patterns(grid):
    rows, cols = 4, 4
    patterns = []

    def kind_of_pattern(path):
        if all(x == path[0][0] for x, y in path):
            return 'row'
        if all(y == path[0][1] for x, y in path):
            return 'col'
        if all(x == y for x, y in path):
            return 'diag_main'
        if all(x + y == 3 for x, y in path):
            return 'diag_anti'
        return 'other'

    def collect(r, c, path):
        if len(path) == 4:
            kind = kind_of_pattern(path)
            patterns.append(
                (kind, 0, ''.join(str(grid[x][y]) for x, y in path), path)
            )
            return
        for dr, dc in [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in path:
                collect(nr, nc, path + [(nr, nc)])

    for i in range(rows):
        for j in range(cols):
            collect(i, j, [(i, j)])

    return patterns


def extract_pattern_flags(grid):
    flat = [int(x) for row in grid for x in row]
    return [1 if v > 0 else 0 for v in flat[:16]]


def compute_pattern_vector_16(grid):
    flat = [int(x) for row in grid for x in row]
    return flat[:16] if len(flat) >= 16 else flat + [0] * (16 - len(flat))


def extract_extended_features(grid):
    flat = [int(x) for row in grid for x in row]
    grid_vals = flat[:16] if len(flat) >= 16 else flat + [0] * (16 - len(flat))
    digit_counts = [grid_vals.count(d) for d in range(10)]
    num_unique = len(set(grid_vals))
    num_repeats = 16 - num_unique
    sum_digits = sum(grid_vals)
    mean_digits = sum_digits / 16
    return grid_vals + digit_counts + [num_unique, num_repeats, sum_digits, mean_digits]


def find_missing_digits(grid):
    all_digits = set(map(str, range(10)))
    used_digits = set(str(cell) for row in grid for cell in row)
    return sorted(all_digits - used_digits)


def find_4digit_patterns(grid):
    return find_all_4digit_patterns(grid)


def highlight_coords_for_patterns(patterns, targets):
    highlights = []
    for kind, idx, p, coords in patterns:
        if p in targets:
            highlights.append(coords)
    return [xy for coords in highlights for xy in coords]


def search_pattern_in_grid(grid, pattern):
    matches = []
    for kind, idx, p, coords in find_4digit_patterns(grid):
        if pattern == p:
            matches.append(coords)
    return matches


# ✅ Reverse grid helpers

def find_reverse_patterns(number: str):
    """Generate reverse grid for number and extract its 4-digit patterns."""
    reverse_grid = generate_reverse_grid(number)
    return find_all_4digit_patterns(reverse_grid)


def find_missing_digits_reverse(number: str):
    """Generate reverse grid for number and extract its missing digits."""
    reverse_grid = generate_reverse_grid(number)
    return find_missing_digits(reverse_grid)
