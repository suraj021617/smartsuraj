# utils/app_grid.py

def _to_str4(number_str):
    """Ensure input is string of length 4 (pad if needed)."""
    s = str(number_str)
    if len(s) < 4:
        s = s.zfill(4)
    return s

def generate_4x4_grid(number_str):
    """
    Main grid (+1 progression)
    Row1 = original number (unchanged)
    Row2 = formula_map applied
    Row3 = Row2 + 1 (wraparound %10)
    Row4 = Row3 + 1 (wraparound %10)
    """
    number_str = _to_str4(number_str)

    formula_map = {
        '0': 5, '1': 6, '2': 7, '3': 8, '4': 9,
        '5': 0, '6': 1, '7': 2, '8': 3, '9': 4
    }

    # Row1: original number (fresh list)
    row1 = [int(d) for d in number_str]

    # Row2: formula map fully applied (fresh list)
    row2 = [formula_map[d] for d in number_str]

    # Row3: row2 + 1 (wraparound) -> create new list from row2 values (do not mutate row2)
    row3 = [ (x + 1) % 10 for x in row2 ]

    # Row4: row3 + 1 (wraparound)
    row4 = [ (x + 1) % 10 for x in row3 ]

    return [row1, row2, row3, row4]


def generate_reverse_grid(number_str):
    """
    Reverse grid (-1 progression)
    Row1 = original number (unchanged)  <-- crucial: do NOT modify this row
    Row2 = formula_map applied fully
    Row3 = Row2 - 1 (wraparound %10)
    Row4 = Row3 - 1 (saturate at 0, no wraparound)
    """
    number_str = _to_str4(number_str)

    formula_map = {
        '0': 5, '1': 6, '2': 7, '3': 8, '4': 9,
        '5': 0, '6': 1, '7': 2, '8': 3, '9': 4
    }

    # Row1: ALWAYS a fresh list of original digits
    row1 = [int(d) for d in number_str]

    # Row2: formula map (fresh list)
    row2 = [formula_map[d] for d in number_str]

    # Row3: row2 -1 with wraparound (fresh list)
    row3 = [ (x - 1) % 10 for x in row2 ]

    # Row4: row3 -1 but saturate at 0 (fresh list)
    row4 = [ (x - 1) if x > 0 else 0 for x in row3 ]

    return [row1, row2, row3, row4]
