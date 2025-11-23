# utils/app_grid.py

def _to_str4(number_str):
    """Ensure input is string of length 4 (pad if needed)."""
    s = str(number_str)
    if len(s) < 4:
        s = s.zfill(4)
    return s

def generate_4x4_grid(number_str):
    """
    Main grid with formula map
    Row1 = original number
    Row2 = formula map applied (0→5, 1→6, 2→7, 3→8, 4→9, 5→0, 6→1, 7→2, 8→3, 9→4)
    Row3 = Row2 + 1
    Row4 = Row3 + 1
    """
    number_str = _to_str4(number_str)
    formula_map = {'0': 5, '1': 6, '2': 7, '3': 8, '4': 9, '5': 0, '6': 1, '7': 2, '8': 3, '9': 4}
    
    row1 = [int(d) for d in number_str]
    row2 = [formula_map[d] for d in number_str]
    row3 = [(x + 1) % 10 for x in row2]
    row4 = [(x + 1) % 10 for x in row3]
    return [row1, row2, row3, row4]


def generate_reverse_grid(number_str):
    """
    Reverse grid with formula map
    Row1 = original number
    Row2 = formula map applied (0→5, 1→6, 2→7, 3→8, 4→9, 5→0, 6→1, 7→2, 8→3, 9→4)
    Row3 = Row2 - 1 (0 stays 0)
    Row4 = Row3 - 1 (0 stays 0)
    """
    number_str = _to_str4(number_str)
    formula_map = {'0': 5, '1': 6, '2': 7, '3': 8, '4': 9, '5': 0, '6': 1, '7': 2, '8': 3, '9': 4}
    
    row1 = [int(d) for d in number_str]
    row2 = [formula_map[d] for d in number_str]
    row3 = [0 if x == 0 else x - 1 for x in row2]
    row4 = [0 if x == 0 else x - 1 for x in row3]
    return [row1, row2, row3, row4]
