from collections import defaultdict

def apply_weekend_bias(cand, draw_date, results, reason_map):
    """Boost score if draw is on Saturday or Sunday"""
    if not draw_date:
        return
    if draw_date.weekday() in [5, 6]:  # 5 = Saturday, 6 = Sunday
        results[cand] += 0.1
        reason_map[cand].add("weekend_bias")

def apply_grid_hotspot(cand, grid, hotspot_map, results, reason_map):
    """Boost score if candidate digits appear in hotspot cells"""
    if not grid or not hotspot_map:
        return

    hit_score = 0
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if str(cell) in cand:
                hit_score += hotspot_map.get((row_idx, col_idx), 0)

    if hit_score > 0:
        boost = round(hit_score * 0.02, 3)
        results[cand] += boost
        reason_map[cand].add("grid_hotspot")