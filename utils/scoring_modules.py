def apply_weekend_bias(cand, draw_date, results, reason_map):
    """Apply weekend bias to predictions"""
    if draw_date and draw_date.weekday() >= 5:  # Saturday or Sunday
        results[cand] *= 1.05
        reason_map[cand].add("weekend")

def apply_grid_hotspot(cand, grid, hotspot_map, results, reason_map):
    """Apply grid hotspot scoring"""
    if not grid or not hotspot_map:
        return
    
    # Check if candidate digits appear in hot positions
    hot_positions = sorted(hotspot_map.items(), key=lambda x: x[1], reverse=True)[:4]
    hot_digits = set(str(grid[r][c]) for (r, c), _ in hot_positions if r < len(grid) and c < len(grid[r]))
    
    overlap = len(set(cand) & hot_digits)
    if overlap >= 2:
        results[cand] *= 1.08
        reason_map[cand].add("hotspot")
