from collections import defaultdict, Counter
from utils.pattern_finder import find_all_4digit_patterns
from utils.app_grid import generate_reverse_grid, generate_4x4_grid
try:
    from utils.scoring_modules import apply_weekend_bias, apply_grid_hotspot
except ImportError:
    def apply_weekend_bias(cand, draw_date, results, reason_map): pass
    def apply_grid_hotspot(cand, grid, hotspot_map, results, reason_map): pass
import datetime
import re
import functools
import time

# Cache for expensive pattern operations
_pattern_cache = {}
_grid_cache = {}

def cached_pattern_operation(func):
    """Decorator for caching pattern operations"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
        if cache_key in _pattern_cache:
            cached_time, cached_result = _pattern_cache[cache_key]
            if time.time() - cached_time < 600:  # 10 minute cache for patterns
                return cached_result
        result = func(*args, **kwargs)
        _pattern_cache[cache_key] = (time.time(), result)
        return result
    return wrapper

@cached_pattern_operation
def predict_top_5(draws, mode="combined", provider=None):
    """
    Deterministic top-5 prediction engine.
    - Uses grid, reverse, missing digits, frequencies.
    - Adds provider-specific bias if provider is passed.
    """
    if not isinstance(draws, list) or not draws:
        return {"combined": [("0000", 0.0, "no-data")]}

    last_draw = draws[-1]
    grid = last_draw.get("grid")
    number = str(last_draw.get("number", "0000")).zfill(4)
    if not grid:
        grid = generate_4x4_grid(number)

    reverse_grid = generate_reverse_grid(number)

    # Convert draw date
    draw_date = None
    if "date" in last_draw:
        try:
            draw_date = datetime.datetime.strptime(str(last_draw["date"]), "%Y-%m-%d")
        except:
            pass

    results = defaultdict(float)
    reason_map = defaultdict(set)

    # Grid + reverse patterns
    grid_patterns = find_all_4digit_patterns(grid)
    reverse_patterns = find_all_4digit_patterns(reverse_grid)

    unique_grid = set(p for _, _, p, _ in grid_patterns)
    unique_reverse = set(p for _, _, p, _ in reverse_patterns)

    for p in unique_grid:
        if len(p) == 4 and p.isdigit():
            results[p] += 0.4
            reason_map[p].add("grid")

    for p in unique_reverse:
        if len(p) == 4 and p.isdigit():
            results[p] += 0.35
            reason_map[p].add("reverse")

    # Missing digits
    missing_digits = find_missing_digits(grid)
    for md in missing_digits:
        cand = str(int(md) * 4).zfill(4)
        if len(cand) == 4 and cand.isdigit():
            results[cand] += 0.25
            reason_map[cand].add("missing")

    missing_digits_reverse = find_missing_digits(reverse_grid)
    for md in missing_digits_reverse:
        cand = str(int(md) * 4).zfill(4)
        if len(cand) == 4 and cand.isdigit():
            results[cand] += 0.20
            reason_map[cand].add("reverse_missing")

    # Frequency scoring
    freq_counter = defaultdict(int)
    for d in draws[:-1]:
        if "grid" in d:
            for _, _, p, _ in find_all_4digit_patterns(d["grid"]):
                freq_counter[p] += 1
        if "reverse_grid" in d:
            for _, _, p, _ in find_all_4digit_patterns(d["reverse_grid"]):
                freq_counter[p] += 1

    for cand in results:
        if freq_counter[cand] > 0:
            boost = freq_counter[cand] * 0.05
            results[cand] += boost
            reason_map[cand].add("freq")

    # Build hotspot map
    hotspot_map = defaultdict(int)
    for d in draws[:-1]:
        g = d.get("grid")
        if g:
            for r in range(4):
                for c in range(4):
                    hotspot_map[(r, c)] += 1 if str(g[r][c]).isdigit() else 0

    # Apply modular scoring
    for cand in results:
        apply_weekend_bias(cand, draw_date, results, reason_map)
        apply_grid_hotspot(cand, grid, hotspot_map, results, reason_map)

    # 🔹 Enhanced provider-specific bias with pattern recognition
    if provider and provider != "all":
        prov_key = re.sub(r"[^a-z0-9]", "", str(provider).lower())
        for cand in list(results.keys()):
            if prov_key and len(prov_key) > 0:
                # Multi-layered provider bias algorithm
                provider_hash = hash(prov_key) % 10000
                provider_digits = str(provider_hash).zfill(4)

                # Layer 1: Direct digit overlap
                overlap = len(set(provider_digits) & set(cand))
                if overlap >= 2:
                    results[cand] *= 1.15  # Increased boost
                    reason_map[cand].add(f"provbias-{prov_key}")
                elif overlap == 1:
                    results[cand] *= 1.08  # Moderate boost

                # Layer 2: Sum-based matching (more sophisticated)
                provider_sum = sum(int(d) for d in provider_digits)
                cand_sum = sum(int(d) for d in cand)
                sum_diff = abs(provider_sum - cand_sum)
                if sum_diff <= 2:  # Close sum match
                    results[cand] *= 1.05
                    reason_map[cand].add(f"provsum-{prov_key}")

                # Layer 3: Pattern similarity (first/last digit match)
                if provider_digits[0] == cand[0] or provider_digits[-1] == cand[-1]:
                    results[cand] *= 1.03
                    reason_map[cand].add(f"provpattern-{prov_key}")

    # Format final predictions
    formatted = []
    for cand, score in results.items():
        if len(cand) == 4 and cand.isdigit():
            capped = min(score, 1.0)
            confidence = round(capped * 100, 2)
            reason = "+".join(sorted(reason_map[cand]))
            formatted.append((cand, confidence, reason))

    sorted_preds = sorted(formatted, key=lambda x: -x[1])[:5]
    
    # Ensure we always return at least 5 predictions
    if len(sorted_preds) < 5:
        # Add fallback predictions from grid patterns
        for p in list(unique_grid)[:5]:
            if len(sorted_preds) >= 5:
                break
            if p not in [pred[0] for pred in sorted_preds]:
                sorted_preds.append((p, 40.0, "grid"))
    
    # Final fallback: generate simple predictions
    if len(sorted_preds) < 5:
        for i in range(5 - len(sorted_preds)):
            fallback_num = str((int(number) + i + 1) % 10000).zfill(4)
            if fallback_num not in [pred[0] for pred in sorted_preds]:
                sorted_preds.append((fallback_num, 20.0, "fallback"))

    return {
        "combined": sorted_preds[:5],
        "grid": [(p, 40.0, "grid") for p in list(unique_grid)[:5]],
        "reverse": [(p, 35.0, "reverse") for p in list(unique_reverse)[:5]],
        "missing": [(str(int(md) * 4).zfill(4), 25.0, "missing") for md in missing_digits[:5] if md.isdigit()],
        "reverse_missing": [(str(int(md) * 4).zfill(4), 20.0, "reverse_missing") for md in missing_digits_reverse[:5] if md.isdigit()],
        "fallback": []
    }

def find_missing_digits(grid):
    all_digits = set(map(str, range(10)))
    used_digits = set(str(cell) for row in grid for cell in row)
    return sorted(all_digits - used_digits)
