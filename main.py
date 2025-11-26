# main.py

from utils.app_grid import generate_4x4_grid, generate_reverse_grid, find_missing_digits
from utils.pattern_finder import find_4digit_patterns
from utils.ai_predictor import predict_top_5

def demo_run(number="1234"):
    print(f"=== DEMO RUN for number {number} ===")

    # Build grids
    grid = generate_4x4_grid(number)
    reverse_grid = generate_reverse_grid(number)

    print("\nGrid:")
    for row in grid:
        print(row)

    print("\nReverse Grid:")
    for row in reverse_grid:
        print(row)

    # Find patterns
    patterns = find_4digit_patterns(grid)
    reverse_patterns = find_4digit_patterns(reverse_grid)

    print(f"\nTotal Patterns Found: {len(patterns)} (normal grid)")
    print(f"Total Patterns Found: {len(reverse_patterns)} (reverse grid)")

    # Missing digits
    missing = find_missing_digits(grid)
    missing_rev = find_missing_digits(reverse_grid)

    print(f"\nMissing digits (grid): {missing}")
    print(f"Missing digits (reverse grid): {missing_rev}")

    # Fake draws list (simulate history for prediction)
    fake_draws = [{
        "date": "2025-08-22",
        "grid": grid,
        "reverse_grid": reverse_grid,
        "patterns": patterns,
        "reverse_patterns": reverse_patterns,
        "missing_digits": missing,
        "missing_digits_reverse": missing_rev,
        "1st_real": number
    }]

    # Predictions
    preds = predict_top_5(fake_draws, mode="history")
    print("\nPredictions (top 5):")
    for cand, score, reason in preds:
        print(f" {cand} | score={score} | reason={reason}")


if __name__ == "__main__":
    demo_run("1234")
