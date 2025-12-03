print("=== QUICK LOGIC TEST ===")

# Test 1: CSV
try:
    import pandas as pd
    df = pd.read_csv('4d_results_history.csv', on_bad_lines='skip')
    print(f"OK CSV: {len(df)} rows")
except:
    print("FAIL CSV")

# Test 2: App
try:
    from app import app
    print("OK App import")
except Exception as e:
    print(f"FAIL App: {e}")

# Test 3: Utils
try:
    from utils.pattern_predictor import PatternPredictor
    print("OK Utils")
except:
    print("FAIL Utils")

print("Test complete")