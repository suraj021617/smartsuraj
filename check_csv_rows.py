import pandas as pd

# Count raw lines
with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

print(f"Total lines in CSV: {total_lines}")

# Load with pandas (mimicking app.py logic)
df = pd.read_csv('4d_results_history.csv', index_col=False, on_bad_lines='skip', 
                 encoding='utf-8', encoding_errors='ignore', dtype=str, 
                 keep_default_na=False, low_memory=False)

print(f"Rows after initial load: {len(df)}")

# Parse dates
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
before_date_filter = len(df)
df = df[df['date_parsed'].notna()].copy()
print(f"Rows with invalid dates removed: {before_date_filter - len(df)}")

# Remove duplicates
before_dedup = len(df)
df = df.drop_duplicates(keep='first')
print(f"Duplicate rows removed: {before_dedup - len(df)}")

print(f"\nFinal loaded rows: {len(df)}")
print(f"Total rows lost: {total_lines - len(df)}")
print(f"\nBreakdown: {total_lines} lines - 1 header = {total_lines-1} data rows")
print(f"After cleaning: {len(df)} rows")
print(f"Difference: {total_lines - 1 - len(df)} rows removed")
