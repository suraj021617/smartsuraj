import pandas as pd
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Load CSV
df = pd.read_csv('4d_results_history.csv', dtype=str, keep_default_na=False, on_bad_lines='skip', encoding='utf-8', encoding_errors='ignore')

print(f"Total rows loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few date values:")
print(df['date'].head(10).tolist())

# Parse dates - try multiple formats
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')

# Remove invalid dates
df = df[df['date_parsed'].notna()]

print(f"Rows with valid dates: {len(df)}")

# Get date range
if not df.empty:
    earliest = df['date_parsed'].min()
    latest = df['date_parsed'].max()
    total_rows = len(df)
    
    print(f"\nCSV DATA SUMMARY")
    print(f"=" * 50)
    print(f"Total Rows: {total_rows}")
    print(f"Earliest Date: {earliest.strftime('%Y-%m-%d')}")
    print(f"Latest Date: {latest.strftime('%Y-%m-%d')}")
    print(f"Date Range: {(latest - earliest).days} days")
    print(f"Months Covered: ~{(latest - earliest).days / 30:.1f} months")
    print(f"=" * 50)
    
    # Show first 5 and last 5 dates
    print(f"\nFirst 5 dates:")
    for date in df['date_parsed'].head(5):
        print(f"  - {date.strftime('%Y-%m-%d')}")
    
    print(f"\nLast 5 dates:")
    for date in df['date_parsed'].tail(5):
        print(f"  - {date.strftime('%Y-%m-%d')}")
else:
    print("No valid data found in CSV")
