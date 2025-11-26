import pandas as pd
import re
import html

print("="*80)
print("FINAL TEST - WHAT IS FLASK ACTUALLY LOADING?")
print("="*80)

# Load EXACTLY like app.py does
csv_paths = ['4d_results_history.csv', 'scraper/4d_results_history.csv']
df = None

for csv_path in csv_paths:
    try:
        df = pd.read_csv(csv_path, header=None, on_bad_lines='skip', encoding='utf-8', engine='python')
        if not df.empty:
            print(f"\nLoaded: {csv_path}")
            print(f"Rows: {len(df)}")
            print(f"Columns: {len(df.columns)}")
            break
    except:
        continue

if df is None:
    print("ERROR: No CSV loaded!")
    exit()

# Assign columns
if len(df.columns) == 8:
    df.columns = ['date', 'provider', 'draw_info', 'draw_number', 'date_info', 'prize_text', 'special', 'consolation']
else:
    df.columns = ['date', 'provider', 'draw_info', 'draw_number', 'prize_text', 'special', 'consolation'][:len(df.columns)]

# Parse dates
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date_parsed'], inplace=True)

# Get latest date
latest_date = df['date_parsed'].max().date()
print(f"\nLatest date: {latest_date}")

# Filter for latest
latest_df = df[df['date_parsed'].dt.date == latest_date].copy()
print(f"Rows for latest date: {len(latest_df)}")

# Extract provider
latest_df['provider_clean'] = latest_df['provider'].str.extract(r'images/([^./\"]+)', expand=False)

# Decode HTML and extract
latest_df['prize_decoded'] = latest_df['prize_text'].fillna('').astype(str).apply(html.unescape)
latest_df['2nd'] = latest_df['prize_decoded'].str.extract(r'2nd\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0]

print("\nWHAT FLASK WILL SHOW:")
print("-"*80)
for i, row in latest_df.iterrows():
    provider = row['provider_clean']
    second = row['2nd']
    if pd.notna(provider) and pd.notna(second):
        print(f"{provider:15} 2nd Prize: {second}")

# Check for 2025
has_2025 = (latest_df['2nd'] == '2025').sum()
print("\n" + "="*80)
if has_2025 > 0:
    print(f"PROBLEM: {has_2025} providers show '2025'")
    print("The CSV itself has bad data OR extraction is failing")
else:
    print("SUCCESS: All 2nd prizes are correct!")
    print("If you still see '2025' in browser, it's a CACHE issue")
print("="*80)
