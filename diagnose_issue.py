import pandas as pd
import re
import html

print("DIAGNOSING THE ISSUE")
print("="*80)

# Load CSV exactly like app.py does
df = pd.read_csv('scraper/4d_results_history.csv', header=None, on_bad_lines='skip', encoding='utf-8', engine='python')
df.columns = ['date', 'provider', 'col3', 'draw_number', 'prize_text', 'special', 'consolation'][:len(df.columns)]

print(f"\nTotal rows in CSV: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Parse dates
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date_parsed'], inplace=True)

# Get latest date
latest_date = df['date_parsed'].max().date()
print(f"\nLatest date: {latest_date}")

# Filter for latest date
filtered = df[df['date_parsed'].dt.date == latest_date]
print(f"Rows for latest date: {len(filtered)}")

# Extract provider names
filtered['provider_name'] = filtered['provider'].str.extract(r'logo_([^./]+)|images/([^./\"]+)', expand=False)[0]
filtered['provider_name'] = filtered['provider_name'].fillna(filtered['provider'].str.extract(r'images/([^./\"]+)', expand=False)[0])

print(f"\nProviders on {latest_date}:")
for p in filtered['provider_name'].unique():
    print(f"  - {p}")

# Test extraction WITH and WITHOUT html.unescape
print("\n" + "="*80)
print("TESTING EXTRACTION")
print("="*80)

for i in range(min(3, len(filtered))):
    row = filtered.iloc[i]
    provider = row['provider_name']
    prize_text_raw = str(row['prize_text'])
    
    print(f"\n{i+1}. Provider: {provider}")
    print(f"   Raw prize_text: {prize_text_raw[:80]}")
    
    # WITHOUT decode
    match1 = re.search(r'2nd\s+Prize\s+(\d{4})', prize_text_raw, re.IGNORECASE)
    print(f"   Without decode: {match1.group(1) if match1 else 'NOT FOUND'}")
    
    # WITH decode
    prize_text_decoded = html.unescape(prize_text_raw)
    match2 = re.search(r'2nd\s+Prize\s+(\d{4})', prize_text_decoded, re.IGNORECASE)
    print(f"   With decode: {match2.group(1) if match2 else 'NOT FOUND'}")

print("\n" + "="*80)
print("CONCLUSION:")
print("If 'With decode' shows correct numbers, the fix is working")
print("If still showing 'NOT FOUND', there's a different issue")
print("="*80)
