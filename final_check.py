import pandas as pd
import re
import html

print("="*80)
print("FINAL COMPREHENSIVE CHECK")
print("="*80)

# Load CSV
df = pd.read_csv('scraper/4d_results_history.csv', header=None, on_bad_lines='skip', encoding='utf-8', engine='python')
df.columns = ['date', 'provider', 'col3', 'draw_number', 'prize_text', 'special', 'consolation'][:len(df.columns)]

# Parse dates
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date_parsed'], inplace=True)

# Get latest date
latest_date = df['date_parsed'].max().date()
print(f"\n1. Latest date in CSV: {latest_date}")

# Filter for latest date
latest_df = df[df['date_parsed'].dt.date == latest_date].copy()
print(f"   Rows for latest date: {len(latest_df)}")

# Extract provider
latest_df['provider_clean'] = latest_df['provider'].str.extract(r'logo_([^./]+)', expand=False)
latest_df['provider_clean'] = latest_df['provider_clean'].fillna(
    latest_df['provider'].str.extract(r'/([^/]+)\.(gif|png)', expand=False)[0]
)

print(f"\n2. Providers on {latest_date}:")
for p in latest_df['provider_clean'].dropna().unique():
    print(f"   - {p}")

# Extract prizes WITH html.unescape
print(f"\n3. Prize extraction test (with HTML decode):")
print("-"*80)

latest_df['prize_text_decoded'] = latest_df['prize_text'].fillna('').astype(str).apply(html.unescape)
latest_df['1st_extracted'] = latest_df['prize_text_decoded'].str.extract(r'1st\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0]
latest_df['2nd_extracted'] = latest_df['prize_text_decoded'].str.extract(r'2nd\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0]
latest_df['3rd_extracted'] = latest_df['prize_text_decoded'].str.extract(r'3rd\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0]

for i, row in latest_df.head(10).iterrows():
    provider = row['provider_clean']
    print(f"\n{provider}:")
    print(f"  1st Prize: {row['1st_extracted']}")
    print(f"  2nd Prize: {row['2nd_extracted']}")
    print(f"  3rd Prize: {row['3rd_extracted']}")

# Check for "2025" in extracted prizes
has_2025_in_2nd = (latest_df['2nd_extracted'] == '2025').sum()
print(f"\n4. Rows with '2025' as 2nd Prize: {has_2025_in_2nd}")

if has_2025_in_2nd > 0:
    print("   ✗ PROBLEM: Some providers have '2025' as 2nd Prize!")
    problem_rows = latest_df[latest_df['2nd_extracted'] == '2025']
    for i, row in problem_rows.iterrows():
        print(f"     - {row['provider_clean']}: {row['prize_text'][:80]}")
else:
    print("   ✓ GOOD: No '2025' found in 2nd Prize!")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(f"✓ CSV loaded: {len(df)} total rows")
print(f"✓ Latest date: {latest_date}")
print(f"✓ Providers on latest date: {len(latest_df['provider_clean'].dropna().unique())}")
print(f"✓ HTML decode: {'Working' if has_2025_in_2nd == 0 else 'NOT working'}")
print(f"✓ Prize extraction: {'Working' if has_2025_in_2nd == 0 else 'FAILED'}")

if has_2025_in_2nd == 0:
    print("\n✓✓✓ ALL CHECKS PASSED! ✓✓✓")
    print("Your Flask app should show correct data for date:", latest_date)
    print("Visit: http://127.0.0.1:5000/")
else:
    print("\n✗✗✗ ISSUES FOUND ✗✗✗")
    print("The CSV has '2025' in the prize_text itself!")

print("="*80)
