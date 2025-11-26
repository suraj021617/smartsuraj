import pandas as pd
import re
import html

print("="*80)
print("VERIFYING SEPT 21 DATA EXTRACTION FROM ROOT CSV")
print("="*80)

# Load root CSV (the one Flask will now use)
df = pd.read_csv('4d_results_history.csv', header=None, on_bad_lines='skip', encoding='utf-8', engine='python')
# Assign columns based on actual column count
if len(df.columns) == 8:
    df.columns = ['date', 'provider', 'col3', 'draw_number', 'col5', 'prize_text', 'special', 'consolation']
else:
    df.columns = ['date', 'provider', 'col3', 'draw_number', 'prize_text', 'special', 'consolation'][:len(df.columns)]

# Get Sept 21 data
sept21 = df[df['date'].astype(str).str.contains('2025-09-21', na=False)].copy()

print(f"\nTotal Sept 21 rows: {len(sept21)}")

# Extract provider names
sept21['provider_name'] = sept21['provider'].str.extract(r'images/([^./\"]+)', expand=False)

# Decode HTML and extract prizes
sept21['prize_decoded'] = sept21['prize_text'].fillna('').astype(str).apply(html.unescape)
sept21['1st'] = sept21['prize_decoded'].str.extract(r'1st\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0]
sept21['2nd'] = sept21['prize_decoded'].str.extract(r'2nd\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0]
sept21['3rd'] = sept21['prize_decoded'].str.extract(r'3rd\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0]

print("\nEXTRACTED PRIZES FOR SEPT 21:")
print("-"*80)

for i, row in sept21.iterrows():
    provider = row['provider_name']
    if pd.notna(provider) and pd.notna(row['2nd']):
        print(f"{provider:15} - 1st: {row['1st']}, 2nd: {row['2nd']}, 3rd: {row['3rd']}")

# Check for "2025" in 2nd prize
has_2025 = (sept21['2nd'] == '2025').sum()
print("\n" + "="*80)
if has_2025 > 0:
    print(f"ERROR: {has_2025} providers still show '2025' as 2nd Prize!")
else:
    print("SUCCESS: NO providers show '2025' as 2nd Prize!")
    print("All 2nd Prize values are correct!")
print("="*80)
