import pandas as pd
import re
import html

df = pd.read_csv('scraper/4d_results_history.csv', header=None, on_bad_lines='skip', encoding='utf-8', engine='python')
df.columns = ['date', 'provider', 'col3', 'draw_number', 'prize_text', 'special', 'consolation'][:len(df.columns)]

# Get Sept 21 data
sept21 = df[df['date'].str.contains('2025-09-21', na=False)]

print("Testing HTML decode on Sept 21 data:")
print("="*80)

for i in range(min(5, len(sept21))):
    row = sept21.iloc[i]
    provider = row['provider']
    prize_text_raw = str(row['prize_text'])
    prize_text_decoded = html.unescape(prize_text_raw)
    
    print(f"\nProvider: {provider}")
    print(f"Raw: {prize_text_raw[:100]}")
    print(f"Decoded: {prize_text_decoded[:100]}")
    
    # Try extraction on decoded
    match = re.search(r'2nd\s+Prize\s+(\d{4})', prize_text_decoded, re.IGNORECASE)
    if match:
        print(f"✓ Extracted 2nd Prize: {match.group(1)}")
    else:
        print(f"✗ Failed to extract 2nd Prize")
        # Show what's around "2nd"
        if '2nd' in prize_text_decoded.lower():
            idx = prize_text_decoded.lower().index('2nd')
            print(f"  Context: ...{prize_text_decoded[max(0,idx-10):idx+30]}...")

print("\n" + "="*80)
