import pandas as pd
import re

# Load CSV
csv_path = 'scraper/4d_results_history.csv'
df = pd.read_csv(csv_path, header=None, on_bad_lines='skip', encoding='utf-8', engine='python')

# Assign columns
df.columns = ['date', 'provider', 'col3', 'draw_number', 'prize_text', 'special', 'consolation'][:len(df.columns)]

print("=" * 80)
print("TESTING PRIZE EXTRACTION")
print("=" * 80)

# Test first 5 rows
for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    print(f"\n--- Row {idx} ---")
    print(f"Date: {row['date']}")
    print(f"Provider: {row['provider']}")
    print(f"Prize Text: {row['prize_text']}")
    
    # Extract prizes
    prize_text = str(row['prize_text'])
    first = re.search(r'1st\s+Prize\s+(\d{4})', prize_text, re.IGNORECASE)
    second = re.search(r'2nd\s+Prize\s+(\d{4})', prize_text, re.IGNORECASE)
    third = re.search(r'3rd\s+Prize\s+(\d{4})', prize_text, re.IGNORECASE)
    
    print(f"Extracted 1st: {first.group(1) if first else 'NOT FOUND'}")
    print(f"Extracted 2nd: {second.group(1) if second else 'NOT FOUND'}")
    print(f"Extracted 3rd: {third.group(1) if third else 'NOT FOUND'}")
    
    print(f"Special column: {str(row['special'])[:80]}")
    print(f"Consolation column: {str(row['consolation'])[:80]}")

print("\n" + "=" * 80)
print("SUMMARY: Check if 2nd Prize extraction is correct above")
print("=" * 80)
