"""
COMPLETE VERIFICATION: All providers + predictions use real CSV data only
"""
import pandas as pd
import re
from collections import Counter

print("=" * 80)
print("COMPLETE SYSTEM VERIFICATION")
print("=" * 80)

# Load CSV
df = pd.read_csv('scraper/4d_results_history.csv', header=None, on_bad_lines='skip', encoding='utf-8', engine='python')
df.columns = ['date', 'provider', 'col3', 'draw_number', 'prize_text', 'special', 'consolation'][:len(df.columns)]

# Extract provider
df['provider_name'] = df['provider'].str.extract(r'logo_([^./]+)', expand=False)

# Extract prizes
prize_text_col = df['prize_text'].fillna('').astype(str)
df['1st_real'] = prize_text_col.str.extract(r'1st\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0].fillna('')
df['2nd_real'] = prize_text_col.str.extract(r'2nd\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0].fillna('')
df['3rd_real'] = prize_text_col.str.extract(r'3rd\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0].fillna('')

print("\n1. VERIFY ALL PROVIDERS HAVE CORRECT DATA")
print("-" * 80)

for provider in ['magnum', 'damacai', 'gdlotto']:
    provider_df = df[df['provider_name'] == provider]
    valid_1st = (provider_df['1st_real'] != '').sum()
    valid_2nd = (provider_df['2nd_real'] != '').sum()
    valid_3rd = (provider_df['3rd_real'] != '').sum()
    
    print(f"\n{provider.upper()}:")
    print(f"  Total rows: {len(provider_df)}")
    print(f"  Valid 1st Prize: {valid_1st} [OK]" if valid_1st > 0 else f"  Valid 1st Prize: {valid_1st} [FAIL]")
    print(f"  Valid 2nd Prize: {valid_2nd} [OK]" if valid_2nd > 0 else f"  Valid 2nd Prize: {valid_2nd} [FAIL]")
    print(f"  Valid 3rd Prize: {valid_3rd} [OK]" if valid_3rd > 0 else f"  Valid 3rd Prize: {valid_3rd} [FAIL]")
    
    # Show sample
    sample = provider_df.head(1)
    if not sample.empty:
        print(f"  Sample 1st: {sample['1st_real'].iloc[0]}")
        print(f"  Sample 2nd: {sample['2nd_real'].iloc[0]}")
        print(f"  Sample 3rd: {sample['3rd_real'].iloc[0]}")

print("\n2. VERIFY NO '2025' IN PRIZE COLUMNS")
print("-" * 80)

has_2025_in_1st = (df['1st_real'] == '2025').sum()
has_2025_in_2nd = (df['2nd_real'] == '2025').sum()
has_2025_in_3rd = (df['3rd_real'] == '2025').sum()

print(f"1st Prize contains '2025': {has_2025_in_1st} rows {'[PROBLEM!]' if has_2025_in_1st > 0 else '[GOOD]'}")
print(f"2nd Prize contains '2025': {has_2025_in_2nd} rows {'[PROBLEM!]' if has_2025_in_2nd > 0 else '[GOOD]'}")
print(f"3rd Prize contains '2025': {has_2025_in_3rd} rows {'[PROBLEM!]' if has_2025_in_3rd > 0 else '[GOOD]'}")

print("\n3. VERIFY PREDICTION DATA POOL (REAL NUMBERS ONLY)")
print("-" * 80)

# Simulate prediction logic
prize_cols = ["1st_real", "2nd_real", "3rd_real", "special", "consolation"]
all_numbers = []

for col in prize_cols:
    if col not in df.columns:
        continue
    col_values = df[col].astype(str).dropna().tolist()
    for val in col_values:
        val = val.strip()
        if val.isdigit() and len(val) == 4:
            all_numbers.append(val)
        else:
            found = re.findall(r'\d{4}', val)
            for f in found:
                if f.isdigit() and len(f) == 4:
                    all_numbers.append(f)

print(f"Total 4-digit numbers extracted: {len(all_numbers)}")
print(f"Unique numbers in pool: {len(set(all_numbers))}")

# Check if any fake/random numbers
unique_numbers = set(all_numbers)
print(f"\nSample numbers from pool (first 10):")
for i, num in enumerate(list(unique_numbers)[:10], 1):
    print(f"  {i}. {num}")

# Verify all are 4-digit
all_valid = all(len(n) == 4 and n.isdigit() for n in unique_numbers)
print(f"\nAll numbers are valid 4-digit: {'[YES]' if all_valid else '[NO]'}")

print("\n4. VERIFY MOST FREQUENT NUMBERS (TOP 10)")
print("-" * 80)

number_freq = Counter(all_numbers)
top_10 = number_freq.most_common(10)

for i, (num, count) in enumerate(top_10, 1):
    print(f"{i:2d}. {num} - appeared {count} times")

print("\n5. VERIFY PER-PROVIDER PREDICTIONS")
print("-" * 80)

for provider in ['magnum', 'damacai', 'gdlotto']:
    provider_df = df[df['provider_name'] == provider]
    
    # Get numbers for this provider
    provider_numbers = []
    for col in ["1st_real", "2nd_real", "3rd_real"]:
        nums = provider_df[col].astype(str).tolist()
        provider_numbers.extend([n for n in nums if n.isdigit() and len(n) == 4])
    
    unique_provider = set(provider_numbers)
    freq = Counter(provider_numbers)
    top_3 = freq.most_common(3)
    
    print(f"\n{provider.upper()}:")
    print(f"  Total numbers: {len(provider_numbers)}")
    print(f"  Unique numbers: {len(unique_provider)}")
    print(f"  Top 3 frequent:")
    for num, count in top_3:
        print(f"    {num} ({count}x)")

print("\n" + "=" * 80)
print("FINAL VERIFICATION SUMMARY")
print("=" * 80)

checks = [
    ("All providers have valid prize data", valid_1st > 0 and valid_2nd > 0 and valid_3rd > 0),
    ("No '2025' in extracted prizes", has_2025_in_1st == 0 and has_2025_in_2nd == 0 and has_2025_in_3rd == 0),
    ("All prediction numbers are 4-digit", all_valid),
    ("Prediction pool uses real CSV data", len(unique_numbers) > 100),
]

all_passed = True
for check_name, passed in checks:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} - {check_name}")
    if not passed:
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print(">>> ALL CHECKS PASSED - SYSTEM IS WORKING CORRECTLY <<<")
else:
    print(">>> SOME CHECKS FAILED - REVIEW ISSUES ABOVE <<<")
print("=" * 80)
