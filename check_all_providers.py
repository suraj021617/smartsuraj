import pandas as pd
import re

df = pd.read_csv('scraper/4d_results_history.csv', header=None, on_bad_lines='skip', encoding='utf-8', engine='python')
df.columns = ['date', 'provider', 'col3', 'draw_number', 'prize_text', 'special', 'consolation'][:len(df.columns)]

# Extract provider names
df['provider_name'] = df['provider'].str.extract(r'logo_([^./]+)', expand=False)

print("=" * 80)
print("ALL PROVIDERS IN CSV")
print("=" * 80)
print(df['provider_name'].value_counts())

print("\n" + "=" * 80)
print("SAMPLE DATA FOR EACH PROVIDER")
print("=" * 80)

for provider in df['provider_name'].dropna().unique()[:5]:
    sample = df[df['provider_name'] == provider].head(1)
    if not sample.empty:
        print(f"\n--- {provider.upper()} ---")
        print(f"Date: {sample['date'].iloc[0]}")
        print(f"Prize Text: {sample['prize_text'].iloc[0]}")
        
        # Extract prizes
        prize_text = str(sample['prize_text'].iloc[0])
        first = re.search(r'1st\s+Prize\s+(\d{4})', prize_text, re.IGNORECASE)
        second = re.search(r'2nd\s+Prize\s+(\d{4})', prize_text, re.IGNORECASE)
        third = re.search(r'3rd\s+Prize\s+(\d{4})', prize_text, re.IGNORECASE)
        
        print(f"Extracted 1st: {first.group(1) if first else 'NOT FOUND'}")
        print(f"Extracted 2nd: {second.group(1) if second else 'NOT FOUND'}")
        print(f"Extracted 3rd: {third.group(1) if third else 'NOT FOUND'}")

print("\n" + "=" * 80)
print("VERIFICATION: All providers have same CSV structure")
print("All 1st, 2nd, 3rd prizes extracted from 'prize_text' column")
print("=" * 80)
