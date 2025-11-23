import pandas as pd
import re
from collections import Counter

df = pd.read_csv('4d_results_history.csv', on_bad_lines='skip')
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date_parsed').tail(5)

print("\n" + "="*60)
print("LATEST RESULTS - Which predictions matched?")
print("="*60)

for idx, row in df.iterrows():
    date = str(row['date_parsed'])[:10]
    provider = str(row.get('provider', ''))
    
    prize_text = str(row.get('3rd', ''))
    first = re.search(r'1st\s+Prize\s+(\d{4})', prize_text, re.I)
    second = re.search(r'2nd\s+Prize\s+(\d{4})', prize_text, re.I)
    third = re.search(r'3rd\s+Prize\s+(\d{4})', prize_text, re.I)
    
    winners = []
    if first: winners.append(first.group(1))
    if second: winners.append(second.group(1))
    if third: winners.append(third.group(1))
    
    if not winners:
        continue
    
    print(f"\n{date} - {provider}")
    print(f"Winners: {', '.join(winners)}")
    print("-" * 40)

print("\n" + "="*60)
