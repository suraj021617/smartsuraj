# -*- coding: utf-8 -*-
import pandas as pd
import re

# Test extraction
df = pd.read_csv('4d_results_history.csv', encoding='utf-8', on_bad_lines='skip', engine='python')
print(f"Columns: {list(df.columns)}")
print(f"Total rows: {len(df)}")

df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
df = df[df['date'].dt.date == pd.to_datetime('2025-09-21').date()]
print(f"Rows for 2025-09-21: {len(df)}")

print("\nSample rows:")
for idx, row in df.head(3).iterrows():
    print(f"\nProvider: {row.iloc[1]}")
    print(f"Column index 5 (3rd): {row.iloc[5] if len(row) > 5 else 'N/A'}")
    
    # Test extraction
    text = str(row.iloc[5]) if len(row) > 5 else ''
    match1 = re.search(r'1st\s+Prize[^\d]*(\d{4})', text, re.IGNORECASE)
    match2 = re.search(r'2nd\s+Prize[^\d]*(\d{4})', text, re.IGNORECASE)
    match3 = re.search(r'3rd\s+Prize[^\d]*(\d{4})', text, re.IGNORECASE)
    
    print(f"1st extracted: {match1.group(1) if match1 else 'NONE'}")
    print(f"2nd extracted: {match2.group(1) if match2 else 'NONE'}")
    print(f"3rd extracted: {match3.group(1) if match3 else 'NONE'}")
