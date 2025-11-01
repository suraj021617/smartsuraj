# generate_grid_training_data.py

import pandas as pd
from utils.pattern_finder import extract_extended_features
from utils.app_grid import generate_4x4_grid
import os

# Load 4D result history
df = pd.read_csv("4d_results_history.csv", on_bad_lines='skip')

# Extract '1st_real' column reliably
def extract_1st_number(row):
    import re
    for col in ['1st', '2nd', '3rd']:
        match = re.search(r'1st(?: Prize)?[^\d]{0,10}(\d{4})', str(row.get(col, '')))
        if match:
            return match.group(1)
    return ''

df['number'] = df.apply(extract_1st_number, axis=1)
df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
df = df.dropna(subset=['number', 'date'])
df = df[df['number'].str.len() == 4]

rows = []

for i in range(len(df) - 1):
    current = df.iloc[i]
    next_row = df.iloc[i + 1]
    
    grid = generate_4x4_grid(current['number'])
    features = extract_extended_features(grid)

    target_numbers = [str(next_row.get(col, '')) for col in ['1st', '2nd', '3rd']]
    is_hit = any(current['number'] in t for t in target_numbers)

    rows.append({
        'date': current['date'],
        **{f'cell_{i}': val for i, val in enumerate(features)},
        'hit': int(is_hit)
    })

out_df = pd.DataFrame(rows)
out_df.to_csv("grid_hit_training_data.csv", index=False)
print(f"✅ Extracted {len(out_df)} rows to 'grid_hit_training_data.csv'")
