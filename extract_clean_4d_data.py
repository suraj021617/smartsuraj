import csv
import re
import pandas as pd

input_file = '4d_results_history.csv'
output_file = 'clean_4d_training_data.csv'

pattern = re.compile(r'1st.*?(\d{4})\D+2nd.*?(\d{4})\D+3rd.*?(\d{4})')

results = []

with open(input_file, encoding='utf-8', errors='ignore') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            results.append({
                '1st': match.group(1),
                '2nd': match.group(2),
                '3rd': match.group(3)
            })

df = pd.DataFrame(results)
df.to_csv(output_file, index=False)
print(f"✅ Extracted {len(df)} rows to '{output_file}'")
