import pandas as pd
from collections import Counter

# Load CSV
df = pd.read_csv('4d_results_history.csv', on_bad_lines='skip')
print(f"Total rows: {len(df)}")

# Parse dates
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date_parsed'])
print(f"Rows with valid dates: {len(df)}")

# Extract prize numbers
import html
df['prize_text'] = df['3rd'].fillna('').astype(str).apply(html.unescape)
df['1st_real'] = df['prize_text'].str.extract(r'1st\s+Prize\s+(\d{4})', flags=2)[0]
df['2nd_real'] = df['prize_text'].str.extract(r'2nd\s+Prize\s+(\d{4})', flags=2)[0]
df['3rd_real'] = df['prize_text'].str.extract(r'3rd\s+Prize\s+(\d{4})', flags=2)[0]

# Check valid 4D numbers
valid_1st = df['1st_real'].dropna()
valid_1st = valid_1st[valid_1st.str.len() == 4]
valid_1st = valid_1st[valid_1st.str.isdigit()]
print(f"Valid 1st prize numbers: {len(valid_1st)}")

# Show latest 5 results
print("\nLatest 5 Results:")
for _, row in df.tail(5).iterrows():
    print(f"  {row['date_parsed'].date()} - 1st: {row['1st_real']}, 2nd: {row['2nd_real']}, 3rd: {row['3rd_real']}")

# Get most frequent numbers
all_nums = []
for col in ['1st_real', '2nd_real', '3rd_real']:
    all_nums.extend([n for n in df[col].astype(str) if n.isdigit() and len(n) == 4])

if all_nums:
    freq = Counter(all_nums).most_common(10)
    print(f"\nTop 10 Most Frequent Numbers:")
    for num, count in freq:
        print(f"  {num}: {count} times")
else:
    print("\nNo valid numbers found!")
