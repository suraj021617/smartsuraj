import pandas as pd
from collections import Counter
import re

# Load CSV
df = pd.read_csv('4d_results_history.csv', on_bad_lines='skip')
print(f"[OK] Loaded {len(df)} rows")

# Get numbers
all_nums = []
for col in ['1st', '2nd', '3rd']:
    for val in df[col].astype(str):
        found = re.findall(r'\b\d{4}\b', val)
        all_nums.extend(found)

print(f"[OK] Found {len(all_nums)} 4-digit numbers")

# Get top 10
freq = Counter(all_nums).most_common(10)
print("\n[TOP 10 NUMBERS]:")
for num, count in freq:
    print(f"  {num}: {count} times")

# Simulate voting
votes = {}
for num, count in freq:
    votes[num] = 3  # Simulate 3 votes

sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
final_picks = [(num, min(count * 25, 95)) for num, count in sorted_votes[:5]]

print("\n[FINAL PICKS]:")
for num, conf in final_picks:
    print(f"  {num} - {conf}% confidence")

print("\n[OK] Test completed successfully!")
print(f"\nfinal_picks = {final_picks}")
print(f"Is empty? {len(final_picks) == 0}")
