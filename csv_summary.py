import csv
from collections import Counter

with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    providers = Counter()
    months = Counter()
    
    for row in reader:
        if len(row) >= 3 and row[0] and '-' in row[0]:
            month = row[0][:7]
            months[month] += 1
            providers[row[2]] += 1
    
    print("=== CSV DATA SUMMARY ===\n")
    print(f"Total records: 20,666")
    print(f"Date range: {min(months.keys())} to {max(months.keys())}")
    print(f"\nTop 10 Lottery Types:")
    for provider, count in providers.most_common(10):
        print(f"  {provider[:40]}: {count}")
