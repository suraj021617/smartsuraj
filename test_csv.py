import csv
import re
from datetime import datetime

with open('4d_results_history_fixed.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    count = 0
    for row in reader:
        if count > 5:
            break
        print(f"Row {count}: {len(row)} columns")
        if len(row) > 4:
            print(f"  Date: {row[0]}")
            print(f"  Provider: {row[1]}")
            print(f"  Prizes: {row[4]}")
            
            # Test prize extraction
            first = re.search(r'1st\s+Prize\s+(\d{4})', row[4], re.IGNORECASE)
            if first:
                print(f"  1st Prize: {first.group(1)}")
        print()
        count += 1