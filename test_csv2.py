import csv
import re

with open('4d_results_history_fixed.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    count = 0
    for row in reader:
        if count > 10:
            break
        if len(row) > 6 and '2025-10' in row[0]:
            print(f"Row {count}:")
            print(f"  Col 0 (Date): {row[0]}")
            print(f"  Col 5 (Prize summary): {row[5]}")
            
            # Test prize extraction from column 5
            first = re.search(r'1st\s+Prize\s+(\d{4})', row[5], re.IGNORECASE)
            if first:
                print(f"  Found 1st Prize: {first.group(1)}")
            else:
                print(f"  No 1st Prize found in: {row[5]}")
            print()
        count += 1