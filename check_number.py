import csv
import re
from itertools import permutations
from datetime import datetime

def get_ibox(num):
    return set([''.join(p) for p in permutations(num)])

def extract_numbers(text):
    return re.findall(r'\b\d{4}\b', text)

def get_provider(url):
    providers = {
        'magnum': 'Magnum', 'damacai': 'Damacai', 'toto': 'Toto',
        'singapore': 'Singapore', 'sandakan': 'Sandakan', 'cashsweep': 'CashSweep',
        'sabah': 'Sabah88', 'gdlotto': 'GD Lotto', 'perdana': 'Perdana', 'harihari': 'HariHari'
    }
    for key, name in providers.items():
        if key in url.lower():
            return name
    return 'Other'

# Get inputs
number = input("Enter number to check: ").strip()
date = input("Enter date (YYYY-MM-DD) or press Enter for all: ").strip()

# Generate ibox
ibox_set = get_ibox(number) if len(number) == 4 else set()

# Search
results = {}
with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    for row in csv.reader(f):
        if len(row) < 6 or not row[0]:
            continue
        if date and row[0] != date:
            continue
        
        # Columns: 0=date, 1=url, 2=type, 3=draw, 4=draw_info, 5=prizes, 6=special, 7=consolation
        nums = extract_numbers(row[5] + ' ' + row[6] + ' ' + (row[7] if len(row) > 7 else ''))
        for num in nums:
            match_type = "EXACT" if num == number else ("IBOX" if num in ibox_set else None)
            if match_type:
                if row[0] not in results:
                    results[row[0]] = []
                provider = get_provider(row[1]) + ' ' + row[2]
                results[row[0]].append((match_type, num, provider.strip()))

# Display
if results:
    for dt in sorted(results.keys()):
        print(f"\n{dt}:")
        for match_type, num, provider in results[dt]:
            print(f"  [{match_type}] {num} - {provider}")
    print(f"\nTotal: {sum(len(v) for v in results.values())} matches")
else:
    print("\nNo matches found")
