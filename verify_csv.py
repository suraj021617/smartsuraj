import csv

with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    total = 0
    valid = 0
    dates = set()
    providers = set()
    
    for row in reader:
        total += 1
        if len(row) >= 3 and row[0]:
            valid += 1
            if row[0].count('-') == 2:
                dates.add(row[0][:7])
            if len(row) > 2:
                providers.add(row[2][:30])
    
    print(f"Total rows: {total}")
    print(f"Valid rows: {valid}")
    print(f"Unique months: {len(dates)}")
    print(f"Sample months: {sorted(list(dates))[:5]}")
    print(f"\nUnique providers: {len(providers)}")
    print(f"Sample providers: {sorted(list(providers))[:10]}")
