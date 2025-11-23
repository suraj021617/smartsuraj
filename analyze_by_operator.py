import csv
import re
from collections import Counter, defaultdict

# Analyze by operator
operators = defaultdict(lambda: {'digits': Counter(), 'numbers': Counter(), 'count': 0})

with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        provider = row.get('provider', '')
        
        # Focus on main operators
        if 'magnum' in provider.lower():
            op = 'Magnum'
        elif 'singapore' in provider.lower():
            op = 'Singapore'
        elif 'gdlotto' in provider.lower() or 'dragon' in provider.lower():
            op = 'Grand Dragon'
        elif 'damacai' in provider.lower():
            op = 'Damacai'
        elif 'toto' in provider.lower() and 'singapore' not in provider.lower():
            op = 'Toto'
        else:
            continue
        
        # Extract 4D numbers from prizes
        for prize in ['1st', '2nd', '3rd']:
            text = row.get(prize, '')
            numbers = re.findall(r'\b\d{4}\b', text)
            for num in numbers:
                # Skip date-like patterns (2015-2025)
                if num.startswith('201') or num.startswith('202'):
                    continue
                operators[op]['numbers'][num] += 1
                operators[op]['count'] += 1
                for digit in num:
                    operators[op]['digits'][digit] += 1

# Display results
for op in sorted(operators.keys()):
    data = operators[op]
    print("=" * 60)
    print(f"{op.upper()} - DIGIT FREQUENCY")
    print("=" * 60)
    total = sum(data['digits'].values())
    print(f"Total 4D numbers analyzed: {data['count']}")
    print(f"Total digits: {total:,}\n")
    
    for digit in sorted(data['digits'].keys()):
        count = data['digits'][digit]
        percent = (count / total * 100) if total > 0 else 0
        expected = 10.0
        bias = percent - expected
        bar = '#' * int(percent)
        print(f"Digit {digit}: {percent:5.2f}% {bar} | Bias: {bias:+.2f}%")
    
    print(f"\nTop 10 Most Frequent Numbers:")
    for num, count in data['numbers'].most_common(10):
        print(f"  {num}: {count} times")
    print()
