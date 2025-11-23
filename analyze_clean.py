import csv
import re
from collections import Counter

# Clean analysis - filter suspicious patterns
digit_count = Counter()
number_count = Counter()
total_numbers = 0
filtered_count = 0

with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for prize in ['1st', '2nd', '3rd']:
            text = row.get(prize, '')
            numbers = re.findall(r'\b\d{4}\b', text)
            for num in numbers:
                total_numbers += 1
                # Filter out date patterns (2015-2025, 1/15, etc)
                if num.startswith('201') or num.startswith('202'):
                    filtered_count += 1
                    continue
                # Filter draw numbers like 4094, 5975, etc
                if int(num) > 9999 or num.startswith('0000'):
                    filtered_count += 1
                    continue
                
                number_count[num] += 1
                for digit in num:
                    digit_count[digit] += 1

print("=" * 60)
print("CLEANED DATA ANALYSIS")
print("=" * 60)
print(f"Total numbers found: {total_numbers:,}")
print(f"Filtered out (dates/invalid): {filtered_count:,}")
print(f"Valid 4D numbers analyzed: {sum(number_count.values()):,}")
print()

print("=" * 60)
print("DIGIT FREQUENCY (0-9) - CLEANED DATA")
print("=" * 60)
total = sum(digit_count.values())
print(f"Total digits: {total:,}\n")

for digit in sorted(digit_count.keys()):
    count = digit_count[digit]
    percent = (count / total * 100)
    expected = 10.0
    bias = percent - expected
    bar = '#' * int(percent)
    status = "HIGH" if bias > 1 else "LOW" if bias < -1 else "OK"
    print(f"Digit {digit}: {percent:5.2f}% {bar} | Bias: {bias:+.2f}% [{status}]")

print("\n" + "=" * 60)
print("TOP 20 MOST FREQUENT 4D NUMBERS (CLEANED)")
print("=" * 60)
for num, count in number_count.most_common(20):
    print(f"{num}: {count} times")

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
max_bias = max(abs((digit_count[d]/total*100) - 10) for d in digit_count)
print(f"Maximum digit bias: {max_bias:.2f}%")
print(f"Unique 4D numbers: {len(number_count)}")
print(f"Average appearances per number: {sum(number_count.values())/len(number_count):.1f}")
