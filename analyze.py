import csv
from collections import Counter

# Read CSV and extract 4D numbers
digit_count = Counter()
number_count = Counter()

with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Extract 1st, 2nd, 3rd prize numbers
        for prize in ['1st', '2nd', '3rd']:
            text = row.get(prize, '')
            # Extract 4-digit numbers
            import re
            numbers = re.findall(r'\b\d{4}\b', text)
            for num in numbers:
                number_count[num] += 1
                # Count each digit
                for digit in num:
                    digit_count[digit] += 1

# Results
print("=" * 50)
print("DIGIT FREQUENCY ANALYSIS (0-9)")
print("=" * 50)
total = sum(digit_count.values())
for digit in sorted(digit_count.keys()):
    count = digit_count[digit]
    percent = (count / total * 100)
    expected = 10.0
    bias = percent - expected
    print(f"Digit {digit}: {count:,} times ({percent:.2f}%) | Bias: {bias:+.2f}%")

print(f"\nTotal digits analyzed: {total:,}")

print("\n" + "=" * 50)
print("TOP 20 MOST FREQUENT 4D NUMBERS")
print("=" * 50)
for num, count in number_count.most_common(20):
    print(f"{num}: appeared {count} times")

print("\n" + "=" * 50)
print("TOP 20 LEAST FREQUENT 4D NUMBERS")
print("=" * 50)
for num, count in list(number_count.most_common())[-20:]:
    print(f"{num}: appeared {count} times")
