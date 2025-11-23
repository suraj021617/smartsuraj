import csv
from collections import Counter

with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)

digit_freq = Counter()
grand_dragon = [row for row in data if 'Grand Dragon' in str(row)]

for row in grand_dragon:
    if len(row) > 4:
        prizes = row[4]
        numbers = [n.strip() for n in prizes.split('|') if 'Prize' in n]
        for num_str in numbers:
            num = ''.join(c for c in num_str if c.isdigit())
            if len(num) == 4:
                for digit in num:
                    digit_freq[digit] += 1

print("Grand Dragon 4D - Digit Frequency Analysis")
print("=" * 50)
total = sum(digit_freq.values())
for digit in sorted(digit_freq.keys()):
    count = digit_freq[digit]
    pct = (count / total * 100) if total > 0 else 0
    print(f"Digit {digit}: {count} times ({pct:.2f}%)")

expected = total / 10
print(f"\nExpected per digit (if random): {expected:.1f} ({10.0:.2f}%)")
print(f"\nDeviation from expected:")
for digit in sorted(digit_freq.keys()):
    deviation = digit_freq[digit] - expected
    print(f"Digit {digit}: {deviation:+.1f}")
