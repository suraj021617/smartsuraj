import re
from collections import Counter

# Read file directly
with open('4d_results_history.csv', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Extract all 4D numbers
numbers = re.findall(r'\b\d{4}\b', content)
numbers = [n for n in numbers if n != '----' and n != '2015' and n != '2025']

print("="*80)
print("4D LOTTERY ANALYSIS")
print("="*80)

# 1. Most Frequent Numbers
print("\n1. TOP 30 MOST FREQUENT NUMBERS:")
print(f"{'Number':<10}{'Frequency':<15}")
print("-"*30)
freq = Counter(numbers)
for num, count in freq.most_common(30):
    print(f"{num:<10}{count:<15}")

# 2. Digit Frequency by Position
print("\n2. DIGIT FREQUENCY BY POSITION:")
pos_freq = [{}, {}, {}, {}]
for num in numbers:
    if len(num) == 4:
        for i, digit in enumerate(num):
            pos_freq[i][digit] = pos_freq[i].get(digit, 0) + 1

for i in range(4):
    print(f"\nPosition {i+1}:")
    sorted_digits = sorted(pos_freq[i].items(), key=lambda x: x[1], reverse=True)
    for digit, count in sorted_digits[:5]:
        print(f"  Digit {digit}: {count} times")

# 3. Pattern Analysis
print("\n3. PATTERN ANALYSIS:")
def get_pattern(num):
    if len(set(num)) == 4: return 'ABCD'
    if len(set(num)) == 1: return 'AAAA'
    if len(set(num)) == 2:
        counts = Counter(num)
        return 'AAAB' if 3 in counts.values() else 'AABB'
    return 'AABC'

patterns = Counter([get_pattern(n) for n in numbers if len(n) == 4])
for pattern, count in patterns.most_common():
    pct = (count / len(numbers)) * 100
    print(f"{pattern}: {count} times ({pct:.1f}%)")

# 4. Least Frequent (Overdue)
print("\n4. TOP 30 LEAST FREQUENT (POTENTIALLY OVERDUE):")
print(f"{'Number':<10}{'Frequency':<15}")
print("-"*30)
for num, count in freq.most_common()[-30:]:
    print(f"{num:<10}{count:<15}")

print("\n" + "="*80)
print(f"Total 4D numbers analyzed: {len(numbers)}")
print(f"Unique 4D numbers: {len(freq)}")
print("="*80)
