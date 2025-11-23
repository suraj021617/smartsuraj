import csv
import re
from collections import Counter

with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    data = f.read()

numbers = re.findall(r'\b\d{4}\b', data)
pos_counts = [Counter(), Counter(), Counter(), Counter()]

for num in numbers:
    if len(num) == 4:
        for i, digit in enumerate(num):
            pos_counts[i][digit] += 1

print("Digit Position Analysis:")
print(f"{'Digit':<8}{'Pos 1':<8}{'Pos 2':<8}{'Pos 3':<8}{'Pos 4':<8}")
for d in '0123456789':
    print(f"{d:<8}{pos_counts[0][d]:<8}{pos_counts[1][d]:<8}{pos_counts[2][d]:<8}{pos_counts[3][d]:<8}")

print(f"\nTotal 4D numbers analyzed: {len(numbers)}")
print("\nMost frequent by position:")
for i in range(4):
    top = pos_counts[i].most_common(3)
    print(f"Pos {i+1}: {', '.join(f'{d}({c})' for d,c in top)}")
