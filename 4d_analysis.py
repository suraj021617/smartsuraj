import pandas as pd
import re
from collections import defaultdict, Counter
from datetime import datetime

# Read CSV with error handling
df = pd.read_csv('4d_results_history.csv', on_bad_lines='skip', encoding='utf-8', low_memory=False)

# Extract 4D numbers from all prize columns
def extract_4d_numbers(text):
    if pd.isna(text):
        return []
    numbers = re.findall(r'\b\d{4}\b', str(text))
    return [n for n in numbers if n != '----']

# Collect all 4D numbers with dates and operators
all_numbers = []
for idx, row in df.iterrows():
    date = row['date']
    operator = row.get('provider', 'Unknown')
    
    for col in ['1st', '2nd', '3rd', 'special', 'consolation']:
        if col in row:
            numbers = extract_4d_numbers(row[col])
            for num in numbers:
                all_numbers.append({
                    'date': date,
                    'number': num,
                    'operator': operator,
                    'prize_type': col
                })

# Create DataFrame
numbers_df = pd.DataFrame(all_numbers)

# 1. GAP ANALYSIS
print("="*80)
print("1. GAP ANALYSIS - Number Frequency & Gaps")
print("="*80)

gap_analysis = {}
for number in numbers_df['number'].unique():
    occurrences = numbers_df[numbers_df['number'] == number].index.tolist()
    
    if len(occurrences) > 1:
        gaps = [occurrences[i+1] - occurrences[i] for i in range(len(occurrences)-1)]
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        latest_gap = len(numbers_df) - occurrences[-1] if occurrences else 0
        
        gap_analysis[number] = {
            'total_hits': len(occurrences),
            'gaps': gaps,
            'avg_gap': round(avg_gap, 2),
            'latest_gap': latest_gap,
            'last_seen': occurrences[-1]
        }

# Top overdue numbers
overdue = sorted(gap_analysis.items(), key=lambda x: x[1]['latest_gap'], reverse=True)[:20]
print("\nTop 20 OVERDUE Numbers (Not appeared for longest):")
print(f"{'Number':<10}{'Total Hits':<12}{'Avg Gap':<12}{'Days Since':<15}")
print("-"*50)
for num, data in overdue:
    print(f"{num:<10}{data['total_hits']:<12}{data['avg_gap']:<12}{data['latest_gap']:<15}")

# 2. DIGIT FREQUENCY ANALYSIS
print("\n" + "="*80)
print("2. DIGIT FREQUENCY BY POSITION")
print("="*80)

position_freq = {0: Counter(), 1: Counter(), 2: Counter(), 3: Counter()}
for num in numbers_df['number']:
    for pos, digit in enumerate(num):
        position_freq[pos][digit] += 1

for pos in range(4):
    print(f"\nPosition {pos+1} (Thousands/Hundreds/Tens/Units):")
    most_common = position_freq[pos].most_common(10)
    for digit, count in most_common:
        print(f"  Digit {digit}: {count} times")

# 3. PATTERN ANALYSIS
print("\n" + "="*80)
print("3. PATTERN ANALYSIS (ABCD, AABC, AABB, etc.)")
print("="*80)

def classify_pattern(num):
    digits = list(num)
    unique = len(set(digits))
    
    if unique == 4:
        return 'ABCD'
    elif unique == 1:
        return 'AAAA'
    elif unique == 2:
        counts = Counter(digits)
        if 3 in counts.values():
            return 'AAAB'
        else:
            return 'AABB'
    else:  # unique == 3
        return 'AABC'

pattern_counts = Counter()
for num in numbers_df['number']:
    pattern = classify_pattern(num)
    pattern_counts[pattern] += 1

print("\nPattern Distribution:")
for pattern, count in pattern_counts.most_common():
    pct = (count / len(numbers_df)) * 100
    print(f"{pattern}: {count} times ({pct:.2f}%)")

# 4. HOT NUMBERS (Recent frequency)
print("\n" + "="*80)
print("4. HOT NUMBERS (Last 100 draws)")
print("="*80)

recent_numbers = numbers_df.tail(100)
hot_numbers = recent_numbers['number'].value_counts().head(20)
print("\nTop 20 Hot Numbers:")
for num, count in hot_numbers.items():
    print(f"Number {num}: appeared {count} times")

# 5. OPERATOR-WISE ANALYSIS
print("\n" + "="*80)
print("5. OPERATOR-WISE STATISTICS")
print("="*80)

operators = numbers_df['operator'].value_counts()
print("\nTotal draws by operator:")
for op, count in operators.items():
    if 'magnum' in op.lower():
        print(f"Magnum: {count}")
    elif 'damacai' in op.lower() or 'da ma cai' in op.lower():
        print(f"Da Ma Cai: {count}")
    elif 'toto' in op.lower():
        print(f"Toto: {count}")
    elif 'singapore' in op.lower():
        print(f"Singapore 4D: {count}")

# 6. MONTH SUMMARY
print("\n" + "="*80)
print("6. MONTHLY SUMMARY")
print("="*80)

numbers_df['date'] = pd.to_datetime(numbers_df['date'], errors='coerce')
numbers_df['month'] = numbers_df['date'].dt.to_period('M')
monthly = numbers_df.groupby('month').size()
print("\nDraws per month:")
for month, count in monthly.items():
    print(f"{month}: {count} numbers drawn")

# 7. PREDICTION CANDIDATES
print("\n" + "="*80)
print("7. PREDICTION CANDIDATES (Based on Gap Analysis)")
print("="*80)

# Numbers due based on average gap
due_numbers = []
for num, data in gap_analysis.items():
    if data['latest_gap'] > data['avg_gap'] * 1.5 and data['total_hits'] >= 3:
        due_numbers.append((num, data))

due_numbers.sort(key=lambda x: x[1]['latest_gap'] / x[1]['avg_gap'], reverse=True)

print("\nTop 30 Numbers DUE (Latest gap > 1.5x average gap):")
print(f"{'Number':<10}{'Hits':<8}{'Avg Gap':<12}{'Current Gap':<15}{'Ratio':<10}")
print("-"*60)
for num, data in due_numbers[:30]:
    ratio = data['latest_gap'] / data['avg_gap'] if data['avg_gap'] > 0 else 0
    print(f"{num:<10}{data['total_hits']:<8}{data['avg_gap']:<12}{data['latest_gap']:<15}{ratio:.2f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nTotal 4D numbers analyzed: {len(numbers_df)}")
print(f"Unique 4D numbers: {numbers_df['number'].nunique()}")
print(f"Date range: {numbers_df['date'].min()} to {numbers_df['date'].max()}")
