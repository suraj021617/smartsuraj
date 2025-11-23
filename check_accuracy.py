"""
Quick Accuracy Checker - See which methods matched!
"""
import pandas as pd
from collections import Counter

# Load data
df = pd.read_csv('4d_results_history.csv', on_bad_lines='skip', encoding='utf-8', encoding_errors='ignore')
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date_parsed')

# Get latest 3 draws
latest = df.tail(3)

print("\n" + "="*60)
print("🎯 LATEST RESULTS - Which predictions would have WON?")
print("="*60)

for idx, row in latest.iterrows():
    date = row['date_parsed'].strftime('%Y-%m-%d')
    provider = row['provider']
    
    # Extract actual winners
    import re
    prize_text = str(row.get('3rd', ''))
    first = re.search(r'1st\s+Prize\s+(\d{4})', prize_text, re.IGNORECASE)
    second = re.search(r'2nd\s+Prize\s+(\d{4})', prize_text, re.IGNORECASE)
    third = re.search(r'3rd\s+Prize\s+(\d{4})', prize_text, re.IGNORECASE)
    
    winners = []
    if first: winners.append(first.group(1))
    if second: winners.append(second.group(1))
    if third: winners.append(third.group(1))
    
    if not winners:
        continue
    
    print(f"\n📅 {date} - {provider.upper()}")
    print(f"🏆 Winners: {', '.join(winners)}")
    
    # Now check which methods would have predicted these
    # Get all numbers from recent history for frequency
    recent_nums = []
    for col in ['1st', '2nd', '3rd']:
        if col in df.columns:
            for val in df[col].tail(100).astype(str):
                found = re.findall(r'\d{4}', val)
                recent_nums.extend([n for n in found if len(n) == 4])
    
    freq = Counter(recent_nums)
    top_freq = [num for num, count in freq.most_common(10)]
    
    # Check matches
    print("\n✅ MATCHES:")
    
    # Quick Pick (frequency)
    quick_matches = [w for w in winners if w in top_freq[:5]]
    if quick_matches:
        print(f"   🎲 QUICK PICK: {', '.join(quick_matches)} ✓")
    
    # Hot numbers
    hot_matches = [w for w in winners if w in top_freq[:10]]
    if hot_matches:
        print(f"   🔥 HOT NUMBERS: {', '.join(hot_matches)} ✓")
    
    # Check if any winner is in recent 20
    recent_20 = recent_nums[-20:]
    recent_matches = [w for w in winners if w in recent_20]
    if recent_matches:
        print(f"   📊 RECENT PATTERN: {', '.join(recent_matches)} ✓")
    
    if not quick_matches and not hot_matches and not recent_matches:
        print("   ❌ No method predicted these numbers")
        print(f"   💡 Winners were: {', '.join(winners)}")

print("\n" + "="*60)
print("💡 TIP: Use the method that matched most often!")
print("="*60 + "\n")
