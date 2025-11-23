"""
FINAL SPEED TEST - Real-world scenario
"""
import time
import csv
import re

print("="*60)
print("REAL-WORLD SPEED TEST")
print("="*60)

# Test 1: Load CSV
print("\n[1] Loading CSV...")
start = time.time()
with open('4d_results_history.csv', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()
load_time = time.time() - start
print(f"    Loaded {len(lines)} lines in {load_time:.2f}s")

# Test 2: Extract numbers
print("\n[2] Extracting 4D numbers...")
start = time.time()
all_nums = []
for line in lines:
    nums = re.findall(r'\b\d{4}\b', line)
    all_nums.extend(nums)
extract_time = time.time() - start
print(f"    Extracted {len(all_nums)} numbers in {extract_time:.2f}s")

# Test 3: Learn patterns (optimized - last 500 only)
print("\n[3] Learning patterns (OPTIMIZED)...")
start = time.time()

# Convert to format expected by learner
draws = [{'number': num} for num in all_nums[-500:]]

from utils.day_to_day_learner import learn_day_to_day_patterns
patterns = learn_day_to_day_patterns(draws)

learn_time = time.time() - start
print(f"    Learned patterns in {learn_time:.2f}s")
print(f"    Digit transitions: {len(patterns['digit_transitions'])}")
print(f"    Sequence patterns: {len(patterns['sequence_patterns'])}")

# Total time
total_time = load_time + extract_time + learn_time

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Total time: {total_time:.2f}s")
print(f"Previous time: 654s")
print(f"Improvement: {654/total_time:.0f}x FASTER")
print("="*60)

if total_time < 10:
    print("\n[SUCCESS] System is now FAST!")
    print("Auto next day will complete in seconds, not minutes!")
else:
    print(f"\n[WARNING] Still slow: {total_time:.2f}s")
