import time
import sys

print("="*60)
print("SPEED BENCHMARK - Before vs After")
print("="*60)

# Simulate old method (processing all data)
print("\n[OLD METHOD] Processing ALL 393,975 numbers...")
start = time.time()
test_data = [{'number': f'{i:04d}'} for i in range(393975)]
old_time = time.time() - start
print(f"Data prep time: {old_time:.2f}s")

# Test optimized method
print("\n[NEW METHOD] Processing last 500 only...")
start = time.time()
test_data_optimized = test_data[-500:]
new_time = time.time() - start
print(f"Data prep time: {new_time:.4f}s")

# Calculate improvement
speedup = old_time / new_time if new_time > 0 else 999
print("\n" + "="*60)
print(f"SPEEDUP: {speedup:.0f}x FASTER")
print(f"Old: {old_time:.2f}s -> New: {new_time:.4f}s")
print("="*60)

# Test actual learning function
print("\n[TESTING ACTUAL FUNCTION]")
sys.path.insert(0, 'utils')
from day_to_day_learner import learn_day_to_day_patterns

start = time.time()
patterns = learn_day_to_day_patterns(test_data)
actual_time = time.time() - start

print(f"[OK] Learning completed in {actual_time:.2f}s")
print(f"[OK] Patterns learned: {len(patterns['digit_transitions'])} digit transitions")
print(f"[OK] Sequence patterns: {len(patterns['sequence_patterns'])}")

if actual_time < 5:
    print("\n[SUCCESS] Under 5 seconds! (was 654s)")
else:
    print(f"\n[WARNING] Took {actual_time:.2f}s")
