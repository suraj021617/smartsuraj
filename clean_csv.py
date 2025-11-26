# Read and filter out empty lines
with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    lines = [line for line in f if line.strip()]

# Write back without empty lines
with open('4d_results_history.csv', 'w', encoding='utf-8', newline='') as f:
    f.writelines(lines)

print(f"CSV cleaned! Removed {20212 - len(lines)} empty lines.")
print(f"Total valid lines: {len(lines)}")
print(f"First 3 lines:")
for i, line in enumerate(lines[:3]):
    print(f"  Line {i+1}: {line[:80]}...")
