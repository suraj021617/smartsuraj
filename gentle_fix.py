import csv

# Read all lines including problematic ones
all_rows = []
with open('4d_results_history.csv', 'r', encoding='utf-8-sig', newline='') as f:
    content = f.read()
    
# Split by newlines and process
lines = content.split('\n')
print(f"Total lines in file: {len(lines)}")

# Parse as CSV
with open('4d_results_history.csv', 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.reader(f)
    all_rows = list(reader)

print(f"Total CSV rows parsed: {len(all_rows)}")
print(f"Header: {all_rows[0]}")
print(f"Last row date: {all_rows[-1][0] if all_rows[-1] else 'empty'}")

# Write back cleanly
with open('4d_results_history.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_rows)

print(f"Done! Wrote {len(all_rows)} rows")
