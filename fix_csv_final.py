import csv

# Read with flexible parsing
rows = []
with open('4d_results_history.csv', 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows.append(header)
    
    for i, row in enumerate(reader, start=2):
        # Ensure exactly 8 columns
        if len(row) > 8:
            # Merge extra columns into last column
            row = row[:7] + [' '.join(row[7:])]
        elif len(row) < 8:
            # Pad with empty strings
            row = row + [''] * (8 - len(row))
        rows.append(row)

# Write clean CSV
with open('4d_results_history.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Fixed! Total rows: {len(rows)}")
print(f"Header: {rows[0]}")
print(f"Last row: {rows[-1][:3]}")
