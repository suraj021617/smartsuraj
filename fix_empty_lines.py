import csv

# Read CSV properly
rows = []
with open('4d_results_history.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if any(cell.strip() for cell in row):  # Skip completely empty rows
            rows.append(row)

# Write back without empty lines
with open('4d_results_history.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Fixed! Total rows: {len(rows)}")
print(f"First 3 rows:")
for i, row in enumerate(rows[:3]):
    print(f"  Row {i+1}: {row[:3]}...")
