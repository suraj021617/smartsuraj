import csv
import re

# Test with a known number from the visible CSV
test_number = "5548"  # From Oct 19, 2025 Singapore 4D

with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

print(f"Total rows in CSV: {len(rows)}")
print(f"\nSearching for {test_number}...\n")

matches = []
for i, row in enumerate(rows):
    if len(row) >= 5:
        # Check all columns for the number
        row_text = ' '.join(row)
        if test_number in row_text:
            date = row[0] if row[0] else "No date"
            provider = row[2] if len(row) > 2 else "Unknown"
            matches.append((i+1, date, provider))

print(f"Found {len(matches)} matches for {test_number}")
print("\nFirst 5 matches:")
for line, date, provider in matches[:5]:
    print(f"  Line {line}: {date} | {provider[:40]}")

print("\nLast 5 matches:")
for line, date, provider in matches[-5:]:
    print(f"  Line {line}: {date} | {provider[:40]}")

# Verify specific row from visible data
print("\n=== VERIFYING VISIBLE DATA ===")
for i, row in enumerate(rows):
    if len(row) > 0 and row[0] == "2025-10-19" and "Singapore 4D" in str(row):
        print(f"Line {i+1}: {row[0]} | {row[2]}")
        print(f"Prizes: {row[4][:80]}...")
        if test_number in row[4]:
            print(f"✓ Found {test_number} in this row!")
        break
