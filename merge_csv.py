import pandas as pd

# Read both CSV files
current = pd.read_csv('4d_results_history.csv')
new_data = pd.read_csv('4d_results_with_oct22.csv')

# Combine and remove duplicates
combined = pd.concat([current, new_data]).drop_duplicates()

# Sort by date
combined = combined.sort_values(combined.columns[0])

# Save to main CSV
combined.to_csv('4d_results_history.csv', index=False)

# Also save to data folder
combined.to_csv('data/4d_results_history.csv', index=False)

print(f"✓ Merged CSV files")
print(f"✓ Total records: {len(combined)}")
print(f"✓ Updated main CSV files")