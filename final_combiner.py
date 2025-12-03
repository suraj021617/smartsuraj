import pandas as pd
import os
from glob import glob

print("=== FINAL DATA COMBINER ===")
print("Combining Nov 1 + Nov 23 + Nov 26 + All CSV data...")

all_dataframes = []

# 1. Nov 1 data
try:
    os.system('git show backup_before_nov1:4d_results_history.csv > temp_nov1.csv 2>nul')
    df_nov1 = pd.read_csv('temp_nov1.csv', on_bad_lines='skip')
    all_dataframes.append(df_nov1)
    print(f"OK Nov 1 data: {len(df_nov1)} records")
    os.remove('temp_nov1.csv')
except:
    print("SKIP Nov 1 data not found")

# 2. Nov 23 data
try:
    os.system('git show morning-backup:4d_results_history.csv > temp_nov23.csv 2>nul')
    df_nov23 = pd.read_csv('temp_nov23.csv', on_bad_lines='skip')
    all_dataframes.append(df_nov23)
    print(f"OK Nov 23 data: {len(df_nov23)} records")
    os.remove('temp_nov23.csv')
except:
    print("SKIP Nov 23 data not found")

# 3. Nov 26 data
try:
    os.system('git show main-backup:4d_results_history.csv > temp_nov26.csv 2>nul')
    df_nov26 = pd.read_csv('temp_nov26.csv', on_bad_lines='skip')
    all_dataframes.append(df_nov26)
    print(f"OK Nov 26 data: {len(df_nov26)} records")
    os.remove('temp_nov26.csv')
except:
    print("SKIP Nov 26 data not found")

# 4. All CSV files
csv_files = glob("csv_data/*.csv")
for file in csv_files:
    try:
        df = pd.read_csv(file, on_bad_lines='skip')
        all_dataframes.append(df)
        print(f"OK {file}: {len(df)} records")
    except:
        print(f"SKIP {file}: error")

# 5. Combine and process
if all_dataframes:
    combined = pd.concat(all_dataframes, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    combined = combined.sort_values(combined.columns[0])
    
    # Save files
    combined.to_csv('COMPLETE_4D_DATA.csv', index=False)
    combined.to_csv('4d_results_history.csv', index=False)
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('scraper', exist_ok=True)
    combined.to_csv('data/4d_results_history.csv', index=False)
    combined.to_csv('scraper/4d_results_history.csv', index=False)
    
    print(f"\nSUCCESS!")
    print(f"Total records: {after}")
    print(f"Duplicates removed: {before - after}")
    print(f"Files updated: COMPLETE_4D_DATA.csv, 4d_results_history.csv, data/, scraper/")
else:
    print("ERROR: No data found")