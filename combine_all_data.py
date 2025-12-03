import pandas as pd
import os
from glob import glob

def combine_all_csv_data():
    all_data = []
    
    # Read all CSV files from csv_data folder
    csv_files = glob("csv_data/*.csv")
    
    for file in csv_files:
        try:
            df = pd.read_csv(file, on_bad_lines='skip')
            if len(df) > 0:
                all_data.append(df)
                print(f"Added: {file} ({len(df)} rows)")
        except:
            print(f"Skipped: {file} (error reading)")
    
    # Read Nov 1 data
    try:
        nov1_df = pd.read_csv("nov1_data.csv", on_bad_lines='skip')
        if len(nov1_df) > 0:
            all_data.append(nov1_df)
            print(f"Added: nov1_data.csv ({len(nov1_df)} rows)")
    except:
        print("No Nov 1 data found")
    
    # Combine all data
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        before_count = len(combined)
        combined = combined.drop_duplicates()
        after_count = len(combined)
        
        # Sort by first column (date)
        combined = combined.sort_values(combined.columns[0])
        
        # Save master file
        combined.to_csv('master_4d_complete.csv', index=False)
        
        # Restore to project folders
        combined.to_csv('4d_results_history.csv', index=False)
        combined.to_csv('data/4d_results_history.csv', index=False)
        combined.to_csv('scraper/4d_results_history.csv', index=False)
        
        print(f"\nCOMPLETE!")
        print(f"Total records: {after_count}")
        print(f"Duplicates removed: {before_count - after_count}")
        print(f"Files updated: master_4d_complete.csv, 4d_results_history.csv, data/, scraper/")
    else:
        print("No data found to combine")

if __name__ == "__main__":
    combine_all_csv_data()