import pandas as pd
import os
from glob import glob
import shutil

def combine_all_data():
    """Combine all CSV data from all sources - Nov 1, Nov 23, Nov 26 and others"""
    print("Starting data combination...")
    
    all_dataframes = []
    
    # 1. Get Nov 1 data from git branch
    try:
        os.system('git show backup_before_nov1:4d_results_history.csv > temp_nov1.csv')
        df_nov1 = pd.read_csv('temp_nov1.csv', on_bad_lines='skip')
        all_dataframes.append(df_nov1)
        print(f"✓ Nov 1 data: {len(df_nov1)} records")
        os.remove('temp_nov1.csv')
    except:
        print("✗ Nov 1 data not found")
    
    # 2. Get Nov 23 data from morning-backup branch
    try:
        os.system('git show morning-backup:4d_results_history.csv > temp_nov23.csv')
        df_nov23 = pd.read_csv('temp_nov23.csv', on_bad_lines='skip')
        all_dataframes.append(df_nov23)
        print(f"✓ Nov 23 data: {len(df_nov23)} records")
        os.remove('temp_nov23.csv')
    except:
        print("✗ Nov 23 data not found")
    
    # 3. Get Nov 26 data from main-backup branch
    try:
        os.system('git show main-backup:4d_results_history.csv > temp_nov26.csv')
        df_nov26 = pd.read_csv('temp_nov26.csv', on_bad_lines='skip')
        all_dataframes.append(df_nov26)
        print(f"✓ Nov 26 data: {len(df_nov26)} records")
        os.remove('temp_nov26.csv')
    except:
        print("✗ Nov 26 data not found")
    
    # 4. Get all CSV files from csv_data folder
    csv_files = glob("csv_data/*.csv")
    for file in csv_files:
        try:
            df = pd.read_csv(file, on_bad_lines='skip')
            all_dataframes.append(df)
            print(f"✓ {file}: {len(df)} records")
        except:
            print(f"✗ {file}: error reading")
    
    # 5. Combine all data
    if all_dataframes:
        combined = pd.concat(all_dataframes, ignore_index=True)
        
        # Remove duplicates
        before = len(combined)
        combined = combined.drop_duplicates()
        after = len(combined)
        
        # Sort by date column
        combined = combined.sort_values(combined.columns[0])
        
        # Save master file
        combined.to_csv('master_complete_4d.csv', index=False)
        
        # Restore to project
        combined.to_csv('4d_results_history.csv', index=False)
        
        # Ensure directories exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('scraper', exist_ok=True)
        
        combined.to_csv('data/4d_results_history.csv', index=False)
        combined.to_csv('scraper/4d_results_history.csv', index=False)
        
        print(f"\n✓ COMPLETE!")
        print(f"✓ Total records: {after}")
        print(f"✓ Duplicates removed: {before - after}")
        print(f"✓ Files updated: master_complete_4d.csv, 4d_results_history.csv, data/, scraper/")
        return True
    else:
        print("✗ No data found to combine")
        return False

def restore_backup(backup_name):
    """Restore from specific backup"""
    try:
        os.system(f'git show {backup_name}:4d_results_history.csv > restored_backup.csv')
        shutil.copy('restored_backup.csv', '4d_results_history.csv')
        shutil.copy('restored_backup.csv', 'data/4d_results_history.csv')
        shutil.copy('restored_backup.csv', 'scraper/4d_results_history.csv')
        os.remove('restored_backup.csv')
        print(f"✓ Restored from {backup_name}")
        return True
    except:
        print(f"✗ Failed to restore from {backup_name}")
        return False

if __name__ == "__main__":
    print("=== DATA MANAGER ===")
    print("1. Combine All Data (Nov 1 + Nov 23 + Nov 26 + All CSV)")
    print("2. Restore from backup_before_nov1")
    print("3. Restore from morning-backup")
    print("4. Restore from main-backup")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == "1":
        combine_all_data()
    elif choice == "2":
        restore_backup("backup_before_nov1")
    elif choice == "3":
        restore_backup("morning-backup")
    elif choice == "4":
        restore_backup("main-backup")
    else:
        print("Invalid choice")
    
    input("Press ENTER to exit...")