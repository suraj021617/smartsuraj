"""
Data Restoration Script
Restores CSV data from November 23 history and today's morning backup
"""

import pandas as pd
import os
from datetime import datetime

def restore_data():
    print("=" * 50)
    print("DATA RESTORATION SCRIPT")
    print("=" * 50)
    
    # Files to restore
    backup_files = [
        "backup_today_morning.csv",
        ".history/backup_today_morning_20251203181038.csv", 
        ".history/backup_today_morning_20251203204453.csv",
        ".history/4d_results_history_20251203204711.csv"
    ]
    
    all_data = []
    
    for file_path in backup_files:
        if os.path.exists(file_path):
            try:
                print(f"Reading {file_path}...")
                df = pd.read_csv(file_path)
                all_data.append(df)
                print(f"✓ Loaded {len(df)} records")
            except Exception as e:
                print(f"✗ Error reading {file_path}: {e}")
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        original_count = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        final_count = len(combined_df)
        
        print(f"\nCombined: {original_count} records")
        print(f"After removing duplicates: {final_count} records")
        
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # Save restored data
        output_file = "data/4d_results_history_restored.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Restored data saved to: {output_file}")
        print(f"✓ Total records restored: {final_count}")
        
        # Show date range
        if 'date' in combined_df.columns:
            dates = pd.to_datetime(combined_df['date'], errors='coerce')
            print(f"Date range: {dates.min()} to {dates.max()}")
        
    else:
        print("✗ No backup files found to restore")

if __name__ == "__main__":
    restore_data()
    input("\nPress ENTER to close...")