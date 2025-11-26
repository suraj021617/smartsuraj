"""
Clean CSV by removing '2025' from special and consolation columns
This fixes the scraper issue where year was inserted into prize columns
"""
import pandas as pd
import re
import os
from datetime import datetime

# Backup original file first
csv_path = 'scraper/4d_results_history.csv'
backup_path = f'scraper/4d_results_history_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

if os.path.exists(csv_path):
    print(f"Creating backup: {backup_path}")
    import shutil
    shutil.copy2(csv_path, backup_path)
    
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, header=None, on_bad_lines='skip', encoding='utf-8', engine='python')
    
    # Assign columns
    df.columns = ['date', 'provider', 'col3', 'draw_number', 'prize_text', 'special', 'consolation'][:len(df.columns)]
    
    print(f"Total rows: {len(df)}")
    
    # Count how many rows have '2025' in special/consolation
    special_count = df['special'].astype(str).str.contains(r'\b2025\b', regex=True).sum()
    consolation_count = df['consolation'].astype(str).str.contains(r'\b2025\b', regex=True).sum()
    
    print(f"Rows with '2025' in special column: {special_count}")
    print(f"Rows with '2025' in consolation column: {consolation_count}")
    
    # Show sample before cleaning
    print("\n--- BEFORE CLEANING (Sample) ---")
    sample = df.head(3)
    for idx in range(len(sample)):
        print(f"Row {idx}:")
        print(f"  Special: {sample['special'].iloc[idx]}")
        print(f"  Consolation: {sample['consolation'].iloc[idx]}")
    
    # Clean: Remove standalone '2025' (word boundary)
    df['special'] = df['special'].astype(str).str.replace(r'\b2025\b', '', regex=True)
    df['consolation'] = df['consolation'].astype(str).str.replace(r'\b2025\b', '', regex=True)
    
    # Clean up extra spaces
    df['special'] = df['special'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['consolation'] = df['consolation'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Replace 'nan' string with empty
    df['special'] = df['special'].replace('nan', '')
    df['consolation'] = df['consolation'].replace('nan', '')
    
    # Show sample after cleaning
    print("\n--- AFTER CLEANING (Sample) ---")
    sample_after = df.head(3)
    for idx in range(len(sample_after)):
        print(f"Row {idx}:")
        print(f"  Special: {sample_after['special'].iloc[idx]}")
        print(f"  Consolation: {sample_after['consolation'].iloc[idx]}")
    
    # Save cleaned CSV
    output_path = csv_path  # Overwrite original
    print(f"\nSaving cleaned CSV to: {output_path}")
    df.to_csv(output_path, index=False, header=False, encoding='utf-8')
    
    print("\n✅ CSV cleaned successfully!")
    print(f"✅ Backup saved to: {backup_path}")
    print("\nNext steps:")
    print("1. Restart Flask: python app.py")
    print("2. Clear browser cache")
    print("3. Check dashboard - 2nd Prize should show correctly")
    
else:
    print(f"❌ CSV file not found: {csv_path}")
    print("Please check the file path and try again.")
