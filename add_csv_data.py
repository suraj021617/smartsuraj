import pandas as pd
import os
from datetime import datetime

def add_csv_data():
    print("=== CSV Data Manager ===")
    print("Current CSV files found:")
    
    # Check existing files
    csv_files = {
        '1': ('4d_results_history.csv', 'Main 4D Results'),
        '2': ('data/4d_training_ready.csv', 'Training Data'),
        '3': ('data/label_mapping.csv', 'Label Mapping'),
        '4': ('smart_history.csv', 'Smart History'),
        '5': ('master_predictions.csv', 'Master Predictions'),
        '6': ('prediction_tracking.csv', 'Prediction Tracking')
    }
    
    for key, (file, desc) in csv_files.items():
        exists = "YES" if os.path.exists(file) else "NO"
        size = ""
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                size = f"({df.shape[0]} rows)"
            except:
                size = "(error reading)"
        print(f"{key}. {desc}: {exists} {size}")
    
    print("\nOptions:")
    print("s - Scan all CSV files")
    print("a - Add new data to existing CSV")
    print("c - Create new CSV file")
    print("q - Quit")
    
    choice = input("\nEnter choice: ").lower()
    
    if choice == 's':
        scan_all_csv()
    elif choice == 'a':
        add_to_existing()
    elif choice == 'c':
        create_new_csv()
    elif choice == 'q':
        return
    else:
        print("Invalid choice")

def scan_all_csv():
    print("\n=== Scanning All CSV Files ===")
    
    files_to_scan = [
        '4d_results_history.csv',
        'data/4d_training_ready.csv', 
        'data/label_mapping.csv',
        'smart_history.csv',
        'master_predictions.csv',
        'prediction_tracking.csv'
    ]
    
    for file in files_to_scan:
        if os.path.exists(file):
            print(f"\n--- {file} ---")
            try:
                df = pd.read_csv(file)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print("Sample data:")
                print(df.head(2))
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"\n--- {file} --- NOT FOUND")

def add_to_existing():
    print("\n=== Add to Existing CSV ===")
    filename = input("Enter CSV filename: ")
    
    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return
    
    try:
        df = pd.read_csv(filename)
        print(f"Current shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Simple data entry
        print("\nEnter new row data (comma-separated):")
        new_data = input("Data: ")
        
        if new_data.strip():
            values = [x.strip() for x in new_data.split(',')]
            if len(values) == len(df.columns):
                new_row = pd.DataFrame([values], columns=df.columns)
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(filename, index=False)
                print(f"Added row to {filename}")
            else:
                print(f"Expected {len(df.columns)} values, got {len(values)}")
    except Exception as e:
        print(f"Error: {e}")

def create_new_csv():
    print("\n=== Create New CSV ===")
    filename = input("Enter new CSV filename: ")
    
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    columns = input("Enter column names (comma-separated): ")
    col_list = [x.strip() for x in columns.split(',')]
    
    df = pd.DataFrame(columns=col_list)
    df.to_csv(filename, index=False)
    print(f"Created {filename} with columns: {col_list}")

if __name__ == "__main__":
    add_csv_data()