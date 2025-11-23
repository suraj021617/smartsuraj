import pandas as pd
import os

def scan_csv_files():
    csv_files = [
        '4d_results_history.csv',
        'data/4d_training_ready.csv',
        'data/label_mapping.csv',
        'smart_history.csv'
    ]
    
    for file in csv_files:
        if os.path.exists(file):
            print(f"\n=== {file} ===")
            try:
                df = pd.read_csv(file)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print("First 3 rows:")
                print(df.head(3))
                print("Last 3 rows:")
                print(df.tail(3))
            except Exception as e:
                print(f"Error reading {file}: {e}")
        else:
            print(f"{file} not found")

if __name__ == "__main__":
    scan_csv_files()