import pandas as pd
import os

files = [
    "4d_results_history.csv",
    "data/4d_training_ready.csv", 
    "data/label_mapping.csv",
    "smart_history.csv",
    "master_predictions.csv"
]

for file in files:
    print(f"\n=== {file} ===")
    if os.path.exists(file):
        try:
            df = pd.read_csv(file, nrows=3)
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print("Sample data:")
            print(df)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("File not found")