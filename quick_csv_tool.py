import pandas as pd
import os

def main():
    print("CSV Data Tool")
    print("1. Scan CSV files")
    print("2. Add data to CSV")
    print("3. Check CSV structure")
    
    choice = input("Choose (1-3): ")
    
    if choice == "1":
        scan_csv()
    elif choice == "2":
        add_data()
    elif choice == "3":
        check_structure()

def scan_csv():
    files = [
        "4d_results_history.csv",
        "data/4d_training_ready.csv", 
        "data/label_mapping.csv",
        "smart_history.csv"
    ]
    
    for file in files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file, nrows=5)
                print(f"\n{file}: {df.shape[0]} rows shown")
                print(df)
            except:
                print(f"\n{file}: Error reading")
        else:
            print(f"\n{file}: Not found")

def add_data():
    filename = input("CSV filename: ")
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        print(f"Columns: {list(df.columns)}")
        
        data = input("Enter data (comma-separated): ")
        values = data.split(",")
        
        if len(values) == len(df.columns):
            new_row = pd.DataFrame([values], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(filename, index=False)
            print("Data added!")
        else:
            print(f"Need {len(df.columns)} values")
    else:
        print("File not found")

def check_structure():
    filename = input("CSV filename: ")
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print("First 3 rows:")
            print(df.head(3))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()