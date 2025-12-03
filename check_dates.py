import os
import pandas as pd
from glob import glob

def check_csv_dates():
    # Check main CSV files
    csv_files = glob("*.csv") + glob(".history/*.csv") + glob("utils/*.csv")
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'Date' in df.columns or any('date' in col.lower() for col in df.columns):
                date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                if date_col:
                    first_date = df[date_col].iloc[0] if len(df) > 0 else "Empty"
                    last_date = df[date_col].iloc[-1] if len(df) > 0 else "Empty"
                    print(f"{file}: {first_date} to {last_date}")
        except:
            pass

if __name__ == "__main__":
    check_csv_dates()