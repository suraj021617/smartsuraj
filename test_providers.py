import pandas as pd
import os

# Load CSV
csv_paths = ['4d_results_history.csv', 'utils/4d_results_history.csv']
df = None

for csv_path in csv_paths:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=False, on_bad_lines='skip')
        if not df.empty:
            print(f"Loaded CSV from: {csv_path}")
            break

if df is not None and not df.empty:
    # Get unique providers
    all_providers = df['provider'].dropna().unique()
    provider_list = sorted([str(p).strip().lower() for p in all_providers if p and str(p).strip() and str(p) != 'nan'])
    
    print(f"\nTotal providers found: {len(provider_list)}")
    print("\nProvider list:")
    for i, provider in enumerate(provider_list, 1):
        print(f"{i}. {provider}")
    
    # Count draws per provider
    print("\nDraws per provider:")
    for provider in provider_list:
        count = len(df[df['provider'].str.lower().str.strip() == provider])
        print(f"{provider}: {count} draws")
else:
    print("No data found!")
