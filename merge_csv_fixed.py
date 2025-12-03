import pandas as pd

try:
    # Read with error handling
    current = pd.read_csv('4d_results_history.csv', on_bad_lines='skip')
    new_data = pd.read_csv('4d_results_with_oct22.csv', on_bad_lines='skip')
    
    # Get column names from new_data (complete data)
    columns = new_data.columns.tolist()
    
    # Ensure both have same columns
    current = current.reindex(columns=columns, fill_value='')
    
    # Combine and remove duplicates
    combined = pd.concat([current, new_data], ignore_index=True)
    combined = combined.drop_duplicates()
    
    # Sort by first column (date)
    combined = combined.sort_values(combined.columns[0])
    
    # Save to main files
    combined.to_csv('4d_results_history.csv', index=False)
    combined.to_csv('data/4d_results_history.csv', index=False)
    
    print(f"✓ Merged CSV files successfully")
    print(f"✓ Total records: {len(combined)}")
    
except Exception as e:
    print(f"Error: {e}")
    # Fallback: just copy the complete file
    import shutil
    shutil.copy('4d_results_with_oct22.csv', '4d_results_history.csv')
    shutil.copy('4d_results_with_oct22.csv', 'data/4d_results_history.csv')
    print("✓ Copied complete data file")