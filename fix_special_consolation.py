#!/usr/bin/env python3
"""
Quick fix for Special and Consolation boxes not showing content
This script will check and fix the data extraction logic
"""

import pandas as pd
import re

def check_csv_data():
    """Check the CSV data to see what's in special and consolation columns"""
    try:
        df = pd.read_csv('4d_results_history_fixed.csv')
        print(f"Total rows: {len(df)}")
        
        # Check the last 5 rows
        print("\n=== LAST 5 ROWS ===")
        for i, row in df.tail(5).iterrows():
            print(f"\nRow {i}:")
            print(f"Date: {row.get('Date', 'N/A')}")
            print(f"Provider: {row.get('Provider', 'N/A')}")
            
            # Check special column (might be different column names)
            special_cols = [col for col in df.columns if 'special' in col.lower()]
            consolation_cols = [col for col in df.columns if 'consolation' in col.lower()]
            
            print(f"Special columns found: {special_cols}")
            print(f"Consolation columns found: {consolation_cols}")
            
            if special_cols:
                special_data = str(row[special_cols[0]])
                print(f"Special data: {special_data[:100]}...")
                # Extract numbers
                numbers = re.findall(r'\b\d{4}\b', special_data)
                print(f"Extracted special numbers: {numbers}")
            
            if consolation_cols:
                consolation_data = str(row[consolation_cols[0]])
                print(f"Consolation data: {consolation_data[:100]}...")
                # Extract numbers
                numbers = re.findall(r'\b\d{4}\b', consolation_data)
                print(f"Extracted consolation numbers: {numbers}")
        
        # Show column names
        print(f"\n=== ALL COLUMNS ===")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
            
    except Exception as e:
        print(f"Error: {e}")

def fix_app_py():
    """Create a fixed version of the data loading logic"""
    
    fix_code = '''
# FIXED VERSION - Replace the load_csv_data function with this:

def load_csv_data():
    """⚡ Load CSV data with FIXED special/consolation parsing"""
    global _csv_cache, _csv_cache_time
    
    if _csv_cache is not None and _csv_cache_time is not None:
        if (datetime.now() - _csv_cache_time).seconds < 60:
            return _csv_cache
    
    import csv
    data = []
    
    try:
        with open('4d_results_history_fixed.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Get header row
            print(f"CSV Headers: {header}")
            
            for row_num, row in enumerate(reader):
                if len(row) < 3 or not row[0]:
                    continue
                
                try:
                    date_parsed = pd.to_datetime(row[0])
                except:
                    continue
                
                # Extract provider
                provider = 'unknown'
                if len(row) > 1 and row[1]:
                    url_text = str(row[1]).lower()
                    if 'magnum' in url_text:
                        provider = 'magnum'
                    elif 'damacai' in url_text:
                        provider = 'damacai'
                    elif 'singapore' in url_text:
                        provider = 'singapore'
                    elif 'toto' in url_text:
                        provider = 'toto'
                
                # Get basic info
                draw_type = row[2] if len(row) > 2 else ''
                draw_info = row[4] if len(row) > 4 else ''
                prizes_text = row[5] if len(row) > 5 else ''
                
                # Extract prizes
                first_num = ''
                second_num = ''
                third_num = ''
                
                # Try multiple patterns
                first_match = re.search(r'1st.*?(\\d{4})', prizes_text, re.IGNORECASE)
                if first_match:
                    first_num = first_match.group(1)
                
                second_match = re.search(r'2nd.*?(\\d{4})', prizes_text, re.IGNORECASE)
                if second_match:
                    second_num = second_match.group(1)
                
                third_match = re.search(r'3rd.*?(\\d{4})', prizes_text, re.IGNORECASE)
                if third_match:
                    third_num = third_match.group(1)
                
                # FIXED: Get special and consolation - try different column indices
                special_text = ''
                consolation_text = ''
                
                # Try columns 6 and 7 first
                if len(row) > 6:
                    special_text = str(row[6]).strip()
                if len(row) > 7:
                    consolation_text = str(row[7]).strip()
                
                # If empty, try to find in other columns
                if not special_text or special_text == 'nan':
                    for i in range(6, min(len(row), 15)):
                        if 'special' in str(row[i]).lower():
                            special_text = str(row[i])
                            break
                
                if not consolation_text or consolation_text == 'nan':
                    for i in range(6, min(len(row), 15)):
                        if 'consolation' in str(row[i]).lower():
                            consolation_text = str(row[i])
                            break
                
                # Clean up the text but keep the numbers
                if special_text and special_text != 'nan':
                    # Remove common non-number text but keep spaces between numbers
                    special_text = re.sub(r'[^0-9\s]', ' ', special_text)
                    special_text = re.sub(r'\s+', ' ', special_text).strip()
                
                if consolation_text and consolation_text != 'nan':
                    # Remove common non-number text but keep spaces between numbers  
                    consolation_text = re.sub(r'[^0-9\s]', ' ', consolation_text)
                    consolation_text = re.sub(r'\s+', ' ', consolation_text).strip()
                
                data.append({
                    'date': row[0],
                    'date_parsed': date_parsed,
                    'provider': provider,
                    'draw_type': draw_type,
                    'draw_info': draw_info,
                    'prizes': prizes_text,
                    'special': special_text,
                    'consolation': consolation_text,
                    '1st_real': first_num,
                    '2nd_real': second_num,
                    '3rd_real': third_num
                })
                
                # Debug: Print first few rows
                if row_num < 3:
                    print(f"Row {row_num}: Special='{special_text}', Consolation='{consolation_text}'")
    
    except Exception as e:
        logger.error(f"CSV error: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    _csv_cache = df
    _csv_cache_time = datetime.now()
    
    logger.info(f"✅ Loaded {len(df)} rows")
    return df
'''
    
    print("=== COPY THIS CODE ===")
    print(fix_code)
    print("=== END CODE ===")

if __name__ == "__main__":
    print("Checking CSV data structure...")
    check_csv_data()
    print("\n" + "="*50)
    print("Here's the fixed code:")
    fix_app_py()