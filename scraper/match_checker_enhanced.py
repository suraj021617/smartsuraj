"""
Enhanced Match Checker with Provider and Date Range Selection
"""

import pandas as pd
from datetime import datetime, timedelta
import re

def select_provider():
    providers = {
        '1': 'magnum',
        '2': 'damacai', 
        '3': 'toto',
        '4': 'all'
    }
    
    print("\nSelect Provider:")
    for key, value in providers.items():
        print(f"[{key}] {value}")
    
    choice = input("\nEnter choice (1-4): ").strip()
    return providers.get(choice, providers['4'])

def select_date_range():
    print("\nSelect Date Range:")
    print("[1] Last 7 days")
    print("[2] Last 30 days") 
    print("[3] Last 90 days")
    print("[4] Custom range")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        days = 7
    elif choice == '3':
        days = 90
    elif choice == '4':
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()
        return start_date, end_date
    else:
        days = 30
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    return start_date, end_date

def load_data():
    """Load 4D results data"""
    try:
        df = pd.read_csv('data/4d_results_history.csv')
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except FileNotFoundError:
        print("❌ Data file not found: data/4d_results_history.csv")
        return None

def check_matches_enhanced():
    """Enhanced match checking using REAL working logic"""
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Get user selections
    provider = select_provider()
    start_date, end_date = select_date_range()
    
    # Filter data
    if provider != 'all':
        df = df[df['provider'].str.contains(provider, case=False, na=False)]
    
    # Filter by date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df['date_parsed'] >= start_dt) & (df['date_parsed'] <= end_dt)]
    
    if df.empty:
        print("❌ No data found for selected filters")
        return
    
    # Get predictions from user
    predictions = input("\nEnter predictions (comma-separated): ").split(',')
    predictions = [p.strip() for p in predictions if p.strip()]
    
    if not predictions:
        print("❌ No predictions entered")
        return
    
    # Extract actual results - REAL logic
    actual = []
    for _, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            if col in row and pd.notna(row[col]):
                num = str(row[col]).strip()
                if len(num) == 4 and num.isdigit():
                    actual.append(num)
    
    print(f"\n{'='*60}")
    print(f"MATCH CHECKER RESULTS")
    print(f"{'='*60}")
    print(f"Provider: {provider.upper()}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Total Draws: {len(df)}")
    print(f"Predictions: {', '.join(predictions)}")
    
    # REAL MATCH LOGIC from existing app
    matches = {
        'exact': [],
        'ibox': [],
        'front3': [],
        'back3': [],
        'digit3': [],
        'digit2': []
    }
    
    for pred in predictions:
        for act in actual:
            if pred == act:
                matches['exact'].append((pred, act))
            elif sorted(pred) == sorted(act):
                matches['ibox'].append((pred, act))
            elif pred[:3] == act[:3]:
                matches['front3'].append((pred, act))
            elif pred[1:] == act[1:]:
                matches['back3'].append((pred, act))
            else:
                digit_matches = sum(1 for i in range(4) if pred[i] == act[i])
                if digit_matches == 3:
                    matches['digit3'].append((pred, act))
                elif digit_matches == 2:
                    matches['digit2'].append((pred, act))
    
    # Display results - REAL format
    print(f"\n=== MATCH RESULTS ===")
    print(f"Exact: {len(matches['exact'])} - {matches['exact']}")
    print(f"iBox: {len(matches['ibox'])} - {matches['ibox']}")
    print(f"Front 3: {len(matches['front3'])} - {matches['front3']}")
    print(f"Back 3: {len(matches['back3'])} - {matches['back3']}")
    print(f"3-Digit: {len(matches['digit3'])} - {matches['digit3']}")
    print(f"2-Digit: {len(matches['digit2'])} - {matches['digit2']}")
    
    total_hits = sum(len(v) for v in matches.values())
    accuracy = (total_hits / len(predictions)) * 100 if predictions else 0
    print(f"\nTotal Hits: {total_hits}/{len(predictions)} ({accuracy:.1f}%)")

def main():
    print("=" * 60)
    print("ENHANCED 4D MATCH CHECKER")
    print("=" * 60)
    
    try:
        check_matches_enhanced()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\nPress ENTER to exit...")

if __name__ == "__main__":
    main()