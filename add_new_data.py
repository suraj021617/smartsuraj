import pandas as pd
from datetime import datetime

def add_prediction_result():
    """Add new prediction result to master_predictions.csv"""
    print("=== Add New Prediction Result ===")
    
    # Get input
    date = input("Date (YYYY-MM-DD) or press Enter for today: ")
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    predicted = input("Predicted numbers (comma-separated): ")
    actual = input("Actual winning numbers (comma-separated): ")
    
    # Calculate matches
    pred_list = [x.strip() for x in predicted.split(',')]
    actual_list = [x.strip() for x in actual.split(',')]
    matches = len(set(pred_list) & set(actual_list))
    
    # Create new row
    new_data = {
        'date': date,
        'predicted_numbers': predicted,
        'actual_numbers': actual,
        'match_count': matches,
        'learned': 'yes',
        'hot_cold_score': 0.5,
        'confidence_score': 0.7,
        'gap_pattern': '+1,+2',
        'pattern_source': 'manual'
    }
    
    # Add to CSV
    try:
        df = pd.read_csv('master_predictions.csv')
        new_row = pd.DataFrame([new_data])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv('master_predictions.csv', index=False)
        print(f"Added prediction result with {matches} matches!")
    except Exception as e:
        print(f"Error: {e}")

def add_smart_history():
    """Add performance tracking data"""
    print("=== Add Smart History ===")
    
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hot_weight = float(input("Hot Weight (0.0-1.0): ") or "0.5")
    pair_weight = float(input("Pair Weight (0.0-1.0): ") or "0.3")
    trans_weight = float(input("Trans Weight (0.0-1.0): ") or "0.2")
    accuracy = float(input("Accuracy %: ") or "50.0")
    
    new_data = {
        'Date': date,
        'HotWeight': hot_weight,
        'PairWeight': pair_weight,
        'TransWeight': trans_weight,
        'Accuracy': accuracy,
        'Top1': None,
        'Provider': None
    }
    
    try:
        df = pd.read_csv('smart_history.csv')
        new_row = pd.DataFrame([new_data])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv('smart_history.csv', index=False)
        print("Added smart history entry!")
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("What do you want to add?")
    print("1. Prediction result")
    print("2. Smart history")
    print("3. View current data")
    
    choice = input("Choose (1-3): ")
    
    if choice == "1":
        add_prediction_result()
    elif choice == "2":
        add_smart_history()
    elif choice == "3":
        view_data()

def view_data():
    """View recent data"""
    print("\n=== Recent Master Predictions ===")
    try:
        df = pd.read_csv('master_predictions.csv')
        print(df.tail(5))
    except:
        print("No data found")
    
    print("\n=== Recent Smart History ===")
    try:
        df = pd.read_csv('smart_history.csv')
        print(df.tail(5))
    except:
        print("No data found")

if __name__ == "__main__":
    main()