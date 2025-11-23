import pandas as pd

def check_csv_matches():
    try:
        df = pd.read_csv('master_predictions.csv')
        print(f"Found {len(df)} prediction records")
        
        total_exact = 0
        total_predictions = 0
        
        for _, row in df.iterrows():
            pred_nums = [x.strip() for x in str(row['predicted_numbers']).split(',')]
            actual_nums = [x.strip() for x in str(row['actual_numbers']).split(',')]
            
            exact_matches = 0
            for pred in pred_nums:
                if pred in actual_nums:
                    exact_matches += 1
            
            total_exact += exact_matches
            total_predictions += len(pred_nums)
            
            print(f"{row['date']}: {exact_matches}/{len(pred_nums)} exact matches")
        
        overall_accuracy = (total_exact / total_predictions) * 100 if total_predictions > 0 else 0
        print(f"\nOverall Accuracy: {total_exact}/{total_predictions} ({overall_accuracy:.1f}%)")
        
    except FileNotFoundError:
        print("master_predictions.csv not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_csv_matches()