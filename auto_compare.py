"""
Auto Compare System - Automatically compares predictions with actual results
Runs whenever new data is added to CSV
"""
import pandas as pd
import json
import os
from datetime import datetime
from collections import Counter

def count_digit_matches(predicted, actual):
    """Count how many digits match between predicted and actual number"""
    if not predicted or not actual or len(predicted) != 4 or len(actual) != 4:
        return 0
    
    pred_digits = list(predicted)
    actual_digits = list(actual)
    
    matches = 0
    for p_digit in pred_digits:
        if p_digit in actual_digits:
            matches += 1
            actual_digits.remove(p_digit)  # Remove to avoid double counting
    
    return matches

def load_predictions():
    """Load saved predictions"""
    pred_file = "prediction_tracking.csv"
    if os.path.exists(pred_file):
        return pd.read_csv(pred_file)
    return pd.DataFrame()

def load_results():
    """Load actual results from CSV"""
    csv_paths = ['4d_results_history.csv', 'utils/4d_results_history.csv']
    for path in csv_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
            return df
    return pd.DataFrame()

def compare_predictions():
    """Compare all pending predictions with actual results"""
    pred_df = load_predictions()
    results_df = load_results()
    
    if pred_df.empty or results_df.empty:
        print("No predictions or results to compare")
        return
    
    # Filter pending predictions
    pending = pred_df[pred_df['hit_status'] == 'pending'].copy()
    
    if pending.empty:
        print("No pending predictions to compare")
        return
    
    updated_count = 0
    comparison_results = []
    
    for idx, pred_row in pending.iterrows():
        try:
            # Parse draw date
            draw_date_str = str(pred_row['draw_date']).split(' ')[0]
            draw_date = pd.to_datetime(draw_date_str, errors='coerce').date()
            
            if draw_date is None:
                continue
            
            provider = str(pred_row['provider']).strip().lower()
            
            # Find matching result
            actual = results_df[
                (results_df['date_parsed'].dt.date == draw_date) & 
                (results_df['provider'] == provider)
            ]
            
            if actual.empty:
                continue
            
            actual_row = actual.iloc[0]
            actual_1st = str(actual_row.get('1st_real', ''))
            actual_2nd = str(actual_row.get('2nd_real', ''))
            actual_3rd = str(actual_row.get('3rd_real', ''))
            
            # Parse predicted numbers
            predicted_str = str(pred_row['predicted_numbers'])
            try:
                if predicted_str.startswith('['):
                    import ast
                    predicted_nums = ast.literal_eval(predicted_str)
                else:
                    import re
                    predicted_nums = re.findall(r'\d{4}', predicted_str)
            except:
                import re
                predicted_nums = re.findall(r'\d{4}', predicted_str)
            
            if not predicted_nums:
                continue
            
            # Compare each predicted number
            best_match = 0
            matched_number = None
            match_type = None
            
            for pred_num in predicted_nums:
                # Check exact match
                if pred_num in [actual_1st, actual_2nd, actual_3rd]:
                    best_match = 4
                    matched_number = pred_num
                    if pred_num == actual_1st:
                        match_type = '1ST PRIZE'
                    elif pred_num == actual_2nd:
                        match_type = '2ND PRIZE'
                    else:
                        match_type = '3RD PRIZE'
                    break
                
                # Check digit matches
                for actual_num in [actual_1st, actual_2nd, actual_3rd]:
                    digit_match = count_digit_matches(pred_num, actual_num)
                    if digit_match > best_match:
                        best_match = digit_match
                        matched_number = pred_num
                        match_type = f'{digit_match}-DIGIT MATCH'
            
            # Update prediction status
            if best_match == 4:
                pred_df.at[idx, 'hit_status'] = f'EXACT - {match_type}'
                pred_df.at[idx, 'accuracy_score'] = 100
            elif best_match == 3:
                pred_df.at[idx, 'hit_status'] = '3-DIGIT MATCH'
                pred_df.at[idx, 'accuracy_score'] = 75
            elif best_match == 2:
                pred_df.at[idx, 'hit_status'] = '2-DIGIT MATCH'
                pred_df.at[idx, 'accuracy_score'] = 50
            elif best_match == 1:
                pred_df.at[idx, 'hit_status'] = '1-DIGIT MATCH'
                pred_df.at[idx, 'accuracy_score'] = 25
            else:
                pred_df.at[idx, 'hit_status'] = 'NO MATCH'
                pred_df.at[idx, 'accuracy_score'] = 0
            
            pred_df.at[idx, 'actual_1st'] = actual_1st
            pred_df.at[idx, 'actual_2nd'] = actual_2nd
            pred_df.at[idx, 'actual_3rd'] = actual_3rd
            
            updated_count += 1
            
            comparison_results.append({
                'date': draw_date_str,
                'provider': provider,
                'predicted': predicted_nums,
                'actual': [actual_1st, actual_2nd, actual_3rd],
                'match': best_match,
                'status': pred_df.at[idx, 'hit_status']
            })
            
            print(f"✅ {draw_date_str} - {provider}: {best_match}-digit match")
            
        except Exception as e:
            print(f"❌ Error processing row {idx}: {e}")
            continue
    
    # Save updated predictions
    if updated_count > 0:
        pred_df.to_csv("prediction_tracking.csv", index=False)
        print(f"\n🎯 Updated {updated_count} predictions")
        
        # Save comparison summary
        summary = {
            'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_compared': updated_count,
            'results': comparison_results
        }
        
        with open('comparison_log.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Comparison log saved to comparison_log.json")
    else:
        print("No predictions were updated")

if __name__ == "__main__":
    print("🔄 Starting automatic comparison...")
    compare_predictions()
    print("✅ Comparison complete!")
