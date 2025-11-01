# auto_evaluate.py
import pandas as pd
from utils.feedback_learner import FeedbackLearner
import json

def auto_evaluate_predictions():
    """Automatically evaluate predictions and learn from results"""
    
    # Load prediction tracking data
    try:
        tracking_df = pd.read_csv('prediction_tracking.csv', on_bad_lines='skip')
    except FileNotFoundError:
        print("❌ No prediction tracking file found!")
        return
    except Exception as e:
        print(f"❌ Error reading tracking file: {e}")
        return
    
    # Load historical results
    try:
        results_df = pd.read_csv('4d_results_history.csv', on_bad_lines='skip')
    except FileNotFoundError:
        print("❌ No results history file found!")
        return
    except Exception as e:
        print(f"❌ Error reading results file: {e}")
        return
    
    learner = FeedbackLearner()
    learner.load_learning_data()
    
    print("🔍 Auto-Evaluating Predictions...\n")
    
    updated_count = 0
    
    for idx, row in tracking_df.iterrows():
        # Skip if already evaluated
        if row['hit_status'] != 'pending':
            continue
        
        draw_date = row['draw_date']
        predicted_numbers = json.loads(row['predicted_numbers'].replace("'", '"'))
        
        # Find matching result
        result = results_df[results_df['date'].str.contains(draw_date, na=False)]
        
        if result.empty:
            continue
        
        # Extract actual numbers
        actual_1st = result.iloc[0].get('1st', '')
        actual_2nd = result.iloc[0].get('2nd', '')
        actual_3rd = result.iloc[0].get('3rd', '')
        
        # Evaluate prediction
        match_type, score, details = learner.evaluate_prediction(
            predicted_numbers, actual_1st, actual_2nd, actual_3rd
        )
        
        # Update tracking data
        tracking_df.at[idx, 'hit_status'] = match_type
        tracking_df.at[idx, 'accuracy_score'] = score
        tracking_df.at[idx, 'actual_1st'] = actual_1st
        tracking_df.at[idx, 'actual_2nd'] = actual_2nd
        tracking_df.at[idx, 'actual_3rd'] = actual_3rd
        
        # Learn from result
        learner.learn_from_result({
            'predicted_numbers': predicted_numbers,
            'predictor_methods': row['predictor_methods'],
            'confidence': row['confidence'],
            'draw_date': draw_date
        }, match_type, score)
        
        updated_count += 1
        
        print(f"✅ {draw_date}: {match_type} (Score: {score})")
        print(f"   Predicted: {predicted_numbers[:3]}")
        print(f"   Actual: {actual_1st}, {actual_2nd}, {actual_3rd}\n")
    
    # Save updated tracking
    tracking_df.to_csv('prediction_tracking.csv', index=False)
    
    # Save learning data
    learner.save_learning_data()
    
    # Print summary
    print("\n" + "="*50)
    print("📊 LEARNING SUMMARY")
    print("="*50)
    print(f"✅ Evaluated: {updated_count} predictions")
    
    best_methods = learner.get_best_methods()
    print("\n🏆 Top Performing Methods:")
    for i, method in enumerate(best_methods, 1):
        print(f"   {i}. {method['method']}: {method['accuracy']:.2f}% "
              f"({method['total_predictions']} predictions)")
    
    print("\n💡 Recommendation: Focus on top-performing methods for future predictions!")

if __name__ == "__main__":
    auto_evaluate_predictions()
