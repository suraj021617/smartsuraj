# add_result_and_learn.py
import pandas as pd
from utils.feedback_learner import FeedbackLearner
import json
from datetime import datetime

def add_result_and_learn():
    """
    Interactive script to add real results and trigger learning
    """
    print("\n" + "="*60)
    print("🎯 ADD REAL RESULT & LEARN")
    print("="*60)
    
    # Load prediction tracking
    pred_file = "prediction_tracking.csv"
    if not pd.io.common.file_exists(pred_file):
        print("❌ No predictions to evaluate!")
        return
    
    pred_df = pd.read_csv(pred_file)
    pending = pred_df[pred_df['hit_status'] == 'pending']
    
    if pending.empty:
        print("✅ All predictions already evaluated!")
        return
    
    print(f"\n📋 Found {len(pending)} pending predictions:\n")
    
    for idx, row in pending.iterrows():
        print(f"{idx}. Date: {row['draw_date']} | Provider: {row['provider']}")
        print(f"   Predicted: {row['predicted_numbers']}")
        print()
    
    # Select which prediction to update
    try:
        choice = int(input("Enter prediction number to update (or -1 to cancel): "))
        if choice == -1:
            return
        
        if choice not in pending.index:
            print("❌ Invalid choice!")
            return
        
        selected = pred_df.loc[choice]
        
        print(f"\n📝 Updating prediction for {selected['draw_date']}")
        print(f"Predicted numbers: {selected['predicted_numbers']}")
        
        # Get actual results
        print("\n🎲 Enter actual results:")
        actual_1st = input("1st Prize (4 digits): ").strip()
        actual_2nd = input("2nd Prize (4 digits): ").strip()
        actual_3rd = input("3rd Prize (4 digits): ").strip()
        
        # Validate inputs
        if not (len(actual_1st) == 4 and actual_1st.isdigit()):
            print("❌ Invalid 1st prize format!")
            return
        
        # Parse predicted numbers
        try:
            predicted_numbers = json.loads(selected['predicted_numbers'].replace("'", '"'))
        except:
            import re
            predicted_numbers = re.findall(r'\\d{4}', selected['predicted_numbers'])
        
        # Evaluate
        learner = FeedbackLearner()
        learner.load_learning_data()
        
        match_type, score, details = learner.evaluate_prediction(
            predicted_numbers, actual_1st, actual_2nd, actual_3rd
        )
        
        # Update tracking
        pred_df.at[choice, 'actual_1st'] = actual_1st
        pred_df.at[choice, 'actual_2nd'] = actual_2nd
        pred_df.at[choice, 'actual_3rd'] = actual_3rd
        pred_df.at[choice, 'hit_status'] = match_type
        pred_df.at[choice, 'accuracy_score'] = score
        
        pred_df.to_csv(pred_file, index=False)
        
        # Learn from result
        learner.learn_from_result({
            'predicted_numbers': predicted_numbers,
            'predictor_methods': selected['predictor_methods'],
            'confidence': selected['confidence'],
            'draw_date': selected['draw_date']
        }, match_type, score)
        
        learner.save_learning_data()
        
        # Show result
        print("\n" + "="*60)
        print("✅ RESULT UPDATED & LEARNED!")
        print("="*60)
        print(f"Match Type: {match_type}")
        print(f"Score: {score}")
        print(f"\nPredicted: {predicted_numbers[:3]}")
        print(f"Actual: {actual_1st}, {actual_2nd}, {actual_3rd}")
        
        # Show learning summary
        best_methods = learner.get_best_methods()
        print("\n🏆 Top Performing Methods:")
        for i, method in enumerate(best_methods, 1):
            print(f"   {i}. {method['method']}: {method['accuracy']:.1f}%")
        
        print("\n💡 AI will use this data to improve future predictions!")
        
    except ValueError:
        print("❌ Invalid input!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    add_result_and_learn()
