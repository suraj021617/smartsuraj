import pandas as pd
import pickle
from collections import Counter
from datetime import datetime

def load_knowledge():
    """Load trained model"""
    try:
        with open('data/trained_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

def learn_new_numbers():
    """Learn only NEW numbers since last training"""
    knowledge = load_knowledge()
    if not knowledge:
        print("❌ No trained model found. Run train_model.py first!")
        return
    
    df = pd.read_csv('4d_results_history.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Get only NEW data
    last_date = pd.to_datetime(knowledge['last_date'])
    new_df = df[df['date'] > last_date]
    
    if len(new_df) == 0:
        print("✅ No new data to learn")
        return
    
    # Extract new numbers
    new_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        if col in new_df.columns:
            new_numbers.extend([n for n in new_df[col].astype(str) if len(n) == 4 and n.isdigit()])
    
    # Update knowledge
    knowledge['numbers'].extend(new_numbers)
    knowledge['frequency'].update(new_numbers)
    knowledge['digit_freq'].update(''.join(new_numbers))
    knowledge['last_trained'] = datetime.now().isoformat()
    knowledge['total_draws'] += len(new_df)
    knowledge['last_date'] = new_df['date'].max().isoformat()
    
    # Save
    with open('data/trained_model.pkl', 'wb') as f:
        pickle.dump(knowledge, f)
    
    print(f"✅ Learned {len(new_numbers)} new numbers from {len(new_df)} draws")
    print(f"📊 Total: {len(knowledge['numbers'])} numbers, {knowledge['total_draws']} draws")

if __name__ == '__main__':
    learn_new_numbers()
