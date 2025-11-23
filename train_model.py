import pandas as pd
import pickle
from collections import Counter
from datetime import datetime

def train_and_save():
    """Train once on 3000 draws, save to file"""
    print("🚀 Starting training on 3000 draws...")
    
    df = pd.read_csv('4d_results_history.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).tail(3000)
    
    # Extract all numbers
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        if col in df.columns:
            all_numbers.extend([n for n in df[col].astype(str) if len(n) == 4 and n.isdigit()])
    
    # Build knowledge
    knowledge = {
        'numbers': all_numbers,
        'frequency': Counter(all_numbers),
        'digit_freq': Counter(''.join(all_numbers)),
        'pairs': Counter([all_numbers[i:i+2] for i in range(len(all_numbers)-1)]),
        'last_trained': datetime.now().isoformat(),
        'total_draws': len(df),
        'last_date': df['date'].max().isoformat()
    }
    
    # Save
    with open('data/trained_model.pkl', 'wb') as f:
        pickle.dump(knowledge, f)
    
    print(f"✅ Trained on {len(all_numbers)} numbers from {len(df)} draws")
    print(f"📅 Last date: {knowledge['last_date']}")
    print(f"💾 Saved to data/trained_model.pkl")

if __name__ == '__main__':
    train_and_save()
