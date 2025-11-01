import json
import os
from datetime import datetime, timedelta
from collections import Counter

PREDICTIONS_FILE = 'predictions_history.json'

def save_predictions(date, provider, predictions, method='combined'):
    """Save today's predictions for tomorrow's draw"""
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = []
    
    data.append({
        'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'target_date': date,
        'provider': provider,
        'predictions': predictions[:10],  # Top 10
        'method': method,
        'checked': False
    })
    
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def check_predictions(df):
    """Check all unchecked predictions against actual results"""
    if not os.path.exists(PREDICTIONS_FILE):
        return []
    
    with open(PREDICTIONS_FILE, 'r') as f:
        data = json.load(f)
    
    results = []
    updated = False
    
    for pred in data:
        if pred['checked']:
            continue
        
        target_date = pred['target_date']
        provider = pred['provider']
        
        # Find actual results
        actual_data = df[df['date_parsed'].dt.strftime('%Y-%m-%d') == target_date]
        if provider != 'all':
            actual_data = actual_data[actual_data['provider'] == provider]
        
        if actual_data.empty:
            continue  # Results not added yet
        
        # Get actual numbers
        actuals = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            nums = actual_data[col].tolist()
            actuals.extend([str(n) for n in nums if len(str(n)) == 4])
        
        # Check matches
        predictions = pred['predictions']
        exact_matches = []
        partial_matches = []
        
        for p in predictions:
            if isinstance(p, dict):
                pred_num = p.get('number', p.get('num', ''))
            else:
                pred_num = p
            
            if pred_num in actuals:
                exact_matches.append(pred_num)
            else:
                # Check 3-digit match
                for actual in actuals:
                    if len(set(pred_num) & set(actual)) >= 3:
                        partial_matches.append({'predicted': pred_num, 'actual': actual})
                        break
        
        results.append({
            'prediction_date': pred['prediction_date'],
            'target_date': target_date,
            'provider': provider,
            'predictions': predictions,
            'actuals': actuals,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'accuracy': len(exact_matches) / len(predictions) * 100 if predictions else 0,
            'method': pred['method']
        })
        
        pred['checked'] = True
        updated = True
    
    if updated:
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    return results

def get_accuracy_stats():
    """Get overall accuracy statistics"""
    if not os.path.exists(PREDICTIONS_FILE):
        return None
    
    with open(PREDICTIONS_FILE, 'r') as f:
        data = json.load(f)
    
    checked = [p for p in data if p.get('checked', False)]
    if not checked:
        return None
    
    total_predictions = sum(len(p['predictions']) for p in checked)
    # Count exact matches from results
    total_exact = 0
    
    return {
        'total_predictions': len(checked),
        'total_numbers': total_predictions,
        'checked': len(checked),
        'unchecked': len([p for p in data if not p.get('checked', False)])
    }

def generate_smart_predictions(df, provider='all', top_n=10):
    """Generate predictions using multiple methods"""
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    # Get recent numbers
    recent = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums = df[col].tail(50).tolist()
        recent.extend([str(n) for n in nums if len(str(n)) == 4])
    
    freq = Counter(recent)
    predictions = [{'number': num, 'confidence': count, 'reason': f'Appeared {count}x'} 
                   for num, count in freq.most_common(top_n)]
    
    return predictions
