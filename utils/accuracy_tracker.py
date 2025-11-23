"""
Accuracy Tracker - Learn which prediction method works best
Tracks: 4-digit exact, 3-digit, 2-digit, 1-digit matches
"""
import json
import os
from datetime import datetime

TRACKER_FILE = "method_accuracy.json"

def count_digit_matches(predicted, actual):
    """Count how many digits match"""
    if len(predicted) != 4 or len(actual) != 4:
        return 0
    
    # Exact match
    if predicted == actual:
        return 4
    
    # Count matching digits (any position)
    pred_digits = set(predicted)
    actual_digits = set(actual)
    matches = len(pred_digits & actual_digits)
    
    return matches

def load_accuracy_data():
    """Load accuracy tracking data"""
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_accuracy_data(data):
    """Save accuracy tracking data"""
    with open(TRACKER_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def track_prediction(method_name, predicted_numbers, actual_winners):
    """
    Track prediction accuracy
    method_name: 'quick_pick', 'ultimate_ai', 'best_predictions', etc.
    predicted_numbers: ['1234', '5678', ...]
    actual_winners: ['1234', '5678', '9012']
    """
    data = load_accuracy_data()
    
    if method_name not in data:
        data[method_name] = {
            'total_predictions': 0,
            'exact_matches': 0,
            '3_digit_matches': 0,
            '2_digit_matches': 0,
            '1_digit_matches': 0,
            'history': []
        }
    
    method_data = data[method_name]
    method_data['total_predictions'] += 1
    
    # Check each prediction
    best_match = 0
    matched_number = None
    
    for pred in predicted_numbers:
        for actual in actual_winners:
            matches = count_digit_matches(pred, actual)
            if matches > best_match:
                best_match = matches
                matched_number = f"{pred} → {actual}"
    
    # Update counters
    if best_match == 4:
        method_data['exact_matches'] += 1
    elif best_match == 3:
        method_data['3_digit_matches'] += 1
    elif best_match == 2:
        method_data['2_digit_matches'] += 1
    elif best_match == 1:
        method_data['1_digit_matches'] += 1
    
    # Add to history
    method_data['history'].append({
        'date': datetime.now().strftime('%Y-%m-%d'),
        'predicted': predicted_numbers[:5],
        'actual': actual_winners,
        'best_match': best_match,
        'matched': matched_number
    })
    
    # Keep only last 50 records
    if len(method_data['history']) > 50:
        method_data['history'] = method_data['history'][-50:]
    
    save_accuracy_data(data)
    return best_match

def get_method_rankings():
    """Get methods ranked by accuracy"""
    data = load_accuracy_data()
    
    rankings = []
    for method, stats in data.items():
        total = stats['total_predictions']
        if total == 0:
            continue
        
        # Calculate score (weighted)
        score = (
            stats['exact_matches'] * 100 +
            stats['3_digit_matches'] * 50 +
            stats['2_digit_matches'] * 20 +
            stats['1_digit_matches'] * 5
        ) / total
        
        accuracy_rate = (
            (stats['exact_matches'] + stats['3_digit_matches']) / total * 100
        ) if total > 0 else 0
        
        rankings.append({
            'method': method,
            'score': round(score, 2),
            'accuracy_rate': round(accuracy_rate, 1),
            'exact': stats['exact_matches'],
            '3_digit': stats['3_digit_matches'],
            '2_digit': stats['2_digit_matches'],
            '1_digit': stats['1_digit_matches'],
            'total': total
        })
    
    rankings.sort(key=lambda x: x['score'], reverse=True)
    return rankings

def get_best_method():
    """Get the best performing method"""
    rankings = get_method_rankings()
    return rankings[0] if rankings else None
