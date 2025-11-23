"""
Auto-validation system that checks predictions against actual results
"""
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

def check_match_type(predicted, actual):
    """Check what type of match occurred"""
    pred_str = str(predicted).zfill(4)
    actual_str = str(actual).zfill(4)
    
    matches = {
        'exact': pred_str == actual_str,
        'front': pred_str[:2] == actual_str[:2],
        'back': pred_str[2:] == actual_str[2:],
        'ibox': sorted(pred_str) == sorted(actual_str),
        'digits_matched': sum(1 for p, a in zip(pred_str, actual_str) if p == a)
    }
    return matches

def extract_all_winning_numbers(row):
    """Extract all winning numbers from a draw result"""
    import re
    winners = []
    
    # 1st, 2nd, 3rd prizes - check both column name formats
    for col in ['1st_real', '2nd_real', '3rd_real', '1st', '2nd', '3rd']:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val.isdigit() and len(val) == 4:
                winners.append(val)
    
    # Special prizes - extract all 4-digit numbers
    if 'special' in row and pd.notna(row['special']):
        special = str(row['special']).strip()
        special_nums = re.findall(r'\b\d{4}\b', special)
        winners.extend(special_nums)
    
    # Consolation prizes - extract all 4-digit numbers
    if 'consolation' in row and pd.notna(row['consolation']):
        consolation = str(row['consolation']).strip()
        consolation_nums = re.findall(r'\b\d{4}\b', consolation)
        winners.extend(consolation_nums)
    
    # Remove duplicates and return
    return list(set([w for w in winners if w and w.isdigit() and len(w) == 4]))

def validate_predictions(csv_path, predictions_dict, provider=None):
    """
    Validate predictions against actual results
    predictions_dict format: {
        'date': 'YYYY-MM-DD',
        'provider': 'sabah88',
        'methods': {
            'quick_pick': [1234, 5678, ...],
            'hot_numbers': [2345, 6789, ...],
            'advanced': [3456, 7890, ...],
            ...
        }
    }
    """
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    # Use date_parsed if available, otherwise try to parse date column
    if 'date_parsed' in df.columns:
        df['date'] = pd.to_datetime(df['date_parsed'], errors='coerce')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # Remove rows with invalid dates
    
    target_date = pd.to_datetime(predictions_dict['date'])
    target_provider = predictions_dict.get('provider', provider)
    
    # Find matching result
    if target_provider:
        result = df[(df['date'] == target_date) & (df['provider'].str.lower() == target_provider.lower())]
    else:
        result = df[df['date'] == target_date]
    
    if result.empty:
        return {'status': 'no_result', 'message': 'Draw result not available yet'}
    
    validation_results = {
        'date': predictions_dict['date'],
        'provider': target_provider,
        'actual_winners': [],
        'method_performance': {}
    }
    
    # Extract actual winning numbers
    for _, row in result.iterrows():
        validation_results['actual_winners'].extend(extract_all_winning_numbers(row))
    
    # Check each prediction method
    for method_name, predicted_numbers in predictions_dict['methods'].items():
        method_results = {
            'predicted': predicted_numbers,
            'matches': [],
            'match_summary': {
                'exact': 0,
                'front': 0,
                'back': 0,
                'ibox': 0,
                'partial': 0,
                'no_match': 0
            }
        }
        
        for pred in predicted_numbers:
            best_match = None
            for actual in validation_results['actual_winners']:
                match_info = check_match_type(pred, actual)
                if match_info['exact']:
                    best_match = {'predicted': pred, 'actual': actual, 'type': 'exact', 'digits': 4}
                    break
                elif match_info['ibox'] and not best_match:
                    best_match = {'predicted': pred, 'actual': actual, 'type': 'ibox', 'digits': 4}
                elif match_info['front'] and (not best_match or best_match['type'] not in ['exact', 'ibox']):
                    best_match = {'predicted': pred, 'actual': actual, 'type': 'front', 'digits': 2}
                elif match_info['back'] and (not best_match or best_match['type'] not in ['exact', 'ibox', 'front']):
                    best_match = {'predicted': pred, 'actual': actual, 'type': 'back', 'digits': 2}
                elif match_info['digits_matched'] > 0 and not best_match:
                    best_match = {'predicted': pred, 'actual': actual, 'type': 'partial', 'digits': match_info['digits_matched']}
            
            if best_match:
                method_results['matches'].append(best_match)
                method_results['match_summary'][best_match['type']] += 1
            else:
                method_results['match_summary']['no_match'] += 1
        
        validation_results['method_performance'][method_name] = method_results
    
    return validation_results

def analyze_prediction_errors(validation_results):
    """AI/ML analysis of why predictions failed"""
    analysis = {
        'overall_accuracy': 0,
        'insights': [],
        'recommendations': []
    }
    
    total_predictions = 0
    total_hits = 0
    
    for method, results in validation_results['method_performance'].items():
        predicted_count = len(results['predicted'])
        hit_count = len(results['matches'])
        total_predictions += predicted_count
        total_hits += hit_count
        
        if predicted_count > 0:
            accuracy = (hit_count / predicted_count) * 100
            
            # Analyze patterns
            if accuracy == 0:
                analysis['insights'].append(f"❌ {method}: 0% accuracy - Complete miss. Logic needs review.")
                analysis['recommendations'].append(f"Review {method} algorithm - may need different data patterns")
            elif accuracy < 20:
                analysis['insights'].append(f"⚠️ {method}: {accuracy:.1f}% accuracy - Very low hit rate")
                analysis['recommendations'].append(f"Adjust {method} weight factors or data window")
            elif accuracy < 50:
                analysis['insights'].append(f"📊 {method}: {accuracy:.1f}% accuracy - Moderate performance")
            else:
                analysis['insights'].append(f"✅ {method}: {accuracy:.1f}% accuracy - Good performance!")
            
            # Check match types
            if results['match_summary']['ibox'] > results['match_summary']['exact']:
                analysis['insights'].append(f"🔄 {method}: More ibox than exact - digits correct but wrong order")
                analysis['recommendations'].append(f"Consider digit frequency over position for {method}")
            
            if results['match_summary']['partial'] > 0:
                analysis['insights'].append(f"🎯 {method}: {results['match_summary']['partial']} partial matches - close but not exact")
    
    if total_predictions > 0:
        analysis['overall_accuracy'] = (total_hits / total_predictions) * 100
    
    return analysis

def save_validation_history(validation_results, analysis):
    """Save validation results for tracking"""
    history_file = 'data/validation_history.json'
    os.makedirs('data', exist_ok=True)
    
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    
    history.append({
        'timestamp': datetime.now().isoformat(),
        'validation': validation_results,
        'analysis': analysis
    })
    
    # Keep last 100 validations
    history = history[-100:]
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

def auto_validate_latest(csv_path):
    """Auto-validate when CSV is updated"""
    # This would be called when CSV updates
    # For now, returns structure for manual validation
    return {
        'status': 'ready',
        'message': 'Call validate_predictions() with your prediction data'
    }
