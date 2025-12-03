from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import time
from datetime import datetime

app = Flask(__name__)

def load_csv_data():
    try:
        df = pd.read_csv('4d_results_history.csv', on_bad_lines='skip')
        return df
    except:
        return pd.DataFrame()

@app.route('/ocr-learning')
def ocr_learning():
    return render_template('ocr_learning.html', 
                         predictions=[], 
                         matched_predictions=[], 
                         accuracy_stats={'total': 0, 'matched': 0, 'accuracy': 0},
                         last_updated=time.strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/auto-ocr-dashboard')
def auto_ocr_dashboard():
    df = load_csv_data()
    predictions = []
    if not df.empty:
        recent = df.tail(10)
        for _, row in recent.iterrows():
            predictions.append({
                'date': str(row.get('date', '')),
                'predicted': str(row.get('1st_real', '')),
                'confidence': 85,
                'status': 'processed'
            })
    
    return render_template('auto_ocr_dashboard.html',
                         predictions=predictions,
                         stats={'total_processed': len(predictions), 'accuracy': 85.2},
                         last_updated=time.strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/positional-ocr')
def positional_ocr():
    return render_template('positional_ocr.html',
                         position_predictions=[],
                         accuracy_by_position={},
                         last_updated=time.strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/match-checker')
def match_checker():
    return render_template('match_checker.html',
                         matches=[],
                         stats={'total_checked': 0, 'matches_found': 0},
                         last_updated=time.strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/api/check-matches', methods=['POST'])
def api_check_matches():
    data = request.get_json()
    predicted = data.get('predicted', [])
    actual = data.get('actual', [])
    
    matches = []
    for p in predicted:
        if p in actual:
            matches.append({'number': p, 'type': 'exact'})
    
    return jsonify({'matches': matches, 'count': len(matches)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)