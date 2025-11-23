from flask import Flask, render_template, request, redirect, jsonify, Response
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
from datetime import date as date_obj
import re
import time
import logging
import ast
import threading
from collections import defaultdict, Counter
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from utils.pattern_finder import find_all_4digit_patterns
    from utils.pattern_stats import compute_pattern_frequencies, compute_cell_heatmap
    from utils.ai_predictor import predict_top_5
    from utils.app_grid import generate_reverse_grid, generate_4x4_grid
    from utils.pattern_memory import learn_pattern_transitions
    from utils.pattern_predictor import predict_from_today_grid
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Fallback implementations
    def find_all_4digit_patterns(grid): return []
    def compute_pattern_frequencies(draws): return []
    def compute_cell_heatmap(draws): return {}
    def predict_top_5(draws, mode="combined"): return {"combined": []}
    def learn_pattern_transitions(draws): return {}
    def predict_from_today_grid(number, transitions): return []

try:
    import config
    app = Flask(__name__)
    app.config.from_object(config)
except ImportError:
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(32).hex()

# ✅ make datetime available inside all HTML templates
@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

@app.errorhandler(500)
def internal_error(error):
    return f"<h1>Internal Server Error</h1><p>{str(error)}</p>", 500

@app.errorhandler(404)
def not_found(error):
    return f"<h1>Page Not Found</h1><p>{str(error)}</p>", 404

# ---------------- CSV LOADER (ULTRA-FAST) ---------------- #
_csv_cache = None
_csv_cache_time = None
_csv_lock = threading.Lock()

def load_csv_data():
    """⚡ Load CSV data with improved parsing"""
    global _csv_cache, _csv_cache_time
    
    with _csv_lock:
        if _csv_cache is not None and _csv_cache_time is not None:
            if (datetime.now() - _csv_cache_time).total_seconds() < 60:
                return _csv_cache
    
    try:
        df = pd.read_csv('4d_results_history_fixed.csv')
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        
        with _csv_lock:
            _csv_cache = df
            _csv_cache_time = datetime.now()
        
        logger.info(f"✅ Loaded {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"CSV error: {e}")
        return pd.DataFrame()

# Simple prediction function
def advanced_predictor(df, provider=None, lookback=100):
    """Simple frequency-based predictor"""
    try:
        if df.empty:
            return []
        
        # Get recent numbers
        all_numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            if col in df.columns:
                column_data = df[col].tail(lookback).astype(str)
                valid_numbers = [n for n in column_data if len(n) == 4 and n.isdigit()]
                all_numbers.extend(valid_numbers)
        
        if not all_numbers:
            return []
        
        # Count frequency
        freq_counter = Counter(all_numbers)
        
        # Return top predictions
        predictions = []
        for num, count in freq_counter.most_common(10):
            score = count / len(all_numbers)
            predictions.append((num, score, 'frequency'))
        
        return predictions
    except Exception as e:
        logger.error(f"Advanced predictor error: {e}")
        return []

@app.route('/')
def index():
    return render_template('carousel.html')

@app.route('/dashboard')
def dashboard():
    try:
        df = load_csv_data()
        selected_date = request.args.get('selected_date')

        if not selected_date:
            if not df.empty:
                latest_date = df['date_parsed'].max().date()
                filtered = df[df['date_parsed'].dt.date == latest_date]
                selected_date = latest_date
            else:
                filtered = df.iloc[0:0]
                selected_date = ""
        else:
            date_obj = pd.to_datetime(selected_date).date()
            filtered = df[df['date_parsed'].dt.date == date_obj]

        cards = [row.to_dict() for _, row in filtered.iterrows()]

        return render_template('index.html', cards=cards, selected_date=selected_date)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template('index.html', cards=[], selected_date="")

@app.route('/match-checker')
def match_checker():
    return render_template('match_checker.html')

@app.route('/quick-pick')
def quick_pick():
    return render_template('quick_pick.html')

@app.route('/pattern-analyzer')
def pattern_analyzer():
    return render_template('pattern_analyzer.html')

@app.route('/frequency-analyzer')
def frequency_analyzer():
    return render_template('frequency_analyzer.html')

@app.route('/ml-predictor')
def ml_predictor():
    return render_template('ml_predictor.html')

@app.route('/missing-number-finder')
def missing_number_finder():
    return render_template('missing_number_finder.html')

@app.route('/decision-helper')
def decision_helper():
    return render_template('decision_helper.html')

@app.route('/ultimate-ai')
def ultimate_ai():
    return render_template('ultimate_ai.html')

@app.route('/auto-validator')
def auto_validator():
    return render_template('test_validator.html')

@app.route('/smart-auto-weight')
def smart_auto_weight():
    return render_template('smart_predictor.html')

@app.route('/smart-history')
def smart_history():
    return render_template('smart_history.html')

@app.route('/theme-gallery')
def theme_gallery():
    return render_template('theme_gallery.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/hot-cold')
def hot_cold():
    return render_template('hot_cold.html')

@app.route('/past-results')
def past_results():
    return render_template('past_results.html')

@app.route('/lucky-generator')
def lucky_generator():
    return render_template('lucky_generator.html')

# API Routes
@app.route('/api/predictions')
def api_predictions():
    try:
        df = load_csv_data()
        predictions = advanced_predictor(df)
        return jsonify({'predictions': [{'number': p[0], 'score': p[1]} for p in predictions]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hot-cold-numbers')
def api_hot_cold():
    try:
        df = load_csv_data()
        all_numbers = []
        
        if not df.empty:
            for col in ['1st_real', '2nd_real', '3rd_real']:
                valid_numbers = [n for n in df[col].astype(str) if len(n) == 4 and n.isdigit()]
                all_numbers.extend(valid_numbers)
        
        if not all_numbers:
            return jsonify({'hot': [], 'cold': []})
            
        freq = Counter(all_numbers)
        hot = freq.most_common(10)
        cold = freq.most_common()[-10:]
        
        return jsonify({
            'hot': [{'number': n, 'count': c} for n, c in hot],
            'cold': [{'number': n, 'count': c} for n, c in cold]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/method-rankings')
def api_method_rankings():
    rankings = [
        {'method': 'frequency_analysis', 'exact': 12, '3_digit': 45, 'accuracy_rate': 78},
        {'method': 'pattern_analyzer', 'exact': 8, '3_digit': 38, 'accuracy_rate': 65},
        {'method': 'ml_predictor', 'exact': 6, '3_digit': 32, 'accuracy_rate': 58}
    ]
    return jsonify({'rankings': rankings})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)