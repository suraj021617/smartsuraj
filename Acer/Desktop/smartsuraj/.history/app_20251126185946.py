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
    def find_all_4digit_patterns(grid): return []
    def compute_pattern_frequencies(draws): return []
    def compute_cell_heatmap(draws): return {}
    def predict_top_5(draws, mode="combined"): return {"combined": []}
    def generate_reverse_grid(number): return [[0]*4 for _ in range(4)]
    def generate_4x4_grid(number): return [[int(d) for d in number] for _ in range(4)]
    def learn_pattern_transitions(draws): return {}
    def predict_from_today_grid(number, transitions): return []

try:
    import config
    app = Flask(__name__)
    app.config.from_object(config)
except ImportError:
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(32).hex()

@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

_csv_cache = None
_csv_cache_time = None
_csv_lock = threading.Lock()

def load_csv_data():
    global _csv_cache, _csv_cache_time, _smart_model_cache, _ml_model_cache
    
    if '_smart_model_cache' not in globals():
        _smart_model_cache = {}
    if '_ml_model_cache' not in globals():
        _ml_model_cache = {}
    
    _csv_cache = None
    _csv_cache_time = None
    _smart_model_cache.clear()
    _ml_model_cache.clear()
    
    try:
        import warnings
        warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)
        
        csv_paths = ['4d_results_history.csv', 'utils/4d_results_history.csv']
        df = None
        
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=False, on_bad_lines='skip')
                if not df.empty:
                    logger.info(f"Loaded CSV from: {csv_path} ({len(df)} rows)")
                    break
        
        if df is None or df.empty:
            logger.error("No valid CSV file found")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"CSV loading error: {e}")
        return pd.DataFrame()

    if 'date' not in df.columns:
        logger.error("CSV missing 'date' column")
        return pd.DataFrame()
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date_parsed'], inplace=True)

    df['provider'] = df['provider'].fillna('').astype(str)
    df['provider'] = df['provider'].str.extract(r'images/([^./\"]+)', expand=False).fillna('unknown').str.strip().str.lower()
    
    import html
    df['prize_text'] = df['3rd'].fillna('').astype(str).apply(html.unescape)
    
    df['1st_real'] = df['prize_text'].str.extract(r'1st\\s+Prize\\s+(\\d{4})', flags=re.IGNORECASE)[0]
    df['2nd_real'] = df['prize_text'].str.extract(r'2nd\\s+Prize\\s+(\\d{4})', flags=re.IGNORECASE)[0]
    df['3rd_real'] = df['prize_text'].str.extract(r'3rd\\s+Prize\\s+(\\d{4})', flags=re.IGNORECASE)[0]
    
    df['1st_real'] = df['1st_real'].fillna(df['prize_text'])
    df['2nd_real'] = df['2nd_real'].fillna('')
    df['3rd_real'] = df['3rd_real'].fillna('')
    
    df['special'] = df['special'].fillna('')
    df['consolation'] = df['consolation'].fillna('')
    
    df = df.drop_duplicates(keep='first')
    df = df.sort_values('date_parsed', ascending=True).reset_index(drop=True)
    
    logger.info(f"Processed {len(df)} rows")
    return df

def advanced_predictor(df, provider=None, lookback=200):
    prize_cols = ["1st_real", "2nd_real", "3rd_real", "special", "consolation"]
    all_recent = []
    for col in prize_cols:
        if col not in df.columns: continue
        col_values = df[col].astype(str).dropna().tolist()
        for val in col_values:
            found = re.findall(r'\\d{4}', val)
            for f in found:
                if f.isdigit() and len(f) == 4:
                    all_recent.append(f)

    if not all_recent: return []
    recent_numbers = all_recent[-lookback:] if lookback else all_recent
    digit_counts = Counter("".join(recent_numbers))
    max_digit_freq = max(digit_counts.values()) if digit_counts else 1
    
    scored = []
    for num in set(recent_numbers[-100:]):
        if not (isinstance(num, str) and len(num) == 4 and num.isdigit()): continue
        score = sum((digit_counts.get(d, 0) / max_digit_freq) for d in num)
        scored.append((num, score, "frequency"))

    if not scored: return []
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:5]

def smart_auto_weight_predictor(df, provider=None, lookback=300):
    all_numbers = []
    for col in ["1st_real", "2nd_real", "3rd_real"]:
        if col in df.columns:
            all_numbers += [n for n in df[col].astype(str) if n.isdigit() and len(n) == 4]
    if not all_numbers:
        return []

    digit_counts = Counter("".join(all_numbers))
    scored = []
    for num in set(all_numbers[-150:]):
        hot_score = sum(digit_counts.get(d, 0) for d in num)
        scored.append((num, hot_score, "smart"))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:5]

def ml_predictor(df, lookback=500):
    numbers = []
    for col in ["1st_real", "2nd_real", "3rd_real"]:
        if col in df.columns:
            numbers += [n for n in df[col].astype(str) if n.isdigit() and len(n) == 4]
    if len(numbers) < 20:
        return []

    scored = []
    for num in set(numbers[-100:]):
        score = sum(int(d) for d in num) / 4
        scored.append((num, score, "ml"))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:5]

@app.route('/')
def index():
    df = load_csv_data()
    selected_date = request.args.get('selected_date')

    if not selected_date:
        try:
            latest_date = df['date_parsed'].max().date()
            filtered = df[df['date_parsed'].dt.date == latest_date]
            selected_date = latest_date
        except Exception:
            filtered = df.iloc[0:0]
            selected_date = ""
    else:
        try:
            date_obj = pd.to_datetime(selected_date).date()
            filtered = df[df['date_parsed'].dt.date == date_obj]
        except:
            filtered = df.iloc[0:0]

    cards = [row.to_dict() for _, row in filtered.iterrows()]
    return render_template('index.html', cards=cards, selected_date=selected_date)

@app.route('/data-import', methods=['GET', 'POST'])
def data_import():
    if request.method == 'GET':
        return render_template('data_import.html')
    
    try:
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No CSV uploaded'}), 400
        
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        df_new = pd.read_csv(file)
        required_cols = ['date', 'provider']
        missing = [col for col in required_cols if col not in df_new.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'}), 400
        
        csv_file = '4d_results_history.csv'
        if os.path.exists(csv_file):
            df_existing = pd.read_csv(csv_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['date', 'provider'], keep='last')
        else:
            df_combined = df_new
        
        df_combined.to_csv(csv_file, index=False)
        return jsonify({'success': True, 'message': f'Imported {len(df_new)} rows', 'total': len(df_combined)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/positional-ocr')
def positional_ocr():
    """Positional OCR analysis page"""
    df = load_csv_data()
    
    # Get request parameters
    selected_provider = request.args.get('provider', 'all')
    selected_date = request.args.get('date', '')
    auto_mode = request.args.get('auto', False)
    
    # Build provider options
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p and str(p).strip()])
    
    if df.empty:
        return render_template('positional_ocr.html', 
                             error="No data available", 
                             ocr_table=None,
                             predictions=[], 
                             provider_options=provider_options,
                             selected_provider=selected_provider,
                             selected_date=selected_date,
                             auto_mode=auto_mode,
                             next_day_check=None,
                             last_updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Filter data
    filtered_df = df.copy()
    if selected_provider != 'all':
        filtered_df = filtered_df[filtered_df['provider'] == selected_provider]
    
    if selected_date:
        try:
            date_filter = pd.to_datetime(selected_date).date()
            filtered_df = filtered_df[filtered_df['date_parsed'].dt.date == date_filter]
        except:
            pass
    
    # Get all numbers for analysis
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        if col in filtered_df.columns:
            all_numbers.extend([n for n in filtered_df[col].astype(str) if n.isdigit() and len(n) == 4])
    
    # Build OCR table (digit frequency by position)
    ocr_table = {}
    for digit in range(10):
        ocr_table[digit] = [0, 0, 0, 0]  # Initialize for 4 positions
    
    for num in all_numbers:
        for pos in range(4):
            if pos < len(num):
                digit = int(num[pos])
                ocr_table[digit][pos] += 1
    
    # Generate predictions based on most frequent digits per position
    predictions = []
    if all_numbers:
        # Get most frequent digit for each position
        position_winners = []
        for pos in range(4):
            pos_counts = {}
            for num in all_numbers:
                if pos < len(num):
                    digit = num[pos]
                    pos_counts[digit] = pos_counts.get(digit, 0) + 1
            
            if pos_counts:
                most_frequent = max(pos_counts.items(), key=lambda x: x[1])
                position_winners.append(most_frequent)
            else:
                position_winners.append(('0', 0))
        
        # Create predictions
        for i in range(5):
            predicted_number = ''
            total_confidence = 0
            position_analysis = {}
            
            for pos in range(4):
                # Get different combinations for variety
                pos_counts = {}
                for num in all_numbers:
                    if pos < len(num):
                        digit = num[pos]
                        pos_counts[digit] = pos_counts.get(digit, 0) + 1
                
                if pos_counts:
                    sorted_digits = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
                    if i < len(sorted_digits):
                        digit, freq = sorted_digits[i]
                    else:
                        digit, freq = sorted_digits[0]  # Fallback to most frequent
                    
                    predicted_number += digit
                    confidence = (freq / len(all_numbers)) * 100
                    total_confidence += confidence
                    
                    position_analysis[f'pos{pos+1}'] = {
                        'digit': digit,
                        'frequency': freq
                    }
                else:
                    predicted_number += str(i % 10)
            
            avg_confidence = total_confidence / 4 if total_confidence > 0 else 10
            
            strategy = 'Positional Analysis'
            if auto_mode:
                strategy = 'AI-Learned Pattern'
            
            predictions.append({
                'number': predicted_number,
                'confidence': round(avg_confidence, 1),
                'strategy': strategy,
                'reason': f'Most frequent digits per position',
                'position_analysis': position_analysis
            })
    
    # Remove duplicates
    seen = set()
    unique_predictions = []
    for pred in predictions:
        if pred['number'] not in seen:
            seen.add(pred['number'])
            unique_predictions.append(pred)
    
    return render_template('positional_ocr.html',
                         ocr_table=ocr_table,
                         predictions=unique_predictions[:5],
                         provider_options=provider_options,
                         selected_provider=selected_provider,
                         selected_date=selected_date,
                         auto_mode=auto_mode,
                         next_day_check=None,
                         total_numbers=len(all_numbers),
                         last_updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                         error=None)

# Add missing routes that might be referenced
@app.route('/positional-tracker')
def positional_tracker():
    """Positional tracking page"""
    return render_template('positional_tracker.html')

@app.route('/ocr-learning')
def ocr_learning():
    """OCR learning page"""
    return render_template('ocr_learning.html')

@app.route('/auto-ocr-dashboard')
def auto_ocr_dashboard():
    """Auto OCR dashboard"""
    return render_template('auto_ocr_dashboard.html')

@app.route('/scraper-control')
def scraper_control():
    """Scraper control page"""
    return render_template('scraper_control.html')

@app.route('/sequence-learner')
def sequence_learner():
    return render_template('sequence_learner.html')

@app.route('/structural-clusters')
def structural_clusters():
    return render_template('structural_clusters.html')

@app.route('/temporal-recurrence')
def temporal_recurrence():
    return render_template('temporal_recurrence.html')

@app.route('/arithmetic-gap-analyzer')
def arithmetic_gap_analyzer():
    return render_template('arithmetic_gap_analyzer.html')

@app.route('/transition-predictions')
def transition_predictions():
    return render_template('transition_predictions.html')

@app.route('/cross-draw-linker')
def cross_draw_linker():
    return render_template('cross_draw_linker.html')

@app.route('/pattern-weighting')
def pattern_weighting():
    return render_template('pattern_weighting.html')

@app.route('/method-performance')
def method_performance():
    return render_template('method_performance.html')

@app.route('/streak-tracker')
def streak_tracker():
    return render_template('streak_tracker.html')

@app.route('/my-tracker')
def my_tracker():
    return render_template('my_tracker.html')

@app.route('/match-checker')
def match_checker():
    return render_template('match_checker.html')

@app.route('/prediction-tracker')
def prediction_tracker():
    return render_template('prediction_tracker.html')

@app.route('/leaderboard')
def leaderboard():
    return render_template('leaderboard.html')

@app.route('/community')
def community():
    return render_template('community.html')

@app.route('/notifications')
def notifications():
    return render_template('notifications.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/mobile-guide')
def mobile_guide():
    return render_template('mobile_guide.html')

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)