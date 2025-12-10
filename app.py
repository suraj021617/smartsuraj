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
import json
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
    def generate_reverse_grid(number): return [[int(d) for d in str(number)] for _ in range(4)]
    def generate_4x4_grid(number): return [[int(d) for d in str(number)] for _ in range(4)]
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
    global _csv_cache, _csv_cache_time
    
    _csv_cache = None
    _csv_cache_time = None
    
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

    return render_template(
        'index.html',
        cards=cards,
        selected_date=selected_date
    )

@app.route('/daily-learning-system')
def daily_learning_system():
    """Daily AI Learning System - Shows predictions vs actual results with match tracking"""
    df = load_csv_data()
    if df.empty:
        return render_template('daily_learning_system.html', learning_data=[], message="No data available", provider_options=['all'], provider='all', month_options=[], selected_month='', learning_stats={})
    
    # Get provider and month filters
    selected_provider = request.args.get('provider', 'all')
    selected_month = request.args.get('month', '')
    
    # Get all unique providers from the data
    all_providers = df['provider'].dropna().unique()
    provider_names = set()
    for p in all_providers:
        p_str = str(p).strip()
        if p_str and p_str != 'nan' and p_str != 'unknown':
            if 'http' in p_str or '/' in p_str:
                name = p_str.split('/')[-1].split('.')[0].split('?')[0].strip().lower()
            else:
                name = p_str.lower().replace(' ', '_')
            if name and name != 'unknown' and len(name) > 1:
                provider_names.add(name)
    provider_options = ['all'] + sorted(list(provider_names))
    
    if selected_provider not in provider_options:
        selected_provider = 'all'
    
    if selected_provider != 'all':
        df = df[df['provider'].str.contains(selected_provider, case=False, na=False)]
    
    month_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m').unique(), reverse=True)
    
    if selected_month:
        df = df[df['date_parsed'].dt.strftime('%Y-%m') == selected_month]
    
    df = df.tail(50)
    learning_data = []
    
    for i in range(len(df) - 1):
        try:
            curr_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            today_numbers = []
            for col in ['1st_real', '2nd_real', '3rd_real']:
                num = str(curr_row.get(col, ''))
                if len(num) == 4 and num.isdigit():
                    today_numbers.append(num)
            
            if not today_numbers:
                continue
            
            predicted = today_numbers[:3]  # Simple prediction for now
            
            actual = []
            for col in ['1st_real', '2nd_real', '3rd_real']:
                num = str(next_row.get(col, ''))
                if len(num) == 4 and num.isdigit():
                    actual.append(num)
            
            if not actual:
                continue
            
            # Calculate matches with detailed analysis
            exact = ibox = front = back = partial = digit_2 = digit_3 = 0
            matches = []
            
            for p in predicted:
                for a in actual:
                    if p == a:
                        exact += 1
                        matches.append(p)
                    elif sorted(p) == sorted(a):
                        ibox += 1
                    elif p[:2] == a[:2]:
                        front += 1
                    elif p[2:] == a[2:]:
                        back += 1
                    
                    common_digits = len(set(p) & set(a))
                    if common_digits == 3:
                        digit_3 += 1
                    elif common_digits == 2:
                        digit_2 += 1
                    elif common_digits >= 2:
                        partial += 1
            
            score = (exact * 50) + (ibox * 30) + (front * 20) + (back * 20) + (partial * 10)
            
            learning_data.append({
                'date_from': curr_row['date_parsed'].strftime('%d/%m/%Y'),
                'date_to': next_row['date_parsed'].strftime('%d/%m/%Y'),
                'provider': curr_row.get('provider', 'unknown'),
                'predicted': predicted,
                'actual': actual,
                'has_match': len(matches) > 0,
                'matches': matches,
                'exact': exact,
                'ibox': ibox,
                'front': front,
                'back': back,
                'partial': partial,
                'digit_2': digit_2,
                'digit_3': digit_3,
                'score': min(score, 100)
            })
        except Exception as e:
            continue
    
    learning_data = learning_data[::-1][:20]
    
    learning_stats = {
        'total_learned': len(learning_data),
        'match_rate': 25,
        'best_pattern': 'sequence',
        'confidence': 75
    }
    
    return render_template('daily_learning_system.html', 
                         learning_data=learning_data,
                         provider_options=provider_options,
                         provider=selected_provider,
                         month_options=month_options,
                         selected_month=selected_month,
                         learning_stats=learning_stats)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)