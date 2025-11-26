# -*- coding: utf-8 -*-
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
import warnings
import sys

# Initialize logger first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config with proper error handling
try:
    import config
except ImportError as import_error:
    logger.warning(f"Config import failed: {import_error}")
    # Create a minimal config object if file doesn't exist
    class Config:
        SECRET_KEY = os.urandom(32).hex()
        DEBUG = False
    config = Config()

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
    def generate_reverse_grid(number): return [[0]*4 for _ in range(4)]
    def generate_4x4_grid(number): return [[int(d) for d in number] for _ in range(4)]
    def learn_pattern_transitions(draws): return {}
    def predict_from_today_grid(number, transitions): return []

app = Flask(__name__)
try:
    app.config.from_object(config)
except Exception as e:
    logger.warning(f"Unable to load config object: {e}")
    app.config['SECRET_KEY'] = os.urandom(32).hex()

# Make datetime available inside all HTML templates
@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

# CSV LOADER with caching
_csv_cache = None
_csv_cache_time = None
_csv_lock = threading.Lock()

def load_csv_data():
    """Load CSV with caching for better performance."""
    global _csv_cache, _csv_cache_time
    
    with _csv_lock:
        # Check if cache is valid (less than 5 minutes old)
        if _csv_cache is not None and _csv_cache_time is not None:
            if (datetime.now() - _csv_cache_time).total_seconds() < 300:
                return _csv_cache.copy()
    
    try:
        warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)
        
        if not os.path.exists('4d_results_history.csv'):
            logger.error("CSV file not found: 4d_results_history.csv")
            return pd.DataFrame()
            
        df = pd.read_csv('4d_results_history.csv', index_col=False, on_bad_lines='skip')
        if df.empty:
            logger.warning("CSV file is empty")
            return df
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error(f"CSV loading error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error loading CSV: {e}")
        return pd.DataFrame()

    # Parse date: try common column names and fallback to first column
    for col in ['date', 'Date', 'draw_date']:
        if col in df.columns:
            df['date_parsed'] = pd.to_datetime(df[col], errors='coerce')
            break
    else:
        # fallback: try to parse first column as date
        df['date_parsed'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

    # Remove rows where the date could not be read
    df.dropna(subset=['date_parsed'], inplace=True)

    # Ensure expected columns exist and are strings
    for col in ['1st', '2nd', '3rd', 'special', 'consolation', 'provider']:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].fillna('')

    # Normalize provider
    prov_series = df['provider'].astype(str)
    extracted = prov_series.str.extract(r'images/([^,/\\s]+)', expand=False)
    df['provider'] = extracted.fillna(prov_series).str.strip().str.lower()
    
    # Remove duplicates
    df.drop_duplicates(subset=['date_parsed', 'provider'], keep='first', inplace=True)
    df.drop_duplicates(subset=['date_parsed', '1st', '2nd', '3rd'], keep='first', inplace=True)

    # Extract real prize numbers
    def extract_number(text):
        text = str(text).strip()
        if len(text) == 4 and text.isdigit():
            return text
        match = re.search(r'(\\d{4})', text)
        return match.group(1) if match else ''
    
    df['1st_real'] = df['1st'].apply(extract_number)
    df['2nd_real'] = df['2nd'].apply(extract_number)
    df['3rd_real'] = df['3rd'].apply(extract_number)

    # Sort data for consistent predictions
    df = df.sort_values(['date_parsed', 'provider'], ascending=[True, True]).reset_index(drop=True)

    # Update cache
    _csv_cache = df.copy()
    _csv_cache_time = datetime.now()

    return df

# HELPER FUNCTIONS
def find_missing_digits(grid):
    all_digits = set(map(str, range(10)))
    used_digits = set(str(cell) for row in grid for cell in row)
    return sorted(all_digits - used_digits)

def advanced_predictor(df, provider=None, lookback=200):
    """Advanced prediction with clear logic explanations"""
    prize_cols = ["1st_real", "2nd_real", "3rd_real", "special", "consolation"]
    all_recent = []
    for col in prize_cols:
        if col not in df.columns: 
            continue
        col_values = df[col].astype(str).dropna().tolist()
        for val in col_values:
            found = re.findall(r'\\d{4}', val)
            for f in found:
                if f.isdigit() and len(f) == 4:
                    all_recent.append(f)

    if not all_recent: 
        return []
    
    recent_numbers = all_recent[-lookback:] if lookback else all_recent
    digit_counts = Counter("".join(recent_numbers))
    max_digit_freq = max(digit_counts.values()) if digit_counts else 1
    
    pair_counts = Counter()
    for num in recent_numbers:
        for i in range(3):
            pair = num[i:i+2]
            pair_counts[pair] += 1
    
    past_draws_for_memory = [{"number": n} for n in recent_numbers]
    transitions = learn_pattern_transitions(past_draws_for_memory) or {}
    
    candidate_pool = set(recent_numbers)
    candidate_pool.update(transitions.keys())
    
    top_pairs = [p for p, c in pair_counts.most_common(20)]
    if top_pairs:
        for p1 in top_pairs[:10]:
            for p2 in top_pairs[:10]:
                candidate_pool.add(p1 + p2)
    
    if len(candidate_pool) < 200:
        for num in recent_numbers[-500:]:
            candidate_pool.add(num)
        if len(candidate_pool) < 200:
            prefixes = {n[:2] for n in recent_numbers[-50:]}
            for p in prefixes:
                for d in range(100):
                    candidate_pool.add(f"{p}{d:02d}")

    if not candidate_pool:
        candidate_pool = set(recent_numbers[-100:]) if recent_numbers else set()
    
    scored = []
    for num in candidate_pool:
        if not (isinstance(num, str) and len(num) == 4 and num.isdigit()): 
            continue
        score = 0.0
        logic_parts = []
        
        # HOT DIGITS LOGIC: Numbers with frequently appearing digits score higher
        hot_score = sum((digit_counts.get(d, 0) / max_digit_freq) for d in num)
        score += hot_score
        hot_digits = [d for d in num if digit_counts.get(d, 0) > max_digit_freq * 0.5]
        if hot_digits:
            logic_parts.append(f"Contains hot digits {','.join(hot_digits)}")
        
        # PAIR FREQUENCY LOGIC: Numbers with common digit pairs score higher
        pair_bonus = 0.0
        strong_pairs = []
        for i in range(3):
            pair = num[i : i + 2]
            cnt = pair_counts.get(pair, 0)
            if cnt >= 3:  # Only count significant pairs
                pair_bonus += cnt * 0.06
                strong_pairs.append(f"{pair}({cnt}x)")
        if pair_bonus:
            score += pair_bonus
            logic_parts.append(f"Strong pairs: {','.join(strong_pairs)}")
        
        # PATTERN TRANSITION LOGIC: Numbers that follow historical patterns
        trans_val = 0.0
        try:
            tval = transitions.get(num, 0)
            if isinstance(tval, (int, float)) and tval > 0:
                trans_val = float(tval)
                score += trans_val * 0.5
                logic_parts.append(f"Follows {int(tval)} historical patterns")
        except Exception:
            pass
        
        # RECENT APPEARANCE LOGIC: Recently drawn numbers get bonus
        if num in recent_numbers[-20:]:
            recent_bonus = 0.1
            score += recent_bonus
            logic_parts.append("Recently drawn (momentum)")
        
        # Build clear explanation
        if not logic_parts:
            logic_parts.append("Base frequency analysis")
        
        explanation = " + ".join(logic_parts)
        scored.append((num, score, explanation))

    if not scored: 
        return []
    
    scored.sort(key=lambda x: x[1], reverse=True)
    top_score = scored[0][1] if scored else 1.0
    normalized = []
    for num, raw_score, reason in scored[:200]:
        confidence = int((raw_score / top_score) * 100) if top_score else 0
        normalized.append((str(num), float(confidence/100), f"Logic: {reason} (Confidence: {confidence}%)"))
    return normalized[:5]

# ROUTES
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
            date_obj_parsed = pd.to_datetime(selected_date).date()
            filtered = df[df['date_parsed'].dt.date == date_obj_parsed]
        except:
            filtered = df.iloc[0:0]

    cards = [row.to_dict() for _, row in filtered.iterrows()]

    # Add predictions with logic to homepage
    homepage_predictions = []
    if not df.empty:
        raw_preds = advanced_predictor(df, lookback=100)[:3]
        for i, (num, score, logic) in enumerate(raw_preds, 1):
            homepage_predictions.append({
                'number': num,
                'rank': i,
                'confidence': int(score * 100),
                'logic': logic.replace('Logic: ', ''),
                'reason': f"Top #{i} pick based on {logic.split('(')[0]}"
            })
    
    return render_template(
        'index.html',
        cards=cards,
        selected_date=selected_date,
        homepage_predictions=homepage_predictions
    )

@app.route('/quick-pick')
def quick_pick():
    df = load_csv_data()
    provider = request.args.get('provider', 'all')
    
    if df.empty:
        return render_template('quick_pick.html', numbers=[], error="No data available", provider_options=['all'], provider='all')
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    # Get predictions with logic
    predictions = advanced_predictor(df, provider=provider, lookback=200)
    
    # Format with clear explanations
    prediction_details = []
    for i, (num, score, logic) in enumerate(predictions[:5], 1):
        prediction_details.append({
            'rank': i,
            'number': num,
            'confidence': int(score * 100),
            'logic': logic,
            'why_selected': f"Ranked #{i} because: {logic}"
        })
    
    # Get next draw date
    if not df.empty:
        last_draw = df.iloc[-1]
        next_draw_date = (last_draw['date_parsed'] + timedelta(days=3)).strftime('%Y-%m-%d (%A)')
    else:
        next_draw_date = "Unknown"
    
    return render_template('quick_pick.html', 
                         numbers=[p['number'] for p in prediction_details],
                         prediction_details=prediction_details,
                         next_draw_date=next_draw_date,
                         error=None,
                         provider_options=provider_options,
                         provider=provider)

@app.route('/statistics')
def statistics():
    df = load_csv_data()
    if df.empty:
        return render_template('statistics.html', total_draws=0, top_20=[], digit_freq=[], predictions=[], provider_options=['all'], provider='all', month_options=[], selected_month='', message="No data available")
    
    provider = request.args.get('provider', 'all')
    selected_month = request.args.get('month', '')
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    month_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m').unique(), reverse=True)
    
    filtered_df = df.copy()
    if provider != 'all':
        filtered_df = filtered_df[filtered_df['provider'] == provider]
    if selected_month:
        filtered_df = filtered_df[filtered_df['date_parsed'].dt.strftime('%Y-%m') == selected_month]
    
    filtered_df = filtered_df.tail(500)
    total_draws = len(filtered_df)
    
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_numbers.extend([n for n in filtered_df[col].astype(str) if n.isdigit() and len(n) == 4])
    
    digit_freq = Counter(''.join(all_numbers)).most_common(10)
    number_freq = Counter(all_numbers)
    top_20 = number_freq.most_common(20)
    
    # Get predictions with detailed logic
    raw_predictions = advanced_predictor(filtered_df, provider=provider, lookback=200)[:5]
    predictions_with_logic = []
    for num, score, logic in raw_predictions:
        predictions_with_logic.append({
            'number': num,
            'confidence': int(score * 100),
            'logic': logic,
            'frequency': all_numbers.count(num),
            'last_seen': 'Recent' if num in all_numbers[-50:] else 'Older'
        })
    
    return render_template('statistics.html', 
                         total_draws=total_draws, 
                         top_20=top_20, 
                         digit_freq=digit_freq, 
                         predictions=predictions_with_logic,
                         provider_options=provider_options, 
                         provider=provider, 
                         month_options=month_options, 
                         selected_month=selected_month, 
                         message="", 
                         last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    app.run(debug=False)