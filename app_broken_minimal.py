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
    config = config.Config()
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

# CSV LOADER with caching - memory leak prevention
_csv_cache = None
_csv_cache_time = None
_csv_lock = threading.Lock()
_cache_max_size = 50 * 1024 * 1024  # 50MB limit

def load_csv_data():
    """Load CSV with caching for better performance."""
    global _csv_cache, _csv_cache_time
    
    try:
        import warnings
        warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)
        
        # Try utils folder first (has the REAL 7.7MB CSV with ALL data)
        csv_paths = ['utils/4d_results_history.csv', '4d_results_history.csv', 'scraper/4d_results_history.csv']
        df = None
        
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                try:
                    # Try loading with header first (utils CSV has header)
                    df_test = pd.read_csv(csv_path, nrows=1, on_bad_lines='skip', encoding='utf-8', engine='python')
                    if 'date' in str(df_test.iloc[0].tolist()).lower():
                        # Has header row, skip it
                        df = pd.read_csv(csv_path, header=0, on_bad_lines='skip', encoding='utf-8', engine='python')
                    else:
                        # No header
                        df = pd.read_csv(csv_path, header=None, on_bad_lines='skip', encoding='utf-8', engine='python')
                    
                    if not df.empty:
                        logger.info(f"✓ Loaded CSV from: {csv_path} ({len(df)} rows)")
                        break
                except Exception as e:
                    logger.warning(f"Failed to load {csv_path}: {e}")
                    continue
        
        if df is None or df.empty:
            logger.error("No valid CSV file found")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Unexpected error loading CSV: {e}")
        return pd.DataFrame()

    # Check if CSV already has proper column names
    if 'date' in df.columns and 'provider' in df.columns:
        # CSV has header, use as-is but rename for consistency
        if '1st' in df.columns:
            df.rename(columns={'1st': 'prize_text_1st', '2nd': 'prize_text_2nd', '3rd': 'prize_text_3rd'}, inplace=True)
    else:
        # No header, assign column names
        num_cols = len(df.columns)
        if num_cols == 8:
            df.columns = ['date', 'provider', 'draw_info', 'draw_number', 'date_info', 'prize_text', 'special', 'consolation']
        elif num_cols == 7:
            df.columns = ['date', 'provider', 'draw_info', 'draw_number', 'prize_text', 'special', 'consolation']
    
    # Ensure prize_text column exists
    if 'prize_text' not in df.columns:
        # Try to find it
        for col in df.columns:
            if '1st Prize' in str(df[col].iloc[0] if len(df) > 0 else ''):
                df['prize_text'] = df[col]
                break
    
    # Parse date
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date_parsed'], inplace=True)

    # Extract provider name from URL
    df['provider'] = df['provider'].fillna('').astype(str)
    df['provider'] = df['provider'].str.extract(r'images/([^./"]+)', expand=False).fillna(
        df['provider'].str.extract(r'logo_([^./]+)', expand=False)
    ).fillna('unknown').str.strip().str.lower()
    
    # CRITICAL: Decode HTML entities BEFORE extraction
    import html
    df['prize_text'] = df['prize_text'].fillna('').astype(str).apply(html.unescape)
    
    # Extract prize numbers
    df['1st_real'] = df['prize_text'].str.extract(r'1st\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0].fillna('')
    df['2nd_real'] = df['prize_text'].str.extract(r'2nd\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0].fillna('')
    df['3rd_real'] = df['prize_text'].str.extract(r'3rd\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0].fillna('')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['date_parsed', 'provider', '1st_real', '2nd_real', '3rd_real'], keep='first')
    
    # Filter out rows without prize data
    df = df[(df['1st_real'] != '') | (df['2nd_real'] != '') | (df['3rd_real'] != '')]
    
    # Sort by date
    df = df.sort_values('date_parsed', ascending=True).reset_index(drop=True)
    
    logger.info(f"✓ Processed {len(df)} rows with prize data")
    logger.info(f"✓ Date range: {df['date_parsed'].min().date()} to {df['date_parsed'].max().date()}")
    logger.info(f"✓ Providers: {', '.join(df['provider'].unique())}")

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
            # Extract 4-digit numbers properly
            val = val.strip()
            if val.isdigit() and len(val) == 4:
                all_recent.append(val)
            else:
                found = re.findall(r'\d{4}', val)
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
    
    # Use ONLY real numbers from CSV - no fake/random generation
    candidate_pool = set(recent_numbers)
    
    # Add historical numbers from transitions (also from real data)
    if transitions:
        candidate_pool.update(k for k in transitions.keys() if isinstance(k, str) and len(k) == 4 and k.isdigit())
    
    # Expand pool with more historical data if needed
    if len(candidate_pool) < 200:
        for num in all_recent[-1000:]:
            candidate_pool.add(num)
    
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
        except (TypeError, ValueError, KeyError):
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
    
    # Sort by score and ensure we have at least 5 unique predictions
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_scored = []
    for item in scored:
        if item[0] not in seen:
            seen.add(item[0])
            unique_scored.append(item)
    
    top_score = unique_scored[0][1] if unique_scored else 1.0
    normalized = []
    for num, raw_score, reason in unique_scored[:5]:
        confidence = int((raw_score / top_score) * 100) if top_score else 0
        normalized.append((str(num), float(confidence/100), f"Logic: {reason} (Confidence: {confidence}%)"))
    
    return normalized

# ROUTES
@app.route('/debug-data')
def debug_data():
    df = load_csv_data()
    latest = df[df['date_parsed'] == df['date_parsed'].max()]
    
    # Show raw CSV columns and extracted data
    debug_info = []
    for _, row in latest.iterrows():
        debug_info.append({
            'provider': row.get('provider'),
            'prize_text_raw': str(row.get('prize_text', ''))[:200],
            'special_raw': str(row.get('special', ''))[:100],
            'consolation_raw': str(row.get('consolation', ''))[:100],
            'extracted_1st': row.get('1st_real'),
            'extracted_2nd': row.get('2nd_real'),
            'extracted_3rd': row.get('3rd_real'),
            'all_columns': list(row.keys())
        })
    
    return jsonify({
        'total_rows': len(df),
        'latest_date': str(df['date_parsed'].max()),
        'debug_details': debug_info
    })

@app.route('/')
def index():
    df = load_csv_data()
    selected_date = request.args.get('selected_date')

    if not selected_date:
        try:
            latest_date = df['date_parsed'].max().date()
            filtered = df[df['date_parsed'].dt.date == latest_date]
            selected_date = latest_date
        except (AttributeError, KeyError, ValueError):
            filtered = df.iloc[0:0]
            selected_date = ""
    else:
        try:
            date_obj_parsed = pd.to_datetime(selected_date).date()
            filtered = df[df['date_parsed'].dt.date == date_obj_parsed]
        except (ValueError, TypeError):
            filtered = df.iloc[0:0]

    cards = [row.to_dict() for _, row in filtered.iterrows()]
    
    # DEBUG: Log what's being sent
    logger.info(f"="*80)
    logger.info(f"INDEX ROUTE - Sending {len(cards)} cards for date {selected_date}")
    if cards:
        for i, card in enumerate(cards):
            logger.info(f"\nCard {i}:")
            logger.info(f"  Provider: {card.get('provider')}")
            logger.info(f"  Date: {card.get('date')}")
            logger.info(f"  1st_real: '{card.get('1st_real')}'")
            logger.info(f"  2nd_real: '{card.get('2nd_real')}'")
            logger.info(f"  3rd_real: '{card.get('3rd_real')}'")
    else:
        logger.info("  NO CARDS TO DISPLAY!")
    logger.info(f"="*80)

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
        homepage_predictions=homepage_predictions,
        total_results=len(cards),
        debug_mode=True,
        test_marker='FLASK_APP_RUNNING_CORRECTLY_' + str(datetime.now().timestamp())
    )

@app.route('/quick-pick')
def quick_pick():
    df = load_csv_data()
    provider = request.args.get('provider', 'all')
    
    if df.empty:
        return render_template('quick_pick.html', numbers=[], error="No data available", provider_options=['all'], provider='all')
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    
    if provider != 'all' and provider in provider_options:
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
    if provider != 'all' and provider in provider_options:
        filtered_df = filtered_df[filtered_df['provider'] == provider]
    if selected_month and selected_month in month_options:
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
    for pred in raw_predictions:
        if isinstance(pred, tuple) and len(pred) >= 2:
            num = pred[0]
            score = pred[1]
            logic = pred[2] if len(pred) > 2 else 'Pattern analysis'
            predictions_with_logic.append({
                'number': num,
                'confidence': int(score * 100) if isinstance(score, float) else score,
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