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

# ✅ make datetime available inside all HTML templates
@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

# ---------------- CSV LOADER (CORRECTED) ---------------- #
_csv_cache = None
_csv_cache_time = None
_csv_lock = threading.Lock()

def load_csv_data():
    """
    Load CSV with caching for better performance.
    """
    global _csv_cache, _csv_cache_time, _smart_model_cache, _ml_model_cache
    
    # Force clear all caches for instant updates
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

    # Parse date
    if 'date' not in df.columns:
        logger.error("CSV missing 'date' column")
        return pd.DataFrame()
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date_parsed'], inplace=True)

    # Normalize provider
    df['provider'] = df['provider'].fillna('').astype(str)
    df['provider'] = df['provider'].str.extract(r'images/([^./"]+)', expand=False).fillna('unknown').str.strip().str.lower()
    
    # Decode HTML entities - the prize text is in the '3rd' column
    import html
    df['prize_text'] = df['3rd'].fillna('').astype(str).apply(html.unescape)
    
    # Extract prize numbers - for standard 4D games, extract from prize text
    # For special games (5D/6D/Lotto/etc), keep the full prize_text as-is
    df['1st_real'] = df['prize_text'].str.extract(r'1st\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0]
    df['2nd_real'] = df['prize_text'].str.extract(r'2nd\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0]
    df['3rd_real'] = df['prize_text'].str.extract(r'3rd\s+Prize\s+(\d{4})', flags=re.IGNORECASE)[0]
    
    # For rows where standard extraction failed, use the full prize_text (special games)
    df['1st_real'] = df['1st_real'].fillna(df['prize_text'])
    df['2nd_real'] = df['2nd_real'].fillna('')
    df['3rd_real'] = df['3rd_real'].fillna('')
    
    # Keep special and consolation columns as-is
    df['special'] = df['special'].fillna('')
    df['consolation'] = df['consolation'].fillna('')
    
    # Only remove exact duplicates, keep all data
    df = df.drop_duplicates(keep='first')
    df = df.sort_values('date_parsed', ascending=True).reset_index(drop=True)
    
    logger.info(f"Processed {len(df)} rows | Date: {df['date_parsed'].min().date()} to {df['date_parsed'].max().date()} | Providers: {', '.join(df['provider'].unique()[:5])}")

    # Cache disabled for instant updates
    # _csv_cache = df.copy()
    # _csv_cache_time = datetime.now()
    return df

# (The rest of your code is unchanged and remains the same as your original)

# ---------------- HELPERS ---------------- #
def find_missing_digits(grid):
    all_digits = set(map(str, range(10)))
    used_digits = set(str(cell) for row in grid for cell in row)
    return sorted(all_digits - used_digits)

def find_4digit_patterns(grid):
    patterns = find_all_4digit_patterns(grid)
    
    # Add Fibonacci sequence detection
    fib_patterns = detect_fibonacci_patterns(grid)
    patterns.extend(fib_patterns)
    
    # Add prime clustering
    prime_patterns = detect_prime_clusters(grid)
    patterns.extend(prime_patterns)
    
    # Add sum range patterns
    sum_patterns = detect_sum_patterns(grid)
    patterns.extend(sum_patterns)
    
    # 🎯 ADVANCED: Golden ratio patterns
    golden_patterns = detect_golden_ratio_patterns(grid)
    patterns.extend(golden_patterns)
    
    # 🎯 ADVANCED: Harmonic sequences
    harmonic_patterns = detect_harmonic_sequences(grid)
    patterns.extend(harmonic_patterns)
    
    # 🎯 ADVANCED: Mirror symmetry
    mirror_patterns = detect_mirror_symmetry(grid)
    patterns.extend(mirror_patterns)
    
    # 🎯 ADVANCED: Arithmetic progressions
    arithmetic_patterns = detect_arithmetic_progressions(grid)
    patterns.extend(arithmetic_patterns)
    
    return patterns

def detect_fibonacci_patterns(grid):
    """Detect Fibonacci sequences in 4D grid"""
    fib_seq = [0, 1, 1, 2, 3, 5, 8]
    patterns = []
    
    for i, row in enumerate(grid):
        for j in range(len(row) - 2):
            seq = [int(str(row[j+k])[0]) for k in range(3) if str(row[j+k]).isdigit()]
            if len(seq) == 3:
                for start in range(len(fib_seq) - 2):
                    if seq == fib_seq[start:start+3]:
                        patterns.append(('fibonacci', f'row_{i}', ''.join(map(str, seq)), [(i, j+k) for k in range(3)]))
    return patterns

def detect_prime_clusters(grid):
    """Detect prime number clustering"""
    primes = [2, 3, 5, 7]
    patterns = []
    
    for i, row in enumerate(grid):
        prime_positions = []
        for j, cell in enumerate(row):
            if str(cell).isdigit() and int(str(cell)[0]) in primes:
                prime_positions.append((i, j))
        
        if len(prime_positions) >= 2:
            prime_nums = ''.join([str(grid[pos[0]][pos[1]]) for pos in prime_positions])
            patterns.append(('prime_cluster', f'row_{i}', prime_nums[:4], prime_positions[:4]))
    return patterns

def detect_sum_patterns(grid):
    """Detect sum range patterns (low/mid/high)"""
    patterns = []
    
    for i, row in enumerate(grid):
        row_sum = sum([int(str(cell)) for cell in row if str(cell).isdigit()])
        if row_sum <= 10:
            range_type = 'low_sum'
        elif row_sum <= 25:
            range_type = 'mid_sum'
        else:
            range_type = 'high_sum'
        
        patterns.append((range_type, f'row_{i}', str(row_sum), [(i, j) for j in range(len(row))]))
    return patterns

def detect_golden_ratio_patterns(grid):
    """🎯 ADVANCED: Detect golden rat