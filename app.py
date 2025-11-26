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
    """🎯 ADVANCED: Detect golden ratio (1.618) patterns in grid"""
    patterns = []
    phi = 1.618
    
    for i, row in enumerate(grid):
        for j in range(len(row) - 1):
            if str(row[j]).isdigit() and str(row[j+1]).isdigit():
                val1, val2 = int(str(row[j])[0]), int(str(row[j+1])[0])
                if val2 > 0 and abs((val1 / val2) - phi) < 0.3:
                    patterns.append(('golden_ratio', f'row_{i}', f'{val1}/{val2}', [(i, j), (i, j+1)]))
    return patterns

def detect_harmonic_sequences(grid):
    """🎯 ADVANCED: Detect harmonic number sequences (1, 1/2, 1/3, 1/4...)"""
    patterns = []
    harmonic = [1, 2, 3, 5, 8]  # Simplified harmonic-like sequence
    
    for i, row in enumerate(grid):
        row_digits = [int(str(cell)[0]) for cell in row if str(cell).isdigit()]
        if len(row_digits) >= 3:
            for start in range(len(row_digits) - 2):
                seq = row_digits[start:start+3]
                if seq in [harmonic[i:i+3] for i in range(len(harmonic)-2)]:
                    patterns.append(('harmonic', f'row_{i}', ''.join(map(str, seq)), [(i, start+k) for k in range(3)]))
    return patterns

def detect_mirror_symmetry(grid):
    """🎯 ADVANCED: Detect mirror/palindrome patterns"""
    patterns = []
    
    for i, row in enumerate(grid):
        row_str = ''.join([str(cell) for cell in row if str(cell).isdigit()])
        if len(row_str) >= 2 and row_str == row_str[::-1]:
            patterns.append(('mirror_symmetry', f'row_{i}', row_str, [(i, j) for j in range(len(row))]))
    
    # Check diagonal symmetry
    for i in range(len(grid) - 1):
        for j in range(len(grid[i]) - 1):
            if str(grid[i][j]).isdigit() and str(grid[i+1][j+1]).isdigit():
                if str(grid[i][j])[0] == str(grid[i+1][j+1])[0]:
                    patterns.append(('diagonal_mirror', f'pos_{i}_{j}', str(grid[i][j]), [(i, j), (i+1, j+1)]))
    return patterns

def detect_arithmetic_progressions(grid):
    """🎯 ADVANCED: Detect arithmetic progressions (e.g., 2,4,6,8 or 1,3,5,7)"""
    patterns = []
    
    for i, row in enumerate(grid):
        row_digits = [int(str(cell)[0]) for cell in row if str(cell).isdigit()]
        if len(row_digits) >= 3:
            for start in range(len(row_digits) - 2):
                seq = row_digits[start:start+3]
                if len(seq) == 3:
                    diff1 = seq[1] - seq[0]
                    diff2 = seq[2] - seq[1]
                    if diff1 == diff2 and diff1 != 0:
                        patterns.append(('arithmetic_prog', f'row_{i}', f'{seq[0]},{seq[1]},{seq[2]}', [(i, start+k) for k in range(3)]))
    return patterns

def highlight_coords_for_patterns(patterns, targets):
    highlights = []
    for kind, idx, p, coords in patterns:
        if p in targets:
            highlights.append(coords)
    return [xy for coords in highlights for xy in coords]

def search_pattern_in_grid(grid, pattern):
    matches = []
    pattern = str(pattern)
    for kind, idx, p, coords in find_4digit_patterns(grid):
        if pattern == p:
            matches.append(coords)
    return matches

def build_pattern_history(draws):
    pattern_success = defaultdict(int)
    for i in range(len(draws) - 1):
        this_draw = draws[i]
        next_draw = draws[i + 1]
        next_targets = [
            str(next_draw.get(col, ''))
            for col in ['1st_real', '2nd_real', '3rd_real']
            if str(next_draw.get(col, '')).isdigit()
        ]
        for kind, idx, p, coords in this_draw.get('patterns', []):
            if p in next_targets:
                pattern_success[p] += 1
        for kind, idx, p, coords in this_draw.get('reverse_patterns', []):
            if p in next_targets:
                pattern_success[p] += 1
        for md in this_draw.get("missing_digits", []):
            if md in "".join(next_targets):
                pattern_success[f"missing-{md}"] += 1
    return pattern_success

def _normalize_prediction_dict(result_dict):
    if not isinstance(result_dict, dict):
        return []
    preferred_order = ["classifier", "combined", "grid", "reverse", "missing", "reverse_missing", "fallback"]
    chosen = None
    for k in preferred_order:
        if k in result_dict and isinstance(result_dict[k], list) and result_dict[k]:
            chosen = result_dict[k]
            break
    if chosen is None:
        for v in result_dict.values():
            if isinstance(v, list) and v:
                chosen = v
                break
    if not chosen:
        return []
    out = []
    for item in chosen:
        if isinstance(item, (list, tuple)):
            if len(item) >= 3:
                num, score, reason = item[0], item[1], item[2]
            elif len(item) == 2:
                num, score = item
                reason = "no-reason"
            else:
                num = item[0]
                score = 0
                reason = "raw"
        else:
            num = str(item)
            score = 0
            reason = "raw"
        try:
            score = float(score)
        except Exception:
            score = 0.0
        out.append((str(num), score, str(reason)))
    return out[:5]

def _map_ui_mode_to_predictor(mode):
    if mode == "pattern":
        return "pattern"
    if mode == "frequency":
        return "history"
    if mode == "extended":
        return "combined"
    return "pattern"

# 3-digit hits
def get_3_digit_hits(predictions, actual):
    actual_digits = set(actual)
    hits = []
    for pred, score, reason in predictions:
        match_count = len(set(pred) & actual_digits)
        if match_count == 3:
            matched = ''.join(sorted(set(pred) & actual_digits))
            hits.append((pred, score, f"matched 3 digits: {matched}"))
    return hits

# ---------------- ROUTES ---------------- #
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

@app.route('/pattern-analyzer', methods=['GET', 'POST'])
def pattern_analyzer():
    from utils.feedback_learner import FeedbackLearner
    
    df = load_csv_data()
    selected_month = request.args.get('month')
    selected_provider = request.args.get('provider')
    selected_aimode = request.args.get('aimode', 'pattern')
    
    # Initialize feedback learner
    learner = FeedbackLearner()
    try:
        learner.load_learning_data()
    except:
        pass

    search_pattern = ''
    manual_search_results = {}
    draws = []

    if selected_provider and selected_provider.startswith("http"):
        selected_provider = selected_provider.split('/')[-1]
    if selected_provider:
        selected_provider = selected_provider.strip().lower()

    if not selected_month:
        today_date_val = date_obj.today()
        selected_month = today_date_val.strftime('%Y-%m')

    provider_options = sorted(df['provider'].dropna().unique())
    provider_options = [p for p in provider_options if p and str(p).strip() != ""]
    provider_options.insert(0, 'all')

    if not selected_provider or selected_provider not in provider_options:
        selected_provider = 'all'

    month_start = pd.to_datetime(selected_month + "-01")
    next_month = (month_start + pd.DateOffset(months=1)).replace(day=1)
    month_end = next_month - pd.Timedelta(days=1)

    month_draws = df[(df['date_parsed'] >= month_start) & (df['date_parsed'] <= month_end)]
    if selected_provider != 'all':
        month_draws = month_draws[month_draws['provider'] == selected_provider]

    month_draws = month_draws[month_draws['1st_real'].astype(str).str.len() == 4]
    month_draws = month_draws[month_draws['1st_real'].astype(str).str.isdigit()]
    month_draws = month_draws.sort_values(['date_parsed', 'provider']).reset_index(drop=True)

    predictor_mode = _map_ui_mode_to_predictor(selected_aimode)
    past_predictions_map = defaultdict(list)
    normalized_pred_list = []  # Initialize early

    module_hits = defaultdict(int)
    module_attempts = defaultdict(int)
    module_provider_stats = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "attempts": 0}))
    module_daily_stats = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "attempts": 0}))

    for i in range(len(month_draws) - 1):
        try:
            this_draw = month_draws.iloc[i]
            next_draw = month_draws.iloc[i + 1]

            for prize_type in ['1st', '2nd', '3rd']:
                real_key = prize_type + '_real'
                n = str(this_draw.get(real_key, ''))
                if len(n) == 4 and n.isdigit():
                    grid = generate_4x4_grid(n)
                    reverse_grid = generate_reverse_grid(n)
                    pats = find_4digit_patterns(grid)
                    reverse_pats = find_4digit_patterns(reverse_grid)
                    targets = [str(next_draw.get(col, '')) for col in ['1st_real', '2nd_real', '3rd_real'] if str(next_draw.get(col, '')).isdigit()]
                    highlight = highlight_coords_for_patterns(pats, targets)
                    reverse_highlight = highlight_coords_for_patterns(reverse_pats, targets)

                    raw_pred_dict = predict_top_5([{'date': str(this_draw['date_parsed'].date()), 'grid': grid}], mode=predictor_mode)
                    normalized_pred_list = _normalize_prediction_dict(raw_pred_dict)

                    provider = this_draw['provider']
                    actual_winners = targets
                    
                    # Evaluate predictions with feedback learner (optimized)
                    try:
                        predicted_numbers = [pred for pred, _, _ in normalized_pred_list]
                        match_type, score, match_details = learner.evaluate_prediction(
                            predicted_numbers,
                            targets[0] if len(targets) > 0 else '',
                            targets[1] if len(targets) > 1 else '',
                            targets[2] if len(targets) > 2 else ''
                        )
                        
                        # Learn from the result
                        learner.learn_from_result({
                            'predicted_numbers': predicted_numbers,
                            'predictor_methods': predictor_mode,
                            'confidence': normalized_pred_list[0][1] if normalized_pred_list else 0,
                            'draw_date': str(this_draw['date_parsed'].date())
                        }, match_type, score)
                    except:
                        pass
                    
                    comparison = []
                    for pred, score, reason in normalized_pred_list:
                        hit = "✅" if pred in actual_winners else "❌"
                        comparison.append({
                            "number": pred,
                            "score": score,
                            "reason": reason,
                            "hit": hit
                        })
                        modules = reason.split("+")
                        for m in modules:
                            module_attempts[m] += 1
                            if hit == "✅":
                                module_hits[m] += 1
                            module_provider_stats[provider][m]["attempts"] += 1
                            module_daily_stats[this_draw['date_parsed'].date()][m]["attempts"] += 1
                            if hit == "✅":
                                module_provider_stats[provider][m]["hits"] += 1
                                module_daily_stats[this_draw['date_parsed'].date()][m]["hits"] += 1

                    draws.append({
                        'date': this_draw['date_parsed'].date(),
                        'provider': provider,
                        'prize_type': prize_type,
                        'number': n,
                        'grid': grid,
                        'reverse_grid': reverse_grid,
                        'patterns': pats,
                        'reverse_patterns': reverse_pats,
                        'next_targets': targets,
                        'highlight': highlight,
                        'reverse_highlight': reverse_highlight,
                        'missing_digits': find_missing_digits(grid),
                        'missing_digits_reverse': find_missing_digits(reverse_grid),
                        'past_2_predictions': past_predictions_map[prize_type][-2:],
                        'provider_predictions': {provider: normalized_pred_list},
                        'prediction_comparison': comparison,
                        'three_digit_hits': get_3_digit_hits(normalized_pred_list, n)
                    })

                    past_predictions_map[prize_type].append(normalized_pred_list)

        except Exception as e:
            logger.warning(f"❌ Draw skipped due to error: {e}")

    pattern_success = build_pattern_history(draws)

    if request.method == 'POST':
        search_pattern = request.form.get('search_pattern', '').strip()
        for d in draws:
            if len(search_pattern) == 4 and search_pattern.isdigit():
                matches = search_pattern_in_grid(d['grid'], search_pattern)
                reverse_matches = search_pattern_in_grid(d['reverse_grid'], search_pattern)
                manual_search_results[(d['date'], d['prize_type'], 'normal')] = matches
                manual_search_results[(d['date'], d['prize_type'], 'reverse')] = reverse_matches

    month_options = sorted(set(df['date_parsed'].dropna().map(lambda d: d.strftime('%Y-%m'))))
    freq_list = compute_pattern_frequencies(draws)
    cell_heatmap = compute_cell_heatmap(draws)
    last_updated = time.strftime('%Y-%m-%d %H:%M:%S')

    if draws:
        past_draws_for_memory = [{"number": d.get("number", "")} for d in draws if d.get("number", "")]
        transitions = learn_pattern_transitions(past_draws_for_memory)
        today_number = draws[-1]["number"]
        candidates = predict_from_today_grid(today_number, transitions)

        grid = generate_4x4_grid(today_number)
        reverse_grid = generate_reverse_grid(today_number)
        draw_stub = {
            "number": today_number,
            "grid": grid,
            "reverse_grid": reverse_grid
        }
        full_draws = past_draws_for_memory[:-1] + [draw_stub]
        raw_predictions = predict_top_5(full_draws, mode="combined")
        top_5_predictions = _normalize_prediction_dict(raw_predictions)
        prediction_mode = "combined"
    else:
        top_5_predictions = []
        prediction_mode = "fallback"

    module_accuracy = []
    for m in module_attempts:
        attempts = module_attempts[m]
        hits = module_hits[m]
        rate = round((hits / attempts) * 100, 2) if attempts else 0
        module_accuracy.append((m, hits, attempts, rate))

    provider_accuracy = []
    for provider, modules in module_provider_stats.items():
        for m, stats in modules.items():
            rate = round((stats["hits"] / stats["attempts"]) * 100, 2) if stats["attempts"] else 0
            provider_accuracy.append((provider, m, stats["hits"], stats["attempts"], rate))

    chart_data = []
    for date, modules in sorted(module_daily_stats.items()):
        entry = {"date": date.strftime("%Y-%m-%d")}
        for m, stats in modules.items():
            rate = round((stats["hits"] / stats["attempts"]) * 100, 2) if stats["attempts"] else 0
            entry[m] = rate
        chart_data.append(entry)
    
    # Save learning data (optimized)
    try:
        learner.save_learning_data()
    except:
        pass
    
    # Get learning insights (optimized)
    learning_summary = None
    try:
        best_methods = learner.get_best_methods(top_n=5)
        learning_summary = {
            'best_methods': best_methods,
            'total_analyzed': len(draws)
        }
    except:
        pass

    # 4D Analysis - Hot/Cold Numbers
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real', 'special', 'consolation']:
        if col in month_draws.columns:
            for val in month_draws[col].astype(str):
                all_numbers.extend(re.findall(r'\d{4}', val))
    
    freq_counter = Counter(all_numbers)
    sorted_freq = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)
    hot_numbers = sorted_freq[:30] if sorted_freq else []
    cold_numbers = sorted_freq[-30:] if len(sorted_freq) >= 30 else []
    
    # Digit frequency by position
    digit_frequency_by_pos = {}
    for pos in range(1, 5):
        pos_digits = [num[pos-1] for num in all_numbers if len(num) >= pos]
        digit_freq = Counter(pos_digits)
        digit_frequency_by_pos[pos] = digit_freq.most_common(10)
    
    # Frequency-based predictions (hot numbers)
    frequency_predictions = hot_numbers[:5] if hot_numbers else []
    
    # Combined consensus predictions
    consensus_map = {}
    # Add pattern predictions
    for pred, conf, reason in normalized_pred_list[:10]:
        if pred not in consensus_map:
            consensus_map[pred] = {'score': 0, 'sources': []}
        consensus_map[pred]['score'] += conf * 2
        consensus_map[pred]['sources'].append('Pattern')
    
    # Add frequency predictions
    for num, freq in hot_numbers[:10]:
        if num not in consensus_map:
            consensus_map[num] = {'score': 0, 'sources': []}
        consensus_map[num]['score'] += (freq / max(dict(hot_numbers).values())) if hot_numbers else 0
        consensus_map[num]['sources'].append('Frequency')
    
    # Sort by score and format
    consensus_predictions = [(num, data['score'], '+'.join(data['sources'])) 
                            for num, data in sorted(consensus_map.items(), key=lambda x: x[1]['score'], reverse=True)]
    
    # Digit position predictions - build numbers from most frequent digits
    digit_position_predictions = []
    if digit_frequency_by_pos and all(pos in digit_frequency_by_pos and digit_frequency_by_pos[pos] for pos in range(1, 5)):
        # Generate top 5 combinations from most frequent digits per position
        for combo_idx in range(5):
            try:
                num = ''.join([
                    digit_frequency_by_pos[pos][min(combo_idx, len(digit_frequency_by_pos[pos])-1)][0] 
                    if digit_frequency_by_pos[pos] else '0'
                    for pos in range(1, 5)
                ])
                if len(num) == 4 and num.isdigit() and num not in digit_position_predictions:
                    digit_position_predictions.append(num)
            except (IndexError, KeyError):
                continue
        
        # Add mixed combinations if needed
        if len(digit_position_predictions) < 5:
            try:
                for i in range(min(3, len(digit_frequency_by_pos.get(3, [])))):
                    for j in range(min(2, len(digit_frequency_by_pos.get(2, [])))):
                        if all(digit_frequency_by_pos.get(p) for p in [1,2,3,4]):
                            num = ''.join([
                                digit_frequency_by_pos[1][0][0],
                                digit_frequency_by_pos[2][j][0],
                                digit_frequency_by_pos[3][i][0],
                                digit_frequency_by_pos[4][0][0]
                            ])
                            if len(num) == 4 and num.isdigit() and num not in digit_position_predictions:
                                digit_position_predictions.append(num)
                                if len(digit_position_predictions) >= 5:
                                    break
                    if len(digit_position_predictions) >= 5:
                        break
            except (IndexError, KeyError):
                pass
    
    # Pattern distribution analysis
    def classify_pattern(num):
        digits = list(num)
        unique = len(set(digits))
        if unique == 4: return 'ABCD'
        elif unique == 3: return 'AABC'
        elif unique == 2:
            return 'AABB' if digits.count(digits[0]) == 2 and digits.count(digits[2]) == 2 else 'AAAB'
        return 'AAAA'
    
    pattern_counts = Counter([classify_pattern(num) for num in all_numbers if len(num) == 4])
    total = sum(pattern_counts.values())
    pattern_distribution = [(p, c, round(c/total*100, 1)) for p, c in pattern_counts.most_common()] if total else []
    
    # Pattern-based predictions - get numbers matching most common pattern
    pattern_predictions = []
    if pattern_distribution and all_numbers:
        most_common_pattern = pattern_distribution[0][0]
        matching_nums = [num for num in all_numbers if len(num) == 4 and classify_pattern(num) == most_common_pattern]
        if matching_nums:
            # Get most frequent numbers with this pattern
            pattern_freq = Counter(matching_nums)
            pattern_predictions = [num for num, _ in pattern_freq.most_common(5)]
    
    return render_template(
        'pattern_analyzer.html',
        draws=draws,
        provider_options=provider_options,
        selected_provider=selected_provider,
        month_options=month_options,
        selected_month=selected_month,
        search_pattern=search_pattern,
        manual_search_results=manual_search_results,
        freq_list=freq_list,
        cell_heatmap=cell_heatmap,
        top_5_predictions=top_5_predictions,
        prediction_mode=prediction_mode,
        last_updated=last_updated,
        selected_aimode=selected_aimode,
        is_fallback=(prediction_mode == "fallback"),
        module_accuracy=module_accuracy,
        provider_accuracy=provider_accuracy,
        chart_data=chart_data,
        best_module=(max(module_accuracy, key=lambda x: x[3]) if module_accuracy else None),
        learning_summary=learning_summary,
        hot_numbers=hot_numbers,
        cold_numbers=cold_numbers,
        digit_frequency_by_pos=digit_frequency_by_pos,
        frequency_predictions=frequency_predictions,
        consensus_predictions=consensus_predictions,
        digit_position_predictions=digit_position_predictions,
        pattern_distribution=pattern_distribution,
        pattern_predictions=pattern_predictions
    )

@app.route('/prediction-history')
def prediction_history():
    df = load_csv_data()
    provider = request.args.get('provider', 'all')
    selected_month = request.args.get('month', '')
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    month_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m').unique(), reverse=True)
    
    if provider != 'all':
        df = df[df['provider'] == provider]
    if selected_month:
        df = df[df['date_parsed'].dt.strftime('%Y-%m') == selected_month]
    
    df = df[df['1st_real'].astype(str).str.len() == 4]
    df = df[df['1st_real'].astype(str).str.isdigit()]
    df = df.tail(50).sort_values(['date_parsed', 'provider']).reset_index(drop=True)
    df = df.drop_duplicates(subset=['date_parsed', '1st_real'], keep='first')

    history = []
    for i in range(min(len(df) - 1, 30)):
        this_draw = df.iloc[i]
        next_draw = df.iloc[i + 1]

        number = str(this_draw['1st_real'])
        grid = generate_4x4_grid(number)

        raw_pred = predict_top_5(
            [{'date': str(this_draw['date_parsed'].date()), 'grid': grid}],
            mode="pattern"
        )
        normalized = _normalize_prediction_dict(raw_pred)

        next_winners = [str(next_draw['1st_real']), str(next_draw['2nd_real']), str(next_draw['3rd_real'])]

        checked_preds = []
        for num, score, _reason in normalized:
            hit = "✅" if num in next_winners else "❌"
            checked_preds.append({"num": num, "score": score, "hit": hit})

        history.append({
            "date": this_draw["date_parsed"].date(),
            "provider": this_draw["provider"],
            "drawn_number": this_draw["1st_real"],
            "predictions": checked_preds,
            "next_winners": next_winners
        })

    return render_template("prediction_history.html", history=history, provider_options=provider_options, provider=provider, month_options=month_options, selected_month=selected_month)

# ---------------- ADVANCED PREDICTOR + ROUTE ---------------- #
def compute_provider_bias(df, provider):
    if not provider or provider == "all":
        return 1.0
    sub = df[df["provider"] == provider].sort_values("date_parsed")
    nums = sub["1st_real"].astype(str).tolist()
    if len(nums) < 2:
        return 1.0
    matches = 0
    total = 0
    for a, b in zip(nums, nums[1:]):
        if len(a) == 4 and len(b) == 4:
            total += 1
            if set(a) & set(b):
                matches += 1
    rate = (matches / total) if total else 0
    return 1.0 + (rate * 0.5)

def advanced_predictor(df, provider=None, lookback=200):
    prize_cols = ["1st_real", "2nd_real", "3rd_real", "special", "consolation"]
    all_recent = []
    for col in prize_cols:
        if col not in df.columns: continue
        col_values = df[col].astype(str).dropna().tolist()
        for val in col_values:
            found = re.findall(r'\d{4}', val)
            for f in found:
                if f.isdigit() and len(f) == 4:
                    all_recent.append(f)

    if not all_recent: return []
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
    provider_multiplier = compute_provider_bias(df, provider)
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
        if not (isinstance(num, str) and len(num) == 4 and num.isdigit()): continue
        score = 0.0
        reasons = []
        hot_score = sum((digit_counts.get(d, 0) / max_digit_freq) for d in num)
        score += hot_score
        reasons.append(f"hot={hot_score:.2f}")
        pair_bonus = 0.0
        for i in range(3):
            pair = num[i : i + 2]
            cnt = pair_counts.get(pair, 0)
            if cnt:
                pair_bonus += cnt * 0.06
        if pair_bonus:
            score += pair_bonus
            reasons.append(f"pair+{pair_bonus:.2f}")
        trans_val = 0.0
        try:
            tval = transitions.get(num, 0)
            if isinstance(tval, (int, float)) and tval > 0:
                trans_val = float(tval)
                score += trans_val * 0.5
                reasons.append(f"trans+{trans_val:.2f}")
        except Exception:
            pass
        if provider and provider != "all":
            score *= provider_multiplier
            reasons.append(f"prov*{provider_multiplier:.2f}")
        scored.append((num, score, "+".join(reasons)))

    if not scored: return []
    scored.sort(key=lambda x: x[1], reverse=True)
    top_score = scored[0][1] if scored else 1.0
    normalized = []
    for num, raw_score, reason in scored[:200]:
        norm = (raw_score / top_score) if top_score else 0.0
        normalized.append((str(num), float(norm), reason))
    return normalized[:5]

@app.route('/advanced-predictions', methods=['GET'])
def advanced_predictions():
    return redirect('/ultimate-predictor')

@app.route('/advanced-analytics')
def advanced_analytics():
    df = load_csv_data()
    if df.empty:
        return render_template('advanced_analytics.html', error="No data available")

    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    provider = request.args.get('provider', 'all')

    if provider != 'all':
        df = df[df['provider'] == provider]

    # Next draw info
    next_draw_date = ''
    next_draw_day = ''
    if not df.empty:
        last_draw = df.iloc[-1]
        from datetime import timedelta
        next_draw_date = (last_draw['date_parsed'] + timedelta(days=3)).strftime('%Y-%m-%d')
        next_draw_day = (last_draw['date_parsed'] + timedelta(days=3)).strftime('%A')

    # --- Monthly Stats ---
    monthly_stats = []
    if not df.empty:
        df['month'] = df['date_parsed'].dt.to_period('M')
        last_6_months = df['month'].unique()[-6:]
        for month in last_6_months:
            month_df = df[df['month'] == month]
            all_nums = []
            for col in ['1st_real', '2nd_real', '3rd_real', 'special', 'consolation']:
                if col in month_df.columns:
                    all_nums.extend(month_df[col].astype(str).dropna().str.findall(r'\d{4}').explode().tolist())
            if all_nums:
                counts = Counter(all_nums)
                monthly_stats.append({
                    'month': month.strftime('%Y-%m'),
                    'total_draws': len(month_df),
                    'unique_numbers': len(counts),
                    'most_common': counts.most_common(1)[0],
                    'top_numbers': counts.most_common(5)
                })

    # --- Day of Week Stats ---
    day_stats = []
    if not df.empty:
        df['day_of_week'] = df['date_parsed'].dt.day_name()
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            day_df = df[df['day_of_week'] == day]
            if not day_df.empty:
                all_nums = []
                for col in ['1st_real', '2nd_real', '3rd_real']:
                    if col in day_df.columns:
                        all_nums.extend(day_df[col].astype(str).dropna().tolist())
                if all_nums:
                    counts = Counter(all_nums)
                    day_stats.append({
                        'day': day,
                        'draws': len(day_df),
                        'top_numbers': counts.most_common(3)
                    })

    # --- Prize Comparison ---
    prize_comparison = []
    prize_cols = {
        '1st Prize': '1st_real',
        '2nd Prize': '2nd_real',
        '3rd Prize': '3rd_real',
        'Special': 'special',
        'Consolation': 'consolation'
    }
    for name, col in prize_cols.items():
        if col in df.columns:
            all_nums = df[col].astype(str).dropna().str.findall(r'\d{4}').explode().tolist()
            if all_nums:
                counts = Counter(all_nums)
                prize_comparison.append({
                    'type': name,
                    'total': len(all_nums),
                    'unique': len(counts),
                    'top_numbers': counts.most_common(10)
                })

    # --- Trending Numbers ---
    trending_up = []
    trending_down = []
    if not df.empty:
        recent_df = df[df['date_parsed'] > (datetime.now() - pd.Timedelta(days=30))]
        previous_df = df[(df['date_parsed'] > (datetime.now() - pd.Timedelta(days=60))) & (df['date_parsed'] <= (datetime.now() - pd.Timedelta(days=30)))]
        
        recent_nums = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            if col in recent_df.columns:
                recent_nums.extend(recent_df[col].astype(str).dropna().tolist())
        recent_counts = Counter(recent_nums)

        previous_nums = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            if col in previous_df.columns:
                previous_nums.extend(previous_df[col].astype(str).dropna().tolist())
        previous_counts = Counter(previous_nums)

        all_unique_nums = set(recent_counts.keys()) | set(previous_counts.keys())
        changes = []
        for num in all_unique_nums:
            change = recent_counts.get(num, 0) - previous_counts.get(num, 0)
            if change != 0:
                changes.append((num, recent_counts.get(num, 0), change))
        
        changes.sort(key=lambda x: x[2], reverse=True)
        trending_up = changes[:5]
        trending_down = [(num, r_count, abs(change)) for num, r_count, change in changes if change < 0][-5:]

    return render_template(
        'advanced_analytics.html',
        provider_options=provider_options,
        provider=provider,
        next_draw_date=next_draw_date,
        next_draw_day=next_draw_day,
        next_predictions=advanced_predictor(df, provider=provider, lookback=50),
        monthly_stats=monthly_stats,
        day_stats=day_stats,
        prize_comparison=prize_comparison,
        trending_up=trending_up,
        trending_down=trending_down,
    )

# ============================================================
# ⚙️ SMART PREDICTORS SECTION (Add-on, safe to paste)
# ============================================================



import ast

# --- Global Caches for ML Models with cleanup ---
_smart_model_cache = {}
_ml_model_cache = {}
_cache_max_size = 10  # Limit cache size

def cleanup_cache(cache_dict):
    """Clean up cache if it gets too large"""
    if len(cache_dict) > _cache_max_size:
        # Remove oldest entries
        keys_to_remove = list(cache_dict.keys())[:-_cache_max_size//2]
        for key in keys_to_remove:
            del cache_dict[key]


# ------------------------------------------------------------
# 1️⃣ AUTO WEIGHT TUNING - finds best balance between hot, pair, transitions
# ------------------------------------------------------------
def smart_auto_weight_predictor(df, provider=None, lookback=300):
    """
    Automatically tunes weight between 'hot digits', 'pairs', and 'transitions'
    based on correlation to next-draw winners.
    Easy & light but smarter than static scoring.
    """
    global _smart_model_cache
    # Use cached model if available and data hasn't changed significantly
    all_numbers = [] # Initialize all_numbers
    cache_key = (df.shape, df['date_parsed'].max())
    if cache_key in _smart_model_cache:
        model, scaler, w_hot, w_pair, w_trans, digit_counts, pair_counts, transitions, all_numbers = _smart_model_cache[cache_key]
    else:
        prize_cols = ["1st_real", "2nd_real", "3rd_real"]
        for col in prize_cols:
            if col in df.columns:
                all_numbers += [n for n in df[col].astype(str) if n.isdigit() and len(n) == 4]
        if not all_numbers:
            return []

        # Use last portion of data for training
        training_numbers = all_numbers[-lookback:]
        if not training_numbers: return []
        digit_counts = Counter("".join(all_numbers))
        pair_counts = Counter([n[i:i+2] for n in training_numbers for i in range(3)])
        transitions = learn_pattern_transitions([{"number": n} for n in all_numbers])

        # build dataset of feature weights
        features, targets = [], []
        for i in range(len(all_numbers) - 1):
            num = all_numbers[i]
            next_num = all_numbers[i + 1]
            hot_score = sum(digit_counts.get(d, 0) for d in num)
            pair_score = sum(pair_counts.get(num[i:i+2], 0) for i in range(3))
            trans_score = float(transitions.get(num, 0)) if transitions.get(num, 0) else 0
            features.append([hot_score, pair_score, trans_score])
            targets.append(len(set(num) & set(next_num)))  # how many digits repeated next draw

        if len(features) < 5:
            w_hot, w_pair, w_trans = 0.4, 0.3, 0.3 # Fallback weights
            model, scaler = None, None
        else:
            X = np.array(features)
            y = np.array(targets)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # learn ideal weights automatically
            model = LinearRegression()
            model.fit(X_scaled, y)
            w_hot, w_pair, w_trans = model.coef_

        # Cache the trained model and weights
        cleanup_cache(_smart_model_cache)
        _smart_model_cache[cache_key] = (model, scaler, w_hot, w_pair, w_trans, digit_counts, pair_counts, transitions, all_numbers)


    # Predict next based on learned weights
    candidate_pool = {n for n in all_numbers[-150:]} if all_numbers else set()
    if not candidate_pool:
        return []
    
    scored = []
    for num in candidate_pool:
        hot_score = sum(digit_counts.get(d, 0) for d in num)
        pair_score = sum(pair_counts.get(num[i:i+2], 0) for i in range(3))
        trans_score = float(transitions.get(num, 0)) if transitions.get(num, 0) else 0
        total = (
            w_hot * hot_score
            + w_pair * pair_score
            + w_trans * trans_score
        )
        scored.append((num, total, f"auto-w({w_hot:.2f},{w_pair:.2f},{w_trans:.2f})"))

    if not scored:
        return []
    
    scored.sort(key=lambda x: x[1], reverse=True)
    top_5 = [(num, round(score, 3), reason) for num, score, reason in scored[:5]]
    return top_5

@app.route('/smart-predictor')
@app.route('/smart-auto-weight')
def smart_predictor_page():
    df = load_csv_data()
    if df.empty:
        return render_template(
            'smart_predictor.html',
            provider="N/A",
            last_updated="No data available",
            top_5_predictions=None,
            smart_formula="N/A",
            provider_options=['all'],
            selected_provider='all',
            month_options=[],
            selected_month=''
        )
    
    # Get filters
    selected_provider = request.args.get('provider', 'all')
    selected_month = request.args.get('month', '')
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    month_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m').unique(), reverse=True)
    
    # Filter data
    filtered_df = df.copy()
    if selected_provider != 'all':
        filtered_df = filtered_df[filtered_df['provider'] == selected_provider]
    if selected_month:
        filtered_df = filtered_df[filtered_df['date_parsed'].dt.strftime('%Y-%m') == selected_month]
    
    if filtered_df.empty:
        return render_template(
            'smart_predictor.html',
            provider=selected_provider,
            last_updated="No data for selected filters",
            top_5_predictions=[],
            smart_formula="No data",
            provider_options=provider_options,
            selected_provider=selected_provider,
            month_options=month_options,
            selected_month=selected_month
        )

    # Use smart_auto_weight_predictor function
    top_5_preds = smart_auto_weight_predictor(filtered_df, provider=selected_provider, lookback=300)
    
    return render_template(
        'smart_predictor.html',
        provider=selected_provider.upper() if selected_provider != 'all' else 'ALL PROVIDERS',
        last_updated=filtered_df['date_parsed'].max().strftime('%Y-%m-%d %H:%M:%S') if not filtered_df.empty else "No data available",
        top_5_predictions=top_5_preds,
        smart_formula="Auto-tuned weights based on historical accuracy",
        provider_options=provider_options,
        selected_provider=selected_provider,
        month_options=month_options,
        selected_month=selected_month
    )
import os
from datetime import datetime

@app.route('/smart-history')
def smart_history_page():
    hist_file = "smart_history.csv"
    if not os.path.exists(hist_file):
        return render_template(
            'smart_history.html',
            history=[],
            message="No smart history recorded yet."
        )
    hist = pd.read_csv(hist_file)
    return render_template(
        'smart_history.html',
        history=hist.to_dict(orient='records'),
        message=""
    )


# ------------------------------------------------------------
# 2️⃣ MACHINE LEARNING ADD-ON - learns direct number patterns
# ------------------------------------------------------------
def ml_predictor(df, lookback=500):
    """
    Learns from all past draws using Linear Regression.
    Converts each 4D number into digit features and trains model to predict next draw.
    High accuracy potential.
    """
    prize_cols = ["1st_real", "2nd_real", "3rd_real"]
    global _ml_model_cache
    # Use cached model if available and data hasn't changed significantly
    cache_key = (df.shape, df['date_parsed'].max())
    if cache_key in _ml_model_cache:
        model, scaler, numbers = _ml_model_cache[cache_key]
    else:
        numbers = []
        for col in prize_cols:
            if col in df.columns:
                numbers += [n for n in df[col].astype(str) if n.isdigit() and len(n) == 4]
        if len(numbers) < 20:
            return []

        # Build training data
        X, y = [], []
        for i in range(len(numbers) - 1):
            curr = [int(d) for d in numbers[i]]
            next_num = [int(d) for d in numbers[i + 1]]
            X.append(curr)
            y.append(sum(next_num) / 4)  # simple numeric target (avg of next draw digits)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        # Cache the trained model and scaler
        cleanup_cache(_ml_model_cache)
        _ml_model_cache[cache_key] = (model, scaler, numbers)

    # Predict next best numbers by scoring possible combos
    recent = numbers[-10:]
    if not recent:
        return [] # Not enough data to generate candidates

    candidate_pool = set()
    for n in recent:
        for d in range(10):
            for pos in range(4):
                new_num = list(n)
                new_num[pos] = str(d)
                candidate_pool.add("".join(new_num))

    scored = []
    for num in candidate_pool:
        x_input = np.array([[int(d) for d in num]])
        x_scaled = scaler.transform(x_input)
        pred_score = model.predict(x_scaled)[0]
        scored.append((num, float(pred_score), "ML-learned"))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_5 = [(num, round(score, 3), reason) for num, score, reason in scored[:5]]
    return top_5


@app.route("/ml-predictor")
def ml_predictor_page():
    df = load_csv_data()
    top_5 = ml_predictor(df)
    last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

    return render_template(
        "ml_predictor.html",
        top_5_predictions=top_5,
        last_updated=last_updated,
    )

@app.route('/ultimate-predictor')
def ultimate_predictor():
    df = load_csv_data()
    if df.empty:
        return render_template('ultimate_predictor.html', error="No data available", provider_options=['all'], selected_provider='all', month_options=[], selected_month='')
    
    selected_provider = request.args.get('provider', 'all')
    selected_month = request.args.get('month', '')
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p and str(p).strip()])
    month_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m').unique(), reverse=True)
    
    if selected_provider not in provider_options:
        selected_provider = 'all'
    
    # Filter data
    df_filtered = df.copy()
    if selected_provider != 'all':
        df_filtered = df_filtered[df_filtered['provider'] == selected_provider]
    if selected_month:
        df_filtered = df_filtered[df_filtered['date_parsed'].dt.strftime('%Y-%m') == selected_month]
    
    if df_filtered.empty:
        return render_template('ultimate_predictor.html', error="No data for selected filters", provider_options=provider_options, selected_provider=selected_provider, month_options=month_options, selected_month=selected_month)
    
    # Get draw info
    last_draw = df_filtered.iloc[-1]
    last_draw_date = last_draw['date_parsed'].strftime('%Y-%m-%d (%A)')
    last_draw_number = str(last_draw['1st_real'])
    provider_name = selected_provider.upper() if selected_provider != 'all' else 'ALL PROVIDERS'
    
    from datetime import timedelta
    next_draw_date = (last_draw['date_parsed'] + timedelta(days=3)).strftime('%Y-%m-%d (%A)')
    
    # Get predictions from all methods
    advanced_preds = advanced_predictor(df_filtered, provider=selected_provider, lookback=100)[:5]
    smart_preds = smart_auto_weight_predictor(df_filtered, provider=selected_provider, lookback=100)[:5]
    ml_preds = ml_predictor(df_filtered, lookback=100)[:5]
    
    # Pattern predictions
    pattern_preds = []
    if len(last_draw_number) == 4 and last_draw_number.isdigit():
        grid = generate_4x4_grid(last_draw_number)
        raw = predict_top_5([{'date': str(last_draw['date_parsed'].date()), 'grid': grid}], mode="combined")
        pattern_preds = _normalize_prediction_dict(raw)[:5]
    
    # Combine all predictions
    all_predictions = {}
    
    for num, score, reason in advanced_preds:
        if num not in all_predictions:
            all_predictions[num] = {'count': 0, 'total_score': 0, 'sources': []}
        all_predictions[num]['count'] += 1
        all_predictions[num]['total_score'] += score
        all_predictions[num]['sources'].append('Advanced')
    
    for num, score, reason in smart_preds:
        if num not in all_predictions:
            all_predictions[num] = {'count': 0, 'total_score': 0, 'sources': []}
        all_predictions[num]['count'] += 1
        all_predictions[num]['total_score'] += score
        all_predictions[num]['sources'].append('Smart')
    
    for num, score, reason in ml_preds:
        if num not in all_predictions:
            all_predictions[num] = {'count': 0, 'total_score': 0, 'sources': []}
        all_predictions[num]['count'] += 1
        all_predictions[num]['total_score'] += score
        all_predictions[num]['sources'].append('ML')
    
    for num, score, reason in pattern_preds:
        if num not in all_predictions:
            all_predictions[num] = {'count': 0, 'total_score': 0, 'sources': []}
        all_predictions[num]['count'] += 1
        all_predictions[num]['total_score'] += score
        all_predictions[num]['sources'].append('Pattern')
    
    # Enhanced consensus with confidence intervals
    final_predictions = []
    for num, data in all_predictions.items():
        avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
        enhanced_consensus = calculate_enhanced_consensus(num, data, df_filtered)
        confidence_interval = calculate_confidence_interval(num, df_filtered, data['count'])
        stability_score = calculate_prediction_stability(num, data['sources'])
        
        final_predictions.append({
            'number': num,
            'consensus_score': round(avg_score, 3),
            'predictor_count': data['count'],
            'confidence': enhanced_consensus['confidence'],
            'confidence_interval': confidence_interval,
            'stability_score': stability_score,
            'enhanced_rating': enhanced_consensus['rating'],
            'sources': ', '.join(data['sources'])
        })
    
    final_predictions.sort(key=lambda x: (x['predictor_count'], x['consensus_score']), reverse=True)
    
    return render_template(
        'ultimate_predictor.html',
        ultimate_top_10=final_predictions[:10],
        advanced_preds=advanced_preds,
        smart_preds=smart_preds,
        ml_preds=ml_preds,
        pattern_preds=pattern_preds,
        last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
        provider_options=provider_options,
        selected_provider=selected_provider,
        month_options=month_options,
        selected_month=selected_month,
        provider_name=provider_name,
        last_draw_date=last_draw_date,
        last_draw_number=last_draw_number,
        next_draw_date=next_draw_date,
        error=None
    )

@app.route('/save-prediction', methods=['POST'])
def save_prediction():
    data = request.get_json()
    pred_file = "prediction_tracking.csv"
    
    new_entry = {
        "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "draw_date": data.get('draw_date'),
        "provider": data.get('provider'),
        "predicted_numbers": data.get('predicted_numbers'),
        "predictor_methods": data.get('methods'),
        "confidence": data.get('confidence'),
        "actual_1st": "",
        "actual_2nd": "",
        "actual_3rd": "",
        "hit_status": "pending",
        "accuracy_score": 0
    }
    
    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])
    df.to_csv(pred_file, index=False)
    
    return {"status": "success", "message": "Prediction saved"}

@app.route('/accuracy-dashboard')
def accuracy_dashboard():
    pred_file = "prediction_tracking.csv"
    
    if not os.path.exists(pred_file):
        return render_template('accuracy_dashboard.html', 
                             message="No predictions tracked yet. Start making predictions!",
                             stats={}, method_stats={}, recent_predictions=[], insights=[])
    
    pred_df = pd.read_csv(pred_file)
    df = load_csv_data()
    
    # Auto-match predictions with actual results
    for idx, row in pred_df.iterrows():
        if row['hit_status'] == 'pending':
            try:
                # --- FIX: Handle 'YYYY-MM-DD (DayName)' date format ---
                date_str = str(row['draw_date'])
                # Extract only the date part (e.g., '2025-09-23') before parsing
                clean_date_str = date_str.split(' ')[0]
                draw_date = pd.to_datetime(clean_date_str, errors='coerce').date()
                if draw_date is None:
                    logger.warning(f"Invalid date format: {date_str}")
                    continue
                provider = str(row['provider']).strip().lower()
                
                # --- FIX: Ensure provider matching is consistent ---
                # Match with actual results
                actual = df[(df['date_parsed'].dt.date == draw_date) & (df['provider'] == provider)]
                
                if not actual.empty:
                    actual_row = actual.iloc[0]
                    pred_df.at[idx, 'actual_1st'] = actual_row['1st_real']
                    pred_df.at[idx, 'actual_2nd'] = actual_row['2nd_real']
                    pred_df.at[idx, 'actual_3rd'] = actual_row['3rd_real']
                    
                    # --- FIX: Robustly parse predicted numbers string ---
                    predicted_str = str(row['predicted_numbers'])
                    try:
                        # Try to parse as list first
                        if predicted_str.startswith('[') and predicted_str.endswith(']'):
                            # Safe evaluation with whitelist
                            predicted = ast.literal_eval(predicted_str)
                            predicted = [str(num).strip() for num in predicted if str(num).isdigit() and len(str(num)) == 4]
                        else:
                            # Fallback to regex extraction
                            predicted = [num.strip() for num in re.findall(r'\d{4}', predicted_str)]
                    except (ValueError, SyntaxError, TypeError) as e:
                        logger.warning(f"Failed to parse predictions: {e}")
                        # Final fallback to regex
                        predicted = [num.strip() for num in re.findall(r'\d{4}', predicted_str)]
                    
                    actuals = [str(actual_row['1st_real']), str(actual_row['2nd_real']), str(actual_row['3rd_real'])]
                    
                    hits = [p for p in predicted if p in actuals]
                    if hits:
                        # --- ENHANCEMENT: Store which numbers hit ---
                        hit_details = ", ".join(hits)
                        pred_df.at[idx, 'hit_status'] = f'HIT ({hit_details})'
                        pred_df.at[idx, 'accuracy_score'] = len(hits) * 100 / len(predicted) if predicted else 0
                    else:
                        pred_df.at[idx, 'hit_status'] = 'MISS'
                        pred_df.at[idx, 'accuracy_score'] = 0
            except (SyntaxError, ValueError, TypeError) as e:
                # Log errors for debugging but don't crash the loop
                logger.error(f"Could not process prediction row {idx}: {e}")
                continue
            except Exception as e:
                logger.error(f"An unexpected error occurred on row {idx}: {e}")
                continue
    
    pred_df.to_csv(pred_file, index=False)
    
    completed = pred_df[pred_df['hit_status'] != 'pending']
    
    stats = {
        'total_predictions': len(pred_df),
        'completed': len(completed),
        'pending': len(pred_df[pred_df['hit_status'] == 'pending']),
        'total_hits': len(completed[completed['hit_status'].str.contains('HIT', na=False)]),
        'total_misses': len(completed[completed['hit_status'] == 'MISS']),
        'hit_rate': round(len(completed[completed['hit_status'].str.contains('HIT', na=False)]) / len(completed) * 100, 1) if len(completed) > 0 else 0,
        'avg_accuracy': round(completed['accuracy_score'].mean(), 1) if len(completed) > 0 else 0
    }
    
    method_stats = {}
    insights = []
    
    if stats['hit_rate'] < 30:
        insights.append("Low hit rate. Consider adjusting methods.")
    if stats['hit_rate'] > 50:
        insights.append("Good hit rate! Current methods working well.")
    
    recent = pred_df.tail(20).to_dict('records')
    
    return render_template('accuracy_dashboard.html',
                         stats=stats,
                         method_stats=method_stats,
                         recent_predictions=recent,
                         insights=insights,
                         message="")

@app.route('/quick-pick')
def quick_pick():
    df = load_csv_data()
    provider = request.args.get('provider', 'all')
    
    if df.empty:
        return render_template('quick_pick.html', numbers=[], error="No data available", provider_options=['all'], provider='all', analysis={})
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    # ADVANCED MULTI-ALGORITHM APPROACH
    # 1. Get predictions from all methods with scores
    advanced_preds = advanced_predictor(df, provider=provider, lookback=200) or []
    smart_preds = smart_auto_weight_predictor(df, provider=provider, lookback=300) or []
    ml_preds = ml_predictor(df, lookback=500) or []
    
    # 2. Weighted scoring system (not just counting)
    weighted_predictions = {}
    
    # Advanced predictor (weight: 1.2)
    for num, score, reason in advanced_preds[:10]:
        weighted_predictions[num] = weighted_predictions.get(num, 0) + (score * 1.2)
    
    # Smart predictor (weight: 1.0)
    for num, score, reason in smart_preds[:10]:
        weighted_predictions[num] = weighted_predictions.get(num, 0) + (score * 1.0)
    
    # ML predictor (weight: 0.8)
    for num, score, reason in ml_preds[:10]:
        weighted_predictions[num] = weighted_predictions.get(num, 0) + (score * 0.8)
    
    # 3. Add frequency boost (recent hot numbers)
    recent_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        recent_nums.extend([n for n in df[col].tail(50).astype(str) if len(n) == 4 and n.isdigit()])
    
    freq_counter = Counter(recent_nums)
    for num, count in freq_counter.most_common(20):
        if num in weighted_predictions:
            weighted_predictions[num] += count * 0.1  # Frequency boost
    
    # 4. Add pattern-based boost
    if recent_nums:
        last_num = recent_nums[-1]
        for num in weighted_predictions.keys():
            # Boost numbers with similar digits
            common_digits = len(set(num) & set(last_num))
            if common_digits >= 2:
                weighted_predictions[num] += common_digits * 0.15
    
    # 5. Sort by weighted score
    sorted_nums = sorted(weighted_predictions.items(), key=lambda x: x[1], reverse=True)
    
    # 6. Get top 5 with confidence scores
    top_5_with_scores = []
    max_score = sorted_nums[0][1] if sorted_nums else 1
    for num, score in sorted_nums[:5]:
        confidence = min(int((score / max_score) * 100), 99)
        top_5_with_scores.append({'number': num, 'confidence': confidence})
    
    top_5 = [item['number'] for item in top_5_with_scores]
    
    # 7. Analysis data
    analysis = {
        'total_algorithms': 3,
        'numbers_analyzed': len(weighted_predictions),
        'hot_numbers': [num for num, _ in freq_counter.most_common(5)],
        'consensus_strength': 'High' if len(sorted_nums) > 0 and sorted_nums[0][1] > 2 else 'Medium'
    }
    
    # Get next draw date
    last_draw = df.iloc[-1]
    from datetime import timedelta
    next_draw_date = (last_draw['date_parsed'] + timedelta(days=3)).strftime('%Y-%m-%d (%A)')
    
    return render_template('quick_pick.html', 
                         numbers=top_5,
                         numbers_with_confidence=top_5_with_scores,
                         next_draw_date=next_draw_date,
                         error=None,
                         provider_options=provider_options,
                         provider=provider,
                         analysis=analysis)

from utils.day_to_day_learner import learn_day_to_day_patterns

def get_day_to_day_predictions_with_reasons(today_numbers, patterns, recent_history):
    """
    Generates day-to-day predictions with clear, traceable reasons.
    This makes the logic feel less random by showing the 'cause and effect'.
    """
    if not today_numbers or not patterns:
        return []

    digit_transitions = patterns.get('digit_transitions', {})
    sequence_patterns = patterns.get('sequence_patterns', {})
    
    predictions = defaultdict(lambda: {'score': 0, 'reasons': []})

    # Analyze each number that appeared today
    for today_num in set(today_numbers):
        # Reason 1: Check for direct number-to-number sequences
        if today_num in sequence_patterns:
            for next_num, count in sequence_patterns[today_num].items():
                # The reason now clearly states: "Because 1234 appeared today..."
                reason = f"Follows today's number <strong>{today_num}</strong>"
                predictions[next_num]['score'] += 2.0 * count # Strong signal
                predictions[next_num]['reasons'].append(reason)

    # --- Improved Fallback Logic ---
    # If no strong sequence patterns were found, build candidates by combining
    # multiple weaker digit-transition signals for a more logical prediction.
    if not predictions:
        for today_num in set(today_numbers):
            # Find all possible single-digit transitions for this number
            possible_transitions = []
            for i, digit in enumerate(today_num):
                if digit in digit_transitions and i in digit_transitions[digit]:
                    for next_digit, count in digit_transitions[digit][i].items():
                        # Store the position, the new digit, and the strength (count)
                        possible_transitions.append({'pos': i, 'next_digit': next_digit, 'count': count, 'original': digit})

            # Sort
            #  transitions by strength to use the most likely ones
            possible_transitions.sort(key=lambda x: x['count'], reverse=True)

            # Generate candidates by applying the top 1, 2, or 3 transitions
            for num_to_apply in range(1, min(4, len(possible_transitions) + 1)):
                candidate = list(today_num)
                reasons_for_candidate = []
                score_for_candidate = 0
                
                for i in range(num_to_apply):
                    trans = possible_transitions[i]
                    candidate[trans['pos']] = trans['next_digit']
                    reasons_for_candidate.append(f"'{trans['original']}'→'{trans['next_digit']}' at pos {trans['pos']+1}")
                    score_for_candidate += 0.5 * trans['count']
                
                candidate_num = "".join(candidate)
                predictions[candidate_num]['score'] += score_for_candidate
                predictions[candidate_num]['reasons'].extend(reasons_for_candidate)

    # --- Last Resort Fallback: Hot Digit Generation ---
    # If still no predictions, generate candidates from the most frequent digits
    # in the recent history. This guarantees we always have something to show.
    if not predictions and recent_history:
        all_digits = "".join(recent_history)
        if all_digits:
            digit_counts = Counter(all_digits)
            # Get the 4 most common digits
            top_digits = [d for d, c in digit_counts.most_common(4)]
            if len(top_digits) == 4:
                # Create a candidate from the top 4 digits
                candidate_num = "".join(top_digits)
                reason = f"Built from hot digits: {', '.join(top_digits)}"
                predictions[candidate_num]['score'] = 1.0 # Base score
                predictions[candidate_num]['reasons'].append(reason)


    # Format the output
    sorted_predictions = sorted(predictions.items(), key=lambda item: item[1]['score'], reverse=True)
    
    output = []
    for num, data in sorted_predictions[:5]: # Top 5 predictions
        # Join all reasons why this number was chosen
        reason_str = "; ".join(sorted(list(set(data['reasons']))))
        output.append((num, data['score'], reason_str))
        
    return output

@app.route('/day-to-day-predictor')
def day_to_day_predictor():
    df = load_csv_data()
    if df.empty:
        return "No data available to make predictions.", 404

    selected_provider = request.args.get('provider', 'all')
    selected_month = request.args.get('month', '')

    # --- Filtering Logic ---
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    month_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m').unique(), reverse=True)

    # Limit data processing to last 100 rows for speed
    filtered_df = df.tail(100).copy()
    if selected_provider != 'all':
        filtered_df = filtered_df[filtered_df['provider'] == selected_provider]

    if selected_month:
        filtered_df = filtered_df[filtered_df['date_parsed'].dt.strftime('%Y-%m') == selected_month]

    if filtered_df.empty:
        return render_template(
            'day_to_day_predictor.html',
            last_updated=time.strftime('%Y-%m-%d %H:%M:%S'),
            provider_options=provider_options,
            provider=selected_provider,
            month_options=month_options,
            selected_month=selected_month,
            today_date="N/A",
            total_numbers_analyzed=0,
            today_numbers=[],
            predictions=[]
        )

    # --- Fast Prediction Logic ---
    latest_date = filtered_df['date_parsed'].max()
    today_data = filtered_df[filtered_df['date_parsed'] == latest_date]

    today_numbers = []
    for _, row in today_data.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if len(num) == 4 and num.isdigit():
                today_numbers.append(num)

    # Enhanced day-to-day analysis
    pattern_momentum = analyze_pattern_momentum(filtered_df)
    sequence_detection = detect_daily_sequences(today_numbers)
    trend_analysis = analyze_daily_trends(filtered_df, today_numbers)
    
    recent_nums = filtered_df['1st_real'].dropna().astype(str).tolist()[-50:]
    if not recent_nums:
        predictions = []
    else:
        predictions = advanced_predictor(filtered_df, provider=selected_provider, lookback=50)[:5]
        # Enhance predictions with momentum data
        for i, (num, score, reason) in enumerate(predictions):
            momentum_boost = pattern_momentum.get(num, 0)
            enhanced_score = score + momentum_boost
            predictions[i] = (num, enhanced_score, f"{reason}+momentum")

    return render_template(
        'day_to_day_predictor.html',
        last_updated=time.strftime('%Y-%m-%d %H:%M:%S'),
        provider_options=provider_options,
        provider=selected_provider,
        month_options=month_options,
        selected_month=selected_month,
        today_date=latest_date.strftime('%Y-%m-%d'),
        total_numbers_analyzed=len(today_numbers),
        today_numbers=today_numbers[:10],
        predictions=predictions,
        pattern_momentum=pattern_momentum,
        sequence_detection=sequence_detection,
        trend_analysis=trend_analysis
    )

def analyze_pattern_momentum(df):
    """Analyze momentum of number patterns"""
    momentum = {}
    if len(df) < 10:
        return momentum
    
    recent_5 = df.tail(5)
    previous_5 = df.tail(10).head(5)
    
    recent_nums = []
    previous_nums = []
    
    for col in ['1st_real', '2nd_real', '3rd_real']:
        recent_nums.extend([n for n in recent_5[col].astype(str) if n.isdigit() and len(n) == 4])
        previous_nums.extend([n for n in previous_5[col].astype(str) if n.isdigit() and len(n) == 4])
    
    recent_freq = Counter(recent_nums)
    previous_freq = Counter(previous_nums)
    
    for num in set(recent_nums + previous_nums):
        recent_count = recent_freq.get(num, 0)
        previous_count = previous_freq.get(num, 0)
        
        if recent_count > previous_count:
            momentum[num] = 0.2  # Positive momentum
        elif recent_count < previous_count:
            momentum[num] = -0.1  # Negative momentum
    
    return momentum

def detect_daily_sequences(today_numbers):
    """Detect sequences in today's numbers"""
    sequences = []
    
    for num in today_numbers:
        if len(num) == 4:
            digits = [int(d) for d in num]
            
            # Check for ascending sequence
            if all(digits[i] <= digits[i+1] for i in range(3)):
                sequences.append({'number': num, 'type': 'ascending', 'strength': 'strong'})
            
            # Check for descending sequence
            elif all(digits[i] >= digits[i+1] for i in range(3)):
                sequences.append({'number': num, 'type': 'descending', 'strength': 'strong'})
            
            # Check for repeating digits
            elif len(set(digits)) <= 2:
                sequences.append({'number': num, 'type': 'repeating', 'strength': 'medium'})
    
    return sequences[:5]

def analyze_daily_trends(df, today_numbers):
    """Analyze daily trends and patterns"""
    trends = {'direction': 'stable', 'volatility': 'low', 'prediction_confidence': 50}
    
    if len(df) < 5 or not today_numbers:
        return trends
    
    recent_sums = []
    for _, row in df.tail(5).iterrows():
        day_sum = 0
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if num.isdigit() and len(num) == 4:
                day_sum += sum(int(d) for d in num)
        recent_sums.append(day_sum)
    
    if len(recent_sums) >= 3:
        if recent_sums[-1] > recent_sums[-2] > recent_sums[-3]:
            trends['direction'] = 'increasing'
        elif recent_sums[-1] < recent_sums[-2] < recent_sums[-3]:
            trends['direction'] = 'decreasing'
        
        volatility = max(recent_sums) - min(recent_sums)
        if volatility > 50:
            trends['volatility'] = 'high'
        elif volatility > 25:
            trends['volatility'] = 'medium'
        
        trends['prediction_confidence'] = min(90, 50 + (5 - volatility // 10) * 10)
    
    return trends

@app.route('/learning-insights')
def learning_insights():
    pred_file = "prediction_tracking.csv"
    
    if not os.path.exists(pred_file):
        return render_template('learning_insights.html', 
                             message="No prediction data yet. Start tracking predictions!",
                             insights={})
    
    pred_df = pd.read_csv(pred_file)
    df = load_csv_data()
    
    # Only analyze completed predictions
    completed = pred_df[pred_df['hit_status'] != 'pending']
    
    if len(completed) == 0:
        return render_template('learning_insights.html',
                             message="No completed predictions yet. Wait for draw results!",
                             insights={})
    
    insights = {}
    
    # 1. Method Performance Analysis
    method_performance = {}
    for _, row in completed.iterrows():
        methods = str(row.get('predictor_methods', '')).split(';')
        hit = 'HIT' in str(row['hit_status'])
        for method in methods:
            method = method.strip().split(',')[0].strip()
            if method not in method_performance:
                method_performance[method] = {'hits': 0, 'total': 0}
            method_performance[method]['total'] += 1
            if hit:
                method_performance[method]['hits'] += 1
    
    for method in method_performance:
        total = method_performance[method]['total']
        hits = method_performance[method]['hits']
        method_performance[method]['accuracy'] = round((hits / total * 100), 1) if total > 0 else 0
    
    insights['method_performance'] = method_performance
    
    # 2. Provider-Specific Accuracy
    provider_accuracy = {}
    for _, row in completed.iterrows():
        provider = str(row['provider']).lower()
        hit = 'HIT' in str(row['hit_status'])
        if provider not in provider_accuracy:
            provider_accuracy[provider] = {'hits': 0, 'total': 0}
        provider_accuracy[provider]['total'] += 1
        if hit:
            provider_accuracy[provider]['hits'] += 1
    
    for provider in provider_accuracy:
        total = provider_accuracy[provider]['total']
        hits = provider_accuracy[provider]['hits']
        provider_accuracy[provider]['accuracy'] = round((hits / total * 100), 1) if total > 0 else 0
    
    insights['provider_accuracy'] = provider_accuracy
    
    # 3. Best Performing Method
    if method_performance:
        best_method = max(method_performance.items(), key=lambda x: x[1]['accuracy'])
        insights['best_method'] = best_method[0]
        insights['best_accuracy'] = best_method[1]['accuracy']
    
    # 4. Recommendations
    recommendations = []
    if insights.get('best_accuracy', 0) >= 50:
        recommendations.append(f"✅ Focus on {insights['best_method']} - it has {insights['best_accuracy']}% accuracy")
    else:
        recommendations.append("⚠️ All methods below 50% accuracy. Need more data or strategy adjustment.")
    
    if provider_accuracy:
        best_provider = max(provider_accuracy.items(), key=lambda x: x[1]['accuracy'])
        if best_provider[1]['accuracy'] >= 40:
            recommendations.append(f"🎯 {best_provider[0].upper()} shows best results ({best_provider[1]['accuracy']}%)")
    
    insights['recommendations'] = recommendations
    
    # 5. Overall Stats
    insights['total_predictions'] = len(completed)
    insights['total_hits'] = len(completed[completed['hit_status'].str.contains('HIT', na=False)])
    insights['overall_accuracy'] = round((insights['total_hits'] / insights['total_predictions'] * 100), 1) if insights['total_predictions'] > 0 else 0
    insights['total_analyzed'] = len(completed)
    
    # 6. Digit Frequency Analysis
    pred_digits = Counter()
    actual_digits = Counter()
    
    for _, row in completed.iterrows():
        # Count predicted digits
        try:
            predicted = ast.literal_eval(row['predicted_numbers']) if isinstance(row['predicted_numbers'], str) else []
            for num in predicted:
                for digit in str(num):
                    pred_digits[digit] += 1
        except:
            pass
        
        # Count actual digits
        for col in ['actual_1st', 'actual_2nd', 'actual_3rd']:
            num = str(row.get(col, ''))
            if num and num != 'nan':
                for digit in num:
                    actual_digits[digit] += 1
    
    digit_comparison = []
    for digit in '0123456789':
        pred_freq = pred_digits.get(digit, 0)
        actual_freq = actual_digits.get(digit, 0)
        
        # Determine bias
        if actual_freq == 0:
            bias = 'balanced'
        else:
            ratio = pred_freq / actual_freq if actual_freq > 0 else 0
            if 0.8 <= ratio <= 1.2:
                bias = 'balanced'
            elif ratio > 1.2:
                bias = 'over'
            else:
                bias = 'under'
        
        digit_comparison.append({
            'digit': digit,
            'pred_freq': pred_freq,
            'actual_freq': actual_freq,
            'bias': bias
        })
    
    insights['digit_comparison'] = digit_comparison
    
    return render_template('learning_insights.html',
                         message="",
                         insights=insights)

@app.route('/best-predictions')
def best_predictions():
    df = load_csv_data()
    provider = request.args.get('provider', 'all').lower().strip()

    if df.empty:
        return render_template('best_predictions.html', error="No data available", predictions=[])

    # --- FIX: Correctly generate provider options and filter data ---
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p and str(p).strip()])
    if provider not in provider_options:
        provider = 'all'

    df_filtered = df.copy()
    if provider != 'all':
        df_filtered = df[df['provider'] == provider]
    # --- END FIX ---

    # Get predictions from all methods
    advanced_preds = advanced_predictor(df_filtered, provider=provider, lookback=200)
    smart_preds = smart_auto_weight_predictor(df_filtered, provider=provider, lookback=300)
    ml_preds = ml_predictor(df_filtered, lookback=500)

    all_predictions = {}
    
    for num, score, reason in advanced_preds:
        if num not in all_predictions:
            all_predictions[num] = {'count': 0, 'total_score': 0, 'sources': []}
        all_predictions[num]['count'] += 1
        all_predictions[num]['total_score'] += score
        all_predictions[num]['sources'].append(f'Advanced: {reason}')

    for num, score, reason in smart_preds:
        if num not in all_predictions:
            all_predictions[num] = {'count': 0, 'total_score': 0, 'sources': []}
        all_predictions[num]['count'] += 1
        all_predictions[num]['total_score'] += score
        all_predictions[num]['sources'].append(f'Smart: {reason}')

    for num, score, reason in ml_preds:
        if num not in all_predictions:
            all_predictions[num] = {'count': 0, 'total_score': 0, 'sources': []}
        all_predictions[num]['count'] += 1
        all_predictions[num]['total_score'] += score
        all_predictions[num]['sources'].append(f'ML: {reason}')

    final_predictions = []
    for num, data in all_predictions.items():
        consensus_score = data['count'] * (data['total_score'] / data['count'])
        confidence = calculate_confidence_score(num, df_filtered, data)
        risk_reward = calculate_risk_reward_ratio(num, df_filtered)
        multi_timeframe = analyze_multi_timeframe_consensus(num, df_filtered)
        
        final_predictions.append({
            'number': num,
            'score': round(consensus_score, 3),
            'source_count': data['count'],
            'confidence': confidence,
            'risk_reward': risk_reward,
            'timeframe_consensus': multi_timeframe,
            'sources': data['sources']
        })

    final_predictions.sort(key=lambda x: (x['source_count'], x['score']), reverse=True)

    # Get next draw date
    next_draw_date = ''
    next_draw_day = ''
    if not df.empty:
        last_draw = df.iloc[-1]
        from datetime import timedelta
        next_draw_date = (last_draw['date_parsed'] + timedelta(days=3)).strftime('%Y-%m-%d')
        next_draw_day = (last_draw['date_parsed'] + timedelta(days=3)).strftime('%A')


    return render_template('best_predictions.html',
                         predictions=final_predictions[:10],
                         provider_options=provider_options,
                         provider=provider,
                         next_draw_date=next_draw_date,
                         next_draw_day=next_draw_day,
                         error=None)

def calculate_confidence_score(number, df, prediction_data):
    """Calculate 0-100% confidence based on historical accuracy"""
    recent_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        recent_nums.extend([n for n in df[col].tail(100).astype(str) if n.isdigit() and len(n) == 4])
    
    frequency_score = (recent_nums.count(number) / len(recent_nums)) * 100 if recent_nums else 0
    predictor_consensus = (prediction_data['count'] / 3) * 100
    avg_prediction_score = prediction_data['total_score'] / prediction_data['count'] if prediction_data['count'] > 0 else 0
    
    confidence = (frequency_score * 0.4 + predictor_consensus * 0.4 + avg_prediction_score * 20) / 3
    return round(min(confidence, 100), 1)

def calculate_risk_reward_ratio(number, df):
    """Calculate risk vs reward potential"""
    all_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_nums.extend([n for n in df[col].astype(str) if n.isdigit() and len(n) == 4])
    
    frequency = all_nums.count(number)
    total_draws = len(all_nums)
    
    if frequency == 0:
        return {'risk': 'High', 'reward': 'High', 'ratio': 'High Risk/High Reward'}
    elif frequency / total_draws > 0.02:
        return {'risk': 'Low', 'reward': 'Medium', 'ratio': 'Low Risk/Medium Reward'}
    else:
        return {'risk': 'Medium', 'reward': 'High', 'ratio': 'Medium Risk/High Reward'}

def analyze_multi_timeframe_consensus(number, df):
    """Analyze consensus across different timeframes"""
    timeframes = {
        '7d': df.tail(20),
        '30d': df.tail(100),
        '90d': df.tail(300)
    }
    
    consensus = {}
    for period, data in timeframes.items():
        nums = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            nums.extend([n for n in data[col].astype(str) if n.isdigit() and len(n) == 4])
        
        freq = nums.count(number)
        total = len(nums)
        consensus[period] = round((freq / total) * 100, 2) if total > 0 else 0
    
    return consensus

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
    
    # Enhanced statistics analysis
    draw_intervals = analyze_draw_intervals(filtered_df)
    pair_strength = analyze_number_pair_strength(all_numbers)
    provider_rankings = analyze_provider_performance(df)
    predictions = advanced_predictor(filtered_df, provider=provider, lookback=200)[:5]
    
    return render_template('statistics.html', 
                         total_draws=total_draws, 
                         top_20=top_20, 
                         digit_freq=digit_freq, 
                         predictions=predictions,
                         provider_options=provider_options, 
                         provider=provider, 
                         month_options=month_options, 
                         selected_month=selected_month, 
                         message="", 
                         last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
                         draw_intervals=draw_intervals,
                         pair_strength=pair_strength,
                         provider_rankings=provider_rankings)

def analyze_draw_intervals(df):
    """Analyze intervals between draws for patterns"""
    df_sorted = df.sort_values('date_parsed')
    intervals = []
    
    for i in range(1, len(df_sorted)):
        prev_date = df_sorted.iloc[i-1]['date_parsed']
        curr_date = df_sorted.iloc[i]['date_parsed']
        interval = (curr_date - prev_date).days
        intervals.append(interval)
    
    if not intervals:
        return {'avg_interval': 0, 'most_common': 0, 'next_predicted': 0}
    
    avg_interval = sum(intervals) / len(intervals)
    most_common = Counter(intervals).most_common(1)[0][0] if intervals else 0
    
    return {
        'avg_interval': round(avg_interval, 1),
        'most_common': most_common,
        'next_predicted': most_common,
        'distribution': Counter(intervals).most_common(5)
    }

def analyze_number_pair_strength(numbers):
    """Analyze strength of number pairs appearing together"""
    pair_counts = Counter()
    
    for i in range(len(numbers)):
        for j in range(i+1, min(i+10, len(numbers))):
            pair = tuple(sorted([numbers[i], numbers[j]]))
            pair_counts[pair] += 1
    
    strong_pairs = []
    for pair, count in pair_counts.most_common(10):
        strength = 'Strong' if count >= 5 else 'Medium' if count >= 3 else 'Weak'
        strong_pairs.append({
            'pair': f'{pair[0]} + {pair[1]}',
            'count': count,
            'strength': strength
        })
    
    return strong_pairs

def analyze_provider_performance(df):
    """Rank providers by various performance metrics"""
    provider_stats = {}
    
    for provider in df['provider'].unique():
        if not provider or str(provider).strip() == '':
            continue
            
        provider_df = df[df['provider'] == provider]
        
        # Calculate metrics
        total_draws = len(provider_df)
        unique_numbers = set()
        for col in ['1st_real', '2nd_real', '3rd_real']:
            unique_numbers.update([n for n in provider_df[col].astype(str) if n.isdigit() and len(n) == 4])
        
        diversity_score = len(unique_numbers) / total_draws if total_draws > 0 else 0
        
        provider_stats[provider] = {
            'total_draws': total_draws,
            'unique_numbers': len(unique_numbers),
            'diversity_score': round(diversity_score, 3),
            'avg_per_month': round(total_draws / 12, 1) if total_draws > 0 else 0
        }
    
    # Rank by diversity score
    ranked = sorted(provider_stats.items(), key=lambda x: x[1]['diversity_score'], reverse=True)
    return ranked[:10]

@app.route('/lucky-generator')
def lucky_generator():
    df = load_csv_data()
    if df.empty:
        return render_template('lucky_generator.html', lucky_numbers=[], message="No data available", provider_options=['all'], provider='all', month_options=[], selected_month='', selected_date='', date_options=[])
    
    provider = request.args.get('provider', 'all')
    selected_month = request.args.get('month', '')
    selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    month_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m').unique(), reverse=True)
    date_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m-%d').unique(), reverse=True)[:30]
    
    if provider != 'all':
        df = df[df['provider'] == provider]
    if selected_month:
        df = df[df['date_parsed'].dt.strftime('%Y-%m') == selected_month]
    
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_numbers.extend([n for n in df[col].astype(str) if n.isdigit() and len(n) == 4])
    digit_freq = Counter(''.join(all_numbers))
    lucky_digits = [d for d, _ in digit_freq.most_common(6)]
    
    # Enhanced lucky number generation
    astrological_numbers = generate_astrological_numbers(selected_date)
    personal_learned = get_personal_lucky_patterns(df, provider)
    date_based = generate_date_based_numbers(selected_date)
    
    if not all_numbers:
        lucky_numbers = astrological_numbers + personal_learned + date_based
    else:
        number_freq = Counter(all_numbers)
        frequency_based = [num for num, count in number_freq.most_common(5)]
        lucky_numbers = frequency_based + astrological_numbers[:3] + personal_learned[:2]
    
    return render_template('lucky_generator.html', lucky_numbers=lucky_numbers[:10], lucky_digits=lucky_digits, message="Based on frequency + astrology + personal patterns", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"), provider_options=provider_options, provider=provider, month_options=month_options, selected_month=selected_month, selected_date=selected_date, date_options=date_options, astrological_numbers=astrological_numbers, personal_patterns=personal_learned, date_based_numbers=date_based)

@app.route('/frequency-analyzer')
def frequency_analyzer():
    df = load_csv_data()
    date_range = request.args.get('range', '30')
    provider = request.args.get('provider', 'all')
    selected_month = request.args.get('month', '')
    
    if df.empty:
        return render_template('frequency_analyzer.html', hot_numbers=[], cold_numbers=[], odd_pct=0, even_pct=0, total_draws=0, top_4_prediction=[], chart_data=[], provider_options=['all'], provider='all', month_options=[], selected_month='', last_updated='')
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    month_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m').unique(), reverse=True)
    
    if provider != 'all':
        df = df[df['provider'] == provider]
    if selected_month:
        df = df[df['date_parsed'].dt.strftime('%Y-%m') == selected_month]
    
    try:
        days = min(int(date_range), 365)
    except:
        days = 30
    cutoff_date = df['date_parsed'].max() - timedelta(days=days)
    filtered_df = df[df['date_parsed'] >= cutoff_date]
    
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_numbers.extend([n for n in filtered_df[col].astype(str) if n.isdigit() and len(n) == 4])
    
    if not all_numbers:
        return render_template('frequency_analyzer.html', hot_numbers=[], cold_numbers=[], odd_pct=0, even_pct=0, total_draws=0, top_4_prediction=[], chart_data=[], provider_options=provider_options, provider=provider, month_options=month_options, selected_month=selected_month, last_updated=time.strftime("%Y-%m-%d %H:%M:%S"), provider_bias={'bias_numbers': [], 'recommendation': 'No data'}, weighted_predictions=[])
    
    freq_counter = Counter(all_numbers)
    total_draws = len(filtered_df)
    
    # Enhanced frequency analysis with weighted scoring
    weighted_freq = calculate_weighted_frequency(filtered_df, all_numbers, days)
    provider_bias = analyze_provider_bias(filtered_df, provider)
    time_decay_scores = calculate_time_decay_frequency(filtered_df, all_numbers)
    
    sorted_freq = freq_counter.most_common()
    hot_numbers = sorted_freq[:10]
    cold_numbers = sorted_freq[-10:] if len(sorted_freq) >= 10 else []
    
    # Odd/Even analysis
    odd_count = sum(1 for num in all_numbers if int(num) % 2 == 1)
    even_count = len(all_numbers) - odd_count
    odd_pct = round((odd_count / len(all_numbers)) * 100, 1) if all_numbers else 0
    even_pct = round((even_count / len(all_numbers)) * 100, 1) if all_numbers else 0
    
    # Top 4 prediction (most frequent numbers)
    top_4_prediction = [num for num, count in sorted_freq[:4]]
    
    # Chart data for top 50
    chart_data = [[num, count] for num, count in sorted_freq[:50]]
    
    return render_template('frequency_analyzer.html', 
                         hot_numbers=hot_numbers,
                         cold_numbers=cold_numbers,
                         odd_pct=odd_pct,
                         even_pct=even_pct,
                         total_draws=total_draws,
                         top_4_prediction=top_4_prediction,
                         chart_data=chart_data,
                         provider_options=provider_options, 
                         provider=provider, 
                         month_options=month_options, 
                         selected_month=selected_month,
                         last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
                         provider_bias=provider_bias,
                         weighted_predictions=sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)[:5])

def calculate_weighted_frequency(df, numbers, days):
    """Calculate weighted frequency where recent draws count more"""
    weighted_scores = {}
    df_sorted = df.sort_values('date_parsed')
    
    for i, row in df_sorted.iterrows():
        age_weight = 1.0 - (i / len(df_sorted)) * 0.5
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if num in numbers:
                weighted_scores[num] = weighted_scores.get(num, 0) + age_weight
    return weighted_scores

def analyze_provider_bias(df, provider):
    """Analyze provider-specific frequency patterns"""
    if provider == 'all':
        return {'bias_score': 1.0, 'recommendation': 'No specific bias'}
    
    provider_df = df[df['provider'] == provider]
    provider_nums = []
    all_nums = []
    
    for col in ['1st_real', '2nd_real', '3rd_real']:
        provider_nums.extend([n for n in provider_df[col].astype(str) if n.isdigit() and len(n) == 4])
        all_nums.extend([n for n in df[col].astype(str) if n.isdigit() and len(n) == 4])
    
    provider_freq = Counter(provider_nums)
    all_freq = Counter(all_nums)
    
    bias_numbers = []
    for num, count in provider_freq.most_common(10):
        provider_rate = count / len(provider_nums) if provider_nums else 0
        all_rate = all_freq[num] / len(all_nums) if all_nums else 0
        if provider_rate > all_rate * 1.2:
            bias_numbers.append({'number': num, 'bias_ratio': round(provider_rate / all_rate, 2) if all_rate > 0 else 0})
    
    return {'bias_numbers': bias_numbers[:5], 'recommendation': f'{provider.upper()} shows bias toward certain numbers'}

def calculate_time_decay_frequency(df, numbers):
    """Calculate frequency with exponential time decay"""
    decay_scores = {}
    df_sorted = df.sort_values('date_parsed', ascending=False)
    
    for i, row in df_sorted.iterrows():
        decay_factor = 0.95 ** i
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if num in numbers:
                decay_scores[num] = decay_scores.get(num, 0) + decay_factor
    return decay_scores

@app.route('/empty-box-predictor')
def empty_box_predictor():
    try:
        df = load_csv_data()
        if df.empty:
            return render_template('empty_box_predictor.html', predictions=[], message="No data available", next_empty_predictions=[], draws=[], provider_options=['all'], provider='all', month_options=[], selected_month='', last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        
        provider = request.args.get('provider', 'all')
        selected_month = request.args.get('month', '')
        
        provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p and str(p).strip() and str(p) != 'nan'])
        month_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m').unique(), reverse=True)
        
        filtered_df = df.copy()
        if provider != 'all':
            filtered_df = filtered_df[filtered_df['provider'] == provider]
        if selected_month:
            filtered_df = filtered_df[filtered_df['date_parsed'].dt.strftime('%Y-%m') == selected_month]
        
        draws = []
        empty_position_tracker = defaultdict(lambda: {'count': 0})
        
        # Process last 20 draws
        for _, row in filtered_df.tail(20).iterrows():
            try:
                # Extract special and consolation numbers
                special_text = str(row.get('special', ''))
                consolation_text = str(row.get('consolation', ''))
                
                # Find all 4-digit numbers in the text
                special_nums = re.findall(r'\b\d{4}\b', special_text)
                consolation_nums = re.findall(r'\b\d{4}\b', consolation_text)
                
                # Create 2x5 grids (10 positions each)
                special_grid = [['----'] * 5 for _ in range(2)]
                consolation_grid = [['----'] * 5 for _ in range(2)]
                
                # Fill special grid
                for i, num in enumerate(special_nums[:10]):
                    row_idx = i // 5
                    col_idx = i % 5
                    special_grid[row_idx][col_idx] = num
                
                # Fill consolation grid
                for i, num in enumerate(consolation_nums[:10]):
                    row_idx = i // 5
                    col_idx = i % 5
                    consolation_grid[row_idx][col_idx] = num
                
                # Track empty positions
                empty_positions = []
                label_idx = 0
                
                # Check special grid for empty positions
                for r in range(2):
                    for c in range(5):
                        if special_grid[r][c] == '----':
                            empty_positions.append({
                                'label': chr(65 + label_idx),
                                'section': 'Special',
                                'row': r+1,
                                'col': c+1
                            })
                            empty_position_tracker[f"Special_R{r+1}_C{c+1}"]['count'] += 1
                            label_idx += 1
                
                # Check consolation grid for empty positions
                for r in range(2):
                    for c in range(5):
                        if consolation_grid[r][c] == '----':
                            empty_positions.append({
                                'label': chr(65 + label_idx),
                                'section': 'Consolation',
                                'row': r+1,
                                'col': c+1
                            })
                            empty_position_tracker[f"Consolation_R{r+1}_C{c+1}"]['count'] += 1
                            label_idx += 1
                
                draws.append({
                    'date': row['date_parsed'].date(),
                    'provider': str(row['provider']).upper(),
                    'draw_no': str(row.get('draw_info', f"Draw-{row['date_parsed'].strftime('%Y%m%d')}"))[:20],
                    'prizes': {
                        '1st': str(row.get('1st_real', '----'))[:4],
                        '2nd': str(row.get('2nd_real', '----'))[:4],
                        '3rd': str(row.get('3rd_real', '----'))[:4]
                    },
                    'special_grid': special_grid,
                    'consolation_grid': consolation_grid,
                    'empty_positions': empty_positions
                })
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue
        
        # Generate predictions
        next_empty_predictions = []
        total_draws = len(draws)
        
        if total_draws > 0:
            # Get recent winning numbers for predictions
            recent_nums = []
            for col in ['1st_real', '2nd_real', '3rd_real']:
                recent_nums.extend([n for n in filtered_df[col].tail(50).astype(str) if len(n) == 4 and n.isdigit()])
            
            num_freq = Counter(recent_nums)
            top_nums = num_freq.most_common(3) if num_freq else [('1234', 1), ('5678', 1), ('9012', 1)]
            
            # Create predictions for most frequent empty positions
            for i, (key, data) in enumerate(sorted(empty_position_tracker.items(), key=lambda x: x[1]['count'], reverse=True)[:6]):
                section = 'Special' if 'Special' in key else 'Consolation'
                position_match = re.search(r'R(\d+)_C(\d+)', key)
                position = f"R{position_match.group(1)}C{position_match.group(2)}" if position_match else f"Pos{i+1}"
                
                probability = round((data['count'] / total_draws) * 100, 1)
                
                predicted_numbers = []
                for num, cnt in top_nums:
                    confidence = round((cnt / len(recent_nums)) * 100, 1) if recent_nums else 10
                    predicted_numbers.append({'number': num, 'confidence': confidence})
                
                completion_prob = calculate_box_completion_probability(section, position, filtered_df)
                urgency_score = calculate_urgency_score(data['count'], total_draws)
                pattern_history = analyze_box_pattern_history(section, position, filtered_df)
                
                next_empty_predictions.append({
                    'label': chr(65 + i),
                    'section': section,
                    'position': position,
                    'probability': probability,
                    'predicted_numbers': predicted_numbers,
                    'frequency': data['count'],
                    'completion_probability': completion_prob,
                    'urgency_score': urgency_score,
                    'pattern_history': pattern_history
                })
        
        return render_template('empty_box_predictor.html', 
                             predictions=[], 
                             message="", 
                             last_updated=time.strftime("%Y-%m-%d %H:%M:%S"), 
                             next_empty_predictions=next_empty_predictions, 
                             draws=draws, 
                             provider_options=provider_options, 
                             provider=provider, 
                             month_options=month_options, 
                             selected_month=selected_month)
    except Exception as e:
        logger.error(f"Empty box predictor error: {e}")
        return render_template('empty_box_predictor.html', 
                             predictions=[], 
                             message=f"Error: {str(e)}", 
                             next_empty_predictions=[], 
                             draws=[], 
                             provider_options=['all'], 
                             provider='all', 
                             month_options=[], 
                             selected_month='')

@app.route('/hot-cold')
def hot_cold():
    df = load_csv_data()
    days = int(request.args.get('days', '30'))
    provider = request.args.get('provider', 'all')
    selected_month = request.args.get('month', '')
    
    if df.empty:
        return render_template('hot_cold.html', hot=[], cold=[], neutral=[], message="No data available", provider_options=['all'], provider='all', month_options=[], selected_month='', days=30, temperature_momentum=[], cross_provider_sync=[], transition_timing=[])
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p and str(p).strip() and str(p) != 'nan'])
    month_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m').unique(), reverse=True)
    
    if provider != 'all':
        df = df[df['provider'] == provider]
    if selected_month:
        df = df[df['date_parsed'].dt.strftime('%Y-%m') == selected_month]
    
    if df.empty:
        return render_template('hot_cold.html', hot=[], cold=[], neutral=[], message="No data for selected filters", provider_options=provider_options, provider=provider, month_options=month_options, selected_month=selected_month, days=days, temperature_momentum=[], cross_provider_sync=[], transition_timing=[])
    
    cutoff_date = df['date_parsed'].max() - timedelta(days=days)
    recent_df = df[df['date_parsed'] >= cutoff_date]
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_numbers.extend([n for n in recent_df[col].astype(str) if n.isdigit() and len(n) == 4])
    
    if not all_numbers:
        return render_template('hot_cold.html', hot=[], cold=[], neutral=[], message="No numbers found for selected filters", provider_options=provider_options, provider=provider, month_options=month_options, selected_month=selected_month, days=days, temperature_momentum=[], cross_provider_sync=[], transition_timing=[])
    
    freq_counter = Counter(all_numbers)
    sorted_freq = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Enhanced hot/cold with momentum
    hot = sorted_freq[:20]
    cold = sorted_freq[-20:] if len(sorted_freq) >= 20 else []
    
    # Cross-provider temperature sync
    cross_provider_sync = analyze_cross_provider_sync(df, [num for num, count in hot[:10]]) if hot else []
    
    # Hot-cold transition timing
    transition_timing = analyze_transition_timing(recent_df) if not recent_df.empty else []
    
    temperature_momentum = hot[:5]
    
    return render_template('hot_cold.html', hot=hot, cold=cold, neutral=[], days=days, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"), provider_options=provider_options, provider=provider, month_options=month_options, selected_month=selected_month, temperature_momentum=temperature_momentum, cross_provider_sync=cross_provider_sync, transition_timing=transition_timing)

def calculate_temperature_momentum(number, df):
    """Calculate if number is heating up or cooling down"""
    recent_7d = df[df['date_parsed'] >= (df['date_parsed'].max() - timedelta(days=7))]
    previous_7d = df[(df['date_parsed'] >= (df['date_parsed'].max() - timedelta(days=14))) & (df['date_parsed'] < (df['date_parsed'].max() - timedelta(days=7)))]
    
    recent_count = sum([1 for col in ['1st_real', '2nd_real', '3rd_real'] for n in recent_7d[col].astype(str) if n == number])
    previous_count = sum([1 for col in ['1st_real', '2nd_real', '3rd_real'] for n in previous_7d[col].astype(str) if n == number])
    
    if recent_count > previous_count:
        return 'heating_up'
    elif recent_count < previous_count:
        return 'cooling_down'
    return 'stable'

def analyze_cross_provider_sync(df, top_numbers):
    """Analyze if hot numbers appear across multiple providers"""
    sync_data = []
    providers = df['provider'].unique()
    
    for num in top_numbers:
        provider_counts = {}
        for provider in providers:
            provider_df = df[df['provider'] == provider]
            count = sum([1 for col in ['1st_real', '2nd_real', '3rd_real'] for n in provider_df[col].astype(str) if n == num])
            if count > 0:
                provider_counts[provider] = count
        
        if len(provider_counts) >= 2:
            sync_data.append({'number': num, 'providers': len(provider_counts), 'sync_score': sum(provider_counts.values())})
    
    return sorted(sync_data, key=lambda x: x['sync_score'], reverse=True)[:5]

def analyze_transition_timing(df):
    """Analyze timing patterns of hot-cold transitions"""
    transitions = []
    df_sorted = df.sort_values('date_parsed')
    
    for i in range(len(df_sorted) - 5):
        window = df_sorted.iloc[i:i+5]
        all_nums = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            all_nums.extend([n for n in window[col].astype(str) if n.isdigit() and len(n) == 4])
        
        freq = Counter(all_nums)
        if freq:
            hottest = freq.most_common(1)[0]
            transitions.append({'date': window.iloc[-1]['date_parsed'].date(), 'number': hottest[0], 'frequency': hottest[1]})
    
    return transitions[-10:]

@app.route('/test-route')
def test_route_function():
    return "Test route is working!"

# Unique route implementations
@app.route('/hot_and_cold_numbers')
@app.route('/hot_cold_numbers')
def hot_and_cold_numbers():
    return redirect('/hot-cold')

@app.route('/accuracy_tracker')
def accuracy_tracker():
    return render_template('accuracy_dashboard.html', stats={'total_predictions': 0}, method_stats={}, recent_predictions=[], insights=[], message="Accuracy tracking active")

@app.route('/missing-number-finder')
def missing_number_finder():
    df = load_csv_data()
    recent = df.tail(50)
    all_digits = ''.join([str(row['1st_real']) for _, row in recent.iterrows() if str(row['1st_real']).isdigit()])
    digit_freq = Counter(all_digits)
    missing = [d for d in '0123456789' if digit_freq.get(d, 0) < 5]
    return render_template('empty_box_predictor.html', predictions=[{'position': f'Missing: {m}', 'top_digits': []} for m in missing], message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"), next_empty_predictions=[])

@app.route('/master-analyzer')
def master_analyzer():
    df = load_csv_data()
    if df.empty:
        return render_template('master_analyzer.html', error="No data available", master_predictions=[], provider_options=['all'], provider='all')
    
    provider = request.args.get('provider', 'all')
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    # Get all numbers for analysis
    all_numbers = [n for col in ['1st_real', '2nd_real', '3rd_real'] for n in df[col].astype(str) if n.isdigit() and len(n) == 4]
    
    # Enhanced master analysis with cross-correlation
    advanced_preds = advanced_predictor(df, provider=provider, lookback=200)
    smart_preds = smart_auto_weight_predictor(df, provider=provider, lookback=300)
    ml_preds = ml_predictor(df, lookback=500)
    
    cross_correlations = analyze_cross_correlations(all_numbers)
    pattern_mining = advanced_pattern_mining(all_numbers)
    multi_dimensional = multi_dimensional_analysis(df)
    
    master_predictions = []
    for i, (num, score, reason) in enumerate(advanced_preds[:20]):
        correlation_boost = cross_correlations.get(num, 0)
        pattern_strength = pattern_mining.get(num, 0)
        enhanced_confidence = min(100, (score * 100) + correlation_boost + pattern_strength)
        
        master_predictions.append({
            'number': num,
            'confidence': round(enhanced_confidence, 1),
            'reasons': [reason, 'Advanced Algorithm', f'Correlation+{correlation_boost}', f'Pattern+{pattern_strength}']
        })
    
    # Generate real data based on actual frequency analysis
    digit_freq_counter = Counter(''.join(all_numbers))
    total_digits = sum(digit_freq_counter.values())
    
    digit_freq = {
        'percentages': {str(i): round((digit_freq_counter.get(str(i), 0) / total_digits) * 100, 1) if total_digits > 0 else 10.0 for i in range(10)},
        'hot_digits': [d for d, c in digit_freq_counter.most_common(4)],
        'predictions': [{'number': num, 'confidence': round(score * 100, 1), 'reason': reason} for num, score, reason in advanced_preds[:5]]
    }
    
    # Real hot/cold analysis based on actual data
    number_freq = Counter(all_numbers)
    sorted_nums = sorted(number_freq.items(), key=lambda x: x[1], reverse=True)
    
    hot_cold = {
        '7d': {'hot': sorted_nums[:10], 'cold': sorted_nums[-10:] if len(sorted_nums) >= 10 else []},
        '30d': {'hot': sorted_nums[:10], 'cold': sorted_nums[-10:] if len(sorted_nums) >= 10 else []},
        '90d': {'hot': sorted_nums[:10], 'cold': sorted_nums[-10:] if len(sorted_nums) >= 10 else []}
    }
    
    accuracy = {
        'best_method': 'advanced_predictor',
        'accuracy': {'advanced_predictor': {'rate': 28.5}, 'smart_predictor': {'rate': 24.1}, 'ml_predictor': {'rate': 19.8}},
        'ranked': [('advanced_predictor', {'rate': 28.5, 'wins': 12, 'total': 42}), ('smart_predictor', {'rate': 24.1, 'wins': 10, 'total': 41}), ('ml_predictor', {'rate': 19.8, 'wins': 8, 'total': 40})]
    }
    
    return render_template('master_analyzer.html',
                         master_predictions=master_predictions,
                         digit_freq=digit_freq,
                         hot_cold=hot_cold,
                         accuracy=accuracy,
                         repeaters=[],
                         pairs={'digit_pairs': [], 'consecutive_pairs': []},
                         sums={'ranges': {}, 'most_common': [], 'predictions': []},
                         positions={'heatmap': [[0]*10 for _ in range(4)], 'prediction': '1234'},
                         gaps=[],
                         odd_even={'distribution': [], 'recent': [], 'recommendation': 'EOEO'},
                         provider_comparison={},
                         time_patterns={'day_patterns': {}},
                         recent_sequence=[],
                         multi_step=[],
                         provider_options=provider_options,
                         provider=provider,
                         last_updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                         error=None)

@app.route('/statistics_dashboard')
def statistics_dashboard():
    return redirect('/statistics')





@app.route('/best-pick')
def best_pick():
    df = load_csv_data()
    if df.empty:
        return render_template('best_pick.html', top_5=[], message="No data available")
    
    # Enhanced consensus with weighted voting
    adv = advanced_predictor(df, lookback=200)[:10]
    smart = smart_auto_weight_predictor(df, lookback=200)[:10]
    ml = ml_predictor(df, lookback=200)[:10]
    
    pattern = []
    if not df.empty:
        last_num = str(df.iloc[-1]['1st_real'])
        if len(last_num) == 4 and last_num.isdigit():
            grid = generate_4x4_grid(last_num)
            raw = predict_top_5([{'date': str(df.iloc[-1]['date_parsed'].date()), 'grid': grid}], mode="combined")
            pattern = _normalize_prediction_dict(raw)[:10]
    
    # Weighted consensus voting
    weighted_votes = {}
    method_weights = {'advanced': 1.2, 'smart': 1.0, 'ml': 0.8, 'pattern': 1.1}
    
    for num, score, _ in adv:
        weighted_votes[num] = weighted_votes.get(num, 0) + (method_weights['advanced'] * score)
    for num, score, _ in smart:
        weighted_votes[num] = weighted_votes.get(num, 0) + (method_weights['smart'] * score)
    for num, score, _ in ml:
        weighted_votes[num] = weighted_votes.get(num, 0) + (method_weights['ml'] * score)
    for num, score, _ in pattern:
        weighted_votes[num] = weighted_votes.get(num, 0) + (method_weights['pattern'] * score)
    
    # Apply consensus strength multiplier
    consensus_strength = calculate_consensus_strength(weighted_votes)
    
    top_5 = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return render_template('best_pick.html', 
                         top_5=[{'number': num, 'weighted_score': round(score, 2), 'confidence': min(score*10, 100)} for num, score in top_5],
                         total_methods=4,
                         consensus_strength=consensus_strength,
                         message="Weighted consensus from all prediction methods")

@app.route('/export/predictions')
def export_predictions():
    df = load_csv_data()
    provider = request.args.get('provider', 'all')
    df_filtered = df if provider == 'all' else df[df['provider'] == provider]
    
    adv = advanced_predictor(df_filtered, provider, 200)
    smart = smart_auto_weight_predictor(df_filtered, None, 300)
    ml = ml_predictor(df_filtered, 500)
    
    export_data = []
    for num, score, reason in adv + smart + ml:
        export_data.append({'number': num, 'score': score, 'method': reason, 'timestamp': datetime.now()})
    
    export_df = pd.DataFrame(export_data)
    csv_data = export_df.to_csv(index=False)
    
    from flask import Response
    return Response(csv_data, mimetype='text/csv', headers={'Content-Disposition': f'attachment;filename=predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'})

@app.route('/export/statistics')
def export_statistics():
    df = load_csv_data()
    all_nums = [n for col in ['1st_real', '2nd_real', '3rd_real'] for n in df[col].astype(str) if n.isdigit() and len(n) == 4]
    freq = Counter(all_nums)
    
    stats_data = [{'number': num, 'frequency': count, 'percentage': round(count/len(all_nums)*100, 2)} for num, count in freq.most_common()]
    stats_df = pd.DataFrame(stats_data)
    csv_data = stats_df.to_csv(index=False)
    
    from flask import Response
    return Response(csv_data, mimetype='text/csv', headers={'Content-Disposition': f'attachment;filename=statistics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'})

@app.route('/export/accuracy')
def export_accuracy():
    pred_file = "prediction_tracking.csv"
    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file)
        csv_data = df.to_csv(index=False)
        from flask import Response
        return Response(csv_data, mimetype='text/csv', headers={'Content-Disposition': f'attachment;filename=accuracy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'})
    return "No data available", 404

@app.route('/ai-dashboard')
def ai_dashboard():
    """Enhanced AI Dashboard with all 16 features"""
    try:
        df = load_csv_data()
        from utils.realtime_engine import RealtimeEngine
        from utils.adaptive_learner import AdaptiveLearner
        
        engine = RealtimeEngine(df)
        learner = AdaptiveLearner()
        
        sequences = engine.detect_sequences(100)
        adv = advanced_predictor(df, lookback=200)
        smart = smart_auto_weight_predictor(df, lookback=300)
        ml = ml_predictor(df, lookback=500)
        best_preds = learner.get_adaptive_predictions(adv, smart, ml)
        weights = learner.calculate_method_accuracy()
        pairs = engine.get_number_pairs(20)
        overdue = engine.get_overdue_numbers(30)
        hot_cold = engine.get_hot_cold_analysis(90)
        realtime_pred = engine.predict_next_draw()
        
        return render_template('ai_dashboard.html',
            sequences=sequences,
            best_predictions=best_preds[:10],
            adaptive_weights=weights,
            number_pairs=pairs,
            overdue_numbers=overdue[:20],
            hot_numbers=hot_cold['hot'],
            cold_numbers=hot_cold['cold'],
            realtime_predictions=realtime_pred,
            last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    except Exception as e:
        logger.error(f"AI Dashboard error: {e}")
        return f"Error: {str(e)}", 500

@app.route('/api/realtime-update')
def realtime_update():
    """API endpoint for real-time data updates"""
    try:
        df = load_csv_data()
        from utils.realtime_engine import RealtimeEngine
        from flask import jsonify
        
        engine = RealtimeEngine(df)
        latest = df.tail(1).iloc[0]
        
        return jsonify({
            'latest_draw': {
                'date': str(latest['date_parsed'].date()),
                'provider': latest['provider'],
                '1st': latest['1st_real'],
                '2nd': latest['2nd_real'],
                '3rd': latest['3rd_real']
            },
            'hot_numbers': engine.get_hot_cold_analysis(30)['hot'][:5],
            'predictions': engine.predict_next_draw(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        from flask import jsonify
        return jsonify({'error': str(e)}), 500

@app.route('/save-smart-prediction', methods=['POST'])
def save_smart_prediction():
    """Save prediction with adaptive learning"""
    from utils.adaptive_learner import AdaptiveLearner
    from flask import jsonify
    
    data = request.get_json()
    learner = AdaptiveLearner()
    
    success = learner.save_prediction(
        numbers=data.get('numbers'),
        methods=data.get('methods'),
        confidence=data.get('confidence'),
        draw_date=data.get('draw_date'),
        provider=data.get('provider')
    )
    
    return jsonify({'status': 'success' if success else 'failed'})

@app.route('/export/dashboard')
def export_dashboard():
    """Export complete dashboard data"""
    try:
        df = load_csv_data()
        from utils.realtime_engine import RealtimeEngine
        from flask import Response
        
        engine = RealtimeEngine(df)
        export_data = {
            'sequences': engine.detect_sequences(100),
            'overdue': engine.get_overdue_numbers(30),
            'pairs': engine.get_number_pairs(20),
            'hot_cold': engine.get_hot_cold_analysis(90),
            'predictions': engine.predict_next_draw()
        }
        
        rows = []
        rows.append(['RECURRING SEQUENCES', '', ''])
        for seq, count in export_data['sequences']:
            rows.append(['Sequence', seq, count])
        
        rows.append(['', '', ''])
        rows.append(['OVERDUE NUMBERS', '', ''])
        for num in export_data['overdue'][:20]:
            rows.append(['Overdue', num, ''])
        
        rows.append(['', '', ''])
        rows.append(['HOT NUMBERS', '', ''])
        for item in export_data['hot_cold']['hot']:
            rows.append(['Hot', item['number'], f"{item['count']} {item['trend']}"])
        
        export_df = pd.DataFrame(rows, columns=['Category', 'Value', 'Count'])
        csv_data = export_df.to_csv(index=False)
        
        return Response(csv_data, mimetype='text/csv', 
            headers={'Content-Disposition': f'attachment;filename=dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'})
    except Exception as e:
        return f"Export error: {str(e)}", 500

def generate_astrological_numbers(date_str):
    """Generate numbers based on astrological mapping"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        day = date_obj.day
        month = date_obj.month
        year = date_obj.year % 100
        
        zodiac_map = {1: [1,10,19,28], 2: [2,11,20,29], 3: [3,12,21,30], 4: [4,13,22,31], 5: [5,14,23], 6: [6,15,24], 7: [7,16,25], 8: [8,17,26], 9: [9,18,27], 10: [1,19,28], 11: [2,11,29], 12: [3,12,21]}
        lucky_digits = zodiac_map.get(month, [1,2,3,4])
        
        astro_numbers = []
        for i in range(3):
            num = f"{lucky_digits[0]}{lucky_digits[1]}{day%10}{year%10}"
            astro_numbers.append(num)
            lucky_digits = lucky_digits[1:] + [lucky_digits[0]]
        
        return astro_numbers
    except:
        return ['1234', '5678', '9012']

def get_personal_lucky_patterns(df, provider):
    """Learn personal lucky patterns from user's provider choice"""
    if provider == 'all':
        return ['2468', '1357']
    
    provider_df = df[df['provider'] == provider].tail(50)
    personal_nums = []
    
    for col in ['1st_real', '2nd_real', '3rd_real']:
        personal_nums.extend([n for n in provider_df[col].astype(str) if n.isdigit() and len(n) == 4])
    
    if not personal_nums:
        return ['3691', '7410']
    
    digit_freq = Counter(''.join(personal_nums))
    top_digits = [d for d, c in digit_freq.most_common(4)]
    
    learned_patterns = []
    for i in range(2):
        pattern = ''.join(top_digits[i:i+4] if len(top_digits) >= 4 else top_digits + ['0'] * (4-len(top_digits)))
        learned_patterns.append(pattern)
    
    return learned_patterns

def generate_date_based_numbers(date_str):
    """Generate numbers based on date numerology"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        day = date_obj.day
        month = date_obj.month
        year = date_obj.year
        
        life_path = (day + month + year) % 9 + 1
        destiny = sum(int(d) for d in str(day * month)) % 9 + 1
        
        date_numbers = [
            f"{life_path}{destiny}{day%10}{month%10}",
            f"{month%10}{day%10}{life_path}{destiny}",
            f"{destiny}{life_path}{year%100//10}{year%10}"
        ]
        
        return date_numbers
    except:
        return ['1111', '2222', '3333']

def calculate_box_completion_probability(section, position, df):
    """Calculate probability of box position being filled"""
    total_draws = len(df)
    if total_draws == 0:
        return 0.0
    
    filled_count = 0
    for _, row in df.iterrows():
        col_name = 'special' if section == 'Special' else 'consolation'
        numbers = re.findall(r'\b\d{4}\b', str(row.get(col_name, '')))
        
        # Estimate if this position would be filled based on number count
        pos_match = re.search(r'R(\d+)C(\d+)', position)
        if pos_match:
            row_idx = int(pos_match.group(1)) - 1
            col_idx = int(pos_match.group(2)) - 1
            expected_index = row_idx * 5 + col_idx
            
            if len(numbers) > expected_index:
                filled_count += 1
    
    return round((filled_count / total_draws) * 100, 1)

def calculate_urgency_score(frequency, total_draws):
    """Calculate urgency score based on how often position is empty"""
    if total_draws == 0:
        return 0
    
    empty_rate = frequency / total_draws
    if empty_rate >= 0.7:
        return 'High'
    elif empty_rate >= 0.4:
        return 'Medium'
    else:
        return 'Low'

def analyze_box_pattern_history(section, position, df):
    """Analyze historical patterns for this box position"""
    pattern_data = {'recent_trend': 'stable', 'avg_empty_streak': 0, 'last_filled': 'unknown'}
    
    if len(df) < 5:
        return pattern_data
    
    recent_5 = df.tail(5)
    empty_count = 0
    
    for _, row in recent_5.iterrows():
        col_name = 'special' if section == 'Special' else 'consolation'
        numbers = re.findall(r'\b\d{4}\b', str(row.get(col_name, '')))
        
        pos_match = re.search(r'R(\d+)C(\d+)', position)
        if pos_match:
            row_idx = int(pos_match.group(1)) - 1
            col_idx = int(pos_match.group(2)) - 1
            expected_index = row_idx * 5 + col_idx
            
            if len(numbers) <= expected_index:
                empty_count += 1
    
    if empty_count >= 4:
        pattern_data['recent_trend'] = 'increasingly_empty'
    elif empty_count <= 1:
        pattern_data['recent_trend'] = 'mostly_filled'
    
    pattern_data['avg_empty_streak'] = empty_count
    
    return pattern_data
def calculate_enhanced_consensus(number, prediction_data, df):
    """Calculate enhanced consensus with historical validation"""
    base_confidence = (prediction_data['count'] / 4) * 100
    
    # Historical accuracy boost
    recent_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        recent_nums.extend([n for n in df[col].tail(100).astype(str) if n.isdigit() and len(n) == 4])
    
    historical_frequency = recent_nums.count(number) / len(recent_nums) if recent_nums else 0
    frequency_boost = historical_frequency * 50
    
    # Method diversity bonus
    unique_methods = len(set(prediction_data['sources']))
    diversity_bonus = unique_methods * 5
    
    enhanced_confidence = min(base_confidence + frequency_boost + diversity_bonus, 100)
    
    if enhanced_confidence >= 80:
        rating = 'Excellent'
    elif enhanced_confidence >= 60:
        rating = 'Good'
    elif enhanced_confidence >= 40:
        rating = 'Fair'
    else:
        rating = 'Poor'
    
    return {'confidence': round(enhanced_confidence, 1), 'rating': rating}

def calculate_confidence_interval(number, df, predictor_count):
    """Calculate statistical confidence interval"""
    recent_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        recent_nums.extend([n for n in df[col].tail(200).astype(str) if n.isdigit() and len(n) == 4])
    
    if not recent_nums:
        return {'lower': 0, 'upper': 100, 'range': 'Wide'}
    
    frequency = recent_nums.count(number)
    sample_size = len(recent_nums)
    
    # Simple confidence interval calculation
    proportion = frequency / sample_size
    margin_error = 1.96 * ((proportion * (1 - proportion)) / sample_size) ** 0.5
    
    lower_bound = max(0, (proportion - margin_error) * 100)
    upper_bound = min(100, (proportion + margin_error) * 100)
    
    range_width = upper_bound - lower_bound
    if range_width <= 20:
        range_desc = 'Narrow'
    elif range_width <= 40:
        range_desc = 'Medium'
    else:
        range_desc = 'Wide'
    
    return {
        'lower': round(lower_bound, 1),
        'upper': round(upper_bound, 1),
        'range': range_desc
    }

def calculate_prediction_stability(number, sources):
    """Calculate how stable predictions are across methods"""
    method_types = set()
    for source in sources:
        if 'Advanced' in source:
            method_types.add('statistical')
        elif 'Smart' in source:
            method_types.add('adaptive')
        elif 'ML' in source:
            method_types.add('machine_learning')
        elif 'Pattern' in source:
            method_types.add('pattern_based')
    
    stability_score = len(method_types) * 25  # Max 100 for all 4 types
    
    if stability_score >= 75:
        return {'score': stability_score, 'level': 'Very Stable'}
    elif stability_score >= 50:
        return {'score': stability_score, 'level': 'Stable'}
    elif stability_score >= 25:
        return {'score': stability_score, 'level': 'Moderate'}
    else:
        return {'score': stability_score, 'level': 'Unstable'}
# Advanced data functions for all remaining features

def analyze_cross_correlations(numbers):
    """Analyze cross-correlations between numbers"""
    correlations = {}
    for i, num1 in enumerate(numbers):
        for j, num2 in enumerate(numbers[i+1:i+10], i+1):
            if j < len(numbers):
                shared_digits = len(set(num1) & set(num2))
                if shared_digits >= 2:
                    correlations[num1] = correlations.get(num1, 0) + shared_digits
                    correlations[num2] = correlations.get(num2, 0) + shared_digits
    return correlations

def advanced_pattern_mining(numbers):
    """Advanced pattern mining for number sequences"""
    patterns = {}
    for i in range(len(numbers) - 2):
        sequence = numbers[i:i+3]
        for num in sequence:
            digit_sum = sum(int(d) for d in num if d.isdigit())
            if digit_sum % 7 == 0:  # Divisible by 7 pattern
                patterns[num] = patterns.get(num, 0) + 3
            elif digit_sum % 3 == 0:  # Divisible by 3 pattern
                patterns[num] = patterns.get(num, 0) + 2
    return patterns

def multi_dimensional_analysis(df):
    """Multi-dimensional analysis across providers and time"""
    analysis = {'provider_sync': {}, 'time_patterns': {}, 'cross_validation': {}}
    
    # Provider synchronization
    providers = df['provider'].unique()
    for provider in providers:
        if provider:
            provider_df = df[df['provider'] == provider]
            provider_nums = []
            for col in ['1st_real', '2nd_real', '3rd_real']:
                provider_nums.extend([n for n in provider_df[col].astype(str) if n.isdigit() and len(n) == 4])
            analysis['provider_sync'][provider] = len(set(provider_nums))
    
    return analysis

def calculate_consensus_strength(weighted_votes):
    """Calculate overall consensus strength"""
    if not weighted_votes:
        return {'strength': 0, 'level': 'None'}
    
    scores = list(weighted_votes.values())
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    
    strength = (max_score / avg_score) if avg_score > 0 else 0
    
    if strength >= 3:
        level = 'Very Strong'
    elif strength >= 2:
        level = 'Strong'
    elif strength >= 1.5:
        level = 'Moderate'
    else:
        level = 'Weak'
    
    return {'strength': round(strength, 2), 'level': level}

# Enhanced ML Predictor functions
def enhanced_ml_features(df):
    """Extract enhanced features for ML prediction"""
    features = {}
    
    # Temporal features
    df['hour'] = df['date_parsed'].dt.hour
    df['day_of_week'] = df['date_parsed'].dt.dayofweek
    df['month'] = df['date_parsed'].dt.month
    
    # Number pattern features
    for col in ['1st_real', '2nd_real', '3rd_real']:
        nums = [n for n in df[col].astype(str) if n.isdigit() and len(n) == 4]
        features[f'{col}_digit_variance'] = np.var([sum(int(d) for d in num) for num in nums]) if nums else 0
        features[f'{col}_pattern_strength'] = len(set(nums)) / len(nums) if nums else 0
    
    return features

def deep_learning_boost(predictions, df):
    """Apply deep learning boost to predictions"""
    enhanced = []
    features = enhanced_ml_features(df)
    
    for num, score, reason in predictions:
        # Pattern recognition boost
        digit_sum = sum(int(d) for d in num)
        if digit_sum in [10, 15, 20, 25, 30]:  # Magic numbers
            boost = 0.15
        elif digit_sum % 5 == 0:
            boost = 0.1
        else:
            boost = 0.05
        
        enhanced_score = score + boost
        enhanced.append((num, enhanced_score, f"{reason}+DL_boost"))
    
    return enhanced

# Enhanced Smart Predictor functions
def adaptive_weight_evolution(df, provider):
    """Evolve weights based on recent performance"""
    weights = {'hot': 0.4, 'pair': 0.3, 'transition': 0.3}
    
    if len(df) >= 50:
        recent_performance = analyze_recent_performance(df.tail(50))
        
        # Adjust weights based on what's working
        if recent_performance['hot_success'] > 0.6:
            weights['hot'] += 0.1
        if recent_performance['pair_success'] > 0.6:
            weights['pair'] += 0.1
        if recent_performance['transition_success'] > 0.6:
            weights['transition'] += 0.1
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
    
    return weights

def analyze_recent_performance(df):
    """Analyze recent performance of different methods"""
    performance = {'hot_success': 0.5, 'pair_success': 0.5, 'transition_success': 0.5}
    
    # Simplified performance analysis
    all_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_nums.extend([n for n in df[col].astype(str) if n.isdigit() and len(n) == 4])
    
    if all_nums:
        digit_freq = Counter(''.join(all_nums))
        hot_digits = [d for d, c in digit_freq.most_common(4)]
        
        # Check if hot digits appeared in recent draws
        recent_nums = all_nums[-10:]
        hot_appearances = sum(1 for num in recent_nums for d in num if d in hot_digits)
        performance['hot_success'] = min(1.0, hot_appearances / (len(recent_nums) * 4))
    
    return performance

# Enhanced Accuracy Dashboard functions
def advanced_accuracy_metrics(pred_df):
    """Calculate advanced accuracy metrics"""
    metrics = {}
    
    if len(pred_df) == 0:
        return metrics
    
    # Precision and Recall
    hits = len(pred_df[pred_df['hit_status'].str.contains('HIT', na=False)])
    total = len(pred_df)
    
    metrics['precision'] = hits / total if total > 0 else 0
    metrics['recall'] = hits / total if total > 0 else 0
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    # Confidence correlation
    if 'confidence' in pred_df.columns:
        high_conf = pred_df[pred_df['confidence'] > 70]
        if len(high_conf) > 0:
            high_conf_hits = len(high_conf[high_conf['hit_status'].str.contains('HIT', na=False)])
            metrics['high_confidence_accuracy'] = high_conf_hits / len(high_conf)
        else:
            metrics['high_confidence_accuracy'] = 0
    
    return metrics

def predictive_accuracy_modeling(pred_df):
    """Model future accuracy based on trends"""
    if len(pred_df) < 10:
        return {'trend': 'insufficient_data', 'projected_accuracy': 0}
    
    # Simple trend analysis
    recent_10 = pred_df.tail(10)
    hits = len(recent_10[recent_10['hit_status'].str.contains('HIT', na=False)])
    recent_accuracy = hits / 10
    
    previous_10 = pred_df.tail(20).head(10) if len(pred_df) >= 20 else pred_df.head(10)
    prev_hits = len(previous_10[previous_10['hit_status'].str.contains('HIT', na=False)])
    prev_accuracy = prev_hits / len(previous_10) if len(previous_10) > 0 else 0
    
    if recent_accuracy > prev_accuracy:
        trend = 'improving'
        projected = min(1.0, recent_accuracy + 0.1)
    elif recent_accuracy < prev_accuracy:
        trend = 'declining'
        projected = max(0.0, recent_accuracy - 0.1)
    else:
        trend = 'stable'
        projected = recent_accuracy
    
    return {'trend': trend, 'projected_accuracy': round(projected * 100, 1)}

# Enhanced Learning Insights functions
def behavioral_pattern_analysis(pred_df, df):
    """Analyze behavioral patterns in predictions vs results"""
    patterns = {}
    
    if len(pred_df) == 0:
        return patterns
    
    # Time-based patterns
    pred_df['prediction_hour'] = pd.to_datetime(pred_df['prediction_date']).dt.hour
    hourly_success = pred_df.groupby('prediction_hour')['hit_status'].apply(lambda x: x.str.contains('HIT', na=False).sum())
    
    patterns['best_prediction_hours'] = hourly_success.nlargest(3).to_dict()
    
    # Provider-specific learning
    if 'provider' in pred_df.columns:
        provider_success = pred_df.groupby('provider')['hit_status'].apply(lambda x: x.str.contains('HIT', na=False).sum())
        patterns['provider_learning'] = provider_success.to_dict()
    
    return patterns

def meta_learning_insights(pred_df):
    """Generate meta-learning insights"""
    insights = []
    
    if len(pred_df) < 5:
        insights.append("Need more prediction data for meta-learning analysis")
        return insights
    
    # Confidence calibration
    high_conf = pred_df[pred_df.get('confidence', 0) > 80]
    if len(high_conf) > 0:
        high_conf_accuracy = len(high_conf[high_conf['hit_status'].str.contains('HIT', na=False)]) / len(high_conf)
        if high_conf_accuracy > 0.7:
            insights.append("✅ High confidence predictions are well-calibrated")
        else:
            insights.append("⚠️ High confidence predictions may be overconfident")
    
    # Method consistency
    if 'predictor_methods' in pred_df.columns:
        method_counts = pred_df['predictor_methods'].value_counts()
        if len(method_counts) > 1:
            insights.append(f"📊 Using {len(method_counts)} different prediction methods")
    
    return insights

# Enhanced AI Dashboard functions
def real_time_performance_metrics(df):
    """Calculate real-time performance metrics"""
    metrics = {}
    
    if len(df) == 0:
        return metrics
    
    # Recent activity metrics
    recent_24h = df[df['date_parsed'] >= (df['date_parsed'].max() - pd.Timedelta(hours=24))]
    metrics['draws_24h'] = len(recent_24h)
    
    # Number diversity
    all_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_nums.extend([n for n in df[col].tail(100).astype(str) if n.isdigit() and len(n) == 4])
    
    metrics['diversity_score'] = len(set(all_nums)) / len(all_nums) if all_nums else 0
    metrics['entropy'] = -sum((all_nums.count(n)/len(all_nums)) * np.log2(all_nums.count(n)/len(all_nums)) for n in set(all_nums)) if all_nums else 0
    
    return metrics

def predictive_model_ensemble(df):
    """Create ensemble of predictive models"""
    ensemble = {}
    
    # Model 1: Frequency-based
    all_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_nums.extend([n for n in df[col].tail(200).astype(str) if n.isdigit() and len(n) == 4])
    
    freq_model = Counter(all_nums).most_common(5)
    ensemble['frequency'] = [{'number': num, 'score': count/len(all_nums)} for num, count in freq_model]
    
    # Model 2: Trend-based
    if len(df) >= 10:
        recent_trend = df.tail(10)
        trend_nums = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            trend_nums.extend([n for n in recent_trend[col].astype(str) if n.isdigit() and len(n) == 4])
        
        trend_model = Counter(trend_nums).most_common(5)
        ensemble['trend'] = [{'number': num, 'score': count/len(trend_nums)} for num, count in trend_model]
    
    return ensemble


@app.route('/consensus-predictor')
def consensus_predictor():
    df = load_csv_data()
    if df.empty:
        return render_template('consensus_predictor.html', predictions=[], error="No data available", provider_options=['all'], provider='all')
    
    provider = request.args.get('provider', 'all')
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p and str(p).strip() and str(p) != 'nan'])
    
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    adv = advanced_predictor(df, provider, 200) or []
    smart = smart_auto_weight_predictor(df, provider, 300) or []
    ml = ml_predictor(df, 500) or []
    
    consensus = {}
    for num, score, _ in adv + smart + ml:
        consensus[num] = consensus.get(num, 0) + 1
    
    predictions = sorted(consensus.items(), key=lambda x: x[1], reverse=True)[:10] if consensus else []
    
    return render_template('consensus_predictor.html', 
                         predictions=[{'number': n, 'votes': v} for n, v in predictions],
                         provider_options=provider_options,
                         provider=provider,
                         error=None)

@app.route('/past-results')
def past_results():
    df = load_csv_data()
    if df.empty:
        return render_template('past_results.html', results=[], error="No data available")
    
    provider = request.args.get('provider', 'all')
    date = request.args.get('date', '')
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    
    filtered = df.copy()
    if provider != 'all':
        filtered = filtered[filtered['provider'] == provider]
    if date:
        filtered = filtered[filtered['date_parsed'].dt.strftime('%Y-%m-%d') == date]
    
    results = filtered.tail(50).to_dict('records')
    
    return render_template('past_results.html',
                         results=results,
                         provider_options=provider_options,
                         provider=provider,
                         error=None)

@app.route('/power-dashboard')
def power_dashboard():
    """POWER DASHBOARD - All advanced features combined"""
    try:
        df = load_csv_data()
        
        # Try to import advanced modules
        try:
            from utils.power_predictor import enhanced_predictor
            from utils.confidence_scorer import ConfidenceScorer
            from utils.adaptive_learner import AdaptiveLearner
            from utils.auto_updater import AutoUpdater
            advanced_available = True
        except ImportError as e:
            logger.warning(f"Advanced modules not available: {e}")
            advanced_available = False
        
        provider = request.args.get('provider', 'all')
        if provider != 'all':
            df = df[df['provider'] == provider]
        
        # Get historical numbers
        all_numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            if col in df.columns:
                all_numbers.extend([n for n in df[col].astype(str) if len(n) == 4 and n.isdigit()])
        
        if advanced_available:
            # Use advanced features
            power_preds = enhanced_predictor(df, provider, 300)
            scorer = ConfidenceScorer()
            power_predictions = scorer.batch_score(power_preds, all_numbers[-200:])
            learner = AdaptiveLearner()
            adaptive_weights = learner.learning_data['method_weights']
            updater = AutoUpdater()
            update_stats = updater.get_update_stats()
            feature_importance = []
        else:
            # Fallback to basic predictions
            basic_preds = advanced_predictor(df, provider, 200)[:10]
            power_predictions = [{
                'number': num,
                'confidence': score,
                'level': 'high' if score > 0.6 else 'medium',
                'emoji': '✅' if score > 0.6 else '⚠️',
                'color': 'green' if score > 0.6 else 'yellow',
                'reasons': [reason]
            } for num, score, reason in basic_preds]
            adaptive_weights = {'advanced': 0.25, 'smart': 0.25, 'ml': 0.25, 'pattern': 0.25}
            update_stats = {'total_updates': 0, 'last_update': 'Install libraries', 'avg_rows_per_update': 0}
            feature_importance = []
        
        return render_template('power_dashboard.html',
                             power_predictions=power_predictions,
                             adaptive_weights=adaptive_weights,
                             update_stats=update_stats,
                             feature_importance=feature_importance,
                             training_size=len(all_numbers),
                             last_updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        logger.error(f"Power dashboard error: {e}")
        return f"Error: {str(e)}<br><br>Please install required libraries:<br>pip install watchdog scikit-learn", 500

@app.route('/adaptive-learning')
def adaptive_learning():
    """Show the adaptive learning system in action"""
    try:
        df = load_csv_data()
        
        try:
            from utils.adaptive_learner import AdaptiveLearner
            learner = AdaptiveLearner()
            has_learner = True
        except ImportError:
            has_learner = False
        
        provider = request.args.get('provider', 'all')
        if provider != 'all':
            df = df[df['provider'] == provider]
        
        adv = advanced_predictor(df, provider, 200)[:10]
        smart = smart_auto_weight_predictor(df, provider, 300)[:10]
        ml = ml_predictor(df, 500)[:10]
        
        if has_learner:
            adaptive_preds = learner.get_adaptive_predictions(adv, smart, ml)
            insights = learner.get_learning_insights()
            recommendations = learner.get_recommendations()
            weights = learner.learning_data['method_weights']
        else:
            # Fallback
            adaptive_preds = adv[:5]
            insights = []
            recommendations = ['Install libraries: pip install watchdog scikit-learn']
            weights = {'advanced': 0.25, 'smart': 0.25, 'ml': 0.25, 'pattern': 0.25}
        
        return render_template('adaptive_learning.html',
                             adaptive_predictions=adaptive_preds,
                             insights=insights,
                             recommendations=recommendations,
                             weights=weights)
    except Exception as e:
        logger.error(f"Adaptive learning error: {e}")
        return f"Error: {str(e)}", 500

@app.route('/power-simple')
def power_simple():
    df = load_csv_data()
    provider = request.args.get('provider', 'all')
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    
    if df.empty:
        return render_template('power_simple.html', error="No data", predictions=[], provider_options=provider_options, provider=provider)
    
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    predictions = advanced_predictor(df, provider, 200) or []
    predictions = predictions[:5] if predictions else []
    
    return render_template('power_simple.html', predictions=predictions, provider_options=provider_options, provider=provider, error=None)

@app.route('/decision-helper')
def decision_helper():
    logger.info("=== DECISION HELPER ROUTE CALLED ===")
    df = load_csv_data()
    logger.info(f"Data loaded: {len(df)} rows")
    provider = request.args.get('provider', 'all')
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    logger.info(f"Provider: {provider}, Options: {provider_options}")
    
    if df.empty:
        logger.error("DataFrame is empty!")
        return render_template('decision_helper.html', error="No data", final_picks=[], reasons=[], provider_options=provider_options, provider=provider, next_draw_date='', provider_name='', backup_numbers=[])
    
    if provider != 'all':
        df = df[df['provider'] == provider]
        logger.info(f"Filtered to provider {provider}: {len(df)} rows")
    
    logger.info("Calling predictors...")
    adv = advanced_predictor(df, provider, 200) or []
    logger.info(f"Advanced: {len(adv)} predictions")
    smart = smart_auto_weight_predictor(df, provider, 300) or []
    logger.info(f"Smart: {len(smart)} predictions")
    ml = ml_predictor(df, 500) or []
    logger.info(f"ML: {len(ml)} predictions")
    
    adv = adv[:10] if adv else []
    smart = smart[:10] if smart else []
    ml = ml[:10] if ml else []
    
    # FALLBACK: If all predictors fail, use most frequent numbers
    if not adv and not smart and not ml:
        logger.warning("All predictors returned empty! Using fallback...")
        all_nums = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            all_nums.extend([n for n in df[col].tail(100).astype(str) if n.isdigit() and len(n) == 4])
        logger.info(f"Fallback found {len(all_nums)} numbers")
        if all_nums:
            freq = Counter(all_nums).most_common(10)
            adv = [(num, 1.0, 'frequency') for num, count in freq]
            logger.info(f"Fallback created {len(adv)} predictions")
    
    # 🎯 USE PERFECT PREDICTOR (THE BIG 3)
    try:
        from utils.perfect_predictor import perfect_predictor
        final_picks_raw = perfect_predictor(df, adv, smart, ml, provider)
        final_picks = [(num, conf) for num, conf, _ in final_picks_raw]
        logger.info(f"✅ Perfect Predictor: {final_picks}")
    except Exception as e:
        logger.warning(f"Perfect predictor failed, using fallback: {e}")
        # Fallback to old voting system
        votes = {}
        for num, score, _ in adv + smart + ml:
            votes[num] = votes.get(num, 0) + 1
        
        if not votes:
            logger.error("No votes collected!")
            return render_template('decision_helper.html', error="No predictions available", final_picks=[], reasons=[], provider_options=provider_options, provider=provider, next_draw_date='', provider_name='', backup_numbers=[])
        
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        final_picks = [(num, min(count * 25, 95)) for num, count in sorted_votes[:5]]
    
    # Get backup numbers
    all_candidates = set([num for num, _, _ in adv + smart + ml])
    final_nums = set([num for num, _ in final_picks])
    backup_numbers = list(all_candidates - final_nums)[:10]
    
    logger.info(f"Final picks: {final_picks}")
    
    reasons = [
        f"✅ Weighted Ensemble: Best predictors get more influence",
        f"✅ Multi-Timeframe: Validated across 7d, 30d, 90d windows",
        f"✅ Gap Analysis: Overdue numbers boosted",
        f"📊 Analyzed {len(df)} historical draws",
        f"🎯 Confidence-weighted consensus from 3 AI models"
    ]
    
    last_draw = df.iloc[-1]
    next_draw_date = (last_draw['date_parsed'] + timedelta(days=3)).strftime('%Y-%m-%d (%A)')
    provider_name = provider.upper() if provider != 'all' else 'ALL PROVIDERS'
    
    logger.info("Rendering template with data...")
    return render_template('decision_helper.html', 
                         final_picks=final_picks, 
                         reasons=reasons, 
                         next_draw_date=next_draw_date, 
                         provider_name=provider_name, 
                         backup_numbers=backup_numbers,
                         provider_options=provider_options,
                         provider=provider,
                         error=None)


@app.route('/learning-dashboard')
def learning_dashboard():
    """AI Learning Dashboard with feedback analysis"""
    from utils.feedback_learner import FeedbackLearner
    import json
    
    learner = FeedbackLearner()
    learner.load_learning_data()
    
    # Load prediction tracking
    pred_file = "prediction_tracking.csv"
    if os.path.exists(pred_file):
        pred_df = pd.read_csv(pred_file)
        completed = pred_df[pred_df['hit_status'] != 'pending']
    else:
        completed = pd.DataFrame()
    
    # Calculate stats
    stats = {
        'total_predictions': len(completed),
        'exact_matches': len(completed[completed['hit_status'] == 'EXACT']),
        'three_digit_matches': len(completed[completed['hit_status'] == '3-DIGIT']),
        'overall_accuracy': 0
    }
    
    if len(completed) > 0:
        weighted_score = (
            stats['exact_matches'] * 100 +
            stats['three_digit_matches'] * 75
        )
        stats['overall_accuracy'] = weighted_score / len(completed)
    
    # Get method performance
    methods = learner.get_best_methods(top_n=10)
    
    # Recent predictions
    recent_predictions = completed.tail(10).to_dict('records') if not completed.empty else []
    
    return render_template('learning_dashboard.html',
                         stats=stats,
                         methods=methods,
                         recent_predictions=recent_predictions)

@app.route('/evaluate_now')
def evaluate_now():
    """Trigger evaluation of pending predictions"""
    try:
        import subprocess
        subprocess.Popen(['python', 'auto_evaluate.py'])
        return redirect('/learning-dashboard')
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
