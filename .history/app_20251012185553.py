from flask import Flask, render_template, request, redirect
import pandas as pd
from datetime import datetime
import os
import numpy as np
from datetime import date
app = Flask(__name__)

# ✅ make datetime available inside all HTML templates
@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

import re
import time
import logging
from collections import defaultdict, Counter
from utils.pattern_finder import find_all_4digit_patterns
from utils.pattern_stats import compute_pattern_frequencies, compute_cell_heatmap
from utils.ai_predictor import predict_top_5
from utils.app_grid import generate_reverse_grid, generate_4x4_grid
from utils.pattern_memory import learn_pattern_transitions
from utils.pattern_predictor import predict_from_today_grid

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- CSV LOADER (CORRECTED) ---------------- #
_csv_cache = None
_csv_cache_time = None

def load_csv_data():
    """
    Load CSV with caching for better performance.
    """
    global _csv_cache, _csv_cache_time
    
    # Check if cache is valid (less than 5 minutes old)
    if _csv_cache is not None and _csv_cache_time is not None:
        if (datetime.now() - _csv_cache_time).seconds < 300:
            return _csv_cache.copy()
    
    df = pd.read_csv('4d_results_history.csv', index_col=False, on_bad_lines='skip')

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
    extracted = prov_series.str.extract(r'images/([^,/\s]+)', expand=False)
    df['provider'] = extracted.fillna(prov_series).str.strip().str.lower()
    
    # <<< THE NEW FIX IS HERE >>>
    # This line removes any rows that have the same date and provider, keeping only the first one it finds.
    df.drop_duplicates(subset=['date_parsed', 'provider'], keep='first', inplace=True)

    # Extract real prize numbers - optimized vectorized approach
    def extract_number(text):
        text = str(text).strip()
        if len(text) == 4 and text.isdigit():
            return text
        match = re.search(r'(\d{4})', text)
        return match.group(1) if match else ''
    
    df['1st_real'] = df['1st'].apply(extract_number)
    df['2nd_real'] = df['2nd'].apply(extract_number)
    df['3rd_real'] = df['3rd'].apply(extract_number)

    # This line sorts the clean, de-duplicated data for consistent predictions
    df = df.sort_values(['date_parsed', 'provider'], ascending=[True, True]).reset_index(drop=True)

    # Update cache
    _csv_cache = df.copy()
    _csv_cache_time = datetime.now()

    return df

# (The rest of your code is unchanged and remains the same as your original)

# ---------------- HELPERS ---------------- #
def find_missing_digits(grid):
    all_digits = set(map(str, range(10)))
    used_digits = set(str(cell) for row in grid for cell in row)
    return sorted(all_digits - used_digits)

def find_4digit_patterns(grid):
    return find_all_4digit_patterns(grid)

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

    damacai_3plus3d = {
        'draw_no': cards[0].get('draw_info', 'DMC0000') if cards else '',
        'date': selected_date,
        'numbers': ['123', '456', '789']
    }

    sports_toto_games = [
        {'name': 'Toto 5D', 'draw_no': 'T5D001', 'date': selected_date, 'result': '12345'},
        {'name': 'Toto 6D', 'draw_no': 'T6D002', 'date': selected_date, 'result': '654321'},
        {'name': 'Toto Lotto', 'draw_no': 'TL003', 'date': selected_date, 'result': '12 23 34 45 56 67'}
    ]

    magnum_life = {
        'draw_no': cards[0].get('draw_info', 'ML0000') if cards else '',
        'date': selected_date,
        'numbers': ['01', '05', '12', '18', '22', '27', '33', '39'],
        'life_number': '07'
    }

    magnum_jackpot_gold = {
        'draw_no': cards[0].get('draw_info', 'MJG0000') if cards else '',
        'date': selected_date,
        'numbers': ['03', '11', '19', '25', '32'],
        'jackpot_amount': '3,000,000'
    }

    return render_template(
        'index.html',
        cards=cards,
        selected_date=selected_date,
        damacai_3plus3d=damacai_3plus3d,
        sports_toto_games=sports_toto_games,
        magnum_life=magnum_life,
        magnum_jackpot_gold=magnum_jackpot_gold
    )

@app.route('/pattern-analyzer', methods=['GET', 'POST'])
def pattern_analyzer():
    df = load_csv_data()
    selected_month = request.args.get('month')
    selected_provider = request.args.get('provider')
    selected_aimode = request.args.get('aimode', 'pattern')

    search_pattern = ''
    manual_search_results = {}
    draws = []

    if selected_provider and selected_provider.startswith("http"):
        selected_provider = selected_provider.split('/')[-1]
    if selected_provider:
        selected_provider = selected_provider.strip().lower()

    if not selected_month:
        today = date.today()
        selected_month = today.strftime('%Y-%m')

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
        best_module=(max(module_accuracy, key=lambda x: x[3]) if module_accuracy else None)
    )

@app.route('/prediction-history')
def prediction_history():
    df = load_csv_data()
    df = df[df['1st_real'].astype(str).str.len() == 4]
    df = df[df['1st_real'].astype(str).str.isdigit()]
    df = df.sort_values(['date_parsed']).reset_index(drop=True)

    history = []
    for i in range(len(df) - 1):
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

        missing_digits = find_missing_digits(grid)

        history.append({
            "date": this_draw["date_parsed"].date(),
            "provider": this_draw["provider"],
            "drawn_number": this_draw["1st_real"],
            "predictions": checked_preds,
            "next_winners": next_winners,
            "missing_digits": missing_digits
        })

    return render_template("prediction_history.html", history=history)

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
    return redirect('/consensus-predictor')
# ============================================================
# ⚙️ SMART PREDICTORS SECTION (Add-on, safe to paste)
# ============================================================

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter

# ------------------------------------------------------------
# 1️⃣ AUTO WEIGHT TUNING - finds best balance between hot, pair, transitions
# ------------------------------------------------------------
def smart_auto_weight_predictor(df, provider=None, lookback=300):
    """
    Automatically tunes weight between 'hot digits', 'pairs', and 'transitions'
    based on correlation to next-draw winners.
    Easy & light but smarter than static scoring.
    """
    prize_cols = ["1st_real", "2nd_real", "3rd_real"]
    all_numbers = []
    for col in prize_cols:
        if col in df.columns:
            all_numbers += [n for n in df[col].astype(str) if n.isdigit() and len(n) == 4]
    if not all_numbers:
        return []

    # Use last portion of data
    all_numbers = all_numbers[-lookback:]
    digit_counts = Counter("".join(all_numbers))
    pair_counts = Counter([n[i:i+2] for n in all_numbers for i in range(3)])
    transitions = learn_pattern_transitions([{"number": n} for n in all_numbers])

    # build dataset of feature weights
    features, targets = [], []
    for i in range(len(all_numbers) - 1):
        num = all_numbers[i]
        next_num = all_numbers[i + 1]
        hot_score = sum(digit_counts.get(d, 0) for d in num)
        pair_score = sum(pair_counts.get(num[i:i+2], 0) for i in range(3))
        trans_score = transitions.get(num, 0)
        features.append([hot_score, pair_score, trans_score])
        targets.append(len(set(num) & set(next_num)))  # how many digits repeated next draw

    if len(features) < 5:
        return []

    X = np.array(features)
    y = np.array(targets)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # learn ideal weights automatically
    model = LinearRegression()
    model.fit(X_scaled, y)
    w_hot, w_pair, w_trans = model.coef_

    # Predict next based on learned weights
    candidate_pool = {n for n in all_numbers[-150:]}
    scored = []
    for num in candidate_pool:
        hot_score = sum(digit_counts.get(d, 0) for d in num)
        pair_score = sum(pair_counts.get(num[i:i+2], 0) for i in range(3))
        trans_score = transitions.get(num, 0)
        total = (
            w_hot * hot_score
            + w_pair * pair_score
            + w_trans * trans_score
        )
        scored.append((num, total, f"auto-w({w_hot:.2f},{w_pair:.2f},{w_trans:.2f})"))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_5 = [(num, round(score, 3), reason) for num, score, reason in scored[:5]]
    return top_5

@app.route('/smart-predictor')
def smart_predictor_page():
    df = load_csv_data()
    if df.empty:
        return render_template(
            'smart_predictor.html',
            provider="N/A",
            last_updated="No data available",
            top_5_predictions=None,
            smart_formula="N/A"
        )

    from itertools import product
    import numpy as np

    # Step 1: Calculate base metrics
    hot_counts = df['1st_real'].astype(str).apply(lambda x: [d for d in x]).explode().value_counts()
    pair_counts = df['1st_real'].astype(str).apply(lambda x: [x[i:i+2] for i in range(3)]).explode().value_counts()

    # Step 2: Generate weights combinations
    combos = [(h, p, t) for h, p, t in product([0.2, 0.3, 0.4, 0.5], repeat=3) if abs((h+p+t)-1) < 0.05]
    best_score = -1
    best_weights = (0.4, 0.3, 0.3)

    # Step 3: Evaluate each combination
    for hot_w, pair_w, trans_w in combos:
        score = 0
        for num in df['1st_real'].astype(str).tail(50):  # check last 50 draws
            digits = [d for d in num]
            pairs = [num[i:i+2] for i in range(3)]
            hot_score = sum(hot_counts.get(d, 0) for d in digits)
            pair_score = sum(pair_counts.get(p, 0) for p in pairs)
            total_score = hot_w * hot_score + pair_w * pair_score + trans_w * len(set(digits))
            score += total_score
        if score > best_score:
            best_score = score
            best_weights = (hot_w, pair_w, trans_w)

    hot_w, pair_w, trans_w = best_weights

    # Step 4: Apply best weights for next prediction
    scores = {}
    for i in range(0, 10000):
        num = f"{i:04d}"
        digits = [d for d in num]
        pairs = [num[i:i+2] for i in range(3)]
        hot_score = sum(hot_counts.get(d, 0) for d in digits)
        pair_score = sum(pair_counts.get(p, 0) for p in pairs)
        total_score = hot_w * hot_score + pair_w * pair_score + trans_w * len(set(digits))
        scores[num] = total_score

    top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    smart_formula = f"🔥 Hot={hot_w:.1f}, 🧩 Pair={pair_w:.1f}, 🔄 Transition={trans_w:.1f}"

    # Save learning history automatically
    hist_file = "smart_history.csv"
    new_entry = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "HotWeight": round(hot_w, 2),
        "PairWeight": round(pair_w, 2),
        "TransWeight": round(trans_w, 2),
        "Accuracy": round(best_score / 10000, 4)  # relative performance index
    }
    if os.path.exists(hist_file):
        hist = pd.read_csv(hist_file)
        hist = pd.concat([hist, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        hist = pd.DataFrame([new_entry])
    hist.to_csv(hist_file, index=False)

    return render_template(
        'smart_predictor.html',
        provider="System Auto-Tuned",
        last_updated=df['date_parsed'].iloc[-1] if not df.empty else "No data available",
        top_5_predictions=[(num, round(score, 2), "Smart weighted prediction") for num, score in top_5],
        smart_formula=smart_formula
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

    # Predict next best numbers by scoring possible combos
    recent = numbers[-10:]
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
    selected_provider = request.args.get('provider', 'all')
    
    if df.empty:
        return render_template('ultimate_predictor.html', error="No data available")
    
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    
    if selected_provider != 'all':
        df_filtered = df[df['provider'] == selected_provider]
    else:
        df_filtered = df
    
    last_draw = df_filtered.iloc[-1]
    last_draw_date = last_draw['date_parsed'].strftime('%Y-%m-%d (%A)')
    last_draw_number = last_draw['1st_real']
    provider_name = last_draw['provider'].upper() if selected_provider != 'all' else 'ALL PROVIDERS'
    
    from datetime import timedelta
    next_draw_date = (last_draw['date_parsed'] + timedelta(days=3)).strftime('%Y-%m-%d (%A)')
    
    advanced_preds = advanced_predictor(df_filtered, provider=selected_provider, lookback=200)
    smart_preds = smart_auto_weight_predictor(df_filtered, provider=None, lookback=300)
    ml_preds = ml_predictor(df_filtered, lookback=500)
    
    recent_draws = df_filtered.tail(10)
    pattern_preds = []
    if not recent_draws.empty:
        last_num = str(recent_draws.iloc[-1]['1st_real'])
        if len(last_num) == 4 and last_num.isdigit():
            grid = generate_4x4_grid(last_num)
            raw = predict_top_5([{'date': str(recent_draws.iloc[-1]['date_parsed'].date()), 'grid': grid}], mode="combined")
            pattern_preds = _normalize_prediction_dict(raw)
    
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
    
    final_predictions = []
    for num, data in all_predictions.items():
        consensus_score = data['count'] * (data['total_score'] / data['count'])
        confidence = (data['count'] / 4) * 100
        final_predictions.append({
            'number': num,
            'consensus_score': round(consensus_score, 3),
            'predictor_count': data['count'],
            'confidence': round(confidence, 1),
            'sources': ', '.join(data['sources'])
        })
    
    final_predictions.sort(key=lambda x: (x['predictor_count'], x['consensus_score']), reverse=True)
    
    return render_template(
        'ultimate_predictor.html',
        ultimate_top_10=final_predictions[:10],
        advanced_preds=advanced_preds[:5],
        smart_preds=smart_preds[:5],
        ml_preds=ml_preds[:5],
        pattern_preds=pattern_preds[:5],
        last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
        provider_options=provider_options,
        selected_provider=selected_provider,
        provider_name=provider_name,
        last_draw_date=last_draw_date,
        last_draw_number=last_draw_number,
        next_draw_date=next_draw_date
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
                # Extract date only (remove day name like "(Tuesday)")
                draw_date_str = str(row['draw_date']).split('(')[0].strip()
                draw_date = pd.to_datetime(draw_date_str).date()
                provider = str(row['provider']).strip().lower()
                
                # Match with actual results
                actual = df[(df['date_parsed'].dt.date == draw_date) & (df['provider'] == provider)]
                
                if not actual.empty:
                    actual_row = actual.iloc[0]
                    pred_df.at[idx, 'actual_1st'] = actual_row['1st_real']
                    pred_df.at[idx, 'actual_2nd'] = actual_row['2nd_real']
                    pred_df.at[idx, 'actual_3rd'] = actual_row['3rd_real']
                    
                    # Parse predicted numbers
                    predicted = eval(row['predicted_numbers']) if isinstance(row['predicted_numbers'], str) else []
                    actuals = [str(actual_row['1st_real']), str(actual_row['2nd_real']), str(actual_row['3rd_real'])]
                    
                    hits = [p for p in predicted if p in actuals]
                    if hits:
                        pred_df.at[idx, 'hit_status'] = f'HIT-{len(hits)}'
                        pred_df.at[idx, 'accuracy_score'] = len(hits) * 100 / len(predicted) if predicted else 0
                    else:
                        pred_df.at[idx, 'hit_status'] = 'MISS'
                        pred_df.at[idx, 'accuracy_score'] = 0
            except Exception as e:
                pass
    
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
    
    if df.empty:
        return render_template('quick_pick.html', numbers=[], error="No data available")
    
    # Get predictions from all methods
    advanced_preds = advanced_predictor(df, provider=None, lookback=200)
    smart_preds = smart_auto_weight_predictor(df, provider=None, lookback=300)
    ml_preds = ml_predictor(df, lookback=500)
    
    # Combine and find consensus
    all_predictions = {}
    for num, score, _ in advanced_preds:
        all_predictions[num] = all_predictions.get(num, 0) + 1
    for num, score, _ in smart_preds:
        all_predictions[num] = all_predictions.get(num, 0) + 1
    for num, score, _ in ml_preds:
        all_predictions[num] = all_predictions.get(num, 0) + 1
    
    # Sort by consensus
    sorted_nums = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    top_5 = [num for num, count in sorted_nums[:5]]
    
    # Get next draw date
    last_draw = df.iloc[-1]
    from datetime import timedelta
    next_draw_date = (last_draw['date_parsed'] + timedelta(days=3)).strftime('%Y-%m-%d (%A)')
    
    return render_template('quick_pick.html', 
                         numbers=top_5,
                         next_draw_date=next_draw_date,
                         error=None)

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

    # --- Fallback Logic ---
    # If no strong sequence patterns were found, use weaker digit transition patterns.
    # This ensures we almost always have some prediction to show.
    if not predictions:
        for today_num in set(today_numbers):
            for i, digit in enumerate(today_num):
                if digit in digit_transitions and i in digit_transitions[digit]:
                    for next_digit, count in digit_transitions[digit][i].items():
                        # Generate candidate numbers based on this single digit transition
                        for j in range(4): # Create 4 candidate numbers
                            candidate = list(today_num)
                            candidate[i] = next_digit
                            # Slightly alter another digit to create variation
                            candidate[j] = str((int(candidate[j]) + count) % 10) 
                            candidate_num = "".join(candidate)
                            
                            reason = f"Digit '{digit}' at pos {i+1} suggests '{next_digit}'"
                            predictions[candidate_num]['score'] += 0.5 * count # Weaker signal
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

    filtered_df = df.copy()
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

    # --- Prediction Logic ---
    patterns = learn_day_to_day_patterns(filtered_df)

    # Get numbers from the most recent draw date in the filtered data
    latest_date = filtered_df['date_parsed'].max()
    today_data = filtered_df[filtered_df['date_parsed'] == latest_date]

    today_numbers = []
    prize_cols = ['1st_real', '2nd_real', '3rd_real', 'special', 'consolation']
    for _, row in today_data.iterrows():
        for col in prize_cols:
            if col in row and pd.notna(row[col]):
                # Split if numbers are space-separated (like in special/consolation)
                nums = str(row[col]).split()
                for num in nums:
                    if len(num) == 4 and num.isdigit():
                        today_numbers.append(num)

    all_recent_numbers = filtered_df['1st_real'].dropna().astype(str).tolist()[-200:]
    predictions = get_day_to_day_predictions_with_reasons(today_numbers, patterns, all_recent_numbers)

    return render_template(
        'day_to_day_predictor.html',
        last_updated=time.strftime('%Y-%m-%d %H:%M:%S'),
        provider_options=provider_options,
        provider=selected_provider,
        month_options=month_options,
        selected_month=selected_month,
        today_date=latest_date.strftime('%Y-%m-%d'),
        total_numbers_analyzed=len(today_numbers),
        today_numbers=today_numbers[:20],
        predictions=predictions
    )

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
            predicted = eval(row['predicted_numbers']) if isinstance(row['predicted_numbers'], str) else []
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

if __name__ == "__main__":
    app.run(debug=True)
