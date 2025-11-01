from flask import Flask, render_template, request, redirect
import pandas as pd
from datetime import datetime, date, timedelta
import os
import numpy as np
import re
import time
import logging
from collections import defaultdict, Counter
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from utils.pattern_finder import find_all_4digit_patterns
from utils.pattern_stats import compute_pattern_frequencies, compute_cell_heatmap
from utils.ai_predictor import predict_top_5
from utils.app_grid import generate_reverse_grid, generate_4x4_grid
from utils.pattern_memory import learn_pattern_transitions
from utils.pattern_predictor import predict_from_today_grid
from utils.positional_pattern_tracker import analyze_positional_patterns
from utils.arithmetic_gap_analyzer import analyze_arithmetic_gaps, find_high_prize_precursors
from utils.temporal_recurrence_mapper import map_temporal_recurrence
from utils.cross_draw_pattern_linker import link_cross_draw_patterns
from utils.structural_motif_clusterer import cluster_structural_motifs
from utils.pattern_predictive_weighting import weight_patterns_predictively

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

_csv_cache = None
_csv_cache_time = None

def load_csv_data():
    global _csv_cache, _csv_cache_time
    if _csv_cache is not None and _csv_cache_time is not None:
        if (datetime.now() - _csv_cache_time).seconds < 300:
            return _csv_cache.copy()
    
    df = pd.read_csv('4d_results_history.csv', index_col=False, on_bad_lines='skip', engine='python')
    for col in ['date', 'Date', 'draw_date']:
        if col in df.columns:
            df['date_parsed'] = pd.to_datetime(df[col], errors='coerce')
            break
    else:
        df['date_parsed'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    
    df.dropna(subset=['date_parsed'], inplace=True)
    for col in ['1st', '2nd', '3rd', 'special', 'consolation', 'provider']:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].fillna('')
    
    prov_series = df['provider'].astype(str)
    extracted = prov_series.str.extract(r'images/([^,/\s]+)', expand=False)
    df['provider'] = extracted.fillna(prov_series).str.strip().str.lower()
    df.drop_duplicates(subset=['date_parsed', 'provider'], keep='first', inplace=True)
    
    def extract_number(text):
        text = str(text).strip()
        if len(text) == 4 and text.isdigit():
            return text
        match = re.search(r'(\d{4})', text)
        return match.group(1) if match else ''
    
    df['1st_real'] = df['1st'].apply(extract_number)
    df['2nd_real'] = df['2nd'].apply(extract_number)
    df['3rd_real'] = df['3rd'].apply(extract_number)
    df = df.sort_values(['date_parsed', 'provider'], ascending=[True, True]).reset_index(drop=True)
    
    _csv_cache = df.copy()
    _csv_cache_time = datetime.now()
    return df

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
        next_targets = [str(next_draw.get(col, '')) for col in ['1st_real', '2nd_real', '3rd_real'] if str(next_draw.get(col, '')).isdigit()]
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

def get_3_digit_hits(predictions, actual):
    actual_digits = set(actual)
    hits = []
    for pred, score, reason in predictions:
        match_count = len(set(pred) & actual_digits)
        if match_count == 3:
            matched = ''.join(sorted(set(pred) & actual_digits))
            hits.append((pred, score, f"matched 3 digits: {matched}"))
    return hits

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
        from datetime import date as date_class
        today = date_class.today()
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
                        comparison.append({"number": pred, "score": score, "reason": reason, "hit": hit})
                        modules = reason.split("+")
                        for m in modules:
                            module_attempts[m] += 1
                            if hit == "✅":
                                module_hits[m] += 1
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
            logger.warning(f"Draw skipped: {e}")
    
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
        grid = generate_4x4_grid(today_number)
        reverse_grid = generate_reverse_grid(today_number)
        draw_stub = {"number": today_number, "grid": grid, "reverse_grid": reverse_grid}
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
    chart_data = []
    
    return render_template('pattern_analyzer.html', draws=draws, provider_options=provider_options,
                         selected_provider=selected_provider, month_options=month_options,
                         selected_month=selected_month, search_pattern=search_pattern,
                         manual_search_results=manual_search_results, freq_list=freq_list,
                         cell_heatmap=cell_heatmap, top_5_predictions=top_5_predictions,
                         prediction_mode=prediction_mode, last_updated=last_updated,
                         selected_aimode=selected_aimode, is_fallback=(prediction_mode == "fallback"),
                         module_accuracy=module_accuracy, provider_accuracy=provider_accuracy,
                         chart_data=chart_data, best_module=(max(module_accuracy, key=lambda x: x[3]) if module_accuracy else None))

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

def smart_auto_weight_predictor(df, provider=None, lookback=300):
    prize_cols = ["1st_real", "2nd_real", "3rd_real"]
    all_numbers = []
    for col in prize_cols:
        if col in df.columns:
            all_numbers += [n for n in df[col].astype(str) if n.isdigit() and len(n) == 4]
    if not all_numbers:
        return []
    all_numbers = all_numbers[-lookback:]
    digit_counts = Counter("".join(all_numbers))
    pair_counts = Counter([n[i:i+2] for n in all_numbers for i in range(3)])
    transitions = learn_pattern_transitions([{"number": n} for n in all_numbers])
    features, targets = [], []
    for i in range(len(all_numbers) - 1):
        num = all_numbers[i]
        next_num = all_numbers[i + 1]
        hot_score = sum(digit_counts.get(d, 0) for d in num)
        pair_score = sum(pair_counts.get(num[i:i+2], 0) for i in range(3))
        trans_score = transitions.get(num, 0)
        features.append([hot_score, pair_score, trans_score])
        targets.append(len(set(num) & set(next_num)))
    if len(features) < 5:
        return []
    X = np.array(features)
    y = np.array(targets)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    w_hot, w_pair, w_trans = model.coef_
    candidate_pool = {n for n in all_numbers[-150:]}
    scored = []
    for num in candidate_pool:
        hot_score = sum(digit_counts.get(d, 0) for d in num)
        pair_score = sum(pair_counts.get(num[i:i+2], 0) for i in range(3))
        trans_score = transitions.get(num, 0)
        total = w_hot * hot_score + w_pair * pair_score + w_trans * trans_score
        scored.append((num, total, f"auto-w({w_hot:.2f},{w_pair:.2f},{w_trans:.2f})"))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_5 = [(num, round(score, 3), reason) for num, score, reason in scored[:5]]
    return top_5

def ml_predictor(df, lookback=500):
    prize_cols = ["1st_real", "2nd_real", "3rd_real"]
    numbers = []
    for col in prize_cols:
        if col in df.columns:
            numbers += [n for n in df[col].astype(str) if n.isdigit() and len(n) == 4]
    if len(numbers) < 20:
        return []
    X, y = [], []
    for i in range(len(numbers) - 1):
        curr = [int(d) for d in numbers[i]]
        next_num = [int(d) for d in numbers[i + 1]]
        X.append(curr)
        y.append(sum(next_num) / 4)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
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

@app.route('/advanced-predictions')
def advanced_predictions():
    df = load_csv_data()
    selected_provider = request.args.get('provider', 'all')
    if df.empty:
        return render_template('advanced_predictions.html', top_5_predictions=[], provider=selected_provider, last_updated="No data")
    top_5 = advanced_predictor(df, provider=selected_provider if selected_provider != 'all' else None, lookback=200)
    return render_template('advanced_predictions.html', top_5_predictions=top_5, provider=selected_provider, last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

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
        final_predictions.append({'number': num, 'consensus_score': round(consensus_score, 3), 'predictor_count': data['count'], 'confidence': round(confidence, 1), 'sources': ', '.join(data['sources'])})
    final_predictions.sort(key=lambda x: (x['predictor_count'], x['consensus_score']), reverse=True)
    return render_template('ultimate_predictor.html', ultimate_top_10=final_predictions[:10], advanced_preds=advanced_preds[:5], smart_preds=smart_preds[:5], ml_preds=ml_preds[:5], pattern_preds=pattern_preds[:5], last_updated=time.strftime("%Y-%m-%d %H:%M:%S"), provider_options=provider_options, selected_provider=selected_provider, provider_name=provider_name, last_draw_date=last_draw_date, last_draw_number=last_draw_number, next_draw_date=next_draw_date)

@app.route('/smart-predictor')
def smart_predictor_page():
    df = load_csv_data()
    if df.empty:
        return render_template('smart_predictor.html', provider="N/A", last_updated="No data available", top_5_predictions=None, smart_formula="N/A")
    hot_counts = df['1st_real'].astype(str).apply(lambda x: [d for d in x]).explode().value_counts()
    pair_counts = df['1st_real'].astype(str).apply(lambda x: [x[i:i+2] for i in range(3)]).explode().value_counts()
    combos = [(h, p, t) for h, p, t in product([0.2, 0.3, 0.4, 0.5], repeat=3) if abs((h+p+t)-1) < 0.05]
    best_score = -1
    best_weights = (0.4, 0.3, 0.3)
    for hot_w, pair_w, trans_w in combos:
        score = 0
        for num in df['1st_real'].astype(str).tail(50):
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
    hist_file = "smart_history.csv"
    new_entry = {"Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "HotWeight": round(hot_w, 2), "PairWeight": round(pair_w, 2), "TransWeight": round(trans_w, 2), "Accuracy": round(best_score / 10000, 4)}
    if os.path.exists(hist_file):
        hist = pd.read_csv(hist_file)
        hist = pd.concat([hist, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        hist = pd.DataFrame([new_entry])
    hist.to_csv(hist_file, index=False)
    return render_template('smart_predictor.html', provider="System Auto-Tuned", last_updated=df['date_parsed'].iloc[-1] if not df.empty else "No data available", top_5_predictions=[(num, round(score, 2), "Smart weighted prediction") for num, score in top_5], smart_formula=smart_formula)

@app.route('/smart-history')
def smart_history_page():
    hist_file = "smart_history.csv"
    if not os.path.exists(hist_file):
        return render_template('smart_history.html', history=[], message="No smart history recorded yet.")
    hist = pd.read_csv(hist_file)
    return render_template('smart_history.html', history=hist.to_dict(orient='records'), message="")

@app.route('/ml-predictor')
def ml_predictor_page():
    df = load_csv_data()
    top_5 = ml_predictor(df)
    return render_template('ml_predictor.html', top_5_predictions=top_5, last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

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
        raw_pred = predict_top_5([{'date': str(this_draw['date_parsed'].date()), 'grid': grid}], mode="pattern")
        normalized = _normalize_prediction_dict(raw_pred)
        next_winners = [str(next_draw['1st_real']), str(next_draw['2nd_real']), str(next_draw['3rd_real'])]
        checked_preds = []
        for num, score, _reason in normalized:
            hit = "✅" if num in next_winners else "❌"
            checked_preds.append({"num": num, "score": score, "hit": hit})
        missing_digits = find_missing_digits(grid)
        history.append({"date": this_draw["date_parsed"].date(), "provider": this_draw["provider"], "drawn_number": this_draw["1st_real"], "predictions": checked_preds, "next_winners": next_winners, "missing_digits": missing_digits})
    return render_template("prediction_history.html", history=history)

@app.route('/consensus-predictor')
def consensus_predictor():
    df = load_csv_data()
    selected_provider = request.args.get('provider', 'all')
    if df.empty:
        return render_template('consensus_predictor.html', error="No data available")
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    advanced_preds = advanced_predictor(df, provider=None, lookback=200)
    smart_preds = smart_auto_weight_predictor(df, provider=None, lookback=300)
    ml_preds = ml_predictor(df, lookback=500)
    consensus_map = {}
    for num, score, reason in advanced_preds:
        if num not in consensus_map:
            consensus_map[num] = {'count': 0, 'total_score': 0, 'methods': []}
        consensus_map[num]['count'] += 1
        consensus_map[num]['total_score'] += score
        consensus_map[num]['methods'].append('Advanced')
    for num, score, reason in smart_preds:
        if num not in consensus_map:
            consensus_map[num] = {'count': 0, 'total_score': 0, 'methods': []}
        consensus_map[num]['count'] += 1
        consensus_map[num]['total_score'] += score
        consensus_map[num]['methods'].append('Smart')
    for num, score, reason in ml_preds:
        if num not in consensus_map:
            consensus_map[num] = {'count': 0, 'total_score': 0, 'methods': []}
        consensus_map[num]['count'] += 1
        consensus_map[num]['total_score'] += score
        consensus_map[num]['methods'].append('ML')
    overall_consensus = []
    for num, data in consensus_map.items():
        confidence = (data['count'] / 3) * 100
        overall_consensus.append({'number': num, 'confidence': round(confidence, 1), 'consensus': data['count'], 'reason': ', '.join(data['methods']), 'methods': data['methods']})
    overall_consensus.sort(key=lambda x: (x['consensus'], x['confidence']), reverse=True)
    provider_predictions = {}
    if selected_provider == 'all':
        for prov in provider_options[1:4]:
            prov_preds = advanced_predictor(df[df['provider'] == prov] if prov != 'all' else df, provider=prov, lookback=100)
            provider_predictions[prov] = [{'number': num, 'confidence': round(score * 100, 1), 'consensus': 1, 'reason': reason, 'methods': ['Advanced']} for num, score, reason in prov_preds[:5]]
    else:
        prov_preds = advanced_predictor(df[df['provider'] == selected_provider], provider=selected_provider, lookback=100)
        provider_predictions[selected_provider] = [{'number': num, 'confidence': round(score * 100, 1), 'consensus': 1, 'reason': reason, 'methods': ['Advanced']} for num, score, reason in prov_preds[:10]]
    return render_template('consensus_predictor.html', overall_consensus=overall_consensus, provider_predictions=provider_predictions, provider_options=provider_options, provider=selected_provider, last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/accuracy-dashboard')
def accuracy_dashboard():
    pred_file = "prediction_tracking.csv"
    if not os.path.exists(pred_file):
        return render_template('accuracy_dashboard.html', message="No predictions tracked yet.", stats={}, method_stats={}, recent_predictions=[], insights=[])
    pred_df = pd.read_csv(pred_file)
    df = load_csv_data()
    for idx, row in pred_df.iterrows():
        if row['hit_status'] == 'pending':
            try:
                draw_date_str = str(row['draw_date']).split('(')[0].strip()
                draw_date = pd.to_datetime(draw_date_str).date()
                provider = str(row['provider']).strip().lower()
                actual = df[(df['date_parsed'].dt.date == draw_date) & (df['provider'] == provider)]
                if not actual.empty:
                    actual_row = actual.iloc[0]
                    pred_df.at[idx, 'actual_1st'] = actual_row['1st_real']
                    pred_df.at[idx, 'actual_2nd'] = actual_row['2nd_real']
                    pred_df.at[idx, 'actual_3rd'] = actual_row['3rd_real']
                    predicted = eval(row['predicted_numbers']) if isinstance(row['predicted_numbers'], str) else []
                    actuals = [str(actual_row['1st_real']), str(actual_row['2nd_real']), str(actual_row['3rd_real'])]
                    hits = [p for p in predicted if p in actuals]
                    if hits:
                        pred_df.at[idx, 'hit_status'] = f'HIT-{len(hits)}'
                        pred_df.at[idx, 'accuracy_score'] = len(hits) * 100 / len(predicted) if predicted else 0
                    else:
                        pred_df.at[idx, 'hit_status'] = 'MISS'
                        pred_df.at[idx, 'accuracy_score'] = 0
            except Exception:
                pass
    pred_df.to_csv(pred_file, index=False)
    completed = pred_df[pred_df['hit_status'] != 'pending']
    stats = {'total_predictions': len(pred_df), 'completed': len(completed), 'pending': len(pred_df[pred_df['hit_status'] == 'pending']), 'total_hits': len(completed[completed['hit_status'].str.contains('HIT', na=False)]), 'total_misses': len(completed[completed['hit_status'] == 'MISS']), 'hit_rate': round(len(completed[completed['hit_status'].str.contains('HIT', na=False)]) / len(completed) * 100, 1) if len(completed) > 0 else 0, 'avg_accuracy': round(completed['accuracy_score'].mean(), 1) if len(completed) > 0 else 0}
    insights = []
    if stats['hit_rate'] < 30:
        insights.append("Low hit rate. Consider adjusting methods.")
    if stats['hit_rate'] > 50:
        insights.append("Good hit rate! Current methods working well.")
    recent = pred_df.tail(20).to_dict('records')
    return render_template('accuracy_dashboard.html', stats=stats, method_stats={}, recent_predictions=recent, insights=insights, message="")

@app.route('/learning-insights')
def learning_insights():
    pred_file = "prediction_tracking.csv"
    if not os.path.exists(pred_file):
        return render_template('learning_insights.html', message="No prediction data yet.", insights={})
    pred_df = pd.read_csv(pred_file)
    completed = pred_df[pred_df['hit_status'] != 'pending']
    if len(completed) == 0:
        return render_template('learning_insights.html', message="No completed predictions yet.", insights={})
    insights = {'total_predictions': len(completed), 'total_hits': len(completed[completed['hit_status'].str.contains('HIT', na=False)]), 'overall_accuracy': round((len(completed[completed['hit_status'].str.contains('HIT', na=False)]) / len(completed) * 100), 1) if len(completed) > 0 else 0}
    return render_template('learning_insights.html', message="", insights=insights)

@app.route('/quick-pick')
def quick_pick():
    df = load_csv_data()
    if df.empty:
        return render_template('quick_pick.html', numbers=[], error="No data available")
    advanced_preds = advanced_predictor(df, provider=None, lookback=200)
    smart_preds = smart_auto_weight_predictor(df, provider=None, lookback=300)
    ml_preds = ml_predictor(df, lookback=500)
    all_predictions = {}
    for num, score, _ in advanced_preds:
        all_predictions[num] = all_predictions.get(num, 0) + 1
    for num, score, _ in smart_preds:
        all_predictions[num] = all_predictions.get(num, 0) + 1
    for num, score, _ in ml_preds:
        all_predictions[num] = all_predictions.get(num, 0) + 1
    sorted_nums = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    top_5 = [num for num, count in sorted_nums[:5]]
    last_draw = df.iloc[-1]
    next_draw_date = (last_draw['date_parsed'] + timedelta(days=3)).strftime('%Y-%m-%d (%A)')
    return render_template('quick_pick.html', numbers=top_5, next_draw_date=next_draw_date, error=None)

@app.route('/save-prediction', methods=['POST'])
def save_prediction():
    data = request.get_json()
    pred_file = "prediction_tracking.csv"
    new_entry = {"prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "draw_date": data.get('draw_date'), "provider": data.get('provider'), "predicted_numbers": data.get('predicted_numbers'), "predictor_methods": data.get('methods'), "confidence": data.get('confidence'), "actual_1st": "", "actual_2nd": "", "actual_3rd": "", "hit_status": "pending", "accuracy_score": 0}
    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])
    df.to_csv(pred_file, index=False)
    return {"status": "success", "message": "Prediction saved"}

@app.route('/day-to-day-predictor')
def day_to_day_predictor():
    df = load_csv_data()
    selected_provider = request.args.get('provider', 'all')
    lookback_days = 14
    if df.empty:
        return render_template('day_to_day_predictor.html', predictions=[], today_numbers=[], provider='all', provider_options=['all'], month_options=[], selected_month='', today_date='N/A', total_numbers_analyzed=0, last_updated='No data')
    provider_options = ['all'] + sorted([p for p in df['provider'].dropna().unique() if p])
    latest_date = df['date_parsed'].max().date()
    cutoff_date = latest_date - timedelta(days=lookback_days)
    recent_df = df[df['date_parsed'].dt.date >= cutoff_date]
    if selected_provider != 'all':
        recent_df = recent_df[recent_df['provider'] == selected_provider]
    today_draws = df[df['date_parsed'].dt.date == latest_date]
    if selected_provider != 'all':
        today_draws = today_draws[today_draws['provider'] == selected_provider]
    today_numbers = []
    for _, row in today_draws.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                today_numbers.append(num)
    transitions = {}
    sorted_recent = recent_df.sort_values('date_parsed')
    for i in range(len(sorted_recent) - 1):
        curr_row = sorted_recent.iloc[i]
        next_row = sorted_recent.iloc[i + 1]
        for col in ['1st_real', '2nd_real', '3rd_real']:
            curr_num = str(curr_row.get(col, ''))
            next_num = str(next_row.get(col, ''))
            if curr_num and next_num and curr_num.isdigit() and next_num.isdigit():
                if curr_num not in transitions:
                    transitions[curr_num] = Counter()
                transitions[curr_num][next_num] += 1
    predictions = []
    if today_numbers:
        next_day_candidates = Counter()
        for num in today_numbers:
            if num in transitions:
                next_day_candidates.update(transitions[num])
        for num, count in next_day_candidates.most_common(5):
            score = count / len(today_numbers)
            predictions.append((num, score, f"Follows {count} times after today's numbers"))
    return render_template('day_to_day_predictor.html', predictions=predictions, today_numbers=today_numbers[:20], provider=selected_provider, provider_options=provider_options, month_options=[], selected_month='', today_date=latest_date, total_numbers_analyzed=len(today_numbers), last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

# Route removed - was showing duplicate data

@app.route('/frequency-analyzer')
def frequency_analyzer():
    df = load_csv_data()
    date_range = request.args.get('range', '30')
    if df.empty:
        return render_template('frequency_analyzer.html', frequencies=[], message="No data available")
    try:
        days = int(date_range)
    except:
        days = 30
    cutoff_date = df['date_parsed'].max() - timedelta(days=days)
    filtered_df = df[df['date_parsed'] >= cutoff_date]
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_numbers.extend([n for n in filtered_df[col].astype(str) if n.isdigit() and len(n) == 4])
    freq_counter = Counter(all_numbers)
    total_draws = len(filtered_df)
    frequencies = []
    for num, count in freq_counter.most_common():
        normalized = count / total_draws if total_draws > 0 else 0
        frequencies.append({'number': num, 'count': count, 'normalized': round(normalized, 4), 'percentage': round(normalized * 100, 2)})
    return render_template('frequency_analyzer.html', frequencies=frequencies, date_range=days, total_draws=total_draws, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/hot-cold')
def hot_cold():
    df = load_csv_data()
    lookback = int(request.args.get('lookback', '30'))
    if df.empty:
        return render_template('hot_cold.html', hot=[], cold=[], neutral=[], message="No data available")
    cutoff_date = df['date_parsed'].max() - timedelta(days=lookback)
    recent_df = df[df['date_parsed'] >= cutoff_date]
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_numbers.extend([n for n in recent_df[col].astype(str) if n.isdigit() and len(n) == 4])
    freq_counter = Counter(all_numbers)
    sorted_freq = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)
    total_unique = len(sorted_freq)
    hot_threshold = int(total_unique * 0.1)
    cold_threshold = int(total_unique * 0.9)
    hot = [{'number': num, 'count': count} for num, count in sorted_freq[:hot_threshold]]
    cold = [{'number': num, 'count': count} for num, count in sorted_freq[cold_threshold:]]
    neutral = [{'number': num, 'count': count} for num, count in sorted_freq[hot_threshold:cold_threshold]]
    return render_template('hot_cold.html', hot=hot, cold=cold, neutral=neutral, lookback=lookback, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/missing-number-finder')
def missing_number_finder():
    df = load_csv_data()
    if df.empty:
        return render_template('missing_number_finder.html', missing=[], message="No data available")
    all_numbers = set()
    last_seen = {}
    for idx, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                all_numbers.add(num)
                last_seen[num] = row['date_parsed']
    latest_date = df['date_parsed'].max()
    missing_data = []
    for num in sorted(all_numbers):
        if num in last_seen:
            days_absent = (latest_date - last_seen[num]).days
            draws_skipped = len(df[df['date_parsed'] > last_seen[num]])
            missing_data.append({'number': num, 'days_absent': days_absent, 'draws_skipped': draws_skipped, 'last_seen': last_seen[num].date()})
    missing_data.sort(key=lambda x: x['days_absent'], reverse=True)
    return render_template('missing_number_finder.html', missing=missing_data[:50], message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/best-predictions')
def best_predictions():
    df = load_csv_data()
    if df.empty:
        return render_template('best_predictions.html', predictions=[], message="No data available")
    advanced_preds = advanced_predictor(df, provider=None, lookback=200)
    smart_preds = smart_auto_weight_predictor(df, provider=None, lookback=300)
    ml_preds = ml_predictor(df, lookback=500)
    cutoff_date = df['date_parsed'].max() - timedelta(days=30)
    recent_df = df[df['date_parsed'] >= cutoff_date]
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_numbers.extend([n for n in recent_df[col].astype(str) if n.isdigit() and len(n) == 4])
    freq_counter = Counter(all_numbers)
    hot_numbers = set([num for num, _ in freq_counter.most_common(20)])
    all_last_seen = {}
    for idx, row in df.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                all_last_seen[num] = row['date_parsed']
    latest_date = df['date_parsed'].max()
    overdue_numbers = set()
    for num, last_date in all_last_seen.items():
        if (latest_date - last_date).days > 60:
            overdue_numbers.add(num)
    scored_predictions = {}
    for num, score, reason in advanced_preds:
        if num not in scored_predictions:
            scored_predictions[num] = {'score': 0, 'reasons': []}
        scored_predictions[num]['score'] += score * 0.3
        scored_predictions[num]['reasons'].append('Advanced')
    for num, score, reason in smart_preds:
        if num not in scored_predictions:
            scored_predictions[num] = {'score': 0, 'reasons': []}
        scored_predictions[num]['score'] += score * 0.3
        scored_predictions[num]['reasons'].append('Smart')
    for num, score, reason in ml_preds:
        if num not in scored_predictions:
            scored_predictions[num] = {'score': 0, 'reasons': []}
        scored_predictions[num]['score'] += score * 0.2
        scored_predictions[num]['reasons'].append('ML')
    for num in hot_numbers:
        if num not in scored_predictions:
            scored_predictions[num] = {'score': 0, 'reasons': []}
        scored_predictions[num]['score'] += 0.1
        scored_predictions[num]['reasons'].append('Hot')
    for num in overdue_numbers:
        if num not in scored_predictions:
            scored_predictions[num] = {'score': 0, 'reasons': []}
        scored_predictions[num]['score'] += 0.1
        scored_predictions[num]['reasons'].append('Overdue')
    final_predictions = []
    for num, data in scored_predictions.items():
        confidence = min(data['score'] * 100, 100)
        final_predictions.append({'number': num, 'confidence': round(confidence, 1), 'score': round(data['score'], 3), 'reasons': ', '.join(set(data['reasons']))})
    final_predictions.sort(key=lambda x: x['score'], reverse=True)
    return render_template('best_predictions.html', predictions=final_predictions[:20], message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/statistics')
def statistics():
    df = load_csv_data()
    if df.empty:
        return render_template('statistics.html', stats={}, message="No data available")
    total_draws = len(df)
    providers = df['provider'].value_counts().to_dict()
    date_range = f"{df['date_parsed'].min().date()} to {df['date_parsed'].max().date()}"
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_numbers.extend([n for n in df[col].astype(str) if n.isdigit() and len(n) == 4])
    digit_freq = Counter(''.join(all_numbers))
    most_common_digits = digit_freq.most_common()
    number_freq = Counter(all_numbers)
    most_common_numbers = number_freq.most_common(10)
    least_common_numbers = number_freq.most_common()[-10:]
    stats = {'total_draws': total_draws, 'providers': providers, 'date_range': date_range, 'total_numbers': len(all_numbers), 'unique_numbers': len(set(all_numbers)), 'digit_frequency': most_common_digits, 'most_common_numbers': most_common_numbers, 'least_common_numbers': least_common_numbers}
    return render_template('statistics.html', stats=stats, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/lucky-generator')
def lucky_generator():
    df = load_csv_data()
    if df.empty:
        return render_template('lucky_generator.html', lucky_numbers=[], message="No data available")
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_numbers.extend([n for n in df[col].astype(str) if n.isdigit() and len(n) == 4])
    digit_freq = Counter(''.join(all_numbers))
    lucky_digits = [d for d, _ in digit_freq.most_common(6)]
    lucky_numbers = []
    for _ in range(10):
        num = ''.join(np.random.choice(lucky_digits, 4))
        lucky_numbers.append(num)
    return render_template('lucky_generator.html', lucky_numbers=lucky_numbers, lucky_digits=lucky_digits, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/empty-box-predictor')
def empty_box_predictor():
    df = load_csv_data()
    if df.empty:
        return render_template('empty_box_predictor.html', predictions=[], message="No data available")
    recent = df.tail(20)
    all_grids = []
    for _, row in recent.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                grid = generate_4x4_grid(num)
                all_grids.append(grid)
    position_freq = {(r, c): Counter() for r in range(4) for c in range(4)}
    for grid in all_grids:
        for r in range(4):
            for c in range(4):
                position_freq[(r, c)][grid[r][c]] += 1
    predictions = []
    for pos, counter in position_freq.items():
        most_common = counter.most_common(3)
        predictions.append({'position': f"Row {pos[0]+1}, Col {pos[1]+1}", 'top_digits': [{'digit': d, 'count': c} for d, c in most_common]})
    return render_template('empty_box_predictor.html', predictions=predictions, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/master-analyzer')
def master_analyzer():
    df = load_csv_data()
    if df.empty:
        return render_template('master_analyzer.html', analysis={}, message="No data available")
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_numbers.extend([n for n in df[col].astype(str) if n.isdigit() and len(n) == 4])
    digit_patterns = Counter()
    for num in all_numbers:
        if len(set(num)) == 4:
            digit_patterns['all_unique'] += 1
        elif len(set(num)) == 1:
            digit_patterns['all_same'] += 1
        elif len(set(num)) == 2:
            digit_patterns['two_pairs'] += 1
        else:
            digit_patterns['mixed'] += 1
    sum_patterns = Counter([sum(int(d) for d in num) for num in all_numbers])
    even_odd = Counter()
    for num in all_numbers:
        evens = sum(1 for d in num if int(d) % 2 == 0)
        even_odd[f"{evens}E-{4-evens}O"] += 1
    analysis = {'total_analyzed': len(all_numbers), 'digit_patterns': dict(digit_patterns), 'sum_distribution': sum_patterns.most_common(10), 'even_odd_patterns': dict(even_odd)}
    return render_template('master_analyzer.html', analysis=analysis, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/smart-auto-weight')
def smart_auto_weight():
    df = load_csv_data()
    if df.empty:
        return render_template('smart_auto_weight.html', predictions=[], weights={}, message="No data available")
    top_5 = smart_auto_weight_predictor(df, provider=None, lookback=300)
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        if col in df.columns:
            all_numbers += [n for n in df[col].astype(str) if n.isdigit() and len(n) == 4]
    if len(all_numbers) < 20:
        return render_template('smart_auto_weight.html', predictions=[], weights={}, message="Insufficient data")
    all_numbers = all_numbers[-300:]
    digit_counts = Counter(''.join(all_numbers))
    pair_counts = Counter([n[i:i+2] for n in all_numbers for i in range(3)])
    transitions = learn_pattern_transitions([{"number": n} for n in all_numbers])
    features, targets = [], []
    for i in range(len(all_numbers) - 1):
        num = all_numbers[i]
        next_num = all_numbers[i + 1]
        hot_score = sum(digit_counts.get(d, 0) for d in num)
        pair_score = sum(pair_counts.get(num[i:i+2], 0) for i in range(3))
        trans_score = transitions.get(num, 0)
        features.append([hot_score, pair_score, trans_score])
        targets.append(len(set(num) & set(next_num)))
    if len(features) >= 5:
        X = np.array(features)
        y = np.array(targets)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        w_hot, w_pair, w_trans = model.coef_
        weights = {'hot': round(w_hot, 3), 'pair': round(w_pair, 3), 'transition': round(w_trans, 3)}
    else:
        weights = {'hot': 0.3, 'pair': 0.5, 'transition': 0.2}
    return render_template('smart_auto_weight.html', predictions=top_5, weights=weights, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/positional-tracker')
def positional_tracker():
    df = load_csv_data()
    if df.empty:
        return render_template('positional_tracker.html', patterns={}, message="No data available")
    patterns = analyze_positional_patterns(df)
    return render_template('positional_tracker.html', position_heatmap=patterns['position_heatmap'], repeating_positions=patterns['repeating_positions'], positional_swaps=patterns['positional_swaps'], placeholder_patterns=patterns['placeholder_patterns'], message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/arithmetic-gap-analyzer')
def arithmetic_gap_analyzer():
    df = load_csv_data()
    if df.empty:
        return render_template('arithmetic_gap_analyzer.html', patterns={}, message="No data available")
    patterns = analyze_arithmetic_gaps(df)
    precursors = find_high_prize_precursors(df)
    return render_template('arithmetic_gap_analyzer.html', gap_frequencies=patterns['gap_frequencies'], arithmetic_progressions=patterns['arithmetic_progressions'], mirrored_gaps=patterns['mirrored_gaps'], high_prize_precursors=precursors, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/temporal-recurrence')
def temporal_recurrence():
    df = load_csv_data()
    if df.empty:
        return render_template('temporal_recurrence.html', recurrences=[], message="No data available")
    recurrences = map_temporal_recurrence(df)
    return render_template('temporal_recurrence.html', recurrences=recurrences, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/cross-draw-linker')
def cross_draw_linker():
    df = load_csv_data()
    if df.empty:
        return render_template('cross_draw_linker.html', patterns={}, message="No data available")
    patterns = link_cross_draw_patterns(df)
    return render_template('cross_draw_linker.html', cross_provider_map=patterns['cross_provider_map'], shared_motifs=patterns['shared_motifs'], message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/structural-clusters')
def structural_clusters():
    df = load_csv_data()
    if df.empty:
        return render_template('structural_clusters.html', clusters=[], message="No data available")
    result = cluster_structural_motifs(df)
    return render_template('structural_clusters.html', clusters=result.get('clusters', []), message=result.get('message', ''), last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/pattern-weighting')
def pattern_weighting():
    df = load_csv_data()
    if df.empty:
        return render_template('pattern_weighting.html', patterns={}, message="No data available")
    patterns = weight_patterns_predictively(df)
    return render_template('pattern_weighting.html', weighted_leaderboard=patterns['weighted_leaderboard'], suggested_combinations=patterns['suggested_combinations'], message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/auto-predictor-dashboard')
def auto_predictor_dashboard():
    pred_file = 'daily_predictions.csv'
    if not os.path.exists(pred_file):
        return render_template('auto_predictor_dashboard.html', predictions=[], stats={}, message="No predictions yet. Run: python auto_predictor.py predict")
    
    pred_df = pd.read_csv(pred_file)
    learned = pred_df[pred_df['learned'] == True]
    
    stats = {
        'total': len(pred_df),
        'learned': len(learned),
        'pending': len(pred_df[pred_df['learned'] == False]),
        'avg_accuracy': round(learned['accuracy'].mean(), 2) if len(learned) > 0 else 0,
        'best_match': int(learned['matches'].max()) if len(learned) > 0 else 0,
        'total_matches': int(learned['matches'].sum()) if len(learned) > 0 else 0
    }
    
    if 'hot_cold_score' in learned.columns and len(learned) > 0:
        stats['avg_hot_cold'] = round(learned['hot_cold_score'].mean(), 4)
    
    recent = pred_df.tail(20).to_dict('records')
    
    return render_template('auto_predictor_dashboard.html', predictions=recent, stats=stats, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/run-auto-predict', methods=['POST'])
def run_auto_predict():
    try:
        import subprocess
        result = subprocess.run(['python', 'auto_predictor.py', 'predict'], capture_output=True, text=True)
        return {'status': 'success', 'output': result.stdout}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.route('/master-dashboard')
def master_dashboard():
    csv_file = 'master_predictions.csv'
    if not os.path.exists(csv_file):
        return render_template('master_dashboard.html', predictions=[], stats={}, latest_prediction=None, module_performance={}, message="No predictions yet. Run: python prediction_engine.py", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    
    pred_df = pd.read_csv(csv_file)
    learned = pred_df[pred_df['learned'] == True]
    
    stats = {
        'total': len(pred_df),
        'learned': len(learned),
        'pending': len(pred_df) - len(learned),
        'avg_matches': round(learned['match_count'].mean(), 2) if len(learned) > 0 else 0,
        'best_match': int(learned['match_count'].max()) if len(learned) > 0 else 0,
        'total_matches': int(learned['match_count'].sum()) if len(learned) > 0 else 0
    }
    
    # Module performance
    module_perf = {}
    if len(learned) > 0:
        for source in learned['pattern_source'].unique():
            source_data = learned[learned['pattern_source'].str.contains(str(source), na=False)]
            if len(source_data) > 0:
                avg = source_data['match_count'].mean()
                module_perf[source] = round((avg / 6) * 100, 1)
    
    latest = pred_df.iloc[-1].to_dict() if len(pred_df) > 0 else None
    recent = pred_df.tail(20).to_dict('records')
    
    return render_template('master_dashboard.html', predictions=recent, stats=stats, latest_prediction=latest, module_performance=module_perf, message="", last_updated=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/run-master-predict', methods=['POST'])
def run_master_predict():
    try:
        import subprocess
        result = subprocess.run(['python', 'prediction_engine.py'], capture_output=True, text=True)
        return {'status': 'success', 'output': result.stdout}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.route('/add-master-result', methods=['POST'])
def add_master_result():
    try:
        data = request.get_json()
        date = data.get('date')
        numbers = data.get('numbers').split(',')
        
        import subprocess
        result = subprocess.run(['python', 'learning_engine.py', 'add_result', date, ','.join(numbers)], capture_output=True, text=True)
        return {'status': 'success', 'message': result.stdout}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.route('/retrain-master-model', methods=['POST'])
def retrain_master_model():
    try:
        import subprocess
        result = subprocess.run(['python', 'learning_engine.py', 'retrain'], capture_output=True, text=True)
        return {'status': 'success', 'message': result.stdout}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.route('/master-stats')
def master_stats():
    try:
        import subprocess
        result = subprocess.run(['python', 'learning_engine.py', 'stats'], capture_output=True, text=True)
        return {'status': 'success', 'stats': result.stdout}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
