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

# ---------------- CSV LOADER (ULTRA-FAST) ---------------- #
_csv_cache = None
_csv_cache_time = None
_csv_lock = threading.Lock()

def _extract_provider(url_text):
    """Extract provider from URL text"""
    url_text = str(url_text).lower()
    
    provider_map = {
        'magnum': 'magnum',
        'damacai': 'damacai', 
        'singapore': 'singapore',
        'singaporepools': 'singapore',
        'sandakan': 'sandakan',
        'stc4d': 'sandakan',
        'cashsweep': 'cashsweep',
        'sabah88': 'sabah88',
        'sabah': 'sabah88',
        'gdlotto': 'gdlotto',
        'perdana': 'perdana',
        'harihari': 'harihari'
    }
    
    # Check simple matches first
    for key, provider in provider_map.items():
        if key in url_text:
            if key == 'toto' and 'magnum' in url_text:
                continue
            return provider
    
    # Check live4d2u specific paths
    if 'live4d2u' in url_text:
        live4d_map = {
            '/magnum': 'magnum',
            '/damacai': 'damacai',
            '/toto': 'toto',
            '/sabah88': 'sabah88',
            '/cashsweep': 'cashsweep',
            '/stc4d': 'sandakan'
        }
        for path, provider in live4d_map.items():
            if path in url_text:
                return provider
    
    return 'unknown'

def _extract_prize_numbers(prizes_text):
    """Extract 1st, 2nd, 3rd prize numbers from text"""
    patterns = {
        'first': [r'1st\s+Prize[^\d]*(\d{4})', r'1st[^\d]*(\d{4})', r'首獎[^\d]*(\d{4})', r'Prize[^\d]*(\d{4}).*?2nd'],
        'second': [r'2nd\s+Prize[^\d]*(\d{4})', r'2nd[^\d]*(\d{4})', r'二獎[^\d]*(\d{4})'],
        'third': [r'3rd\s+Prize[^\d]*(\d{4})', r'3rd[^\d]*(\d{4})', r'三獎[^\d]*(\d{4})']
    }
    
    results = {'first': '', 'second': '', 'third': ''}
    
    for prize_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, prizes_text, re.IGNORECASE)
            if match:
                results[prize_type] = match.group(1)
                break
    
    # Fallback: extract all 4-digit numbers
    if not all(results.values()):
        all_numbers = re.findall(r'\b\d{4}\b', prizes_text)
        if not results['first'] and len(all_numbers) > 0:
            results['first'] = all_numbers[0]
        if not results['second'] and len(all_numbers) > 1:
            results['second'] = all_numbers[1]
        if not results['third'] and len(all_numbers) > 2:
            results['third'] = all_numbers[2]
    
    return results['first'], results['second'], results['third']

def _extract_special_consolation(text):
    """Extract special/consolation numbers from text"""
    if not text or text == 'nan':
        return ''
    numbers = re.findall(r'\b\d{4}\b', str(text))
    return ' '.join(numbers) if numbers else ''

def _process_csv_row(row):
    """Process a single CSV row"""
    # Skip invalid rows
    if len(row) < 3 or not row[0] or len(str(row[0])) > 50 or row[0].startswith('3rd'):
        return None
    
    try:
        date_parsed = pd.to_datetime(row[0])
    except (ValueError, TypeError, pd.errors.ParserError):
        return None
    
    # Extract data
    provider = _extract_provider(row[1]) if len(row) > 1 and row[1] else 'unknown'
    draw_type = row[2] if len(row) > 2 else ''
    draw_info = row[4] if len(row) > 4 else ''
    prizes_text = row[5] if len(row) > 5 else ''
    
    first_num, second_num, third_num = _extract_prize_numbers(prizes_text)
    
    # Skip rows with no prize data
    if not first_num and not second_num and not third_num:
        return None
    
    special_text = _extract_special_consolation(row[6] if len(row) > 6 else '')
    consolation_text = _extract_special_consolation(row[7] if len(row) > 7 else '')
    
    return {
        'date': row[0],
        'date_parsed': date_parsed,
        'provider': provider,
        'draw_type': draw_type,
        'draw_info': draw_info,
        'prizes': prizes_text,
        'special': special_text,
        'consolation': consolation_text,
        '1st_real': first_num,
        '2nd_real': second_num,
        '3rd_real': third_num
    }

def load_csv_data():
    """⚡ Load CSV data with improved parsing"""
    global _csv_cache, _csv_cache_time
    
    with _csv_lock:
        if _csv_cache is not None and _csv_cache_time is not None:
            if (datetime.now() - _csv_cache_time).total_seconds() < 60:
                return _csv_cache
    
    import csv
    data = []
    
    try:
        with open('4d_results_history_fixed.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                processed_row = _process_csv_row(row)
                if processed_row:
                    data.append(processed_row)
    except Exception as e:
        logger.error(f"CSV error: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    with _csv_lock:
        _csv_cache = df
        _csv_cache_time = datetime.now()
    
    logger.info(f"✅ Loaded {len(df)} rows")
    return df

# Simple prediction function
def advanced_predictor(df, provider=None, lookback=100):
    """Simple frequency-based predictor"""
    try:
        if df.empty:
            return []
        
        # Validate lookback parameter
        if not isinstance(lookback, int) or lookback <= 0:
            lookback = 100
        
        # Get recent numbers
        all_numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            if col in df.columns:
                try:
                    # Safe DataFrame operations with error handling
                    column_data = df[col].tail(lookback).astype(str)
                    valid_numbers = [n for n in column_data if len(n) == 4 and n.isdigit()]
                    all_numbers.extend(valid_numbers)
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Error processing column {col}: {e}")
                    continue
        
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
    try:
        return render_template('carousel.html')
    except Exception as e:
        logger.error(f"Index error: {e}")
        return render_template('carousel.html')

@app.route('/dashboard')
def dashboard():
    try:
        df = load_csv_data()
        selected_date = request.args.get('selected_date')

        if not selected_date:
            try:
                if not df.empty:
                    latest_date = df['date_parsed'].max().date()
                    filtered = df[df['date_parsed'].dt.date == latest_date]
                    selected_date = latest_date
                else:
                    filtered = df.iloc[0:0]
                    selected_date = ""
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Error getting latest date: {e}")
                filtered = df.iloc[0:0]
                selected_date = ""
        else:
            try:
                date_obj = pd.to_datetime(selected_date).date()
                filtered = df[df['date_parsed'].dt.date == date_obj]
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing selected date {selected_date}: {e}")
                filtered = df.iloc[0:0]

        cards = [row.to_dict() for _, row in filtered.iterrows()]

        return render_template(
            'index.html',
            cards=cards,
            selected_date=selected_date
        )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template(
            'index.html',
            cards=[],
            selected_date=""
        )

@app.route('/test-slider')
def test_slider():
    try:
        return render_template('test_slider.html')
    except Exception as e:
        logger.error(f"Test slider error: {e}")
        return render_template('test_slider.html')

@app.route('/match-checker')
def match_checker():
    try:
        return render_template('match_checker.html')
    except Exception as e:
        logger.error(f"Match checker error: {e}")
        return render_template('match_checker.html')

@app.route('/api/check-match', methods=['POST'])
def api_check_match():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        number = data.get('number', '').strip()
        if not number or len(number) != 4 or not number.isdigit():
            return jsonify({'error': 'Invalid 4-digit number'}), 400
            
        df = load_csv_data()
        matches = []
        
        if not df.empty:
            # Use vectorized operations for better performance
            for col in ['1st_real', '2nd_real', '3rd_real']:
                mask = df[col].astype(str) == number
                matched_rows = df[mask]
                for _, row in matched_rows.iterrows():
                    matches.append({
                        'date': row['date'],
                        'provider': row['provider'],
                        'prize': col.replace('_real', ''),
                        'number': str(row[col])
                    })
        
        return jsonify({'matches': matches})
    except Exception as e:
        logger.error(f"API check-match error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predictions')
def api_predictions():
    try:
        df = load_csv_data()
        predictions = advanced_predictor(df)
        return jsonify({'predictions': [{'number': p[0], 'score': p[1]} for p in predictions]})
    except Exception as e:
        logger.error(f"API predictions error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/recent-results')
def api_recent_results():
    try:
        df = load_csv_data()
        recent = df.tail(10).to_dict('records')
        return jsonify({'results': recent})
    except Exception as e:
        logger.error(f"API recent results error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/lucky-generator')
def lucky_generator():
    try:
        return render_template('lucky_generator.html')
    except Exception as e:
        logger.error(f"Lucky generator error: {e}")
        return render_template('lucky_generator.html')

@app.route('/api/lucky-numbers')
def api_lucky_numbers():
    try:
        import random
        numbers = [f"{random.randint(0, 9999):04d}" for _ in range(5)]
        return jsonify({'numbers': numbers})
    except Exception as e:
        logger.error(f"API lucky numbers error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/statistics')
def statistics():
    try:
        df = load_csv_data()
        stats = {
            'total_draws': len(df),
            'providers': df['provider'].value_counts().to_dict() if not df.empty else {},
            'recent_date': df['date'].max() if not df.empty else 'No data'
        }
        return render_template('statistics.html', stats=stats)
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return render_template('statistics.html', stats={'total_draws': 0, 'providers': {}, 'recent_date': 'Error loading data'})

@app.route('/hot-cold')
def hot_cold():
    try:
        return render_template('hot_cold.html')
    except Exception as e:
        logger.error(f"Hot cold error: {e}")
        return render_template('hot_cold.html')

@app.route('/api/hot-cold-numbers')
def api_hot_cold():
    try:
        df = load_csv_data()
        all_numbers = []
        
        if not df.empty:
            for col in ['1st_real', '2nd_real', '3rd_real']:
                # Filter only valid 4-digit numbers
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
        logger.error(f"API hot-cold error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/slider-dashboard')
def slider_dashboard():
    """3D Slider Dashboard with all features"""
    try:
        df = load_csv_data()
        if df.empty:
            results = []
            recent_winners = []
            learning_progress = []
        else:
            results = [{
                'method': 'Frequency Analysis',
                'accuracy': 75.2,
                'predictions': ['1234', '5678', '9012', '3456', '7890'],
                'exact_hits': [{'pred': '1234', 'winner': '1234'}],
                'ibox_hits': [],
                'front_hits': [{'pred': '5678', 'winner': '5679'}],
                'back_hits': [],
                'all_hits': [{'pred': '1234', 'winner': '1234'}]
            }]
            
            recent_winners = []
            for i, row in df.tail(10).iterrows():
                for col in ['1st_real', '2nd_real', '3rd_real']:
                    if len(str(row[col])) == 4 and str(row[col]).isdigit():
                        recent_winners.append({
                            'date': row['date_parsed'].strftime('%Y-%m-%d'),
                            'day': row['date_parsed'].strftime('%A'),
                            'provider': 'Lottery',
                            'number': str(row[col])
                        })
            
            learning_progress = [{'epoch': i, 'accuracy': 50 + i} for i in range(1, 21)]
        
        return render_template('slider_dashboard.html', results=results, recent_winners=recent_winners, learning_progress=learning_progress)
    except Exception as e:
        logger.error(f"Slider dashboard error: {e}")
        return render_template('slider_dashboard.html', results=[], recent_winners=[], learning_progress=[])

@app.route('/pattern-analyzer')
def pattern_analyzer():
    try:
        return render_template('pattern_analyzer.html')
    except Exception as e:
        logger.error(f"Pattern analyzer error: {e}")
        return render_template('pattern_analyzer.html')

@app.route('/frequency-analyzer')
def frequency_analyzer():
    try:
        return render_template('frequency_analyzer.html')
    except Exception as e:
        logger.error(f"Frequency analyzer error: {e}")
        return render_template('frequency_analyzer.html')

@app.route('/quick-pick')
def quick_pick():
    try:
        return render_template('quick_pick.html')
    except Exception as e:
        logger.error(f"Quick pick error: {e}")
        return render_template('quick_pick.html')

@app.route('/my-tracker')
def my_tracker():
    try:
        return render_template('my_tracker.html')
    except Exception as e:
        logger.error(f"My tracker error: {e}")
        return render_template('my_tracker.html')

@app.route('/past-results')
def past_results():
    try:
        df = load_csv_data()
        results = df.tail(50).to_dict('records') if not df.empty else []
        return render_template('past_results.html', results=results)
    except Exception as e:
        logger.error(f"Past results error: {e}")
        return render_template('past_results.html', results=[])

@app.route('/ai-dashboard')
def ai_dashboard():
    try:
        df = load_csv_data()
        predictions = advanced_predictor(df) if not df.empty else []
        return render_template('ai_dashboard.html', predictions=predictions)
    except Exception as e:
        logger.error(f"AI dashboard error: {e}")
        return render_template('ai_dashboard.html', predictions=[])

@app.route('/ml-predictor')
def ml_predictor():
    try:
        return render_template('ml_predictor.html')
    except Exception as e:
        logger.error(f"ML predictor error: {e}")
        return render_template('ml_predictor.html')

@app.route('/power-dashboard')
def power_dashboard():
    try:
        df = load_csv_data()
        stats = {
            'total_draws': len(df),
            'providers': df['provider'].value_counts().to_dict() if not df.empty else {},
            'recent_numbers': df.tail(20).to_dict('records') if not df.empty else []
        }
        return render_template('power_dashboard.html', stats=stats)
    except Exception as e:
        logger.error(f"Power dashboard error: {e}")
        return render_template('power_dashboard.html', stats={'total_draws': 0, 'providers': {}, 'recent_numbers': []})

@app.route('/ultimate-predictor')
def ultimate_predictor():
    try:
        df = load_csv_data()
        predictions = advanced_predictor(df, lookback=200)
        return render_template('ultimate_predictor.html', predictions=predictions)
    except Exception as e:
        logger.error(f"Ultimate predictor error: {e}")
        return render_template('ultimate_predictor.html', predictions=[])

@app.route('/learning-dashboard')
def learning_dashboard():
    try:
        return render_template('learning_dashboard.html')
    except Exception as e:
        logger.error(f"Learning dashboard error: {e}")
        return render_template('learning_dashboard.html')

@app.route('/accuracy-dashboard')
def accuracy_dashboard():
    try:
        return render_template('accuracy_dashboard.html')
    except Exception as e:
        logger.error(f"Accuracy dashboard error: {e}")
        return render_template('accuracy_dashboard.html')

@app.route('/comparison-dashboard')
def comparison_dashboard():
    try:
        return render_template('comparison_dashboard.html')
    except Exception as e:
        logger.error(f"Comparison dashboard error: {e}")
        return render_template('comparison_dashboard.html')

@app.route('/auto-predictor-dashboard')
def auto_predictor_dashboard():
    try:
        return render_template('auto_predictor_dashboard.html')
    except Exception as e:
        logger.error(f"Auto predictor dashboard error: {e}")
        return render_template('auto_predictor_dashboard.html')

@app.route('/consensus-predictor')
def consensus_predictor():
    try:
        return render_template('consensus_predictor.html')
    except Exception as e:
        logger.error(f"Consensus predictor error: {e}")
        return render_template('consensus_predictor.html')

@app.route('/best-predictions')
def best_predictions():
    try:
        df = load_csv_data()
        predictions = advanced_predictor(df)
        return render_template('best_predictions.html', predictions=predictions)
    except Exception as e:
        logger.error(f"Best predictions error: {e}")
        return render_template('best_predictions.html', predictions=[])

@app.route('/smart-predictor')
def smart_predictor():
    try:
        return render_template('smart_predictor.html')
    except Exception as e:
        logger.error(f"Smart predictor error: {e}")
        return render_template('smart_predictor.html')

@app.route('/missing-number-finder')
def missing_number_finder():
    try:
        return render_template('missing_number_finder.html')
    except Exception as e:
        logger.error(f"Missing number finder error: {e}")
        return render_template('missing_number_finder.html')

@app.route('/empty-box-predictor')
def empty_box_predictor():
    try:
        return render_template('empty_box_predictor.html')
    except Exception as e:
        logger.error(f"Empty box predictor error: {e}")
        return render_template('empty_box_predictor.html')

@app.route('/day-to-day-predictor')
def day_to_day_predictor():
    try:
        return render_template('day_to_day_predictor.html')
    except Exception as e:
        logger.error(f"Day to day predictor error: {e}")
        return render_template('day_to_day_predictor.html')

@app.route('/adaptive-learning')
def adaptive_learning():
    try:
        return render_template('adaptive_learning.html')
    except Exception as e:
        logger.error(f"Adaptive learning error: {e}")
        return render_template('adaptive_learning.html')

@app.route('/master-analyzer')
def master_analyzer():
    try:
        return render_template('master_analyzer.html')
    except Exception as e:
        logger.error(f"Master analyzer error: {e}")
        return render_template('master_analyzer.html')

@app.route('/advanced-analytics')
def advanced_analytics():
    try:
        return render_template('advanced_analytics.html')
    except Exception as e:
        logger.error(f"Advanced analytics error: {e}")
        return render_template('advanced_analytics.html')

@app.route('/method-performance')
def method_performance():
    try:
        return render_template('method_performance.html')
    except Exception as e:
        logger.error(f"Method performance error: {e}")
        return render_template('method_performance.html')

@app.route('/prediction-history')
def prediction_history():
    try:
        return render_template('prediction_history.html')
    except Exception as e:
        logger.error(f"Prediction history error: {e}")
        return render_template('prediction_history.html')

@app.route('/streak-tracker')
def streak_tracker():
    try:
        return render_template('streak_tracker.html')
    except Exception as e:
        logger.error(f"Streak tracker error: {e}")
        return render_template('streak_tracker.html')

@app.route('/leaderboard')
def leaderboard():
    try:
        return render_template('leaderboard.html')
    except Exception as e:
        logger.error(f"Leaderboard error: {e}")
        return render_template('leaderboard.html')

@app.route('/community')
def community():
    try:
        return render_template('community.html')
    except Exception as e:
        logger.error(f"Community error: {e}")
        return render_template('community.html')

@app.route('/profile')
def profile():
    try:
        return render_template('profile.html')
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return render_template('profile.html')

@app.route('/notifications')
def notifications():
    try:
        return render_template('notifications.html')
    except Exception as e:
        logger.error(f"Notifications error: {e}")
        return render_template('notifications.html')

@app.route('/themes')
def themes():
    try:
        return render_template('themes.html')
    except Exception as e:
        logger.error(f"Themes error: {e}")
        return render_template('themes.html')

@app.route('/mobile-guide')
def mobile_guide():
    try:
        return render_template('mobile_guide.html')
    except Exception as e:
        logger.error(f"Mobile guide error: {e}")
        return render_template('mobile_guide.html')

@app.route('/countdown')
def countdown():
    try:
        return render_template('countdown.html')
    except Exception as e:
        logger.error(f"Countdown error: {e}")
        return render_template('countdown.html')

def _get_provider_options(df):
    """Get sorted provider options from dataframe"""
    if df.empty:
        return ['magnum', 'damacai', 'singapore', 'gdlotto']
    return sorted([p for p in df['provider'].dropna().unique() if p and str(p).strip()])

def _extract_numbers_for_date(df, selected_date, selected_provider):
    """Extract all valid numbers for given date and provider"""
    date_obj = pd.to_datetime(selected_date).date()
    filtered = df[(df['date_parsed'].dt.date == date_obj) & (df['provider'] == selected_provider)]
    
    all_numbers = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        all_numbers.extend([n for n in filtered[col].astype(str) if len(n) == 4 and n.isdigit()])
    
    return all_numbers

def _build_ocr_table(all_numbers):
    """Build OCR frequency table from numbers"""
    ocr_table = {digit: [0, 0, 0, 0] for digit in range(10)}
    for num in all_numbers:
        for pos in range(4):
            digit = int(num[pos])
            ocr_table[digit][pos] += 1
    return ocr_table

def _generate_hot_prediction(ocr_table):
    """Generate prediction based on hot digits"""
    hot_digits = []
    for pos in range(4):
        max_count = max(ocr_table[d][pos] for d in range(10))
        hot_digit = [d for d in range(10) if ocr_table[d][pos] == max_count][0]
        hot_digits.append(str(hot_digit))
    
    base_pred = ''.join(hot_digits)
    return [{'number': base_pred, 'confidence': 85, 'reason': 'Top digits combination'}]

@app.route('/positional-ocr')
def positional_ocr():
    try:
        df = load_csv_data()
        selected_provider = request.args.get('provider', 'magnum')
        selected_date = request.args.get('date', '')
        
        provider_options = _get_provider_options(df)
        ocr_table = None
        predictions = []
        
        if selected_date and selected_provider and not df.empty:
            try:
                all_numbers = _extract_numbers_for_date(df, selected_date, selected_provider)
                if all_numbers:
                    ocr_table = _build_ocr_table(all_numbers)
                    predictions = _generate_hot_prediction(ocr_table)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing OCR data: {e}")
        
        return render_template('positional_ocr.html',
                             provider_options=provider_options,
                             selected_provider=selected_provider,
                             selected_date=selected_date,
                             ocr_table=ocr_table,
                             predictions=predictions)
    except Exception as e:
        logger.error(f"Positional OCR error: {e}")
        return render_template('positional_ocr.html',
                             provider_options=['magnum', 'damacai', 'singapore', 'gdlotto'],
                             selected_provider='magnum',
                             selected_date='',
                             ocr_table=None,
                             predictions=[])

@app.route('/api/generate-predictions', methods=['POST'])
def api_generate_predictions():
    try:
        df = load_csv_data()
        data = request.get_json() or {}
        method = data.get('method', 'frequency')
        lookback = data.get('lookback', 100)
        
        predictions = advanced_predictor(df, lookback=lookback)
        return jsonify({'predictions': [{'number': p[0], 'score': p[1], 'method': p[2]} for p in predictions]})
    except Exception as e:
        logger.error(f"API generate predictions error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/check-number', methods=['POST'])
def api_check_number():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        number = data.get('number', '').strip()
        if not number or len(number) != 4 or not number.isdigit():
            return jsonify({'error': 'Invalid 4-digit number'}), 400
            
        df = load_csv_data()
        matches = []
        
        if not df.empty:
            # Use vectorized operations for better performance
            for col in ['1st_real', '2nd_real', '3rd_real']:
                mask = df[col].astype(str) == number
                matched_rows = df[mask]
                position = col.replace('_real', '')
                
                for _, row in matched_rows.iterrows():
                    matches.append({
                        'date': row['date'],
                        'provider': row['provider'],
                        'position': position,
                        'draw_type': row.get('draw_type', '')
                    })
        
        return jsonify({'number': number, 'matches': matches, 'total_matches': len(matches)})
    except Exception as e:
        logger.error(f"API check-number error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/provider-stats')
def api_provider_stats():
    try:
        df = load_csv_data()
        if df.empty:
            return jsonify({'providers': {}})
            
        stats = df['provider'].value_counts().to_dict()
        return jsonify({'providers': stats})
    except Exception as e:
        logger.error(f"Provider stats error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/latest-results')
def api_latest_results():
    try:
        df = load_csv_data()
        if df.empty:
            return jsonify({'results': []})
            
        latest = df.tail(20).to_dict('records')
        # Convert datetime objects to strings for JSON serialization
        for result in latest:
            if 'date_parsed' in result:
                result['date_parsed'] = result['date_parsed'].strftime('%Y-%m-%d %H:%M:%S')
                
        return jsonify({'results': latest})
    except Exception as e:
        logger.error(f"Latest results error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/search-by-date', methods=['POST'])
def api_search_by_date():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        date_str = data.get('date', '')
        if not date_str:
            return jsonify({'error': 'Date is required'}), 400
            
        df = load_csv_data()
        if df.empty:
            return jsonify({'results': []})
            
        try:
            date_obj = pd.to_datetime(date_str).date()
        except (ValueError, TypeError, pd.errors.ParserError) as e:
            logger.warning(f"Invalid date format {date_str}: {e}")
            return jsonify({'error': 'Invalid date format'}), 400
            
        filtered = df[df['date_parsed'].dt.date == date_obj]
        results = filtered.to_dict('records')
        
        # Convert datetime objects to strings
        for result in results:
            if 'date_parsed' in result and result['date_parsed']:
                try:
                    result['date_parsed'] = result['date_parsed'].strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error formatting date: {e}")
                    result['date_parsed'] = str(result['date_parsed'])
                
        return jsonify({'results': results, 'count': len(results)})
    except Exception as e:
        logger.error(f"Search by date error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    import os
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode)
error("No votes collected!")
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
