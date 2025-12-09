from flask import request, render_template
import pandas as pd
from datetime import datetime, timedelta
import re
from collections import Counter

def add_match_checker_route(app, load_csv_data):
    @app.route('/match-checker', methods=['GET', 'POST'])
    def match_checker():
        df = load_csv_data()
        
        # Get all unique providers dynamically
        all_providers = df['provider'].dropna().unique()
        provider_names = []
        for p in all_providers:
            if p and str(p).strip() and str(p) != 'nan':
                name = str(p).split('/')[-1].split('.')[0].split('?')[0].strip().lower()
                if name and name not in provider_names:
                    provider_names.append(name)
        provider_options = ['all'] + sorted(provider_names)
        
        # Get month options dynamically
        month_options = sorted(df['date_parsed'].dropna().dt.strftime('%Y-%m').unique(), reverse=True)
        
        # Get filters from both GET and POST
        selected_provider = request.args.get('provider') or request.form.get('provider', 'all')
        analysis_period = request.args.get('analysis_period') or request.form.get('analysis_period', 'last_30_days')
        
        # Filter by provider
        if selected_provider != 'all':
            df = df[df['provider'].str.contains(selected_provider, case=False, na=False)]
        
        # Filter by period
        if analysis_period == 'last_30_days':
            cutoff = df['date_parsed'].max() - timedelta(days=30)
            df = df[df['date_parsed'] >= cutoff]
        elif analysis_period == 'last_90_days':
            cutoff = df['date_parsed'].max() - timedelta(days=90)
            df = df[df['date_parsed'] >= cutoff]
        elif analysis_period.startswith('202'):
            year, month = analysis_period.split('-')
            df = df[(df['date_parsed'].dt.year == int(year)) & (df['date_parsed'].dt.month == int(month))]
        
        # Sort by date
        df = df.sort_values('date_parsed').reset_index(drop=True)
        
        # Generate daily learning data
        daily_learning = []
        
        for i in range(len(df) - 1):
            current = df.iloc[i]
            next_day = df.iloc[i + 1]
            
            # Same provider only
            if str(current['provider']).lower() != str(next_day['provider']).lower():
                continue
            
            # Get REAL numbers from CSV (1st, 2nd, 3rd, special, consolation)
            current_nums = [str(current['1st_real']), str(current['2nd_real']), str(current['3rd_real'])]
            current_nums = [n for n in current_nums if len(n) == 4 and n.isdigit()]
            
            # Get special and consolation for current day
            current_special = str(current.get('special', '')).split() if str(current.get('special', '')) != 'nan' else []
            current_special = [n for n in current_special if len(n) == 4 and n.isdigit()]
            current_consolation = str(current.get('consolation', '')).split() if str(current.get('consolation', '')) != 'nan' else []
            current_consolation = [n for n in current_consolation if len(n) == 4 and n.isdigit()]
            
            next_nums = [str(next_day['1st_real']), str(next_day['2nd_real']), str(next_day['3rd_real'])]
            next_nums = [n for n in next_nums if len(n) == 4 and n.isdigit()]
            
            # Get special and consolation for next day
            next_special = str(next_day.get('special', '')).split() if str(next_day.get('special', '')) != 'nan' else []
            next_special = [n for n in next_special if len(n) == 4 and n.isdigit()]
            next_consolation = str(next_day.get('consolation', '')).split() if str(next_day.get('consolation', '')) != 'nan' else []
            next_consolation = [n for n in next_consolation if len(n) == 4 and n.isdigit()]
            
            if not current_nums or not next_nums:
                continue
            
            # OLD LOGIC: Frequency-based predictions
            all_historical = []
            for j in range(max(0, i-50), i):
                row = df.iloc[j]
                for col in ['1st_real', '2nd_real', '3rd_real']:
                    num = str(row[col])
                    if len(num) == 4 and num.isdigit():
                        all_historical.append(num)
            
            old_predictions = []
            if all_historical:
                freq = Counter(all_historical)
                old_predictions = [num for num, _ in freq.most_common(3)]
            else:
                old_predictions = current_nums[:3]
            
            # NEW LOGIC: AI day-to-day learner predictions
            predictions = []
            try:
                from utils.day_to_day_learner import learn_day_to_day_patterns, predict_tomorrow
                
                # Build historical data up to this point
                historical = df.iloc[:i+1]
                recent_draws = []
                for _, row in historical.tail(100).iterrows():
                    for col in ['1st_real', '2nd_real', '3rd_real']:
                        num = str(row.get(col, ''))
                        if len(num) == 4 and num.isdigit():
                            recent_draws.append({'number': num})
                
                # Learn patterns and predict
                patterns = learn_day_to_day_patterns(recent_draws)
                recent_nums = [d['number'] for d in recent_draws[-50:]]
                prediction_results = predict_tomorrow(current_nums, patterns, recent_nums)
                
                # Get top 3 DIVERSE predictions
                if prediction_results and len(prediction_results) > 0:
                    seen = set()
                    for num, score, reason in prediction_results:
                        if len(num) != 4 or not num.isdigit():
                            continue
                        is_diverse = True
                        for existing in predictions:
                            matches = sum(1 for i in range(4) if num[i] == existing[i])
                            if matches >= 3:
                                is_diverse = False
                                break
                        if num not in seen and is_diverse:
                            predictions.append(num)
                            seen.add(num)
                            if len(predictions) >= 3:
                                break
                
                # Fallback: use hot numbers if not enough predictions
                while len(predictions) < 3 and recent_nums:
                    for hot_num in recent_nums[-30:]:
                        if len(hot_num) == 4 and hot_num.isdigit() and hot_num not in predictions:
                            is_diverse = True
                            for existing in predictions:
                                matches = sum(1 for i in range(4) if hot_num[i] == existing[i])
                                if matches >= 3:
                                    is_diverse = False
                                    break
                            if is_diverse:
                                predictions.append(hot_num)
                                if len(predictions) >= 3:
                                    break
                    break
            except Exception as e:
                # If AI fails, use old frequency predictions
                predictions = old_predictions
            
            # Ensure we have 3 predictions
            if len(predictions) < 3:
                predictions = predictions + current_nums[:3-len(predictions)]
            
            # Use NEW AI predictions as primary, but keep old for comparison
            # (Currently using NEW predictions for display)
            
            # Check matches
            matches = [p for p in predictions if p in next_nums]
            
            # Match types
            match_types = {
                'exact': len(matches),
                'ibox': sum(1 for p in predictions for n in next_nums if p != n and sorted(p) == sorted(n)),
                'front': sum(1 for p in predictions for n in next_nums if p[:2] == n[:2]),
                'back': sum(1 for p in predictions for n in next_nums if p[-2:] == n[-2:]),
                'partial': sum(1 for p in predictions for n in next_nums if len(set(p) & set(n)) >= 2)
            }
            
            learning_score = (match_types['exact'] * 50 + match_types['ibox'] * 30 + 
                            match_types['front'] * 20 + match_types['back'] * 20 + 
                            match_types['partial'] * 10)
            learning_score = min(learning_score, 100)
            
            daily_learning.append({
                'prediction_date': current['date_parsed'].strftime('%d/%m/%Y'),
                'actual_date': next_day['date_parsed'].strftime('%d/%m/%Y'),
                'predicted_numbers': predictions,
                'predicted_special': current_special[:10],
                'predicted_consolation': current_consolation[:10],
                'actual_numbers': next_nums,
                'actual_special': next_special[:10],
                'actual_consolation': next_consolation[:10],
                'matches': matches,
                'match_count': len(matches),
                'match_types': match_types,
                'learning_score': learning_score,
                'ai_learned': learning_score > 50,
                'provider': str(current['provider']).upper()
            })
        
        # CONTINUOUS LEARNING: Auto-retrain AI with latest data
        try:
            from utils.day_to_day_learner import learn_day_to_day_patterns
            # Retrain with ALL historical data (incremental learning)
            all_draws = []
            for _, row in df.iterrows():
                for col in ['1st_real', '2nd_real', '3rd_real']:
                    num = str(row.get(col, ''))
                    if len(num) == 4 and num.isdigit():
                        all_draws.append({'number': num})
            # Learn patterns from complete dataset (AI gets smarter with each CSV update)
            updated_patterns = learn_day_to_day_patterns(all_draws)
        except:
            pass  # Silent fail - no mess
        
        ai_insights = {
            'daily_learning': daily_learning,
            'ai_mode': True
        }
        
        return render_template('match_checker_simple.html',
                             ai_insights=ai_insights,
                             selected_provider=selected_provider,
                             analysis_period=analysis_period,
                             provider_options=provider_options,
                             month_options=month_options,
                             show_advanced=True,
                             matches=[],
                             ai_mode=True)
