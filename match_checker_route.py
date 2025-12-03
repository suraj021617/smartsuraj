from flask import request, render_template
import pandas as pd
from datetime import datetime, timedelta
import re

def add_match_checker_route(app, load_csv_data):
    @app.route('/match-checker', methods=['GET', 'POST'])
    def match_checker():
        df = load_csv_data()
        matches = []
        
        if request.method == 'POST':
            provider = request.form.get('provider', 'all')
            date_range = int(request.form.get('date_range', '30'))
            predictions = request.form.get('predictions', '').split(',')
            predictions = [p.strip() for p in predictions if p.strip()]
            
            # Filter data
            if provider != 'all':
                df = df[df['provider'].str.contains(provider, case=False, na=False)]
            
            # Date filtering
            cutoff_date = df['date_parsed'].max() - timedelta(days=date_range)
            df = df[df['date_parsed'] >= cutoff_date]
            
            # Check matches
            for pred in predictions:
                for _, row in df.iterrows():
                    actual = []
                    for col in ['1st_real', '2nd_real', '3rd_real']:
                        if pd.notna(row[col]):
                            actual.append(str(row[col]))
                    
                    if pred in actual:
                        matches.append({
                            'date': row['date_parsed'].date(),
                            'predictions': [pred],
                            'all_winners': actual,
                            'exact': 1,
                            'special': 0,
                            'consolation': 0,
                            'accuracy': 100
                        })
        
        return render_template('match_checker.html', matches=matches)