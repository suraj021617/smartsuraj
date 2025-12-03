"""
Fixed Positional OCR Route with proper provider and date filtering
"""

from flask import request, render_template
from utils.auto_ocr_predictor import analyze_next_day_ocr, check_next_day_results
from datetime import datetime, timedelta
import pandas as pd

def positional_ocr_route(app, load_csv_data):
    @app.route('/positional-ocr')
    def positional_ocr():
        df = load_csv_data()
        
        # Get parameters
        provider = request.args.get('provider', 'all')
        selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        auto_mode = request.args.get('auto', False)
        
        # Provider options - FIXED
        provider_options = ['all']
        if not df.empty:
            unique_providers = df['provider'].dropna().unique()
            provider_options.extend([p for p in unique_providers if p and str(p).strip() and str(p) != 'nan'])
            provider_options = sorted(list(set(provider_options)))
        
        # Filter by provider - FIXED
        filtered_df = df.copy()
        if provider != 'all' and provider in provider_options:
            filtered_df = filtered_df[filtered_df['provider'] == provider]
        
        # Initialize variables
        ocr_table = {}
        predictions = []
        next_day_check = None
        
        if not filtered_df.empty:
            try:
                # Get OCR analysis
                predictions, ocr_table, hot_per_position = analyze_next_day_ocr(
                    filtered_df, 
                    provider=provider, 
                    lookback_days=30
                )
                
                # Check next day results if requested
                if selected_date:
                    try:
                        check_date = pd.to_datetime(selected_date) + timedelta(days=1)
                        next_day_check = check_next_day_results(
                            filtered_df, 
                            predictions, 
                            check_date.strftime('%Y-%m-%d')
                        )
                    except:
                        next_day_check = None
                        
            except Exception as e:
                print(f"OCR Analysis Error: {e}")
                predictions = []
                ocr_table = {}
        
        # Ensure ocr_table has proper format
        if not ocr_table or not isinstance(ocr_table, dict):
            ocr_table = {i: [0, 0, 0, 0] for i in range(10)}
        
        return render_template('positional_ocr.html',
                             provider_options=provider_options,
                             selected_provider=provider,
                             selected_date=selected_date,
                             ocr_table=ocr_table,
                             predictions=predictions,
                             auto_mode=auto_mode,
                             next_day_check=next_day_check,
                             last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))