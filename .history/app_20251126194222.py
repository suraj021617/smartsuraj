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
