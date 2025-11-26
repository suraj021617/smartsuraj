from flask import Flask, render_template
import pandas as pd
from collections import Counter
import re

app = Flask(__name__)

@app.route('/test-decision')
def test_decision():
    # Load data
    df = pd.read_csv('4d_results_history.csv', on_bad_lines='skip')
    
    # Get numbers
    all_nums = []
    for col in ['1st', '2nd', '3rd']:
        for val in df[col].astype(str):
            found = re.findall(r'\b\d{4}\b', val)
            all_nums.extend(found)
    
    # Get top 5
    freq = Counter(all_nums).most_common(5)
    final_picks = [(num, 75) for num, count in freq]
    
    reasons = [
        "Test reason 1",
        "Test reason 2",
        "Test reason 3"
    ]
    
    return render_template('decision_helper.html',
                         final_picks=final_picks,
                         reasons=reasons,
                         next_draw_date='2025-01-20 (Monday)',
                         provider_name='TEST',
                         backup_numbers=['1111', '2222', '3333'],
                         provider_options=['all', 'test'],
                         provider='all',
                         error=None)

if __name__ == '__main__':
    print("Starting test Flask server...")
    print("Visit: http://127.0.0.1:5001/test-decision")
    app.run(debug=True, port=5001)
