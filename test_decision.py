from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Test Server Running</h1><a href="/decision-helper">Click to test decision-helper</a>'

@app.route('/decision-helper')
def decision_helper():
    # Minimal working version
    final_picks = [('1234', 75), ('5678', 75), ('9012', 50), ('3456', 50), ('7890', 25)]
    reasons = ['Test reason 1', 'Test reason 2']
    
    return render_template('decision_helper.html',
                         final_picks=final_picks,
                         reasons=reasons,
                         next_draw_date='2025-10-28 (Monday)',
                         provider_name='ALL PROVIDERS',
                         backup_numbers=['1111', '2222', '3333'],
                         provider_options=['all'],
                         provider='all',
                         error=None)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
