from flask import Flask, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/results')
def get_results():
    try:
        df = pd.read_csv('4d_results_history.csv')
        return jsonify(df.to_dict('records'))
    except FileNotFoundError:
        return jsonify({'error': 'Results file not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)