from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Working!</h1><a href="/dashboard">Go to Dashboard</a>'

@app.route('/dashboard')
def dashboard():
    return '''
    <h1>Dashboard</h1>
    <a href="/quick-pick">Quick Pick</a><br>
    <a href="/pattern-analyzer">Pattern Analyzer</a><br>
    <a href="/frequency-analyzer">Frequency Analyzer</a><br>
    <a href="/ml-predictor">ML Predictor</a><br>
    '''

@app.route('/quick-pick')
def quick_pick():
    return '<h1>Quick Pick</h1><p>Numbers: 1234, 5678, 9012</p>'

@app.route('/pattern-analyzer')
def pattern_analyzer():
    return '<h1>Pattern Analyzer</h1><p>Pattern analysis working</p>'

@app.route('/frequency-analyzer')
def frequency_analyzer():
    return '<h1>Frequency Analyzer</h1><p>Frequency analysis working</p>'

@app.route('/ml-predictor')
def ml_predictor():
    return '<h1>ML Predictor</h1><p>ML predictions working</p>'

if __name__ == '__main__':
    app.run(debug=True, port=5001)