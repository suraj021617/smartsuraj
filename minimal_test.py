from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask is working!"

@app.route('/decision-helper')
def decision():
    return "Decision helper route exists!"

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Flask on http://127.0.0.1:5000")
    print("Visit: http://127.0.0.1:5000/decision-helper")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)
