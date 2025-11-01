from flask import Flask, render_template, request
from pattern_engine import build_grid, detect_patterns

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    number = ""
    grid = []
    patterns = []
    pattern_cells = []

    if request.method == 'POST':
        number = request.form['number']
        if len(number) == 4 and number.isdigit():
            grid = build_grid(number)
            patterns = detect_patterns(grid)

            # Highlight cells for some patterns
            # Diagonal ↘
            if "Diagonal ↘" in patterns:
                pattern_cells += [f"{i}-{i}" for i in range(4)]

            # Box corners
            if "Box Corners Match" in patterns:
                pattern_cells += ["0-0", "0-3", "3-0", "3-3"]

            # L-Shape ┏
            if "L-Shape ┏" in patterns:
                pattern_cells += ["0-0", "0-1", "0-2", "1-0"]

            # Same Column
            for p in patterns:
                if p.startswith("Same Column"):
                    col = int(p.split()[-1]) - 1
                    pattern_cells += [f"{r}-{col}" for r in range(4)]

    return render_template(
        "predict.html",
        number=number,
        grid=grid,
        patterns=patterns,
        pattern_cells=pattern_cells
    )

if __name__ == '__main__':
    app.run(debug=True)
