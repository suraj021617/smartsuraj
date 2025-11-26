# evaluate_prediction_accuracy.py

import pandas as pd
from utils.ai_predictor import predict_top_5
from app import generate_4x4_grid, load_csv_data

def evaluate_predictions():
    df = load_csv_data()
    df = df[df['1st_real'].str.len() == 4]
    df = df[df['1st_real'].str.isdigit()]
    df = df.sort_values('date_parsed').reset_index(drop=True)

    total = 0
    hits = 0

    print("🔎 Evaluating AI predictions vs actual next-day results...\n")

    for i in range(len(df) - 1):
        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]

        number = today['1st_real']
        if not number.isdigit() or len(number) != 4:
            continue

        # Format draw data for ai_predictor
        draw_data = [{
            "number": number,
            "grid": generate_4x4_grid(number),
            "date": today['date_parsed'].strftime('%Y-%m-%d')
        }]
        predicted = [x[0] for x in predict_top_5(draw_data)]

        actuals = [
            str(tomorrow.get('1st_real', '')),
            str(tomorrow.get('2nd_real', '')),
            str(tomorrow.get('3rd_real', ''))
        ]

        hit = any(p in actuals for p in predicted)
        result = "✅ HIT" if hit else "❌ MISS"
        print(f"[{today['date_parsed'].date()}] Predicted: {predicted} | Actual: {actuals} => {result}")

        total += 1
        if hit:
            hits += 1

    accuracy = (hits / total) * 100 if total else 0
    print("\n📊 Prediction Evaluation Summary")
    print(f"   → Total Evaluated Days: {total}")
    print(f"   → Total Hits: {hits}")
    print(f"   → Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_predictions()
