import pandas as pd
from prediction_engine import predict_tomorrow
from app import load_csv_data

def test_prediction_engine():
    """Test the full prediction engine against historical data"""
    df = load_csv_data()
    df = df[df['1st_real'].str.len() == 4]
    df = df[df['1st_real'].str.isdigit()]
    df = df.sort_values('date_parsed').reset_index(drop=True)

    total = 0
    hits = 0

    print("🔄 Testing Full Prediction Engine (Ensemble)...\n")

    # Test on a subset of recent data to avoid long runtime
    test_days = 10  # Test last 10 days

    for i in range(len(df) - test_days, len(df) - 1):
        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]

        number = today['1st_real']
        if not number.isdigit() or len(number) != 4:
            continue

        # Get prediction from full engine
        try:
            prediction_result = predict_tomorrow()
            predicted = prediction_result.get('predicted_numbers', '').split(',')
            predicted = [p.strip() for p in predicted if p.strip()]
        except Exception as e:
            print(f"Error getting prediction for {today['date_parsed'].date()}: {e}")
            continue

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
    print("\n📊 Full Engine Evaluation Summary")
    print(f"   → Total Evaluated Days: {total}")
    print(f"   → Total Hits: {hits}")
    print(f"   → Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test_prediction_engine()
