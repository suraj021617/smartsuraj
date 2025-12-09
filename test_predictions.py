from utils.day_to_day_learner import predict_tomorrow, learn_day_to_day_patterns

# Test with sample data
test_draws = [
    {'number': '1234'},
    {'number': '5678'},
    {'number': '9012'},
    {'number': '3456'},
    {'number': '7890'}
]

patterns = learn_day_to_day_patterns(test_draws)
print(f"Patterns learned: {len(patterns)}")

preds = predict_tomorrow(['1234'], patterns, ['5678', '9012', '3456'])
print(f"\nPredictions generated: {len(preds)}")

if preds:
    print("\nTop 5 predictions:")
    for i, (num, score, reason) in enumerate(preds[:5], 1):
        print(f"{i}. {num} (score: {score:.3f}) - {reason}")
else:
    print("No predictions generated!")
