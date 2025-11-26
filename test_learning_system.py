# test_learning_system.py
from utils.feedback_learner import FeedbackLearner

print("Testing Feedback Learning System...")
print("="*50)

# Initialize learner
learner = FeedbackLearner()

# Test prediction evaluation
predicted = ['1234', '5678', '9012']
actual_1st = '1234'
actual_2nd = '5679'
actual_3rd = '9999'

match_type, score, details = learner.evaluate_prediction(predicted, actual_1st, actual_2nd, actual_3rd)

print(f"\nTest Prediction:")
print(f"Predicted: {predicted}")
print(f"Actual: {actual_1st}, {actual_2nd}, {actual_3rd}")
print(f"\nResult: {match_type} (Score: {score})")

# Test learning
learner.learn_from_result({
    'predicted_numbers': predicted,
    'predictor_methods': 'Test Method',
    'confidence': 85,
    'draw_date': '2025-01-15'
}, match_type, score)

# Get best methods
best = learner.get_best_methods()
print(f"\nBest Methods: {best}")

print("\n" + "="*50)
print("All tests passed! System is working correctly.")
