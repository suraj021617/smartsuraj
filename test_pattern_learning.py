# test_pattern_learning.py
print("Testing Pattern Analyzer Learning Integration...")
print("="*50)

# Test 1: Import check
try:
    from utils.feedback_learner import FeedbackLearner
    print("[OK] FeedbackLearner imported successfully")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    exit(1)

# Test 2: Initialize learner
try:
    learner = FeedbackLearner()
    print("[OK] Learner initialized")
except Exception as e:
    print(f"[FAIL] Initialization failed: {e}")
    exit(1)

# Test 3: Test evaluation
try:
    predicted = ['1234', '5678', '9012']
    actual_1st = '1234'
    actual_2nd = '5679'
    actual_3rd = '9999'
    
    match_type, score, details = learner.evaluate_prediction(
        predicted, actual_1st, actual_2nd, actual_3rd
    )
    
    print(f"[OK] Evaluation works: {match_type} (Score: {score})")
except Exception as e:
    print(f"[FAIL] Evaluation failed: {e}")
    exit(1)

# Test 4: Test learning
try:
    learner.learn_from_result({
        'predicted_numbers': predicted,
        'predictor_methods': 'pattern',
        'confidence': 85,
        'draw_date': '2025-01-15'
    }, match_type, score)
    
    print("[OK] Learning works")
except Exception as e:
    print(f"[FAIL] Learning failed: {e}")
    exit(1)

# Test 5: Get best methods
try:
    best = learner.get_best_methods()
    print(f"[OK] Best methods: {best}")
except Exception as e:
    print(f"[FAIL] Get best methods failed: {e}")
    exit(1)

print("\n" + "="*50)
print("[SUCCESS] ALL TESTS PASSED!")
print("Pattern Analyzer Learning Integration is ready!")
print("\nNext steps:")
print("1. Run: python app.py")
print("2. Visit: http://127.0.0.1:5000/pattern-analyzer")
print("3. The system will now learn from every prediction!")
print("4. View learning insights at: http://127.0.0.1:5000/learning-dashboard")
