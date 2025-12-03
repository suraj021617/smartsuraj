import sys
import traceback

def test_app_logic():
    """Test if app logic works"""
    try:
        from app import app
        print("✓ App import: SUCCESS")
        
        # Test basic routes
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("✓ Home route: SUCCESS")
            else:
                print(f"✗ Home route: FAILED ({response.status_code})")
        
        return True
    except Exception as e:
        print(f"✗ App logic: FAILED - {str(e)}")
        traceback.print_exc()
        return False

def test_csv_logic():
    """Test CSV data logic"""
    try:
        import pandas as pd
        df = pd.read_csv('4d_results_history.csv')
        print(f"✓ CSV read: SUCCESS ({len(df)} rows)")
        
        # Check date column
        if len(df) > 0:
            last_date = df.iloc[-1, 0]
            print(f"✓ Last date: {last_date}")
        
        return True
    except Exception as e:
        print(f"✗ CSV logic: FAILED - {str(e)}")
        return False

def test_prediction_logic():
    """Test prediction logic"""
    try:
        from utils.pattern_predictor import PatternPredictor
        predictor = PatternPredictor()
        print("✓ Prediction import: SUCCESS")
        return True
    except Exception as e:
        print(f"✗ Prediction logic: FAILED - {str(e)}")
        return False

if __name__ == "__main__":
    print("=== LOGIC TEST ===")
    
    csv_ok = test_csv_logic()
    app_ok = test_app_logic()
    pred_ok = test_prediction_logic()
    
    if csv_ok and app_ok and pred_ok:
        print("\n✓ ALL LOGIC WORKING")
    else:
        print("\n✗ SOME LOGIC FAILED")
        print("Need to restore from working backup")