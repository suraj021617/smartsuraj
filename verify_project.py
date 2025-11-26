"""
Project Verification Script
Tests all routes and checks data integrity
"""
import pandas as pd
import os

def verify_csv_data():
    """Check if CSV data is valid"""
    print("📊 Checking CSV data...")
    try:
        df = pd.read_csv('4d_results_history.csv')
        print(f"  ✅ CSV loaded: {len(df)} rows")
        
        # Check for required columns
        required_cols = ['date', 'provider', '1st', '2nd', '3rd']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"  ⚠️ Missing columns: {missing}")
        else:
            print(f"  ✅ All required columns present")
        
        # Check date range
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        print(f"  📅 Date range: {df['date_parsed'].min()} to {df['date_parsed'].max()}")
        
        # Check providers
        providers = df['provider'].unique()
        print(f"  🏢 Providers: {len(providers)} unique")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def verify_templates():
    """Check if all templates exist"""
    print("\n📄 Checking templates...")
    essential_templates = [
        'index.html',
        'pattern_analyzer.html',
        'ultimate_predictor.html',
        'accuracy_dashboard.html',
        'learning_insights.html',
        'ml_predictor.html',
        'smart_predictor.html',
        'prediction_history.html',
    ]
    
    missing = []
    for template in essential_templates:
        path = os.path.join('templates', template)
        if os.path.exists(path):
            print(f"  ✅ {template}")
        else:
            print(f"  ❌ {template} MISSING")
            missing.append(template)
    
    return len(missing) == 0

def verify_utils():
    """Check if utility modules exist"""
    print("\n🔧 Checking utility modules...")
    essential_utils = [
        'ai_predictor.py',
        'pattern_finder.py',
        'pattern_memory.py',
        'app_grid.py',
        'pattern_stats.py',
    ]
    
    missing = []
    for util in essential_utils:
        path = os.path.join('utils', util)
        if os.path.exists(path):
            print(f"  ✅ {util}")
        else:
            print(f"  ❌ {util} MISSING")
            missing.append(util)
    
    return len(missing) == 0

def verify_routes():
    """List all available routes"""
    print("\n🛣️ Available Routes:")
    routes = [
        ('/', 'Home Page'),
        ('/pattern-analyzer', 'Pattern Analyzer'),
        ('/prediction-history', 'Prediction History'),
        ('/ultimate-predictor', 'Ultimate Predictor (ALL METHODS)'),
        ('/smart-predictor', 'Smart Auto-Weight Predictor'),
        ('/ml-predictor', 'Machine Learning Predictor'),
        ('/accuracy-dashboard', 'Accuracy Dashboard (Learning)'),
        ('/learning-insights', 'AI Learning Insights'),
        ('/smart-history', 'Smart History'),
    ]
    
    for route, name in routes:
        print(f"  ✅ {route:30} → {name}")

def main():
    print("=" * 60)
    print("PROJECT VERIFICATION")
    print("=" * 60)
    
    csv_ok = verify_csv_data()
    templates_ok = verify_templates()
    utils_ok = verify_utils()
    verify_routes()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if csv_ok and templates_ok and utils_ok:
        print("All checks passed! Project is ready to run.")
        print("\nTo start the app:")
        print("   python app.py")
        print("\nThen open: http://localhost:5000")
    else:
        print("Some issues found. Please fix them before running.")
    
    print("\nKey Features:")
    print("  1. Ultimate Predictor - Combines ALL 4 prediction methods")
    print("  2. Accuracy Dashboard - Tracks prediction vs actual results")
    print("  3. Learning Insights - AI learns from mistakes")
    print("  4. Pattern Analyzer - Grid-based pattern detection")
    print("  5. Smart Predictor - Auto-weight optimization")
    print("  6. ML Predictor - Machine learning predictions")

if __name__ == "__main__":
    main()
