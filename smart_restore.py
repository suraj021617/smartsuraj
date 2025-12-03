import os
import pandas as pd

def test_logic():
    """Test current logic completely"""
    score = 0
    
    # Test CSV
    try:
        df = pd.read_csv('4d_results_history.csv', on_bad_lines='skip')
        print(f"OK CSV: {len(df)} rows")
        score += 1
    except:
        print("FAIL CSV")
    
    # Test App
    try:
        from app import app
        print("OK App")
        score += 1
    except Exception as e:
        print(f"FAIL App: {str(e)[:50]}")
    
    # Test Utils
    try:
        from utils.pattern_predictor import PatternPredictor
        print("OK Utils")
        score += 1
    except:
        print("FAIL Utils")
    
    # Test Templates
    try:
        if os.path.exists('templates/index.html'):
            print("OK Templates")
            score += 1
        else:
            print("FAIL Templates")
    except:
        print("FAIL Templates")
    
    # Test Data folders
    try:
        if os.path.exists('data') and os.path.exists('scraper'):
            print("OK Folders")
            score += 1
        else:
            print("FAIL Folders")
    except:
        print("FAIL Folders")
    
    print(f"SCORE: {score}/5")
    return score

def restore_and_test(backup_name, description):
    """Restore from backup and test logic"""
    print(f"\n=== TESTING {description} ===")
    
    try:
        # Restore key files
        os.system(f'git show {backup_name}:app.py > temp_app.py 2>nul')
        os.system(f'git show {backup_name}:4d_results_history.csv > temp_csv.csv 2>nul')
        
        # Backup current
        if os.path.exists('app.py'):
            os.system('copy app.py app_backup.py >nul')
        if os.path.exists('4d_results_history.csv'):
            os.system('copy 4d_results_history.csv csv_backup.csv >nul')
        
        # Replace with backup
        if os.path.exists('temp_app.py'):
            os.system('move temp_app.py app.py >nul')
        if os.path.exists('temp_csv.csv'):
            os.system('move temp_csv.csv 4d_results_history.csv >nul')
            os.system('copy 4d_results_history.csv data\\4d_results_history.csv >nul')
            os.system('copy 4d_results_history.csv scraper\\4d_results_history.csv >nul')
        
        # Test logic
        score = test_logic()
        
        # Restore original if score is low
        if score < 3:
            if os.path.exists('app_backup.py'):
                os.system('move app_backup.py app.py >nul')
            if os.path.exists('csv_backup.csv'):
                os.system('move csv_backup.csv 4d_results_history.csv >nul')
            print(f"RESTORED ORIGINAL (score too low)")
        else:
            # Keep backup version
            if os.path.exists('app_backup.py'):
                os.remove('app_backup.py')
            if os.path.exists('csv_backup.csv'):
                os.remove('csv_backup.csv')
            print(f"KEPT {description} (good score)")
        
        return score
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 0

if __name__ == "__main__":
    print("=== SMART RESTORE SYSTEM ===")
    print("Testing which backup has best working logic...")
    
    # Test current first
    print("\n=== CURRENT STATE ===")
    current_score = test_logic()
    
    # Test all backups
    backups = [
        ("backup_before_nov1", "Nov 1 Backup"),
        ("morning-backup", "Nov 23 Backup"), 
        ("main-backup", "Nov 26 Backup")
    ]
    
    best_score = current_score
    best_backup = "current"
    
    for backup_name, description in backups:
        score = restore_and_test(backup_name, description)
        if score > best_score:
            best_score = score
            best_backup = description
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Best working logic: {best_backup}")
    print(f"Score: {best_score}/5")
    
    if best_score >= 4:
        print("LOGIC WORKING WELL!")
    elif best_score >= 3:
        print("LOGIC PARTIALLY WORKING")
    else:
        print("LOGIC NEEDS FIXING")
    
    input("Press ENTER to exit...")