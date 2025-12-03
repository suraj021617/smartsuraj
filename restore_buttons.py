import os

def restore_from_backup(backup_name):
    """Restore from git backup"""
    try:
        # Restore main files
        os.system(f'git show {backup_name}:app.py > app.py')
        os.system(f'git show {backup_name}:4d_results_history.csv > 4d_results_history.csv')
        
        # Copy to folders
        os.system('copy "4d_results_history.csv" "data\\4d_results_history.csv"')
        os.system('copy "4d_results_history.csv" "scraper\\4d_results_history.csv"')
        
        print(f"OK Restored from {backup_name}")
        return True
    except:
        print(f"FAIL Restore from {backup_name}")
        return False

def test_current_logic():
    """Test if current logic works"""
    try:
        from app import app
        import pandas as pd
        df = pd.read_csv('4d_results_history.csv', on_bad_lines='skip')
        print(f"OK Current logic works - CSV: {len(df)} rows")
        return True
    except Exception as e:
        print(f"FAIL Current logic: {e}")
        return False

if __name__ == "__main__":
    print("=== RESTORE BUTTONS ===")
    print("1. Test Current Logic")
    print("2. Restore from backup_before_nov1")
    print("3. Restore from morning-backup (Nov 23)")
    print("4. Restore from main-backup (Nov 26)")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == "1":
        test_current_logic()
    elif choice == "2":
        restore_from_backup("backup_before_nov1")
        test_current_logic()
    elif choice == "3":
        restore_from_backup("morning-backup")
        test_current_logic()
    elif choice == "4":
        restore_from_backup("main-backup")
        test_current_logic()
    else:
        print("Invalid choice")
    
    input("Press ENTER to exit...")