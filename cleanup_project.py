"""
Project Cleanup Script
Removes unnecessary files and folders
"""
import os
import shutil

# Folders to remove
folders_to_remove = [
    '.history',  # Version history backups
    '__pycache__',  # Python cache
]

# Files to remove (keep only essential)
files_to_remove = [
    'how 17f3fd4app.py',
    'how 17f3fd4app.py  old_app.py',
    'python grid_generator.py',
    'debug_calendar.py',
    'debug_grid_check.py',
    'debug_page.html',
    'test_ai_learning.py',
    'test_app.py',
    'test_calendar.py',
    'test_empty_box.py',
    'test_empty_direct.py',
    'test_predictions.py',
    'check_extract.py',
    'check_rows.py',
    'clear_cache.py',
    'prediction_cache.json',
    'predictions_cache.json',
    'predictions_history.json',
    'provider_scores.json',
]

def cleanup():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🧹 Starting cleanup...")
    
    # Remove folders
    for folder in folders_to_remove:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"✅ Removed folder: {folder}")
            except Exception as e:
                print(f"❌ Error removing {folder}: {e}")
    
    # Remove files
    for file in files_to_remove:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✅ Removed file: {file}")
            except Exception as e:
                print(f"❌ Error removing {file}: {e}")
    
    print("\n✨ Cleanup complete!")
    print("\n📁 Essential files kept:")
    print("  - app.py (main application)")
    print("  - 4d_results_history.csv (data)")
    print("  - templates/ (HTML pages)")
    print("  - utils/ (prediction logic)")
    print("  - static/ (CSS, images)")
    print("  - requirements.txt (dependencies)")

if __name__ == "__main__":
    cleanup()
