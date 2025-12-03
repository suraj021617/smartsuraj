"""
Restore November 23 App Files
Copies the best app version from Nov 23 restored folder
"""

import shutil
import os

def restore_nov23_app():
    print("RESTORING NOVEMBER 23 APP FILES")
    print("=" * 40)
    
    source_dir = "restored_23nov"
    
    if not os.path.exists(source_dir):
        print("✗ November 23 restored folder not found")
        return
    
    # Find the latest/best app file from Nov 23
    app_files = [f for f in os.listdir(source_dir) if f.startswith("app_")]
    
    if not app_files:
        print("✗ No app files found in restored folder")
        return
    
    # Use the latest one
    latest_app = sorted(app_files)[-1]
    source_file = os.path.join(source_dir, latest_app)
    
    # Copy to main directory as app.py
    try:
        shutil.copy2(source_file, "app.py")
        print(f"✓ Restored {latest_app} as app.py")
        
        # Also backup current
        backup_name = f"app_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        if os.path.exists("app.py"):
            shutil.copy2("app.py", backup_name)
            print(f"✓ Current app backed up as {backup_name}")
            
    except Exception as e:
        print(f"✗ Error restoring app: {e}")

if __name__ == "__main__":
    from datetime import datetime
    restore_nov23_app()
    input("Press ENTER to close...")