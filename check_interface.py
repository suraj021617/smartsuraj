import os
from flask import Flask

def check_interface():
    """Check if interface files exist and app can start"""
    
    # Check templates
    templates = ['index.html', 'predict.html', 'results.html']
    template_ok = 0
    for template in templates:
        if os.path.exists(f'templates/{template}'):
            print(f"OK templates/{template}")
            template_ok += 1
        else:
            print(f"MISSING templates/{template}")
    
    # Check static files
    static_files = ['style.css', 'main.js']
    static_ok = 0
    for file in static_files:
        if os.path.exists(f'static/{file}'):
            print(f"OK static/{file}")
            static_ok += 1
        else:
            print(f"MISSING static/{file}")
    
    # Check app
    try:
        from app import app
        print("OK App imports")
        app_ok = True
    except Exception as e:
        print(f"FAIL App: {e}")
        app_ok = False
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"Templates: {template_ok}/{len(templates)}")
    print(f"Static: {static_ok}/{len(static_files)}")
    print(f"App: {'OK' if app_ok else 'FAIL'}")
    
    if template_ok >= 2 and app_ok:
        print("INTERFACE READY - Can start Flask")
        return True
    else:
        print("INTERFACE BROKEN - Need restore")
        return False

if __name__ == "__main__":
    print("=== INTERFACE CHECK ===")
    ready = check_interface()
    
    if ready:
        print("\nTo start interface:")
        print("python app.py")
        print("Then open: http://localhost:5000")
    else:
        print("\nNeed to restore working backup first")
    
    input("Press ENTER...")