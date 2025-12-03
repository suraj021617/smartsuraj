import os
import re
import subprocess

def deep_search():
    # 1. Check git stash
    try:
        result = subprocess.run(['git', 'stash', 'list'], capture_output=True, text=True)
        if result.stdout:
            print("Git stashes found:")
            print(result.stdout)
    except:
        pass
    
    # 2. Check git branches
    try:
        result = subprocess.run(['git', 'branch', '-a'], capture_output=True, text=True)
        print("\nGit branches:")
        print(result.stdout)
    except:
        pass
    
    # 3. Check all commits for CSV files
    try:
        result = subprocess.run(['git', 'log', '--all', '--name-only', '--grep=csv', '-i'], capture_output=True, text=True)
        if result.stdout:
            print("\nCommits with CSV:")
            print(result.stdout)
    except:
        pass
    
    # 4. Search in all files for 2025-10 or 2025-11 or 2025-12
    pattern = r'2025-(1[0-2])-\d{2}'
    found_files = []
    
    for root, dirs, files in os.walk('.'):
        # Skip .git and .venv
        dirs[:] = [d for d in dirs if d not in ['.git', '.venv', '__pycache__']]
        
        for file in files:
            if file.endswith(('.csv', '.txt', '.json', '.py', '.md')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        matches = re.findall(pattern, content)
                        if matches:
                            found_files.append((filepath, matches))
                except:
                    pass
    
    if found_files:
        print("\nFiles with dates after Sep 2025:")
        for file, matches in found_files:
            print(f"{file}: {set(matches)}")
    
    # 5. Check temp/backup directories
    temp_dirs = ['temp', 'backup', 'bak', 'old', 'archive']
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            print(f"\nFound {temp_dir} directory - checking...")
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.csv'):
                        print(f"  CSV: {os.path.join(root, file)}")

deep_search()