import os
import re
from pathlib import Path

def find_dates_after_sep26():
    pattern = r'2025-(09-2[7-9]|1[0-2]-\d{2})'
    found = []
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.csv'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        matches = re.findall(pattern, content)
                        if matches:
                            found.append((os.path.join(root, file), matches))
                except:
                    pass
    
    if found:
        for file, matches in found:
            print(f"{file}: {set(matches)}")
    else:
        print("No dates after 2025-09-26 found")

find_dates_after_sep26()