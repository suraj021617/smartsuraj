import csv
import json
import sys
from datetime import datetime

def get_results(provider, month):
    """Get all winning numbers for a provider and month"""
    results = []
    
    with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) < 6 or not row[0]:
                continue
            
            # Check if matches provider and month
            if month and not row[0].startswith(month):
                continue
            
            if provider.lower() not in row[2].lower():
                continue
            
            # Extract winning numbers from column 5 (prizes)
            import re
            prizes = re.findall(r'\d{4}', row[5])
            
            if prizes:
                results.append({
                    'date': row[0],
                    'provider': row[2],
                    'numbers': prizes[:10]  # Top 10 numbers
                })
    
    return results

if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "TOTO"
    month = sys.argv[2] if len(sys.argv) > 2 else "2025-10"
    
    results = get_results(provider, month)
    print(json.dumps(results, indent=2))
