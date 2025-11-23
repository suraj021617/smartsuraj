import csv
import sys

def get_csv_data(provider, month):
    """Get CSV data by provider and month"""
    results = []
    
    with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get('date', '')
            prov = row.get('provider', '').lower()
            
            # Filter by month
            if month and not date.startswith(month):
                continue
            
            # Filter by provider (match any part)
            if provider != 'all':
                # Extract provider name from URL or text
                prov_clean = prov.split('/')[-1].lower() if '/' in prov else prov.lower()
                if provider.lower() not in prov_clean and provider.lower() not in row.get('draw_info', '').lower():
                    continue
            
            # Extract winning numbers from columns 4, 5, 6, 7
            numbers = []
            for col in ['3rd', 'special', 'consolation']:
                if col in row:
                    import re
                    nums = re.findall(r'\b\d{4}\b', row[col])
                    numbers.extend(nums)
            
            if numbers:
                results.append({
                    'date': date,
                    'provider': row.get('draw_info', ''),
                    'numbers': numbers[:10]  # Limit to 10 numbers
                })
    
    return results

if __name__ == '__main__':
    provider = sys.argv[1] if len(sys.argv) > 1 else 'all'
    month = sys.argv[2] if len(sys.argv) > 2 else ''
    
    data = get_csv_data(provider, month)
    
    import json
    print(json.dumps(data, indent=2))
