import re

with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

results = []
for line in lines:
    if line.startswith('2025-08-') or line.startswith('2025-09-'):
        parts = line.strip().split(',')
        if len(parts) >= 6:
            date = parts[0]
            provider = parts[1].replace('https://www.live4d2u.net/images/', '').replace('"', '')
            
            # Extract prizes from the combined field
            prize_field = ','.join(parts[4:6])
            first = re.search(r'1st[^0-9]*(\d{4})', prize_field)
            second = re.search(r'2nd[^0-9]*(\d{4})', prize_field)
            third = re.search(r'3rd[^0-9]*(\d{4})', prize_field)
            
            special = parts[6].replace('"', '') if len(parts) > 6 else ''
            consolation = parts[7].replace('"', '') if len(parts) > 7 else ''
            
            results.append(f"{date},{provider},{first.group(1) if first else ''},{second.group(1) if second else ''},{third.group(1) if third else ''},{special},{consolation}\n")

with open('4d_results_clean.csv', 'w', encoding='utf-8') as f:
    f.write('date,provider,1st,2nd,3rd,special,consolation\n')
    f.writelines(results)

print(f"Saved {len(results)} rows")
