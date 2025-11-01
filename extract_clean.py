import re

with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

results = []
for line in lines[20451:]:  # Start from where new data begins
    if line.strip() and (line.startswith('2025-08-') or line.startswith('2025-09-')):
        parts = line.strip().split(',')
        if len(parts) == 6:
            date, provider, draw_no, prizes, special, consolation = parts
            
            # Extract individual prizes
            first = re.search(r'1st (\d{4})', prizes)
            second = re.search(r'2nd (\d{4})', prizes)
            third = re.search(r'3rd (\d{4})', prizes)
            
            results.append(f"{date},{provider},{first.group(1) if first else ''},{second.group(1) if second else ''},{third.group(1) if third else ''},{special},{consolation}\n")

with open('4d_results_clean.csv', 'w', encoding='utf-8') as f:
    f.write('date,provider,1st,2nd,3rd,special,consolation\n')
    f.writelines(results)

print(f"Saved {len(results)} rows to 4d_results_clean.csv")
