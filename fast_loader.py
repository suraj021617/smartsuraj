import csv
from collections import defaultdict

def load_csv_fast(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 5:
                data.append(row)
    return data

def search_by_date(data, date):
    return [row for row in data if row[0] == date]

def search_by_lottery(data, lottery_name):
    return [row for row in data if lottery_name.lower() in row[2].lower()]

def search_by_number(data, number):
    results = []
    for row in data:
        if len(row) >= 5 and number in str(row[4]):
            results.append(row)
    return results

if __name__ == "__main__":
    data = load_csv_fast('4d_results_history.csv')
    print(f"Loaded {len(data)} records")
    
    # Example: Get Singapore 4D results
    sg_results = search_by_lottery(data, "Singapore 4D")
    for row in sg_results[:3]:
        print(f"{row[0]} | {row[2]} | {row[4]}")
