import random
from datetime import datetime

# Test date-based seed
dates = ['2025-09-27', '2025-10-05', '2025-10-18']

for date_str in dates:
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    date_seed = int(date_obj.strftime('%Y%m%d'))
    random.seed(date_seed)
    
    # Generate some random predictions
    predictions = []
    for _ in range(3):
        num = ''.join([str(random.randint(0, 9)) for _ in range(4)])
        predictions.append(num)
    
    print(f"{date_str}: {predictions}")
