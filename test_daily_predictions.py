import pandas as pd
from utils.day_to_day_learner import learn_day_to_day_patterns, predict_tomorrow

# Load CSV
df = pd.read_csv('4d_results_history.csv', index_col=False, on_bad_lines='skip')
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date_parsed'])
df = df.sort_values('date_parsed').tail(10)

print("Testing prediction diversity...")
print("=" * 50)

for i in range(len(df) - 1):
    curr = df.iloc[i]
    
    # Get today's numbers
    today_nums = []
    for col in ['1st_real', '2nd_real', '3rd_real']:
        num = str(curr.get(col, ''))
        if len(num) == 4 and num.isdigit():
            today_nums.append(num)
    
    if not today_nums:
        continue
    
    # Build historical data
    historical = df.iloc[:i+1]
    recent_draws = []
    for _, row in historical.iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if len(num) == 4 and num.isdigit():
                recent_draws.append({'number': num})
    
    # Learn and predict
    patterns = learn_day_to_day_patterns(recent_draws)
    recent_nums = [d['number'] for d in recent_draws[-50:]]
    predictions = predict_tomorrow(today_nums, patterns, recent_nums)
    
    print(f"\nDate: {curr['date_parsed'].date()}")
    print(f"Today's numbers: {today_nums[:3]}")
    print(f"Predictions: {[num for num, _, _ in predictions[:3]]}")
    print("-" * 50)
