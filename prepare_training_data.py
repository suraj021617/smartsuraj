import pandas as pd
import os
import re
from utils.pattern_finder import create_4x4_grid

INPUT_CSV = "4d_results_history.csv"
OUTPUT_CSV = "data/4d_training_ready.csv"

os.makedirs("data", exist_ok=True)
df = pd.read_csv(INPUT_CSV)

training_rows = []

def extract_4d(text):
    """Extract first 4-digit number from messy string"""
    matches = re.findall(r"\b\d{4}\b", str(text))
    return matches[0] if matches else None

for i in range(len(df) - 1):
    today = df.iloc[i]
    next_day = df.iloc[i + 1]

    try:
        # Extract all next day winning numbers
        next_numbers = []
        for col in ["1st", "2nd", "3rd", "special", "consolation"]:
            raw = str(next_day.get(col, ""))
            nums = re.findall(r"\b\d{4}\b", raw)
            next_numbers.extend(nums)

        # Extract and train from today's 1st/2nd/3rd
        for col in ["1st", "2nd", "3rd"]:
            num = extract_4d(today.get(col, ""))
            if not num: continue

            grid = create_4x4_grid(num)
            if grid:
                flat_grid = [cell for row in grid for cell in row]
                next_win = 1 if num in next_numbers else 0

                training_rows.append({
                    "date": today["date"],
                    "grid": str(flat_grid),
                    "next_win": next_win
                })
    except Exception as e:
        print(f"❌ Row {i} error: {e}")

if training_rows:
    pd.DataFrame(training_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Prepared {len(training_rows)} training rows → {OUTPUT_CSV}")
else:
    print("❌ Still no valid rows. Check your CSV for correct formats.")
