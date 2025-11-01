import pandas as pd
from xgboost import XGBRegressor
from joblib import dump

df = pd.read_csv("clean_4d_training_data.csv")

# Validate input format
if not set(['1st', '2nd', '3rd']).issubset(df.columns):
    raise ValueError(f"❌ Unexpected CSV format: {list(df.columns)}")

X = []
y = []

def extract_features_from_number(n):
    digits = [int(d) for d in str(n).zfill(4)]
    grid = [digits]
    grid.append([(d + 5) % 10 for d in digits])
    grid.append([(d + 6) % 10 for d in digits])
    grid.append([(d + 7) % 10 for d in digits])
    return [cell for row in grid for cell in row][:16]

for _, row in df.iterrows():
    for col in ['1st', '2nd', '3rd']:
        val = str(row[col]).strip()
        if val.isdigit() and len(val) == 4:
            X.append(extract_features_from_number(val))
            y.append(int(val))

X_df = pd.DataFrame(X)
y_series = pd.Series(y)

model = XGBRegressor(n_estimators=50, max_depth=5, verbosity=0)
model.fit(X_df, y_series)

dump(model, "utils/real_model.joblib")
print("✅ Model trained and saved to utils/real_model.joblib")
