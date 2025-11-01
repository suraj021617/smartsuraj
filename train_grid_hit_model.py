# train_grid_hit_model.py

import pandas as pd
from xgboost import XGBClassifier
from joblib import dump

df = pd.read_csv("grid_hit_training_data.csv")
X = df[[col for col in df.columns if col.startswith('cell_')]]
y = df['hit']

model = XGBClassifier(n_estimators=50, max_depth=4, eval_metric='logloss')
model.fit(X, y)

dump(model, "utils/grid_hit_model.joblib")
print("✅ AI Classifier trained and saved to 'utils/grid_hit_model.joblib'")
