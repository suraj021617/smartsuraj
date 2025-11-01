# 🎯 Auto-Learning 4D Prediction System

## What It Does:
1. **Predicts** tomorrow's lottery numbers using AI
2. **Saves** predictions to CSV
3. **Learns** from actual results
4. **Improves** accuracy over time

---

## 📋 Quick Start Guide

### Step 1: Make Your First Prediction
```bash
python auto_predictor.py predict
```
This will:
- Analyze historical data
- Predict 6 numbers for tomorrow
- Save to `daily_predictions.csv`

### Step 2: Add Actual Results (After Draw)
```bash
python auto_predictor.py add_result 2025-01-15 1234,5678,9012
```
Replace:
- `2025-01-15` with actual draw date
- `1234,5678,9012` with actual winning numbers

### Step 3: View Performance
```bash
python auto_predictor.py stats
```
Shows:
- Total predictions
- Average accuracy
- Best matches
- Recent performance

### Step 4: Train AI Model
```bash
python auto_predictor.py train
```
After 10+ learned predictions, this trains the ML model

---

## 🔄 Daily Workflow

### Every Day:
1. **Run prediction** (morning)
   ```bash
   python auto_predictor.py predict
   ```

2. **Add results** (after draw)
   ```bash
   python auto_predictor.py add_result YYYY-MM-DD num1,num2,num3
   ```

3. **Check stats** (optional)
   ```bash
   python auto_predictor.py stats
   ```

### Or Use Batch File:
Double-click `run_daily_prediction.bat` to predict automatically!

---

## 📊 CSV Format: `daily_predictions.csv`

| prediction_date | draw_date | predicted_numbers | actual_numbers | matches | accuracy | learned |
|----------------|-----------|-------------------|----------------|---------|----------|---------|
| 2025-01-14 10:00 | 2025-01-15 | 1234,5678,9012 | 1234,5555,9012 | 2 | 33.33 | True |

---

## 🧠 How It Learns

### Features Used:
- Digit frequency (0-9)
- Recent number gaps
- Day of week patterns
- Historical trends

### Learning Process:
1. Compare predicted vs actual
2. Count matches
3. Calculate accuracy
4. Train RandomForest model
5. Use model for next prediction

### Improvement Over Time:
- First 10 predictions: Simple frequency-based
- After 10 predictions: ML model kicks in
- More data = Better predictions

---

## 🎯 Example Usage

### Day 1: First Prediction
```bash
> python auto_predictor.py predict
🚀 Starting Auto-Prediction System...
📊 Loaded 500 historical draws
✅ Prediction saved for 2025-01-15: ['1234', '5678', '9012', '3456', '7890', '2468']

🎯 Tomorrow's Predictions: ['1234', '5678', '9012', '3456', '7890', '2468']
```

### Day 2: Add Results & Learn
```bash
> python auto_predictor.py add_result 2025-01-15 1234,5555,9012
✅ Actual results added for 2025-01-15: ['1234', '5555', '9012']
📊 Learned: 2025-01-15 - 2 matches (33.3% accuracy)
```

### Day 11: Model Trained
```bash
> python auto_predictor.py train
✅ Model trained with 30 samples
```

### Anytime: Check Stats
```bash
> python auto_predictor.py stats

📈 PREDICTION PERFORMANCE
==================================================
Total Predictions: 15
Learned: 12
Average Accuracy: 28.5%
Best Match: 3 numbers
Total Matches: 34
```

---

## 🔧 Advanced Options

### Customize Lookback Period
Edit `auto_predictor.py`:
```python
def extract_features(df, lookback=30):  # Change 30 to 60, 90, etc.
```

### Change Number of Predictions
```python
predicted_numbers = [num for num, _ in freq.most_common(6)]  # Change 6 to 10
```

### Adjust Model Parameters
```python
model = RandomForestClassifier(n_estimators=100)  # Change 100 to 200
```

---

## 📁 Files Created

- `daily_predictions.csv` - All predictions and results
- `learning_model.pkl` - Trained ML model
- `scaler.pkl` - Feature scaler
- `run_daily_prediction.bat` - Quick run script

---

## 💡 Tips for Best Results

1. **Run daily** - More data = better learning
2. **Add results promptly** - Don't skip days
3. **Wait for 10+ predictions** before expecting ML improvements
4. **Check stats weekly** to track progress
5. **Retrain model** after every 20-30 new predictions

---

## 🚀 Automation (Optional)

### Windows Task Scheduler:
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Daily at 9:00 AM
4. Action: Start Program
5. Program: `C:\Users\Acer\Desktop\smartsuraj\run_daily_prediction.bat`

Now it runs automatically every day!

---

## ❓ Troubleshooting

### "No predictions yet"
- Run `python auto_predictor.py predict` first

### "Insufficient training data"
- Need at least 10 learned predictions
- Keep adding results daily

### Model not improving
- Check if results are being added correctly
- Verify CSV format
- Retrain: `python auto_predictor.py train`

---

## 📞 Support

If you need help:
1. Check `daily_predictions.csv` format
2. Verify historical data in `4d_results_history.csv`
3. Run `python auto_predictor.py stats` to diagnose

---

**That's it! Your AI is now learning and improving every day! 🎉**
