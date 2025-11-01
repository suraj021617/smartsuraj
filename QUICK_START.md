# 🚀 Quick Start Guide

## Available Routes

### 📊 Prediction Routes
- `/best-predictions` - Top AI predictions
- `/ultimate-predictor` - Multi-model consensus
- `/quick-pick` - Fast number selection
- `/ml-predictor` - Machine learning predictions
- `/smart-predictor` - Auto-tuned predictions
- `/day-to-day-predictor` - Daily updates
- `/consensus-predictor` - Combined predictions

### 📈 Analysis Routes
- `/pattern-analyzer` - Pattern detection
- `/frequency-analyzer` - Number frequency
- `/statistics` - Overall statistics
- `/hot-cold` - Hot/cold numbers
- `/missing-number-finder` - Overdue numbers
- `/empty-box-predictor` - Empty slots

### 📜 Tracking Routes
- `/accuracy-dashboard` - Prediction accuracy
- `/prediction-history` - Past predictions
- `/learning-insights` - Learning analytics
- `/smart-history` - Smart predictor history

### 💾 Export Routes (NEW!)
- `/export/predictions` - Download predictions CSV
- `/export/statistics` - Download statistics CSV
- `/export/accuracy` - Download accuracy CSV

## Usage Examples

### Get Predictions
```
http://localhost:5000/best-predictions?provider=magnum
```

### Export Data
```
http://localhost:5000/export/predictions?provider=all
```

### View Statistics
```
http://localhost:5000/statistics
```

## CSV Export Format

### Predictions Export
```csv
number,score,method,timestamp
1234,0.95,hot+pair+trans,2025-01-18 10:30:00
5678,0.87,ML-learned,2025-01-18 10:30:00
```

### Statistics Export
```csv
number,frequency,percentage
1234,45,2.5
5678,38,2.1
```

### Accuracy Export
```csv
prediction_date,draw_date,provider,predicted_numbers,hit_status,accuracy_score
2025-01-15,2025-01-18,magnum,"1234,5678",HIT (1234),50.0
```

## Tips

1. **Cache**: Data cached for 5 minutes for performance
2. **Providers**: Filter by magnum, toto, damacai, or all
3. **Export**: Use export routes to download data anytime
4. **Accuracy**: Track predictions in `/accuracy-dashboard`
5. **Learning**: System adapts based on hit/miss rates
