"""
Real-time Data Ingestion & Processing Engine
"""
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import numpy as np

class RealtimeEngine:
    def __init__(self, df):
        self.df = df
        self.last_update = datetime.now()
    
    def detect_sequences(self, lookback=100):
        """Detect recurring number sequences"""
        nums = self.df['1st_real'].tail(lookback).astype(str).tolist()
        sequences = {}
        for i in range(len(nums)-2):
            seq = f"{nums[i]}->{nums[i+1]}->{nums[i+2]}"
            sequences[seq] = sequences.get(seq, 0) + 1
        return sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def get_overdue_numbers(self, threshold=30):
        """Find numbers that haven't appeared recently - optimized for performance"""
        # Single query to get recent data
        recent_data = self.df.tail(threshold)[['1st_real', '2nd_real', '3rd_real']]
        appeared = set(recent_data.values.flatten())
        appeared = {str(n) for n in appeared if str(n).isdigit() and len(str(n)) == 4}
        
        # Single query for historical candidates
        historical_data = self.df.tail(200)[['1st_real', '2nd_real', '3rd_real']]
        historical_nums = set(historical_data.values.flatten())
        historical_nums = {str(n) for n in historical_nums if str(n).isdigit() and len(str(n)) == 4}
        
        candidate_pool = historical_nums - appeared
        
        # Generate additional candidates if needed
        if len(candidate_pool) < 50:
            for i in range(1000, 2000):  # Start from 1000 to avoid common low numbers
                num = f"{i:04d}"
                if num not in appeared:
                    candidate_pool.add(num)
                    if len(candidate_pool) >= 100:
                        break
        
        return list(candidate_pool)[:50]
    
    def get_number_pairs(self, top_n=20):
        """Analyze frequently occurring number pairs"""
        all_nums = [n for col in ['1st_real', '2nd_real', '3rd_real'] 
                    for n in self.df[col].astype(str) if n.isdigit() and len(n) == 4]
        pairs = []
        for i in range(len(all_nums)-1):
            pairs.append(f"{all_nums[i]}-{all_nums[i+1]}")
        return Counter(pairs).most_common(top_n)
    
    def calculate_trend_score(self, number, days=30):
        """Calculate trending score for a number"""
        recent = self.df[self.df['date_parsed'] > (datetime.now() - timedelta(days=days))]
        old = self.df[(self.df['date_parsed'] > (datetime.now() - timedelta(days=days*2))) & 
                      (self.df['date_parsed'] <= (datetime.now() - timedelta(days=days)))]
        
        recent_count = sum([1 for col in ['1st_real', '2nd_real', '3rd_real'] 
                           if (recent[col].astype(str) == number).any()])
        old_count = sum([1 for col in ['1st_real', '2nd_real', '3rd_real'] 
                        if (old[col].astype(str) == number).any()])
        
        return recent_count - old_count
    
    def get_hot_cold_analysis(self, lookback=90):
        """Advanced hot/cold analysis with trend direction"""
        recent = self.df.tail(lookback)
        all_nums = [n for col in ['1st_real', '2nd_real', '3rd_real'] 
                    for n in recent[col].astype(str) if n.isdigit() and len(n) == 4]
        freq = Counter(all_nums)
        
        hot = []
        cold = []
        for num, count in freq.most_common():
            trend = self.calculate_trend_score(num, 30)
            if count >= np.percentile(list(freq.values()), 90):
                hot.append({'number': num, 'count': count, 'trend': '📈' if trend > 0 else '📉'})
            elif count <= np.percentile(list(freq.values()), 10):
                cold.append({'number': num, 'count': count, 'trend': '📈' if trend > 0 else '📉'})
        
        return {'hot': hot[:20], 'cold': cold[:20]}
    
    def predict_next_draw(self, method='ensemble'):
        """Real-time prediction using latest data"""
        latest = self.df.tail(50)
        all_nums = [n for col in ['1st_real', '2nd_real', '3rd_real'] 
                    for n in latest[col].astype(str) if n.isdigit() and len(n) == 4]
        
        # Frequency-based
        freq = Counter(all_nums)
        freq_preds = [n for n, c in freq.most_common(10)]
        
        # Pattern-based
        sequences = self.detect_sequences(50)
        pattern_preds = [seq.split('->')[-1] for seq, _ in sequences[:10]]
        
        # Combine
        combined = Counter(freq_preds + pattern_preds)
        return [n for n, _ in combined.most_common(5)]
