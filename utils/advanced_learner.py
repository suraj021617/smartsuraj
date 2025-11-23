import numpy as np
import pandas as pd
import re
import time
from collections import defaultdict, Counter
from scipy import stats

class AdvancedLearner:
    def __init__(self, csv_path='4d_results_history_fixed.csv'):
        self.csv_path = csv_path
        self.data = None
        self.patterns = {}
        
    def load_data_safe(self):
        """Load CSV with malformed line handling"""
        print("Loading CSV data...")
        start = time.time()
        
        # Read as raw text
        try:
            with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            self.data = lines
            print(f"[OK] Loaded {len(self.data)} lines in {time.time()-start:.2f}s")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load CSV: {e}")
            return False
    
    def extract_numbers(self):
        """Extract all 4-digit numbers with progress"""
        print("\nExtracting 4D numbers...")
        all_nums = []
        total = len(self.data)
        start = time.time()
        
        for idx, line in enumerate(self.data, 1):
            nums = re.findall(r'\b\d{4}\b', line)
            all_nums.extend(nums)
            
            if idx % 1000 == 0:
                elapsed = time.time() - start
                rate = idx / elapsed if elapsed > 0 else 1
                eta = (total - idx) / rate
                print(f"\r  Progress: {idx}/{total} ({idx*100//total}%) | ETA: {int(eta)}s", end='')
        
        print(f"\n[OK] Extracted {len(all_nums)} numbers in {time.time()-start:.2f}s")
        return all_nums
    
    def learn_markov_chains(self, numbers):
        """Advanced: Markov chain transition probabilities"""
        print("\nLearning Markov chains...")
        start = time.time()
        
        # Digit-level transitions
        digit_trans = defaultdict(lambda: defaultdict(int))
        
        # Position-specific transitions
        pos_trans = [defaultdict(lambda: defaultdict(int)) for _ in range(4)]
        
        total = len(numbers) - 1
        for i in range(total):
            if len(numbers[i]) == 4 and len(numbers[i+1]) == 4:
                # Overall digit transitions
                for d1 in numbers[i]:
                    for d2 in numbers[i+1]:
                        digit_trans[d1][d2] += 1
                
                # Position-specific
                for pos in range(4):
                    d1, d2 = numbers[i][pos], numbers[i+1][pos]
                    pos_trans[pos][d1][d2] += 1
            
            if (i+1) % 5000 == 0:
                print(f"\r  Processed: {i+1}/{total} ({(i+1)*100//total}%)", end='')
        
        # Convert to probabilities
        for d1 in digit_trans:
            total_trans = sum(digit_trans[d1].values())
            for d2 in digit_trans[d1]:
                digit_trans[d1][d2] /= total_trans
        
        print(f"\n[OK] Markov chains learned in {time.time()-start:.2f}s")
        return {'digit': digit_trans, 'position': pos_trans}
    
    def learn_frequency_patterns(self, numbers):
        """Statistical frequency analysis"""
        print("\nAnalyzing frequency patterns...")
        start = time.time()
        
        # Position frequency
        pos_freq = [Counter() for _ in range(4)]
        
        # Pair frequency (consecutive digits)
        pair_freq = Counter()
        
        # Hot/Cold analysis (recent vs historical)
        recent = numbers[-1000:]
        historical = numbers[:-1000] if len(numbers) > 1000 else []
        
        for num in numbers:
            if len(num) == 4:
                for i, d in enumerate(num):
                    pos_freq[i][d] += 1
                for i in range(3):
                    pair_freq[num[i:i+2]] += 1
        
        print(f"[OK] Frequency analysis done in {time.time()-start:.2f}s")
        return {
            'position': pos_freq,
            'pairs': pair_freq,
            'recent': Counter(recent),
            'historical': Counter(historical)
        }
    
    def learn_time_series(self, numbers):
        """Time-series pattern detection"""
        print("\nDetecting time-series patterns...")
        start = time.time()
        
        # Convert to numeric for analysis
        numeric = [int(n) for n in numbers if n.isdigit() and len(n) == 4]
        
        if len(numeric) < 100:
            return {}
        
        # Moving averages
        window = 50
        ma = np.convolve(numeric, np.ones(window)/window, mode='valid')
        
        # Trend detection
        recent_trend = np.polyfit(range(len(numeric[-100:])), numeric[-100:], 1)[0]
        
        # Cyclical patterns (day of week, etc)
        cycles = defaultdict(list)
        for i, num in enumerate(numeric):
            cycles[i % 7].append(num)  # Weekly cycle
        
        print(f"[OK] Time-series analysis done in {time.time()-start:.2f}s")
        return {
            'moving_avg': ma[-1] if len(ma) > 0 else 0,
            'trend': recent_trend,
            'cycles': {k: np.mean(v) for k, v in cycles.items()}
        }
    
    def learn_all(self):
        """Master learning function with full progress tracking"""
        total_start = time.time()
        
        if not self.load_data_safe():
            return False
        
        numbers = self.extract_numbers()
        
        self.patterns['markov'] = self.learn_markov_chains(numbers)
        self.patterns['frequency'] = self.learn_frequency_patterns(numbers)
        self.patterns['timeseries'] = self.learn_time_series(numbers)
        self.patterns['raw_numbers'] = numbers
        
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"[COMPLETE] All learning done in {total_time:.2f}s")
        print(f"{'='*60}")
        
        return True
    
    def predict_advanced(self, top_n=10):
        """Generate predictions using all learned patterns"""
        predictions = []
        
        if not self.patterns:
            return []
        
        recent = self.patterns['raw_numbers'][-10:]
        last_num = recent[-1] if recent else '0000'
        
        # Method 1: Markov chain prediction
        if len(last_num) == 4:
            pred = ''
            for pos in range(4):
                d = last_num[pos]
                trans = self.patterns['markov']['digit'].get(d, {})
                if trans:
                    next_d = max(trans, key=trans.get)
                    pred += next_d
                else:
                    pred += d
            predictions.append((pred, 0.85, 'markov'))
        
        # Method 2: Frequency-based
        freq = self.patterns['frequency']['position']
        pred = ''.join(freq[i].most_common(1)[0][0] for i in range(4))
        predictions.append((pred, 0.75, 'frequency'))
        
        # Method 3: Hot numbers (recent frequency)
        hot = self.patterns['frequency']['recent'].most_common(5)
        for num, _ in hot:
            if len(num) == 4:
                predictions.append((num, 0.70, 'hot'))
        
        # Method 4: Time-series trend
        ts = self.patterns['timeseries']
        if 'moving_avg' in ts:
            pred = str(int(ts['moving_avg'])).zfill(4)
            predictions.append((pred, 0.65, 'trend'))
        
        # Remove duplicates
        seen = set()
        unique = []
        for num, conf, method in predictions:
            if num not in seen and len(num) == 4 and num.isdigit():
                seen.add(num)
                unique.append((num, conf, method))
        
        return unique[:top_n]
