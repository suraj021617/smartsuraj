"""
SAFE ADVANCED LEARNER - Isolated, won't affect existing code
"""
import numpy as np
import re
import time
from collections import defaultdict, Counter

class SafeAdvancedLearner:
    def __init__(self, csv_path='4d_results_history_fixed.csv'):
        self.csv_path = csv_path
        self.patterns = None
        
    def learn_silent(self):
        """Learn patterns silently without breaking anything"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Extract numbers
            all_nums = []
            for line in lines:
                nums = re.findall(r'\b\d{4}\b', line)
                all_nums.extend(nums)
            
            # Quick Markov chains
            digit_trans = defaultdict(lambda: defaultdict(int))
            for i in range(len(all_nums) - 1):
                if len(all_nums[i]) == 4 and len(all_nums[i+1]) == 4:
                    for d1 in all_nums[i]:
                        for d2 in all_nums[i+1]:
                            digit_trans[d1][d2] += 1
            
            # Convert to probabilities
            for d1 in digit_trans:
                total = sum(digit_trans[d1].values())
                for d2 in digit_trans[d1]:
                    digit_trans[d1][d2] /= total
            
            # Position frequency
            pos_freq = [Counter() for _ in range(4)]
            for num in all_nums:
                if len(num) == 4:
                    for i, d in enumerate(num):
                        pos_freq[i][d] += 1
            
            self.patterns = {
                'markov': digit_trans,
                'position': pos_freq,
                'recent': all_nums[-50:],
                'all': all_nums
            }
            return True
        except:
            return False
    
    def get_predictions(self, top_n=10):
        """Get predictions safely"""
        if not self.patterns:
            if not self.learn_silent():
                return []
        
        predictions = []
        recent = self.patterns['recent']
        last_num = recent[-1] if recent else '0000'
        
        # Markov prediction
        if len(last_num) == 4:
            pred = ''
            for d in last_num:
                trans = self.patterns['markov'].get(d, {})
                pred += max(trans, key=trans.get) if trans else d
            predictions.append((pred, 0.85, 'markov'))
        
        # Frequency prediction
        pred = ''.join(self.patterns['position'][i].most_common(1)[0][0] for i in range(4))
        predictions.append((pred, 0.75, 'frequency'))
        
        # Hot numbers
        hot = Counter(recent).most_common(5)
        for num, _ in hot:
            if len(num) == 4:
                predictions.append((num, 0.70, 'hot'))
        
        # Remove duplicates
        seen = set()
        unique = []
        for num, conf, method in predictions:
            if num not in seen and len(num) == 4 and num.isdigit():
                seen.add(num)
                unique.append((num, conf, method))
        
        return unique[:top_n]

# Global instance - safe to use anywhere
_learner = None

def get_safe_predictions(top_n=10):
    """Safe function to get predictions without breaking anything"""
    global _learner
    if _learner is None:
        _learner = SafeAdvancedLearner()
    return _learner.get_predictions(top_n)
