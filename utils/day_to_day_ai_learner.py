"""
AI Learning System for Day-to-Day Predictions
Tracks matches, learns patterns, and improves predictions
"""
import json
import os
from collections import defaultdict, Counter
from datetime import datetime

class DayToDayAILearner:
    def __init__(self):
        self.learning_file = 'data/day_to_day_learning.json'
        self.learning_data = self.load_learning_data()
    
    def load_learning_data(self):
        """Load learning data from file"""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'total_predictions': 0,
            'total_matches': 0,
            'pattern_success': {},
            'digit_transition_success': {},
            'sequence_success': {},
            'last_match': None,
            'match_history': []
        }
    
    def save_learning_data(self):
        """Save learning data to file"""
        os.makedirs('data', exist_ok=True)
        with open(self.learning_file, 'w') as f:
            json.dump(self.learning_data, f, indent=2)
    
    def record_prediction(self, predicted_numbers, actual_numbers, patterns_used):
        """Record a prediction and check for matches"""
        self.learning_data['total_predictions'] += 1
        
        matches = []
        match_types = {}
        
        for pred in predicted_numbers:
            for actual in actual_numbers:
                if pred == actual:
                    matches.append(pred)
                    match_types[pred] = 'EXACT'
                    self.learning_data['total_matches'] += 1
                elif sorted(pred) == sorted(actual):
                    matches.append(pred)
                    match_types[pred] = 'iBox'
                elif pred[:2] == actual[:2]:
                    matches.append(pred)
                    match_types[pred] = 'Front'
                elif pred[2:] == actual[2:]:
                    matches.append(pred)
                    match_types[pred] = 'Back'
        
        # Learn from matches
        if matches:
            self.learning_data['last_match'] = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'predicted': predicted_numbers[:3],
                'actual': actual_numbers,
                'matches': matches,
                'types': match_types
            }
            
            # Update pattern success rates
            for pattern_type in patterns_used:
                if pattern_type not in self.learning_data['pattern_success']:
                    self.learning_data['pattern_success'][pattern_type] = {'hits': 0, 'total': 0}
                self.learning_data['pattern_success'][pattern_type]['total'] += 1
                if any(m in matches for m in predicted_numbers):
                    self.learning_data['pattern_success'][pattern_type]['hits'] += 1
            
            # Store match history
            self.learning_data['match_history'].append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'matches': matches,
                'types': match_types
            })
            
            # Keep only last 100 matches
            if len(self.learning_data['match_history']) > 100:
                self.learning_data['match_history'] = self.learning_data['match_history'][-100:]
        
        self.save_learning_data()
        return matches, match_types
    
    def get_learning_stats(self):
        """Get current learning statistics"""
        total_preds = self.learning_data['total_predictions']
        total_matches = self.learning_data['total_matches']
        
        match_rate = round((total_matches / total_preds * 100), 1) if total_preds > 0 else 0
        
        # Find best performing pattern
        best_pattern = 'sequence'
        best_rate = 0
        for pattern, stats in self.learning_data['pattern_success'].items():
            if stats['total'] > 0:
                rate = stats['hits'] / stats['total']
                if rate > best_rate:
                    best_rate = rate
                    best_pattern = pattern
        
        # Calculate confidence based on recent performance
        recent_matches = self.learning_data['match_history'][-10:]
        confidence = min(95, 50 + len(recent_matches) * 4)
        
        # Generate learning message
        if match_rate >= 30:
            learning_message = "AI is performing excellently! High accuracy achieved."
        elif match_rate >= 20:
            learning_message = "AI is learning well. Good prediction patterns detected."
        elif match_rate >= 10:
            learning_message = "AI is improving. Building pattern database."
        else:
            learning_message = "AI is in early learning phase. Collecting data."
        
        last_match_info = None
        if self.learning_data['last_match']:
            lm = self.learning_data['last_match']
            last_match_info = f"{lm['date']}: {', '.join(lm['matches'])}"
        
        return {
            'total_learned': total_preds,
            'match_rate': match_rate,
            'best_pattern': best_pattern,
            'confidence': confidence,
            'last_match': last_match_info,
            'learning_message': learning_message
        }
    
    def get_boosted_predictions(self, predictions):
        """Boost predictions based on learned patterns"""
        if not predictions:
            return predictions
        
        boosted = []
        pattern_weights = {}
        
        # Calculate pattern weights from success rates
        for pattern, stats in self.learning_data['pattern_success'].items():
            if stats['total'] > 0:
                pattern_weights[pattern] = stats['hits'] / stats['total']
        
        for num, score, reason in predictions:
            boost = 1.0
            
            # Boost based on pattern type
            for pattern, weight in pattern_weights.items():
                if pattern in reason.lower():
                    boost += weight * 0.3
            
            # Boost if similar to past matches
            for match_record in self.learning_data['match_history'][-20:]:
                for matched_num in match_record['matches']:
                    # Check digit similarity
                    similarity = sum(1 for i in range(4) if num[i] == matched_num[i])
                    if similarity >= 2:
                        boost += 0.1
            
            boosted_score = min(score * boost, 1.0)
            boosted.append((num, boosted_score, reason))
        
        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted
