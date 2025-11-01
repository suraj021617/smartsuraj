# utils/feedback_learner.py
import pandas as pd
import json
from datetime import datetime
from collections import defaultdict

class FeedbackLearner:
    def __init__(self):
        self.learning_data = defaultdict(lambda: {
            'total_predictions': 0,
            'exact_matches': 0,
            'three_digit_matches': 0,
            'two_digit_matches': 0,
            'success_patterns': [],
            'failed_patterns': []
        })
    
    def count_matching_digits(self, predicted, actual):
        """Count how many digits match between predicted and actual"""
        if not predicted or not actual:
            return 0
        
        pred_str = str(predicted).zfill(4)
        actual_str = str(actual).zfill(4)
        
        # Position-based matching
        position_matches = sum(1 for i in range(4) if pred_str[i] == actual_str[i])
        
        # Any position matching
        pred_digits = set(pred_str)
        actual_digits = set(actual_str)
        any_matches = len(pred_digits & actual_digits)
        
        return max(position_matches, any_matches)
    
    def evaluate_prediction(self, predicted_numbers, actual_1st, actual_2nd, actual_3rd):
        """
        Evaluate prediction against actual results
        Returns: match_type, score, details
        """
        actuals = [actual_1st, actual_2nd, actual_3rd]
        best_match = 0
        match_details = []
        
        for pred in predicted_numbers:
            for actual in actuals:
                matches = self.count_matching_digits(pred, actual)
                if matches > best_match:
                    best_match = matches
                match_details.append({
                    'predicted': pred,
                    'actual': actual,
                    'matches': matches
                })
        
        # Determine match type
        if best_match == 4:
            match_type = 'EXACT'
            score = 100
        elif best_match == 3:
            match_type = '3-DIGIT'
            score = 75
        elif best_match == 2:
            match_type = '2-DIGIT'
            score = 50
        else:
            match_type = 'MISS'
            score = 0
        
        return match_type, score, match_details
    
    def learn_from_result(self, prediction_data, match_type, score):
        """Learn from prediction result"""
        method = prediction_data.get('predictor_methods', 'Unknown')
        
        self.learning_data[method]['total_predictions'] += 1
        
        if match_type == 'EXACT':
            self.learning_data[method]['exact_matches'] += 1
            self.learning_data[method]['success_patterns'].append(prediction_data)
        elif match_type == '3-DIGIT':
            self.learning_data[method]['three_digit_matches'] += 1
        elif match_type == '2-DIGIT':
            self.learning_data[method]['two_digit_matches'] += 1
        else:
            self.learning_data[method]['failed_patterns'].append(prediction_data)
    
    def get_method_accuracy(self, method):
        """Calculate accuracy for a specific method"""
        data = self.learning_data[method]
        if data['total_predictions'] == 0:
            return 0
        
        weighted_score = (
            data['exact_matches'] * 100 +
            data['three_digit_matches'] * 75 +
            data['two_digit_matches'] * 50
        )
        
        return weighted_score / data['total_predictions']
    
    def get_best_methods(self, top_n=3):
        """Get top performing prediction methods"""
        methods = []
        for method in self.learning_data.keys():
            accuracy = self.get_method_accuracy(method)
            methods.append({
                'method': method,
                'accuracy': accuracy,
                'total_predictions': self.learning_data[method]['total_predictions']
            })
        
        return sorted(methods, key=lambda x: x['accuracy'], reverse=True)[:top_n]
    
    def save_learning_data(self, filepath='learning_history.json'):
        """Save learning data to file"""
        with open(filepath, 'w') as f:
            json.dump(dict(self.learning_data), f, indent=2, default=str)
    
    def load_learning_data(self, filepath='learning_history.json'):
        """Load learning data from file"""
        try:
            with open(filepath, 'r') as f:
                self.learning_data = defaultdict(lambda: {
                    'total_predictions': 0,
                    'exact_matches': 0,
                    'three_digit_matches': 0,
                    'two_digit_matches': 0,
                    'success_patterns': [],
                    'failed_patterns': []
                }, json.load(f))
        except FileNotFoundError:
            pass
