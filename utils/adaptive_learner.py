"""
Adaptive Learning System - Learns from prediction mismatches
Automatically adjusts weights and strategies based on what actually works
"""
import pandas as pd
import json
import os
from datetime import datetime
from collections import defaultdict, Counter

class AdaptiveLearner:
    def __init__(self):
        self.learning_file = "adaptive_learning.json"
        self.load_learning_data()
    
    def load_learning_data(self):
        """Load learning data from file"""
        if os.path.exists(self.learning_file):
            with open(self.learning_file, 'r') as f:
                self.learning_data = json.load(f)
        else:
            self.learning_data = {
                'method_weights': {'advanced': 0.25, 'smart': 0.25, 'ml': 0.25, 'pattern': 0.25},
                'method_accuracy': {'advanced': [], 'smart': [], 'ml': [], 'pattern': []},
                'digit_patterns': {},
                'provider_insights': {},
                'miss_analysis': [],
                'total_predictions': 0,
                'total_hits': 0
            }
    
    def save_learning_data(self):
        """Save learning data to file"""
        with open(self.learning_file, 'w') as f:
            json.dump(self.learning_data, f, indent=2)
    
    def analyze_miss(self, predicted_numbers, actual_winner, provider, date):
        """
        When prediction doesn't match, analyze WHY
        This is the core learning function
        """
        analysis = {
            'date': str(date),
            'provider': provider,
            'predicted': predicted_numbers,
            'actual': actual_winner,
            'insights': []
        }
        
        # Analyze digit patterns
        predicted_digits = set(''.join(predicted_numbers))
        actual_digits = set(actual_winner)
        
        # What digits did we miss?
        missed_digits = actual_digits - predicted_digits
        if missed_digits:
            analysis['insights'].append(f"Missed digits: {', '.join(missed_digits)}")
            # Learn: These digits are important
            for digit in missed_digits:
                if digit not in self.learning_data['digit_patterns']:
                    self.learning_data['digit_patterns'][digit] = {'importance': 1}
                else:
                    self.learning_data['digit_patterns'][digit]['importance'] += 1
        
        # Analyze digit sum pattern
        predicted_sums = [sum(int(d) for d in num) for num in predicted_numbers]
        actual_sum = sum(int(d) for d in actual_winner)
        
        if actual_sum not in range(min(predicted_sums)-5, max(predicted_sums)+5):
            analysis['insights'].append(f"Sum mismatch: predicted {min(predicted_sums)}-{max(predicted_sums)}, actual {actual_sum}")
        
        # Analyze odd/even pattern
        actual_odd_count = sum(1 for d in actual_winner if int(d) % 2 == 1)
        predicted_odd_counts = [sum(1 for d in num if int(d) % 2 == 1) for num in predicted_numbers]
        
        if actual_odd_count not in predicted_odd_counts:
            analysis['insights'].append(f"Odd/even mismatch: actual has {actual_odd_count} odd digits")
        
        # Store analysis
        self.learning_data['miss_analysis'].append(analysis)
        
        # Keep only last 100 analyses
        if len(self.learning_data['miss_analysis']) > 100:
            self.learning_data['miss_analysis'] = self.learning_data['miss_analysis'][-100:]
        
        self.save_learning_data()
        return analysis
    
    def update_method_accuracy(self, method_name, was_correct):
        """Update accuracy tracking for a method"""
        self.learning_data['method_accuracy'][method_name].append(1 if was_correct else 0)
        
        # Keep only last 50 predictions per method
        if len(self.learning_data['method_accuracy'][method_name]) > 50:
            self.learning_data['method_accuracy'][method_name] = self.learning_data['method_accuracy'][method_name][-50:]
        
        # Recalculate weights based on recent accuracy
        self.adjust_weights()
        self.save_learning_data()
    
    def adjust_weights(self):
        """Automatically adjust method weights based on accuracy"""
        accuracies = {}
        
        for method, results in self.learning_data['method_accuracy'].items():
            if len(results) >= 10:  # Need at least 10 predictions
                accuracy = sum(results) / len(results)
                accuracies[method] = accuracy
            else:
                accuracies[method] = 0.25  # Default weight
        
        # Normalize weights (total = 1.0)
        total = sum(accuracies.values())
        if total > 0:
            for method in accuracies:
                self.learning_data['method_weights'][method] = accuracies[method] / total
    
    def get_adaptive_predictions(self, advanced_preds, smart_preds, ml_preds, pattern_preds=None):
        """
        Combine predictions using LEARNED weights
        Give more importance to methods that have been accurate
        """
        weights = self.learning_data['method_weights']
        
        # Score each number based on weighted votes
        number_scores = defaultdict(float)
        
        for num, score, _ in advanced_preds:
            number_scores[num] += weights['advanced'] * score
        
        for num, score, _ in smart_preds:
            number_scores[num] += weights['smart'] * score
        
        for num, score, _ in ml_preds:
            number_scores[num] += weights['ml'] * score
        
        if pattern_preds:
            for num, score, _ in pattern_preds:
                number_scores[num] += weights['pattern'] * score
        
        # Apply learned digit importance
        for num in number_scores:
            for digit in num:
                if digit in self.learning_data['digit_patterns']:
                    importance = self.learning_data['digit_patterns'][digit]['importance']
                    number_scores[num] += importance * 0.01  # Small boost
        
        # Sort by score
        sorted_predictions = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(num, score, f"Adaptive (weights: A:{weights['advanced']:.2f} S:{weights['smart']:.2f} M:{weights['ml']:.2f})") 
                for num, score in sorted_predictions[:10]]
    
    def get_learning_insights(self):
        """Get insights about what the system has learned"""
        insights = []
        
        # Method performance
        for method, results in self.learning_data['method_accuracy'].items():
            if len(results) >= 5:
                accuracy = sum(results) / len(results) * 100
                weight = self.learning_data['method_weights'][method] * 100
                insights.append({
                    'type': 'method_performance',
                    'method': method,
                    'accuracy': round(accuracy, 1),
                    'weight': round(weight, 1),
                    'predictions': len(results)
                })
        
        # Important digits learned
        if self.learning_data['digit_patterns']:
            sorted_digits = sorted(self.learning_data['digit_patterns'].items(), 
                                 key=lambda x: x[1]['importance'], reverse=True)
            insights.append({
                'type': 'important_digits',
                'digits': [(d, info['importance']) for d, info in sorted_digits[:5]]
            })
        
        # Recent miss patterns
        if self.learning_data['miss_analysis']:
            recent_misses = self.learning_data['miss_analysis'][-10:]
            common_insights = Counter()
            for miss in recent_misses:
                for insight in miss['insights']:
                    common_insights[insight] += 1
            
            insights.append({
                'type': 'common_miss_patterns',
                'patterns': common_insights.most_common(5)
            })
        
        # Overall stats
        insights.append({
            'type': 'overall_stats',
            'total_predictions': self.learning_data['total_predictions'],
            'total_hits': self.learning_data['total_hits'],
            'hit_rate': round(self.learning_data['total_hits'] / self.learning_data['total_predictions'] * 100, 1) 
                       if self.learning_data['total_predictions'] > 0 else 0
        })
        
        return insights
    
    def record_prediction_result(self, predicted_numbers, actual_winners, methods_used, provider, date):
        """
        Record a prediction result and learn from it
        """
        self.learning_data['total_predictions'] += 1
        
        # Check if any prediction hit
        hit = any(pred in actual_winners for pred in predicted_numbers)
        
        if hit:
            self.learning_data['total_hits'] += 1
            # Update accuracy for methods that contributed to the hit
            for method in methods_used:
                self.update_method_accuracy(method, True)
        else:
            # Analyze why we missed
            self.analyze_miss(predicted_numbers, actual_winners[0], provider, date)
            # Update accuracy for methods that missed
            for method in methods_used:
                self.update_method_accuracy(method, False)
        
        self.save_learning_data()
        return hit
    
    def get_recommendations(self):
        """Get recommendations based on learning"""
        recommendations = []
        
        # Best performing method
        best_method = max(self.learning_data['method_weights'].items(), key=lambda x: x[1])
        recommendations.append(f"✅ Trust '{best_method[0]}' method most (weight: {best_method[1]*100:.1f}%)")
        
        # Worst performing method
        worst_method = min(self.learning_data['method_weights'].items(), key=lambda x: x[1])
        if worst_method[1] < 0.15:
            recommendations.append(f"⚠️ '{worst_method[0]}' method underperforming (weight: {worst_method[1]*100:.1f}%)")
        
        # Important digits
        if self.learning_data['digit_patterns']:
            top_digits = sorted(self.learning_data['digit_patterns'].items(), 
                              key=lambda x: x[1]['importance'], reverse=True)[:3]
            digit_str = ', '.join([d for d, _ in top_digits])
            recommendations.append(f"🔢 Focus on digits: {digit_str}")
        
        # Overall performance
        if self.learning_data['total_predictions'] >= 10:
            hit_rate = self.learning_data['total_hits'] / self.learning_data['total_predictions'] * 100
            if hit_rate > 30:
                recommendations.append(f"🎯 System performing well ({hit_rate:.1f}% hit rate)")
            elif hit_rate < 15:
                recommendations.append(f"📊 System needs more data ({hit_rate:.1f}% hit rate, {self.learning_data['total_predictions']} predictions)")
        
        return recommendations
