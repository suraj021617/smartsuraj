"""
CONFIDENCE SCORER - Calculates confidence levels for predictions
"""
import numpy as np
from collections import Counter

class ConfidenceScorer:
    def __init__(self):
        self.confidence_levels = {
            'very_high': (0.8, 1.0),
            'high': (0.6, 0.8),
            'medium': (0.4, 0.6),
            'low': (0.2, 0.4),
            'very_low': (0.0, 0.2)
        }
    
    def calculate_confidence(self, number, historical_data, method_accuracies=None):
        """
        Calculate confidence score for a prediction
        Returns: (confidence_score, confidence_level, reasons)
        """
        scores = []
        reasons = []
        
        # 1. Frequency-based confidence
        freq_score = self._frequency_confidence(number, historical_data)
        scores.append(freq_score)
        reasons.append(f"Frequency: {freq_score:.2f}")
        
        # 2. Recency-based confidence
        recency_score = self._recency_confidence(number, historical_data)
        scores.append(recency_score)
        reasons.append(f"Recency: {recency_score:.2f}")
        
        # 3. Pattern-based confidence
        pattern_score = self._pattern_confidence(number, historical_data)
        scores.append(pattern_score)
        reasons.append(f"Pattern: {pattern_score:.2f}")
        
        # 4. Method accuracy boost
        if method_accuracies:
            method_score = np.mean(list(method_accuracies.values()))
            scores.append(method_score)
            reasons.append(f"Method: {method_score:.2f}")
        
        # Calculate final confidence
        final_confidence = np.mean(scores)
        confidence_level = self._get_confidence_level(final_confidence)
        
        return final_confidence, confidence_level, reasons
    
    def _frequency_confidence(self, number, historical_data):
        """Confidence based on how often number appeared"""
        if not historical_data:
            return 0.5
        
        count = historical_data.count(number)
        max_count = max(Counter(historical_data).values())
        
        if max_count == 0:
            return 0.5
        
        return min(count / max_count, 1.0)
    
    def _recency_confidence(self, number, historical_data):
        """Confidence based on how recently number appeared"""
        if not historical_data or number not in historical_data:
            return 0.3
        
        # Find last occurrence
        last_index = len(historical_data) - 1 - historical_data[::-1].index(number)
        recency = (len(historical_data) - last_index) / len(historical_data)
        
        # Recent = higher confidence
        return 1.0 - recency
    
    def _pattern_confidence(self, number, historical_data):
        """Confidence based on digit patterns"""
        if not historical_data:
            return 0.5
        
        # Analyze digit patterns in recent history
        recent = historical_data[-50:]
        
        # Check if number's digits are trending
        number_digits = set(number)
        recent_digits = Counter(''.join(recent))
        
        # Calculate how "hot" the digits are
        digit_scores = []
        for digit in number_digits:
            digit_freq = recent_digits.get(digit, 0) / len(''.join(recent))
            digit_scores.append(digit_freq)
        
        return np.mean(digit_scores) * 10  # Scale up
    
    def _get_confidence_level(self, score):
        """Convert score to confidence level"""
        for level, (min_score, max_score) in self.confidence_levels.items():
            if min_score <= score < max_score:
                return level
        return 'medium'
    
    def get_confidence_color(self, level):
        """Get color for confidence level"""
        colors = {
            'very_high': 'green',
            'high': 'lightgreen',
            'medium': 'yellow',
            'low': 'orange',
            'very_low': 'red'
        }
        return colors.get(level, 'gray')
    
    def get_confidence_emoji(self, level):
        """Get emoji for confidence level"""
        emojis = {
            'very_high': '🔥',
            'high': '✅',
            'medium': '⚠️',
            'low': '❓',
            'very_low': '❌'
        }
        return emojis.get(level, '❔')
    
    def batch_score(self, predictions, historical_data, method_accuracies=None):
        """
        Score multiple predictions at once
        Returns: [(number, confidence, level, reasons), ...]
        """
        scored_predictions = []
        
        for pred in predictions:
            if isinstance(pred, tuple):
                number = pred[0]
            else:
                number = pred
            
            confidence, level, reasons = self.calculate_confidence(
                number, historical_data, method_accuracies
            )
            
            scored_predictions.append({
                'number': number,
                'confidence': round(confidence, 3),
                'level': level,
                'emoji': self.get_confidence_emoji(level),
                'color': self.get_confidence_color(level),
                'reasons': reasons
            })
        
        # Sort by confidence
        scored_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return scored_predictions
