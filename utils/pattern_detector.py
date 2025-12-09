"""
Pattern Detector Utility
Detects various patterns in 4D lottery numbers
"""

from collections import Counter
import re

def detect_ascending_pattern(number):
    """Detect if digits are in ascending order"""
    num_str = str(number).strip()
    if not num_str.isdigit() or len(num_str) != 4:
        return False
    digits = [int(d) for d in num_str]
    return all(digits[i] <= digits[i+1] for i in range(len(digits)-1))

def detect_descending_pattern(number):
    """Detect if digits are in descending order"""
    num_str = str(number).strip()
    if not num_str.isdigit() or len(num_str) != 4:
        return False
    digits = [int(d) for d in num_str]
    return all(digits[i] >= digits[i+1] for i in range(len(digits)-1))

def detect_repeating_pattern(number):
    """Detect repeating digits"""
    num_str = str(number).strip()
    if not num_str.isdigit() or len(num_str) != 4:
        return False
    return len(set(num_str)) < len(num_str)

def detect_consecutive_pattern(number):
    """Detect consecutive digits"""
    num_str = str(number).strip()
    if not num_str.isdigit() or len(num_str) != 4:
        return False
    digits = [int(d) for d in num_str]
    consecutive_count = 0
    for i in range(len(digits)-1):
        if abs(digits[i] - digits[i+1]) == 1:
            consecutive_count += 1
    return consecutive_count >= 2

def detect_sum_pattern(number):
    """Detect sum-based patterns"""
    num_str = str(number).strip()
    if not num_str.isdigit() or len(num_str) != 4:
        return 'invalid'
    digits = [int(d) for d in num_str]
    total = sum(digits)
    
    if total <= 10:
        return 'low_sum'
    elif total <= 20:
        return 'medium_sum'
    else:
        return 'high_sum'

def detect_all_patterns(number):
    """Detect all patterns for a number"""
    patterns = []
    
    if detect_ascending_pattern(number):
        patterns.append('ascending')
    
    if detect_descending_pattern(number):
        patterns.append('descending')
    
    if detect_repeating_pattern(number):
        patterns.append('repeating')
    
    if detect_consecutive_pattern(number):
        patterns.append('consecutive')
    
    sum_pattern = detect_sum_pattern(number)
    patterns.append(sum_pattern)
    
    return patterns

def analyze_pattern_frequency(numbers):
    """Analyze frequency of patterns in a list of numbers"""
    pattern_counts = Counter()
    
    for number in numbers:
        patterns = detect_all_patterns(number)
        for pattern in patterns:
            pattern_counts[pattern] += 1
    
    return pattern_counts

def get_pattern_strength(number, historical_numbers):
    """Get pattern strength based on historical frequency"""
    patterns = detect_all_patterns(number)
    pattern_freq = analyze_pattern_frequency(historical_numbers)
    
    strength = 0
    for pattern in patterns:
        strength += pattern_freq.get(pattern, 0)
    
    return strength / len(historical_numbers) if historical_numbers else 0

def detect_patterns(numbers):
    """Detect patterns in a list of numbers - main function"""
    return analyze_pattern_frequency(numbers)