"""
Markov Chain Predictor - Learn number transitions
If 1234 appeared, what number came NEXT?
"""
from collections import defaultdict, Counter

def build_transition_matrix(numbers):
    """Learn: After number X, what comes next?"""
    transitions = defaultdict(list)
    
    for i in range(len(numbers) - 1):
        current = numbers[i]
        next_num = numbers[i + 1]
        transitions[current].append(next_num)
    
    return transitions

def predict_next_numbers(last_number, transitions, top_n=5):
    """Predict based on what came after this number before"""
    if last_number not in transitions:
        return []
    
    # Count frequency of next numbers
    next_nums = transitions[last_number]
    freq = Counter(next_nums)
    
    predictions = []
    for num, count in freq.most_common(top_n):
        confidence = (count / len(next_nums)) * 100
        predictions.append((num, confidence, f"After {last_number}: {count}x"))
    
    return predictions

def markov_predictor(df, provider='all', lookback=500):
    """Main predictor using Markov chains"""
    if provider != 'all':
        df = df[df['provider'] == provider]
    
    # Extract all numbers in sequence
    all_nums = []
    for _, row in df.tail(lookback).iterrows():
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row[col])
            if len(num) == 4 and num.isdigit():
                all_nums.append(num)
    
    if len(all_nums) < 2:
        return []
    
    # Build transition matrix
    transitions = build_transition_matrix(all_nums)
    
    # Get last number
    last_num = all_nums[-1]
    
    # Predict what comes next
    predictions = predict_next_numbers(last_num, transitions, top_n=10)
    
    # If no direct transitions, use most common followers
    if not predictions:
        all_followers = []
        for followers in transitions.values():
            all_followers.extend(followers)
        freq = Counter(all_followers)
        predictions = [(num, 50, 'Common follower') for num, _ in freq.most_common(10)]
    
    return predictions[:5]
