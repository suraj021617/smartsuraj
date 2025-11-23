def match_check(predictions, actual):
    matches = {'exact': 0, 'ibox': 0, 'front3': 0, 'back3': 0, 'digit3': 0, 'digit2': 0}
    
    for pred in predictions:
        for act in actual:
            if pred == act:
                matches['exact'] += 1
            elif sorted(pred) == sorted(act):
                matches['ibox'] += 1
            elif pred[:3] == act[:3]:
                matches['front3'] += 1
            elif pred[1:] == act[1:]:
                matches['back3'] += 1
            else:
                digit_matches = sum(1 for i in range(4) if pred[i] == act[i])
                if digit_matches == 3:
                    matches['digit3'] += 1
                elif digit_matches == 2:
                    matches['digit2'] += 1
    
    total = sum(matches.values())
    accuracy = (total / len(predictions)) * 100 if predictions else 0
    
    print(f"Exact: {matches['exact']}, iBox: {matches['ibox']}, Front3: {matches['front3']}")
    print(f"Back3: {matches['back3']}, 3-Digit: {matches['digit3']}, 2-Digit: {matches['digit2']}")
    print(f"Total Hits: {total}/{len(predictions)} ({accuracy:.1f}%)")
    return matches

# Example usage
predictions = ['1234', '5678', '9012']
actual = ['1234', '5679', '9013']
match_check(predictions, actual)