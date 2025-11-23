import pandas as pd
from datetime import datetime

def check_matches():
    predictions = input("Enter predictions (comma-separated): ").split(',')
    actual = input("Enter actual results (comma-separated): ").split(',')
    
    predictions = [p.strip() for p in predictions]
    actual = [a.strip() for a in actual]
    
    matches = {
        'exact': [],
        'ibox': [],
        'front3': [],
        'back3': [],
        'digit3': [],
        'digit2': []
    }
    
    for pred in predictions:
        for act in actual:
            if pred == act:
                matches['exact'].append((pred, act))
            elif sorted(pred) == sorted(act):
                matches['ibox'].append((pred, act))
            elif pred[:3] == act[:3]:
                matches['front3'].append((pred, act))
            elif pred[1:] == act[1:]:
                matches['back3'].append((pred, act))
            else:
                digit_matches = sum(1 for i in range(4) if pred[i] == act[i])
                if digit_matches == 3:
                    matches['digit3'].append((pred, act))
                elif digit_matches == 2:
                    matches['digit2'].append((pred, act))
    
    print("\n=== MATCH RESULTS ===")
    print(f"Exact: {len(matches['exact'])} - {matches['exact']}")
    print(f"iBox: {len(matches['ibox'])} - {matches['ibox']}")
    print(f"Front 3: {len(matches['front3'])} - {matches['front3']}")
    print(f"Back 3: {len(matches['back3'])} - {matches['back3']}")
    print(f"3-Digit: {len(matches['digit3'])} - {matches['digit3']}")
    print(f"2-Digit: {len(matches['digit2'])} - {matches['digit2']}")
    
    total_hits = sum(len(v) for v in matches.values())
    accuracy = (total_hits / len(predictions)) * 100 if predictions else 0
    print(f"\nTotal Hits: {total_hits}/{len(predictions)} ({accuracy:.1f}%)")

if __name__ == "__main__":
    check_matches()