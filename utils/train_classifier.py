import numpy as np

def extract_features(grid):
    grid = np.array(grid).reshape(4, 4)
    features = []

    # Raw cell values
    features.extend(grid.flatten())

    # Digit frequency
    digit_counts = [np.sum(grid == i) for i in range(10)]
    features.extend(digit_counts)

    # Grid stats
    features.append(np.sum(grid))
    features.append(np.mean(grid))
    features.append(len(np.unique(grid)))
    features.append(16 - len(np.unique(grid)))  # repeat count

    # Row/column/diagonal sums
    features.extend(np.sum(grid, axis=0))  # columns
    features.extend(np.sum(grid, axis=1))  # rows
    features.append(np.trace(grid))        # main diagonal
    features.append(np.trace(np.fliplr(grid)))  # anti-diagonal

    # Entropy
    probs = np.array(digit_counts) / 16
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
    features.append(entropy)

    return features