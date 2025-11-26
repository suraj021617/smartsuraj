def build_grid(number):
    """Return a 4x4 grid based on +5/+6/+7 pattern"""
    row1 = [int(d) for d in number]
    row2 = [(d + 5) % 10 for d in row1]
    row3 = [(d + 6) % 10 for d in row1]
    row4 = [(d + 7) % 10 for d in row1]
    return [row1, row2, row3, row4]

def detect_patterns(grid):
    """Return list of patterns detected inside the 4x4 grid"""
    patterns = []

    # Diagonal ↘ pattern
    if grid[0][0] == grid[1][1] == grid[2][2] == grid[3][3]:
        patterns.append("Diagonal ↘")

    # Box corners
    corners = [grid[0][0], grid[0][3], grid[3][0], grid[3][3]]
    if len(set(corners)) <= 2:
        patterns.append("Box Corners Match")

    # L-shape ┏
    if grid[0][0] == grid[0][1] == grid[0][2] == grid[1][0]:
        patterns.append("L-Shape ┏")

    # Same column
    for col in range(4):
        if all(grid[row][col] == grid[0][col] for row in range(1, 4)):
            patterns.append(f"Same Column {col+1}")

    return patterns

# Test run
if __name__ == "__main__":
    number = "5445"
    grid = build_grid(number)

    for row in grid:
        print(row)

    results = detect_patterns(grid)
    print("Detected Patterns:", results)

