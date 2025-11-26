def detect_patterns(grid):
    patterns = []

    if len(grid) != 4 or any(len(row) != 4 for row in grid):
        return ["invalid_grid"]

    # 🔹 Diagonals
    if all(grid[i][i] == grid[0][0] for i in range(4)):
        patterns.append("diagonal_main")
    if all(grid[i][3 - i] == grid[0][3] for i in range(4)):
        patterns.append("diagonal_anti")

    # 🔹 Snake (zigzag)
    snake = [grid[0][0], grid[1][1], grid[2][2], grid[3][3]]
    if len(set(snake)) == 1:
        patterns.append("snake_diagonal")

    # 🔹 Box corners
    corners = [grid[0][0], grid[0][3], grid[3][0], grid[3][3]]
    if len(set(corners)) == 1:
        patterns.append("box_corners")

    # 🔹 Center block (2x2)
    center = [grid[1][1], grid[1][2], grid[2][1], grid[2][2]]
    if len(set(center)) == 1:
        patterns.append("center_block")

    # 🔹 Cross
    mid_row = grid[1]
    mid_col = [grid[i][1] for i in range(4)]
    if len(set(mid_row)) == 1 and len(set(mid_col)) == 1:
        patterns.append("cross")

    # 🔹 Horizontal repeats
    for i, row in enumerate(grid):
        if len(set(row)) == 1:
            patterns.append(f"horizontal_repeat_row{i}")

    # 🔹 Vertical repeats
    for col in range(4):
        column = [grid[row][col] for row in range(4)]
        if len(set(column)) == 1:
            patterns.append(f"vertical_repeat_col{col}")

    # 🔹 Edge symmetry
    if grid[0] == grid[3]:
        patterns.append("edge_top_bottom")

    # 🔹 Corner symmetry
    if grid[0][0] == grid[3][3] and grid[0][3] == grid[3][0]:
        patterns.append("corner_symmetry")

    # 🔹 L-shapes
    if grid[0][0] == grid[1][0] == grid[1][1]:
        patterns.append("L_top_left")
    if grid[0][3] == grid[1][3] == grid[1][2]:
        patterns.append("L_top_right")
    if grid[3][0] == grid[2][0] == grid[2][1]:
        patterns.append("L_bottom_left")
    if grid[3][3] == grid[2][3] == grid[2][2]:
        patterns.append("L_bottom_right")

    # 🔹 T-shapes
    if grid[0][1] == grid[0][2] == grid[1][1]:
        patterns.append("T_top")
    if grid[3][1] == grid[3][2] == grid[2][1]:
        patterns.append("T_bottom")
    if grid[1][0] == grid[2][0] == grid[1][1]:
        patterns.append("T_left")
    if grid[1][3] == grid[2][3] == grid[1][2]:
        patterns.append("T_right")

    # 🔹 Plus shape
    center = grid[1][1]
    if center == grid[0][1] == grid[2][1] == grid[1][0] == grid[1][2]:
        patterns.append("plus_center")

    # 🔹 X-shape
    if grid[0][0] == grid[1][1] == grid[2][2] == grid[3][3] and \
       grid[0][3] == grid[1][2] == grid[2][1] == grid[3][0]:
        patterns.append("x_cross")

    return patterns if patterns else ["no_pattern"]