import csv
import re

def generate_grid_from_number(number_str):
    # Ensure the number is a 4-digit string
    if len(number_str) != 4 or not number_str.isdigit():
        return None

    formula_map = {'0': 5, '1': 6, '2': 7, '3': 8, '4': 9,
                   '5': 0, '6': 1, '7': 2, '8': 3, '9': 4}

    grid = [[int(d) for d in number_str]]
    second_row = [formula_map[d] for d in number_str]
    third_row = [(x + 1) % 10 for x in second_row]
    fourth_row = [(x + 2) % 10 for x in second_row]
    grid.extend([second_row, third_row, fourth_row])
    return grid

def extract_4d_numbers(prize_col):
    # Use regex to find 4-digit numbers in the 6th column
    return re.findall(r'\b\d{4}\b', prize_col)

with open('yourfile.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 6:
            continue  # Skip if row is too short
        date = row[0]
        prize_col = row[5]  # 6th column (index 5)
        numbers = extract_4d_numbers(prize_col)

        print(f"\nDate: {date}")
        for idx, title in enumerate(['1st', '2nd', '3rd']):
            if idx < len(numbers):
                num = numbers[idx]
                grid = generate_grid_from_number(num)
                print(f"{title} Prize ({num}):")
                if grid:
                    for g_row in grid:
                        print(g_row)
                else:
                    print("Invalid number!")
                print()
            else:
                print(f"{title} Prize: Not found\n")
