import csv

# Read the broken CSV
with open('4d_results_history.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Write fixed CSV with proper headers
with open('4d_results_history.csv', 'w', encoding='utf-8', newline='') as f:
    # Write proper header
    f.write('date,provider,draw_info,1st,2nd,3rd,special,consolation\n')
    
    # Skip the broken first line and write the rest
    for line in lines[1:]:
        f.write(line)

print("CSV file fixed! Headers added successfully.")
