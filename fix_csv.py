import csv
import re

def clean_csv_file(input_file, output_file):
    """Clean and fix the 4D results CSV file"""
    
    cleaned_rows = []
    
    # Define proper headers
    headers = ['date', 'provider', 'lottery_name', 'draw_number', 'draw_date', 'prize_summary', 'special_numbers', 'consolation_numbers']
    cleaned_rows.append(headers)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Split into lines and process each
        lines = content.split('\n')
        
        for i, line in enumerate(lines[1:], 1):  # Skip header
            if not line.strip():
                continue
                
            # Split by comma but handle quoted fields
            try:
                # Use csv module to properly parse the line
                reader = csv.reader([line])
                fields = next(reader)
                
                # Ensure we have at least 8 fields
                while len(fields) < 8:
                    fields.append('')
                
                # Clean each field
                cleaned_fields = []
                for j, field in enumerate(fields[:8]):  # Only take first 8 fields
                    # Remove excessive whitespace and special characters
                    cleaned_field = re.sub(r'\s+', ' ', field.strip())
                    
                    # Clean prize summary field (remove Chinese characters and excessive text)
                    if j == 5:  # prize_summary field
                        cleaned_field = re.sub(r'[\u4e00-\u9fff]+', '', cleaned_field)  # Remove Chinese
                        cleaned_field = re.sub(r'4D Jackpot.*?Prize.*?\|', '', cleaned_field)  # Remove jackpot info
                        cleaned_field = re.sub(r'Jackpot.*?RM.*?\|', '', cleaned_field)  # Remove jackpot amounts
                        cleaned_field = re.sub(r'\s*\|\s*$', '', cleaned_field)  # Remove trailing |
                        cleaned_field = cleaned_field.strip()
                    
                    # Fix provider field
                    if j == 1 and not cleaned_field.startswith('http'):
                        if 'Singapore' in cleaned_field:
                            cleaned_field = 'https://www.singaporepools.com.sg'
                        elif cleaned_field == '':
                            cleaned_field = 'https://unknown-provider.com'
                    
                    # Fix lottery name field
                    if j == 2:
                        if cleaned_field.startswith('http'):
                            cleaned_field = 'Unknown Lottery'
                        elif cleaned_field == '4D':
                            cleaned_field = 'Singapore 4D'
                    
                    cleaned_fields.append(cleaned_field)
                
                # Validate date format
                if cleaned_fields[0] and not re.match(r'\d{4}-\d{2}-\d{2}', cleaned_fields[0]):
                    continue  # Skip invalid date rows
                
                cleaned_rows.append(cleaned_fields)
                
            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    # Write cleaned data
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(cleaned_rows)
        
        print(f"Successfully cleaned CSV file. Output: {output_file}")
        print(f"Total rows processed: {len(cleaned_rows) - 1}")  # Exclude header
        return True
        
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

if __name__ == "__main__":
    input_file = "4d_results_history_fixed.csv"
    output_file = "4d_results_history_cleaned.csv"
    
    success = clean_csv_file(input_file, output_file)
    if success:
        print("CSV file has been successfully cleaned!")
    else:
        print("Failed to clean CSV file.")