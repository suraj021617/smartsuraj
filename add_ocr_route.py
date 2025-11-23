# Add this route to app.py before if __name__ == "__main__":

@app.route('/positional-ocr')
def positional_ocr():
    df = load_csv_data()
    selected_provider = request.args.get('provider', 'magnum')
    selected_date = request.args.get('date', '')
    
    provider_options = sorted([p for p in df['provider'].dropna().unique() if p and str(p).strip()])
    if not provider_options:
        provider_options = ['magnum', 'damacai', 'singapore', 'gdlotto']
    
    ocr_table = None
    predictions = []
    
    if selected_date and selected_provider:
        # Filter by date and provider
        date_obj = pd.to_datetime(selected_date).date()
        filtered = df[(df['date_parsed'].dt.date == date_obj) & (df['provider'] == selected_provider)]
        
        # Collect all numbers
        all_numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            all_numbers.extend([n for n in filtered[col].astype(str) if len(n) == 4 and n.isdigit()])
        
        # Extract from special and consolation
        for _, row in filtered.iterrows():
            for col in ['special', 'consolation']:
                nums = re.findall(r'\b\d{4}\b', str(row.get(col, '')))
                all_numbers.extend(nums)
        
        if all_numbers:
            # Build OCR table
            ocr_table = {digit: [0, 0, 0, 0] for digit in range(10)}
            for num in all_numbers:
                for pos in range(4):
                    digit = int(num[pos])
                    ocr_table[digit][pos] += 1
            
            # Find hot digits per position
            hot_digits = []
            for pos in range(4):
                max_count = max(ocr_table[d][pos] for d in range(10))
                hot_digit = [d for d in range(10) if ocr_table[d][pos] == max_count][0]
                hot_digits.append(str(hot_digit))
            
            # Generate predictions
            base_pred = ''.join(hot_digits)
            predictions.append({'number': base_pred, 'confidence': 85, 'reason': f'Top digits: Pos1={hot_digits[0]}, Pos2={hot_digits[1]}, Pos3={hot_digits[2]}, Pos4={hot_digits[3]}'})
            
            # Alternative combinations
            for pos in range(4):
                sorted_digits = sorted(range(10), key=lambda d: ocr_table[d][pos], reverse=True)
                if len(sorted_digits) > 1:
                    alt_digits = hot_digits.copy()
                    alt_digits[pos] = str(sorted_digits[1])
                    alt_num = ''.join(alt_digits)
                    if alt_num != base_pred:
                        predictions.append({'number': alt_num, 'confidence': 75 - len(predictions)*5, 'reason': f'Alternative at Pos{pos+1}'})
                if len(predictions) >= 10:
                    break
    
    return render_template('positional_ocr.html',
                         provider_options=provider_options,
                         selected_provider=selected_provider,
                         selected_date=selected_date,
                         ocr_table=ocr_table,
                         predictions=predictions)
