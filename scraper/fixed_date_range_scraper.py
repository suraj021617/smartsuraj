from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import re
from datetime import datetime, timedelta
import time

def scrape_single_date(driver, date_str):
    try:
        print(f"Scraping {date_str}...")
        
        # Navigate to the page
        driver.get("https://www.live4d2u.net/past-results")
        time.sleep(3)
        
        # Close any popups
        try:
            if len(driver.window_handles) > 1:
                driver.switch_to.window(driver.window_handles[-1])
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
        except:
            pass
        
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        target_month = dt.strftime('%b')
        target_year = str(dt.year)
        target_day = str(dt.day)
        
        # Wait for calendar to be clickable
        wait = WebDriverWait(driver, 10)
        calendar_img = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".ui-datepicker-trigger")))
        
        # Scroll to calendar and click
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", calendar_img)
        time.sleep(1)
        driver.execute_script("arguments[0].click();", calendar_img)
        time.sleep(2)
        
        # Select month
        month_select = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".ui-datepicker-month")))
        Select(month_select).select_by_visible_text(target_month)
        time.sleep(1)
        
        # Select year
        year_select = driver.find_element(By.CSS_SELECTOR, ".ui-datepicker-year")
        Select(year_select).select_by_visible_text(target_year)
        time.sleep(1)
        
        # Click day
        day_links = driver.find_elements(By.CSS_SELECTOR, ".ui-datepicker-calendar a")
        for link in day_links:
            if link.text.strip() == target_day:
                link.click()
                break
        
        time.sleep(5)
        
        # Wait for results to load
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".result-table, .results-container, table")))
        
        results = []
        
        # Try different selectors for result tables
        table_selectors = [
            "table",
            ".result-table", 
            ".results-container table",
            "[class*='result'] table",
            "[id*='result'] table"
        ]
        
        tables = []
        for selector in table_selectors:
            try:
                found_tables = driver.find_elements(By.CSS_SELECTOR, selector)
                tables.extend(found_tables)
            except:
                continue
        
        if not tables:
            # Fallback: get all text and parse
            page_text = driver.find_element(By.TAG_NAME, "body").text
            results = parse_text_results(page_text, date_str)
        else:
            # Parse tables
            for table in tables:
                try:
                    table_results = parse_table_results(table, date_str)
                    results.extend(table_results)
                except Exception as e:
                    print(f"Error parsing table: {e}")
                    continue
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for result in results:
            key = (result['date'], result['provider'], result['draw_no'])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results
        
    except Exception as e:
        print(f"Error scraping {date_str}: {e}")
        return []

def parse_table_results(table, date_str):
    results = []
    try:
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) < 3:
                continue
                
            # Extract provider from images or text
            provider = "Unknown"
            provider_cell = cells[0] if cells else None
            
            if provider_cell:
                # Check for images
                imgs = provider_cell.find_elements(By.TAG_NAME, "img")
                if imgs:
                    img_src = imgs[0].get_attribute("src") or ""
                    provider = extract_provider_from_image(img_src)
                else:
                    # Use text content
                    text = provider_cell.text.strip()
                    if text:
                        provider = clean_provider_name(text)
            
            # Extract other data from remaining cells
            draw_no = ""
            prizes = ""
            special = ""
            consolation = ""
            
            for i, cell in enumerate(cells[1:], 1):
                text = cell.text.strip()
                if "Draw" in text or "/" in text:
                    draw_no = extract_draw_number(text)
                elif "Prize" in text or any(num in text for num in ["1st", "2nd", "3rd"]):
                    prizes = extract_prizes(text)
                elif len(text) > 20 and any(c.isdigit() for c in text):
                    if not special:
                        special = extract_numbers(text)
                    else:
                        consolation = extract_numbers(text)
            
            if provider != "Unknown" and (draw_no or prizes):
                results.append({
                    'date': date_str,
                    'provider': provider,
                    'draw_no': draw_no,
                    'prizes': prizes,
                    'special': special,
                    'consolation': consolation
                })
                
    except Exception as e:
        print(f"Error parsing table: {e}")
    
    return results

def parse_text_results(page_text, date_str):
    results = []
    
    # Provider patterns
    providers = {
        'Magnum': ['magnum', 'Magnum 4D'],
        'Damacai': ['damacai', 'Da Ma Cai', 'DaMaCai'],
        'Toto': ['toto', 'SportsToto', 'Sports Toto'],
        'Sandakan': ['sandakan', 'Sandakan 4D'],
        'CashSweep': ['cashsweep', 'Cash Sweep', 'Special CashSweep'],
        'Grand Dragon': ['grand dragon', 'granddragon', 'Grand Dragon 4D'],
        'Perdana': ['perdana', 'Perdana Lottery'],
        'HariHari': ['harihari', 'Lucky HariHari']
    }
    
    lines = page_text.split('\n')
    current_provider = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Detect provider
        for provider, patterns in providers.items():
            if any(pattern.lower() in line.lower() for pattern in patterns):
                current_provider = provider
                break
        
        # Extract draw info and prizes
        if current_provider and ("Draw" in line or "Prize" in line):
            draw_no = extract_draw_number(line)
            prizes = extract_prizes(line)
            
            # Look for special and consolation in next few lines
            special = ""
            consolation = ""
            
            for j in range(i+1, min(i+5, len(lines))):
                next_line = lines[j].strip()
                if len(next_line) > 10 and any(c.isdigit() for c in next_line):
                    if not special:
                        special = extract_numbers(next_line)
                    elif not consolation:
                        consolation = extract_numbers(next_line)
                        break
            
            if draw_no or prizes:
                results.append({
                    'date': date_str,
                    'provider': current_provider,
                    'draw_no': draw_no,
                    'prizes': prizes,
                    'special': special,
                    'consolation': consolation
                })
    
    return results

def extract_provider_from_image(img_src):
    """Extract provider name from image source"""
    if not img_src:
        return "Unknown"
    
    img_src = img_src.lower()
    
    if 'magnum' in img_src:
        return 'Magnum'
    elif 'damacai' in img_src:
        return 'Damacai'
    elif 'toto' in img_src:
        return 'Toto'
    elif 'sandakan' in img_src:
        return 'Sandakan'
    elif 'cashsweep' in img_src:
        return 'CashSweep'
    elif 'grand' in img_src or 'dragon' in img_src:
        return 'Grand Dragon'
    elif 'perdana' in img_src:
        return 'Perdana'
    elif 'hari' in img_src:
        return 'HariHari'
    else:
        return 'Unknown'

def clean_provider_name(text):
    """Clean provider name from text"""
    text = text.lower().strip()
    
    if 'magnum' in text:
        return 'Magnum'
    elif 'damacai' in text or 'da ma cai' in text:
        return 'Damacai'
    elif 'toto' in text:
        return 'Toto'
    elif 'sandakan' in text:
        return 'Sandakan'
    elif 'cashsweep' in text or 'cash sweep' in text:
        return 'CashSweep'
    elif 'grand dragon' in text or 'granddragon' in text:
        return 'Grand Dragon'
    elif 'perdana' in text:
        return 'Perdana'
    elif 'hari' in text:
        return 'HariHari'
    else:
        return text.title()

def extract_draw_number(text):
    """Extract draw number from text"""
    # Look for patterns like "Draw No: 123/45" or "123/45"
    match = re.search(r'(\d+/\d+|\d{3,})', text)
    return match.group(1) if match else ""

def extract_prizes(text):
    """Extract prize information from text"""
    prizes = []
    
    # Look for 1st, 2nd, 3rd prize patterns
    for prize in ['1st', '2nd', '3rd']:
        pattern = rf'{prize}[^0-9]*(\d{{4}})'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            prizes.append(f"{prize} {match.group(1)}")
    
    return ' | '.join(prizes)

def extract_numbers(text):
    """Extract 4-digit numbers from text"""
    numbers = re.findall(r'\b\d{4}\b', text)
    return ' '.join(numbers[:10])  # Limit to first 10 numbers

def scrape_date_range(start_date='2025-08-06', end_date='2025-09-17'):
    print(f"Scraping from {start_date} to {end_date}...")
    
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    all_results = []
    total_days = (end - current).days + 1
    
    try:
        day_count = 0
        while current <= end:
            day_count += 1
            date_str = current.strftime('%Y-%m-%d')
            print(f"[{day_count}/{total_days}] Processing {date_str}...")
            
            results = scrape_single_date(driver, date_str)
            all_results.extend(results)
            print(f"Found {len(results)} results for {date_str}")
            
            current += timedelta(days=1)
            time.sleep(3)  # Longer delay between requests
            
    finally:
        driver.quit()
    
    return all_results

def save_to_csv(results, filename='4d_results_history_fixed.csv'):
    if not results:
        print("No results to save!")
        return
    
    df = pd.DataFrame(results)
    
    # Clean up the data
    df = df.drop_duplicates()
    df = df[df['provider'] != 'Unknown']  # Remove unknown providers
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Save with headers
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} unique results to {filename}")
    
    # Print summary
    print("\nSummary by provider:")
    print(df['provider'].value_counts())