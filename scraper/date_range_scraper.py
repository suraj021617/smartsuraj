from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
import pandas as pd
import re
from datetime import datetime, timedelta
import time

def scrape_single_date(driver, date_str):
    try:
        # Close popups
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
        
        # Click calendar
        driver.switch_to.window(driver.window_handles[0])
        calendar_img = driver.find_element(By.CSS_SELECTOR, ".ui-datepicker-trigger")
        
        # Scroll to calendar icon to avoid navbar blocking
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", calendar_img)
        time.sleep(1)
        
        # Use JavaScript click to avoid interception
        driver.execute_script("arguments[0].click();", calendar_img)
        time.sleep(2)
        
        # Select month
        selects = driver.find_elements(By.TAG_NAME, "select")
        for sel in selects:
            try:
                opts = [o.text for o in Select(sel).options]
                if 'Jan' in opts:
                    Select(sel).select_by_visible_text(target_month)
                    break
            except:
                continue
        time.sleep(1)
        
        # Select year
        selects = driver.find_elements(By.TAG_NAME, "select")
        for sel in selects:
            try:
                opts = [o.text for o in Select(sel).options]
                if target_year in opts:
                    Select(sel).select_by_visible_text(target_year)
                    break
            except:
                continue
        time.sleep(1)
        
        # Click day
        day_links = driver.find_elements(By.TAG_NAME, "a")
        for link in day_links:
            if link.text.strip() == target_day:
                try:
                    link.click()
                    break
                except:
                    continue
        
        time.sleep(6)
        
        # Get page text
        page_text = driver.find_element(By.TAG_NAME, "body").text
        
        results = []
        
        # Split by provider sections
        providers = {
            'Magnum 4D': 'Magnum',
            'Da Ma Cai 1+3D': 'Damacai',
            'SportsToto 4D': 'Toto',
            'Sandakan 4D': 'Sandakan',
            'Special CashSweep': 'CashSweep',
            'Grand Dragon 4D': 'Grand Dragon',
            'Perdana Lottery 4D': 'Perdana',
            'Lucky HariHari': 'HariHari'
        }
        
        for provider_name, short_name in providers.items():
            if provider_name in page_text:
                # Find section for this provider
                start = page_text.find(provider_name)
                # Find next provider or end
                end = len(page_text)
                for other_provider in providers.keys():
                    if other_provider != provider_name:
                        next_pos = page_text.find(other_provider, start + 1)
                        if next_pos > start and next_pos < end:
                            end = next_pos
                
                section = page_text[start:end]
                
                # Extract draw number
                draw_match = re.search(r'Draw No[:\s]*(\d+/\d+|\d+)', section)
                draw_no = draw_match.group(1) if draw_match else ''
                
                # Extract prizes
                prizes = {}
                for prize in ['1st', '2nd', '3rd']:
                    pattern = rf'{prize}\s+Prize[^\d]*(\d{{4}})'
                    match = re.search(pattern, section)
                    if match:
                        prizes[prize] = match.group(1)
                
                if prizes:
                    prize_str = ' | '.join([f"{k} {v}" for k, v in prizes.items()])
                    
                    # Extract special numbers
                    special_match = re.search(r'Special[^C]*?Consolation', section, re.DOTALL)
                    special = ''
                    if special_match:
                        special_text = special_match.group(0)
                        special_nums = re.findall(r'\b\d{4}\b', special_text)
                        special = ' '.join(special_nums[:10])
                    
                    # Extract consolation numbers
                    consolation = ''
                    cons_start = section.find('Consolation')
                    if cons_start > 0:
                        cons_text = section[cons_start:cons_start+200]
                        cons_nums = re.findall(r'\b\d{4}\b', cons_text)
                        consolation = ' '.join(cons_nums[:10])
                    
                    results.append({
                        'date': date_str,
                        'provider': short_name,
                        'draw_no': draw_no,
                        'prizes': prize_str,
                        'special': special,
                        'consolation': consolation
                    })
        
        return results
    except Exception as e:
        print(f"Error: {e}")
        return []

def scrape_date_range(start_date='2025-08-06', end_date='2025-09-17'):
    print(f"Scraping from {start_date} to {end_date}...")
    
    chrome_options = Options()
    chrome_options.page_load_strategy = 'eager'
    driver = webdriver.Chrome(options=chrome_options)
    
    driver.get("https://www.live4d2u.net/past-results")
    time.sleep(15)
    
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    all_results = []
    total_days = (end - current).days + 1
    
    try:
        day_count = 0
        while current <= end:
            day_count += 1
            date_str = current.strftime('%Y-%m-%d')
            print(f"[{day_count}/{total_days}] {date_str}...", end=' ')
            
            results = scrape_single_date(driver, date_str)
            all_results.extend(results)
            print(f"{len(results)} results")
            
            current += timedelta(days=1)
            time.sleep(2)
    finally:
        driver.quit()
    
    return all_results

def save_to_csv(results, filename='4d_results_history.csv'):
    if not results:
        print("No results!")
        return
    
    df = pd.DataFrame(results)
    try:
        df.to_csv(filename, mode='a', index=False, header=False)
        print(f"\nSaved {len(df)} to {filename}")
    except:
        df.to_csv(filename, index=False, header=False)
        print(f"\nCreated {filename}")
