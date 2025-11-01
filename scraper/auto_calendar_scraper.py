"""
Automated Calendar Scraper - Opens website, clicks calendar dates, scrapes all data
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta

def setup_driver():
    options = Options()
    options.add_argument('--headless')  # Remove to see browser
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)
    return driver

def scrape_date(driver, date_str):
    """Scrape data for a specific date"""
    url = f"https://www.live4d2u.net/?date={date_str}"
    driver.get(url)
    time.sleep(2)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    results = []
    
    # Find all result cards
    cards = soup.find_all('div', class_='card')
    
    for card in cards:
        try:
            # Extract provider
            img = card.find('img')
            provider = img['src'] if img else ''
            
            # Extract game name
            game_name = card.find('h5', class_='card-title')
            game = game_name.text.strip() if game_name else ''
            
            # Extract draw info
            draw_info = card.find('p', class_='card-text')
            draw = draw_info.text.strip() if draw_info else ''
            
            # Extract prizes
            prize_divs = card.find_all('div', class_='prize')
            prizes = {}
            for p in prize_divs:
                text = p.text.strip()
                if '1st' in text:
                    prizes['1st'] = text
                elif '2nd' in text:
                    prizes['2nd'] = text
                elif '3rd' in text:
                    prizes['3rd'] = text
            
            # Extract special and consolation
            special = ''
            consolation = ''
            tables = card.find_all('table')
            if len(tables) > 0:
                special_rows = tables[0].find_all('td')
                special = ' '.join([td.text.strip() for td in special_rows])
            if len(tables) > 1:
                cons_rows = tables[1].find_all('td')
                consolation = ' '.join([td.text.strip() for td in cons_rows])
            
            results.append({
                'date': date_str,
                'provider': provider,
                'game': game,
                'draw_info': draw,
                'prizes': str(prizes),
                'special': special,
                'consolation': consolation
            })
        except Exception as e:
            print(f"Error parsing card: {e}")
            continue
    
    return results

def scrape_date_range(start_date, end_date, output_file='4d_results_history.csv'):
    """Scrape all dates in range"""
    driver = setup_driver()
    all_results = []
    
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    print(f"Scraping from {start_date} to {end_date}...")
    
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        print(f"Scraping {date_str}...")
        
        try:
            results = scrape_date(driver, date_str)
            all_results.extend(results)
            print(f"  Found {len(results)} results")
        except Exception as e:
            print(f"  Error: {e}")
        
        current += timedelta(days=1)
        time.sleep(1)  # Be nice to server
    
    driver.quit()
    
    # Save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Load existing data
        try:
            existing = pd.read_csv(output_file)
            df = pd.concat([existing, df]).drop_duplicates(subset=['date', 'provider', 'game'])
        except:
            pass
        
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(df)} results to {output_file}")
    
    return all_results

if __name__ == '__main__':
    # Scrape last 7 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    scrape_date_range(start_date, end_date)
