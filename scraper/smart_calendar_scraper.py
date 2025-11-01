"""
Smart Calendar Scraper - Clicks calendar widget and scrapes all available dates
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

class CalendarScraper:
    def __init__(self, headless=True):
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1920,1080')
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)
    
    def open_website(self, url='https://www.live4d2u.net'):
        print(f"Opening {url}...")
        self.driver.get(url)
        time.sleep(3)
    
    def click_calendar(self):
        """Click calendar button to open date picker"""
        try:
            # Try different selectors
            selectors = [
                "//button[contains(@class, 'calendar')]",
                "//input[@type='date']",
                "//div[contains(@class, 'datepicker')]",
                "//a[contains(@href, 'date')]"
            ]
            
            for selector in selectors:
                try:
                    element = self.driver.find_element(By.XPATH, selector)
                    element.click()
                    print("Calendar opened!")
                    time.sleep(1)
                    return True
                except:
                    continue
            
            print("Calendar button not found, using direct URL method")
            return False
        except Exception as e:
            print(f"Error clicking calendar: {e}")
            return False
    
    def get_available_dates(self):
        """Get all available dates from calendar"""
        try:
            # Look for date links or calendar cells
            date_elements = self.driver.find_elements(By.XPATH, 
                "//a[contains(@href, 'date=')]")
            
            dates = []
            for elem in date_elements:
                href = elem.get_attribute('href')
                match = re.search(r'date=(\d{4}-\d{2}-\d{2})', href)
                if match:
                    dates.append(match.group(1))
            
            return list(set(dates))
        except Exception as e:
            print(f"Error getting dates: {e}")
            return []
    
    def scrape_current_page(self):
        """Scrape all results from current page"""
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        results = []
        
        # Method 1: Find result containers
        containers = soup.find_all(['div', 'section'], class_=re.compile(r'result|card|draw'))
        
        for container in containers:
            try:
                result = self.parse_result_container(container)
                if result:
                    results.append(result)
            except Exception as e:
                continue
        
        return results
    
    def parse_result_container(self, container):
        """Parse a single result container"""
        result = {}
        
        # Get date
        date_elem = container.find(['span', 'div', 'p'], class_=re.compile(r'date'))
        if date_elem:
            result['date'] = date_elem.text.strip()
        
        # Get provider from image
        img = container.find('img')
        if img and 'src' in img.attrs:
            result['provider'] = img['src']
        
        # Get game name
        title = container.find(['h1', 'h2', 'h3', 'h4', 'h5'])
        if title:
            result['game'] = title.text.strip()
        
        # Get prizes
        prize_text = container.get_text()
        prizes = {}
        
        # Extract 1st, 2nd, 3rd prizes
        for prize_type in ['1st', '2nd', '3rd']:
            pattern = rf'{prize_type}[^\d]*(\d{{4}})'
            match = re.search(pattern, prize_text, re.IGNORECASE)
            if match:
                prizes[prize_type] = match.group(1)
        
        result['prizes'] = ' | '.join([f"{k} Prize {v}" for k, v in prizes.items()])
        
        # Get special numbers
        special = []
        tables = container.find_all('table')
        if tables:
            for td in tables[0].find_all('td'):
                text = td.text.strip()
                if text and (len(text) == 4 or text == '----'):
                    special.append(text)
        
        result['special'] = ' '.join(special) if special else ''
        
        # Get consolation
        consolation = []
        if len(tables) > 1:
            for td in tables[1].find_all('td'):
                text = td.text.strip()
                if text and (len(text) == 4 or text == '----'):
                    consolation.append(text)
        
        result['consolation'] = ' '.join(consolation) if consolation else ''
        
        return result if result.get('prizes') else None
    
    def scrape_all(self, output_file='4d_results_auto.csv'):
        """Main scraping function"""
        self.open_website()
        
        # Try to get dates from calendar
        self.click_calendar()
        dates = self.get_available_dates()
        
        if not dates:
            print("No dates found in calendar, scraping current page only")
            results = self.scrape_current_page()
        else:
            print(f"Found {len(dates)} dates to scrape")
            results = []
            
            for date in dates[:30]:  # Limit to 30 dates
                print(f"Scraping {date}...")
                self.driver.get(f"https://www.live4d2u.net/?date={date}")
                time.sleep(2)
                
                page_results = self.scrape_current_page()
                results.extend(page_results)
                print(f"  Found {len(page_results)} results")
        
        # Save results
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"\nSaved {len(results)} results to {output_file}")
        else:
            print("No results found!")
        
        self.driver.quit()
        return results

def main():
    scraper = CalendarScraper(headless=False)  # Set True to hide browser
    results = scraper.scrape_all()
    print(f"\nTotal results scraped: {len(results)}")

if __name__ == '__main__':
    main()
