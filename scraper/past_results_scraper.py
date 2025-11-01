"""
Scraper for https://www.live4d2u.net/past-results
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

def scrape_past_results():
    url = "https://www.live4d2u.net/past-results"
    print(f"Scraping {url}...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    results = []
    
    # Find all rows in tables or divs with results
    rows = soup.find_all(['tr', 'div'], class_=re.compile(r'result|row|draw'))
    
    for row in rows:
        text = row.get_text()
        
        # Look for date pattern
        date_match = re.search(r'(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})', text)
        if not date_match:
            continue
        
        date = date_match.group(1)
        
        # Find provider image
        img = row.find('img')
        provider = img['src'] if img and 'src' in img.attrs else ''
        
        # Extract 4-digit numbers
        numbers = re.findall(r'\b(\d{4})\b', text)
        
        if len(numbers) >= 3:
            result = {
                'date': date,
                'provider': provider,
                'game': '',
                'draw_no': '',
                'prizes': f"1st Prize {numbers[0]} | 2nd Prize {numbers[1]} | 3rd Prize {numbers[2]}",
                'special': ' '.join(numbers[3:13]) if len(numbers) > 3 else '',
                'consolation': ' '.join(numbers[13:23]) if len(numbers) > 13 else ''
            }
            results.append(result)
    
    print(f"Found {len(results)} results")
    return results

def save_to_csv(results, filename='4d_results_history.csv'):
    if not results:
        print("No results to save!")
        return
    
    df = pd.DataFrame(results)
    
    # Append to existing file
    try:
        df.to_csv(filename, mode='a', index=False, header=False)
        print(f"Appended {len(df)} results to {filename}")
    except:
        df.to_csv(filename, index=False, header=False)
        print(f"Created {filename} with {len(df)} results")

if __name__ == '__main__':
    results = scrape_past_results()
    if results:
        save_to_csv(results)
