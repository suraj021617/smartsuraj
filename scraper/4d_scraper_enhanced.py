"""
4D Results Scraper - Enhanced Version
Scrapes 4D lottery results with provider selection and date range options
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scraper.date_range_scraper import scrape_date_range, save_to_csv
from datetime import datetime, timedelta

def select_provider():
    providers = {
        '1': 'live4d2u.net',
        '2': 'magnum4d.com',
        '3': 'damacai.com',
        '4': 'toto.com.my'
    }
    
    print("\nSelect Provider:")
    for key, value in providers.items():
        print(f"[{key}] {value}")
    
    choice = input("\nEnter choice (1-4): ").strip()
    return providers.get(choice, providers['1'])

def select_date_range():
    print("\nSelect Date Range:")
    print("[1] Last 7 days")
    print("[2] Last 30 days")
    print("[3] Last 90 days")
    print("[4] Custom range")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        days = 7
    elif choice == '3':
        days = 90
    elif choice == '4':
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()
        return start_date, end_date
    else:
        days = 30
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    return start_date, end_date

def main():
    print("=" * 60)
    print("4D RESULTS SCRAPER - ENHANCED")
    print("=" * 60)
    
    provider = select_provider()
    start_date, end_date = select_date_range()
    
    print(f"\nProvider: {provider}")
    print(f"Scraping dates: {start_date} to {end_date}")
    print("Chrome browser will open automatically...\n")

    input("Press ENTER to start scraping...")

    results = scrape_date_range(start_date, end_date)
    print(f"\n✓ Total results scraped: {len(results)}")

    # Save to data directory
    csv_path = "data/4d_results_history.csv"
    save_to_csv(results, csv_path)

    print(f"\n✓ DONE! Data saved to {csv_path}")
    input("\nPress ENTER to close...")

if __name__ == "__main__":
    main()