"""
4D Results Scraper
Scrapes 4D lottery results from live4d2u.net and saves to data/4d_results_history.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scraper.date_range_scraper import scrape_date_range, save_to_csv
from datetime import datetime, timedelta

def main():
    # Default to scrape last 30 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    print("=" * 60)
    print("4D RESULTS SCRAPER")
    print("=" * 60)
    print(f"\nScraping dates: {start_date} to {end_date}")
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
