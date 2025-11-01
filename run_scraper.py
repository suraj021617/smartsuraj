"""
DOUBLE CLICK THIS FILE TO START SCRAPING
"""
import sys
sys.path.append('c:\\Users\\Acer\\Desktop\\smartsuraj')

from scraper.date_range_scraper import scrape_date_range, save_to_csv

print("=" * 60)
print("4D RESULTS SCRAPER")
print("=" * 60)
print("\nScraping dates: 2025-08-06 to 2025-09-17")
print("Chrome browser will open automatically...\n")

input("Press ENTER to start scraping...")

results = scrape_date_range('2025-08-06', '2025-09-17')
print(f"\n✓ Total results scraped: {len(results)}")

save_to_csv(results)

print("\n✓ DONE! Data saved to 4d_results_history.csv")
input("\nPress ENTER to close...")
