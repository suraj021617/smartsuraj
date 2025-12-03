"""
Test script for the fixed scraper
Tests a small date range to verify the fixes work
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from scraper.fixed_date_range_scraper import scrape_date_range, save_to_csv
from datetime import datetime, timedelta

def test_scraper():
    # Test with just 3 days to verify it works
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')

    print("=" * 60)
    print("TESTING FIXED 4D SCRAPER")
    print("=" * 60)
    print(f"Testing dates: {start_date} to {end_date}")
    print("This will test the fixes for:")
    print("- Proper provider name extraction")
    print("- Duplicate removal")
    print("- Better data parsing")
    print("- Improved error handling")
    print()

    input("Press ENTER to start test...")

    results = scrape_date_range(start_date, end_date)
    print(f"\n✓ Total results scraped: {len(results)}")

    if results:
        # Save to test file
        test_file = "data/test_4d_results.csv"
        save_to_csv(results, test_file)
        print(f"\n✓ Test data saved to {test_file}")
        
        # Show sample results
        print("\nSample results:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. {result['date']} - {result['provider']} - {result['prizes']}")
    else:
        print("\n⚠ No results found - check if the website is accessible")

    input("\nPress ENTER to close...")

if __name__ == "__main__":
    test_scraper()