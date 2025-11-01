# tasks/daily_job.py
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scraper import live4d_scraper

if __name__ == "__main__":
    live4d_scraper.run_scraper()
