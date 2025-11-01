import os
import csv
import time
import pandas as pd
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright

CSV_PATH = 'data/4d_results_history.csv'
START_DATE = datetime(2025, 7, 17)
END_DATE = datetime.now()

# === Ensure CSV Exists ===
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            "date", "provider", "game_title", "draw_info",
            "1st", "2nd", "3rd", "special", "consolation", "extra_info", "note"
        ])

existing = set()
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    existing = set(df['date'].dropna().unique())

# === Start Playwright ===
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # set to True if you want headless mode
    context = browser.new_context()
    page = context.new_page()

    print("🚀 Starting Playwright scraper...")

    current = START_DATE
    while current <= END_DATE:
        date_str = current.strftime('%Y-%m-%d')

        if date_str in existing:
            print(f"⏩ Skipping {date_str}")
            current += timedelta(days=1)
            continue

        try:
            page.goto("https://www.live4d2u.net/past-results", timeout=60000)
            page.wait_for_selector("#date")

            # === Click calendar icon ===
            page.click("img.ui-datepicker-trigger")
            page.wait_for_selector(".ui-datepicker-calendar")

            # === Navigate to correct year/month ===
            target_year = int(current.strftime('%Y'))
            target_month = current.strftime('%B')
            target_day = str(int(current.strftime('%d')))

            while True:
                visible_month = page.locator(".ui-datepicker-month").text_content()
                visible_year = int(page.locator(".ui-datepicker-year").text_content())
                if visible_month == target_month and visible_year == target_year:
                    break
                page.click(".ui-datepicker-prev")
                time.sleep(0.2)

            # === Click day ===
            found = False
            for el in page.locator(".ui-datepicker-calendar td a").all():
                if el.inner_text() == target_day:
                    el.click()
                    found = True
                    break
            if not found:
                print(f"❌ Could not find day {target_day}")
                current += timedelta(days=1)
                continue

            # Wait for result to load
            page.wait_for_selector(".result-container", timeout=10000)
            blocks = page.locator(".result-container")
            count = blocks.count()
            print(f"✅ {date_str}: Found {count} blocks")

            rows = []
            for i in range(count):
                block = blocks.nth(i)
                try:
                    provider = block.locator("img").get_attribute("src")
                    game_title = block.locator(".game-title").text_content().strip()
                    draw_info = block.locator(".drawdate").text_content().strip()
                    first = block.locator(".first-prize .number").text_content().strip()
                    second = block.locator(".second-prize .number").text_content().strip()
                    third = block.locator(".third-prize .number").text_content().strip()
                    special = ' '.join([e.text_content().strip() for e in block.locator(".special .number").all()])
                    consolation = ' '.join([e.text_content().strip() for e in block.locator(".consolation .number").all()])
                    extra_info = ' | '.join([e.text_content().strip() for e in block.locator(".additional-info, .result-footer").all() if e.text_content().strip()])
                    note = ' | '.join([e.text_content().strip() for e in block.locator(".note").all() if e.text_content().strip()])
                    rows.append([
                        date_str, provider, game_title, draw_info,
                        first, second, third, special, consolation, extra_info, note
                    ])
                except Exception as err:
                    print(f"⚠️ Failed to parse block {i}: {err}")

            # Save to CSV
            if rows:
                with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(rows)
                print(f"💾 Saved {len(rows)} rows for {date_str}")
            else:
                print(f"⚠️ No rows to save for {date_str}")

        except Exception as e:
            print(f"❌ Failed on {date_str}: {e}")

        current += timedelta(days=1)

    browser.close()
    print("✅ All Done.")
