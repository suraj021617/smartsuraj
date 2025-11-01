import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import os

# === CONFIG ===
CSV_PATH = "data/4d_results_history.csv"  # Change path as needed
URL = "https://www.live4d2u.net/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/114.0.0.0 Safari/537.36"
}

def fetch_results():
    try:
        response = requests.get(URL, headers=HEADERS, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        draw_date_tag = soup.select_one(".drawdate")
        draw_date = draw_date_tag.get_text(strip=True) if draw_date_tag else datetime.now().strftime("%Y-%m-%d")

        provider_blocks = soup.select("div.result4d")
        if not provider_blocks:
            print("⚠️ No result blocks found — maybe IP is blocked.")
            return []

        all_data = []

        for block in provider_blocks:
            provider_img = block.select_one("img")
            provider = provider_img['src'] if provider_img else ""

            draw_info = block.select_one(".drawnumber")
            draw_info = draw_info.get_text(strip=True) if draw_info else ""

            prizes = block.select(".resultnum")
            prizes_text = [p.get_text(strip=True) for p in prizes]
            while len(prizes_text) < 3:
                prizes_text.append("")
            first, second, third = prizes_text[:3]

            special = block.select_one(".resultbottom .special")
            special = special.get_text(" ", strip=True) if special else ""

            consolation = block.select_one(".resultbottom .consolation")
            consolation = consolation.get_text(" ", strip=True) if consolation else ""

            all_data.append([
                draw_date, provider, draw_info, first, second, third, special, consolation
            ])

        return all_data

    except Exception as e:
        print("❌ Scraping error:", e)
        return []

def update_csv(new_rows):
    if not new_rows:
        print("⚠️ No new data to write.")
        return

    try:
        # ✅ Create folder if not exists
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

        # ✅ Avoid duplicate row (based on draw date + provider)
        existing = set()
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    existing.add((row[0], row[1]))  # (date, provider)

        with open(CSV_PATH, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            new_count = 0
            for row in new_rows:
                if (row[0], row[1]) not in existing:
                    writer.writerow(row)
                    new_count += 1

        print(f"✅ Appended {new_count} new rows to: {CSV_PATH}")

    except Exception as e:
        print("❌ CSV update failed:", e)

if __name__ == "__main__":
    print("🚀 Fetching latest 4D results...")
    data = fetch_results()
    if data:
        update_csv(data)
    else:
        print("❌ No valid data scraped.")
