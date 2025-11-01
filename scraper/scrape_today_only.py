import time
import csv
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Output CSV path
CSV_FILE = "data/4d_results_history.csv"

# Create CSV if not exists
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "provider", "game_type", "draw_number", "draw_info",
            "main_prizes", "special", "consolation", "jackpot_bonus", "extra"
        ])

# Scroll to bottom slowly
def scroll_to_bottom(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollBy(0, 600);")
        time.sleep(0.8)
        new_height = driver.execute_script("return window.scrollY + window.innerHeight")
        if new_height >= last_height:
            break
        last_height = new_height

# Extract draw date, draw number, extra info
def extract_draw_info(box):
    lines = box.text.split("\n")
    draw_date, draw_no, extra = "", "", ""
    for line in lines:
        if "Date:" in line:
            draw_date = line.split("Date:")[1].strip()
        if "Draw No" in line:
            draw_no = line.split("Draw No:")[1].strip()
        if any(word in line for word in ["Zodiac", "Tiger", "Rooster", "Rabbit"]):
            extra += line + " "
    return draw_date, draw_no, extra.strip()

# Main scraper for one day
def scrape_one_day(driver, target_date):
    driver.get("https://www.live4d2u.net/past-results")
    time.sleep(2)

    # Pick the date using JavaScript
    date_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "datepicker"))
    )
    date_str = target_date.strftime("%Y-%m-%d")
    driver.execute_script("arguments[0].value = arguments[1];", date_input, date_str)
    time.sleep(1)
    driver.find_element(By.ID, "frmPS").submit()
    time.sleep(3)

    scroll_to_bottom(driver)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "outerbox"))
    )

    provider_blocks = driver.find_elements(By.CLASS_NAME, "outerbox")
    rows = []
    for box in provider_blocks:
        try:
            provider = "(Unknown)"
            img = box.find_element(By.TAG_NAME, "img")
            provider = img.get_attribute("alt") or img.get_attribute("src")

            draw_date, draw_no, extra = extract_draw_info(box)
            block_text = box.text

            game_type = "4D"
            for tag in ["5D", "6D", "Jackpot", "Life", "Lotto"]:
                if tag in block_text:
                    game_type = tag
                    break

            main = " | ".join([line for line in block_text.splitlines() if any(x in line for x in ["1st", "2nd", "3rd"])])
            special = " ".join([line for line in block_text.splitlines() if "Special" in line])
            conso = " ".join([line for line in block_text.splitlines() if "Consolation" in line])
            jackpot = " | ".join([line for line in block_text.splitlines() if any(k in line for k in ["Jackpot", "Bonus"])])

            rows.append([
                date_str, provider, game_type, draw_no, draw_date,
                main.strip(), special.strip(), conso.strip(),
                jackpot.strip(), extra.strip()
            ])
        except Exception as e:
            print("Error:", e)
    return rows

# Run
def main():
    options = Options()
    options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(options=options)

    target_date = datetime.strptime("2025-07-29", "%Y-%m-%d")
    print("📅 Scraping:", target_date.strftime("%Y-%m-%d"))

    data = scrape_one_day(driver, target_date)

    if data:
        with open(CSV_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)
                print("✅", row[1], "|", row[2], "|", row[3])
    else:
        print("⚠️ No results found")

    input("✅ Done scraping. Press Enter to close...")
    driver.quit()

if __name__ == "__main__":
    main()
