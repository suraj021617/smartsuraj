import os
import time
import csv
from datetime import date, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

CSV_FILE = "4d_results_history.csv"

def ensure_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "date", "provider", "game_type", "draw_number", "draw_info",
                "main_prizes", "special", "consolation", "jackpot_bonus", "extra"
            ])

def extract_draw_info(box):
    """Get draw date, number and extra info from the block."""
    lines = box.text.split("\n")
    draw_date, draw_no, extra = "", "", ""
    for line in lines:
        if "Date:" in line:
            draw_date = line.split("Date:")[1].strip()
        if "Draw No" in line:
            draw_no = line.split("Draw No:")[1].strip()
        if "Zodiac" in line or "Tiger" in line or "Rooster" in line or "Rabbit" in line:
            extra += line + " "
        if "Jackpot" in line or "Bonus" in line or "Grand Prize" in line or "Prize :" in line or "Power Toto" in line or "Supreme Toto" in line or "Star Toto" in line:
            extra += line + " "
    return draw_date, draw_no, extra.strip()

def scrape_one_day(driver, target_date):
    url = "https://www.live4d2u.net/past-results"
    driver.get(url)
    time.sleep(2)

    # Set the datepicker value (date format: yyyy-mm-dd)
    input_elem = driver.find_element(By.ID, "datepicker")
    driver.execute_script("arguments[0].value = arguments[1]", input_elem, target_date.strftime("%Y-%m-%d"))
    time.sleep(0.5)
    driver.find_element(By.ID, "frmPS").submit()
    time.sleep(2)

    # Wait for result sections to appear
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "outerbox"))
    )

    provider_blocks = driver.find_elements(By.CLASS_NAME, "outerbox")
    all_rows = []
    for box in provider_blocks:
        try:
            # PROVIDER NAME
            provider_name = "(Unknown)"
            tables = box.find_elements(By.TAG_NAME, "table")
            if tables and len(tables) > 0:
                img_elems = tables[0].find_elements(By.TAG_NAME, "img")
                if img_elems:
                    provider_name = img_elems[0].get_attribute("alt") or img_elems[0].get_attribute("src")
                else:
                    # fallback, provider might be text in first row or h4
                    label_elem = tables[0].find_elements(By.CLASS_NAME, "resultm4dlable")
                    if label_elem:
                        provider_name = label_elem[0].text.strip()
            if provider_name == "(Unknown)" or not provider_name.strip():
                # fallback, first strong or h4 inside box
                try:
                    provider_name = box.find_element(By.TAG_NAME, "h4").text.strip()
                except:
                    try:
                        provider_name = box.find_element(By.TAG_NAME, "strong").text.strip()
                    except:
                        provider_name = box.text.split("\n")[0].strip()

            provider_name = provider_name.replace("logo_", "").replace(".gif", "").replace(".png", "").strip()

            # DRAW INFO (date, draw number, extra tags like Zodiac, Bonus, etc)
            draw_date, draw_no, extra_info = extract_draw_info(box)

            # GAME TYPE (try to extract "4D", "6D", "Life", "Jackpot", etc from visible block)
            block_text = box.text
            type_keys = ["4D", "5D", "6D", "Lotto", "Jackpot", "Life", "3D"]
            game_type = ""
            for key in type_keys:
                if key in block_text:
                    game_type += key + " "
            game_type = game_type.strip() or provider_name

            # MAIN PRIZES (all lines with 1st/2nd/3rd/bonus)
            main_prizes = ""
            for line in block_text.splitlines():
                if "1st" in line or "2nd" in line or "3rd" in line or "Bonus" in line:
                    main_prizes += line.replace("\t", " ").strip() + " | "

            # SPECIAL PRIZE
            special = ""
            in_special = False
            for line in block_text.splitlines():
                if "Special" in line:
                    in_special = True
                    continue
                if "Consolation" in line or "Jackpot" in line or "Grand Prize" in line or "Prize :" in line:
                    in_special = False
                if in_special:
                    special += line.replace("\t", " ").strip() + " "
            special = special.strip()

            # CONSOLATION PRIZE
            consolation = ""
            in_conso = False
            for line in block_text.splitlines():
                if "Consolation" in line:
                    in_conso = True
                    continue
                if "Jackpot" in line or "Grand Prize" in line or "Prize :" in line:
                    in_conso = False
                if in_conso:
                    consolation += line.replace("\t", " ").strip() + " "
            consolation = consolation.strip()

            # JACKPOT / BONUS
            jackpot_bonus = ""
            for line in block_text.splitlines():
                if "Jackpot" in line or "Grand Prize" in line or "Bonus" in line or "Prize :" in line or "Partially Won" in line:
                    jackpot_bonus += line.replace("\t", " ").strip() + " | "

            # EXTRA (any leftover lines with a lot of numbers, power/supreme/star lotto, etc)
            extra = ""
            for line in block_text.splitlines():
                if any(s in line for s in ["Star Toto", "Power Toto", "Supreme Toto"]):
                    extra += line.strip() + " | "

            # Skip blocks with no meaningful info
            if any([main_prizes, special, consolation, jackpot_bonus]):
                row = [
                    target_date.strftime("%Y-%m-%d"),
                    provider_name,
                    game_type,
                    draw_no,
                    draw_date,
                    main_prizes,
                    special,
                    consolation,
                    jackpot_bonus,
                    (extra_info + " " + extra).strip()
                ]
                all_rows.append(row)
        except Exception as e:
            print("Box parse error:", e)
    return all_rows

def main():
    ensure_csv()
    chrome_options = Options()
    chrome_options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(options=chrome_options)

    start_date = date(2015, 1, 1)
    end_date = date.today()
    delta = timedelta(days=1)
    current_date = start_date

    with open(CSV_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        while current_date <= end_date:
            # Only scrape Wed/Sat/Sun (or remove this filter to scrape all days)
            if current_date.weekday() in [2, 5, 6]:
                try:
                    print(f"Scraping {current_date}...")
                    rows = scrape_one_day(driver, current_date)
                    for r in rows:
                        writer.writerow(r)
                        print("  ", r[1], "|", r[2], "|", r[3], "|", r[4][:50], "...")
                except Exception as e:
                    print(f"Error on {current_date}: {e}")
            else:
                print(f"Skipping {current_date} (not draw day)")
            current_date += delta
    driver.quit()

if __name__ == "__main__":
    main()
