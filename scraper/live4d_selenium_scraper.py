import os
import time
import csv
from datetime import date, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

CSV_FILE = "data/4d_results_history.csv"

# Create the CSV file if it does not exist
def ensure_csv():
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "date", "provider", "game_type", "draw_number", "draw_info",
                "main_prizes", "special", "consolation", "jackpot_bonus", "extra"
            ])

# Extract draw date, draw number, extra info from result block
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
        if any(word in line for word in ["Jackpot", "Bonus", "Grand Prize", "Prize :", "Power Toto", "Supreme Toto", "Star Toto"]):
            extra += line + " "
    return draw_date, draw_no, extra.strip()

# Scrape one day's result from the site
def scrape_one_day(driver, target_date):
    url = "https://www.live4d2u.net/past-results"
    driver.get(url)
    time.sleep(2)

    try:
        input_elem = driver.find_element(By.ID, "datepicker")
        driver.execute_script("arguments[0].value = arguments[1]", input_elem, target_date.strftime("%Y-%m-%d"))
        time.sleep(1)
        driver.find_element(By.ID, "frmPS").submit()
        time.sleep(3)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "outerbox"))
        )
    except Exception as e:
        print(f"Date picker failed: {target_date} -> {e}")
        return []

    provider_blocks = driver.find_elements(By.CLASS_NAME, "outerbox")
    all_rows = []
    for box in provider_blocks:
        try:
            provider_name = "(Unknown)"
            tables = box.find_elements(By.TAG_NAME, "table")
            if tables:
                img_elems = tables[0].find_elements(By.TAG_NAME, "img")
                if img_elems:
                    provider_name = img_elems[0].get_attribute("alt") or img_elems[0].get_attribute("src")
                else:
                    label_elem = tables[0].find_elements(By.CLASS_NAME, "resultm4dlable")
                    if label_elem:
                        provider_name = label_elem[0].text.strip()
            if provider_name == "(Unknown)" or not provider_name.strip():
                try:
                    provider_name = box.find_element(By.TAG_NAME, "h4").text.strip()
                except:
                    try:
                        provider_name = box.find_element(By.TAG_NAME, "strong").text.strip()
                    except:
                        provider_name = box.text.split("\n")[0].strip()

            provider_name = provider_name.replace("logo_", "").replace(".gif", "").replace(".png", "").strip()

            draw_date, draw_no, extra_info = extract_draw_info(box)

            block_text = box.text
            type_keys = ["4D", "5D", "6D", "Lotto", "Jackpot", "Life", "3D"]
            game_type = " ".join([key for key in type_keys if key in block_text]).strip() or provider_name

            main_prizes = " | ".join([line.replace("\t", " ").strip() for line in block_text.splitlines() if any(k in line for k in ["1st", "2nd", "3rd", "Bonus"])])

            special, consolation, jackpot_bonus, extra = "", "", "", ""
            in_special = in_conso = False
            for line in block_text.splitlines():
                if "Special" in line:
                    in_special = True
                    continue
                if any(k in line for k in ["Consolation", "Jackpot", "Grand Prize", "Prize :"]):
                    in_special = False
                if in_special:
                    special += line.replace("\t", " ").strip() + " "

                if "Consolation" in line:
                    in_conso = True
                    continue
                if any(k in line for k in ["Jackpot", "Grand Prize", "Prize :"]):
                    in_conso = False
                if in_conso:
                    consolation += line.replace("\t", " ").strip() + " "

                if any(k in line for k in ["Jackpot", "Grand Prize", "Bonus", "Prize :", "Partially Won"]):
                    jackpot_bonus += line.replace("\t", " ").strip() + " | "

                if any(s in line for s in ["Star Toto", "Power Toto", "Supreme Toto"]):
                    extra += line.strip() + " | "

            if any([main_prizes, special, consolation, jackpot_bonus]):
                all_rows.append([
                    target_date.strftime("%Y-%m-%d"),
                    provider_name,
                    game_type,
                    draw_no,
                    draw_date,
                    main_prizes.strip(),
                    special.strip(),
                    consolation.strip(),
                    jackpot_bonus.strip(),
                    (extra_info + " " + extra).strip()
                ])
        except Exception as e:
            print("Box parse error:", e)
    return all_rows

# Main runner
def main():
    ensure_csv()
    chrome_options = Options()
    chrome_options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(options=chrome_options)

    start_date = date(2025, 7, 17)
    end_date = date.today()
    delta = timedelta(days=1)
    current_date = start_date

    with open(CSV_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        while current_date <= end_date:
            if current_date.weekday() in [2, 5, 6]:
                try:
                    print(f"Scraping {current_date}...")
                    rows = scrape_one_day(driver, current_date)
                    for r in rows:
                        writer.writerow(r)
                        print(f"  {r[1]} | {r[2]} | {r[3]} | {r[4][:50]}...")
                except Exception as e:
                    print(f"Error on {current_date}: {e}")
            else:
                print(f"Skipping {current_date} (not draw day)")
            current_date += delta
    driver.quit()

if __name__ == "__main__":
    main()
