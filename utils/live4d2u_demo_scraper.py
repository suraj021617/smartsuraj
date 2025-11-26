import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

TARGET_DATE = "2024-07-03"  # <--- change to any yyyy-mm-dd

def close_ads(driver):
    try:
        # Try generic modal close (ads may change)
        close_btns = driver.find_elements(By.CSS_SELECTOR, ".close, .close-btn, .btn-close, .modal-close")
        for btn in close_btns:
            try:
                btn.click()
                print("Closed an ad popup.")
                time.sleep(0.5)
            except Exception:
                pass
    except Exception:
        pass

def scroll_page(driver):
    # Scroll down a few times to make sure all blocks load (mobile view sometimes lazy-loads)
    for _ in range(8):
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)

def main():
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        print("Opening browser...")
        driver.get("https://www.live4d2u.net/past-results")
        print("Page loaded")
        time.sleep(2)

        # Wait & close any immediate popups/ads
        close_ads(driver)

        # Set the date in input box (readonly so need JS)
        driver.execute_script(f"document.getElementById('datepicker').removeAttribute('readonly');")
        date_input = driver.find_element(By.ID, "datepicker")
        date_input.clear()
        date_input.send_keys(TARGET_DATE)
        time.sleep(0.5)

        # Submit form to load results
        driver.find_element(By.ID, "frmPS").submit()
        print(f"Set date to {TARGET_DATE} and submitted form")
        time.sleep(3)
        close_ads(driver)
        scroll_page(driver)
        time.sleep(1)
        close_ads(driver)

        # Wait for results blocks to load
        time.sleep(2)

        print("\n==============================")
        print(f"SCRAPED RESULTS FOR {TARGET_DATE}")
        print("==============================")

        # Scrape each result block (each provider section/outerbox)
        for outer in driver.find_elements(By.CSS_SELECTOR, ".outerbox"):
            # --- Robust Provider Detection ---
            provider = ""
            try:
                provider = outer.find_element(By.CSS_SELECTOR, ".resultm4dlable").text.strip()
            except Exception:
                pass
            if not provider:
                try:
                    provider = outer.find_element(By.XPATH, ".//table[1]//tr[1]/td[2]").text.strip()
                except Exception:
                    pass
            if not provider:
                try:
                    img = outer.find_element(By.TAG_NAME, "img")
                    provider = img.get_attribute("alt") or img.get_attribute("src").split("/")[-1].split(".")[0]
                except Exception:
                    pass
            if not provider:
                provider = "(Unknown)"

            # Draw Info (date, draw no)
            try:
                draw_info = outer.find_element(By.CSS_SELECTOR, ".resultdrawdate").text.strip()
            except Exception:
                draw_info = ""

            print(f"Provider: {provider}")
            print(f"Draw Info: {draw_info}")

            # Main Prizes (1st, 2nd, 3rd)
            main_prizes = {}
            try:
                for row in outer.find_elements(By.XPATH, ".//table[contains(@class,'resultTable2')][3]//tr"):
                    tds = row.find_elements(By.TAG_NAME, "td")
                    if len(tds) == 2:
                        key = tds[0].text.strip()
                        val = tds[1].text.strip()
                        main_prizes[key] = val
            except Exception:
                pass
            if main_prizes:
                print("Main Prizes:")
                for k, v in main_prizes.items():
                    print(f"   {k}: {v}")

            # Special Prizes
            specials = []
            try:
                for sp_table in outer.find_elements(By.XPATH, ".//table[contains(@class,'resultTable2')]"):
                    if sp_table.text.strip().startswith("Special"):
                        for row in sp_table.find_elements(By.TAG_NAME, "tr")[1:]:
                            for td in row.find_elements(By.TAG_NAME, "td"):
                                val = td.text.strip()
                                if val and val != "----":
                                    specials.append(val)
            except Exception:
                pass
            if specials:
                print("Special:", ", ".join(specials))

            # Consolation Prizes
            consolations = []
            try:
                for sp_table in outer.find_elements(By.XPATH, ".//table[contains(@class,'resultTable2')]"):
                    if sp_table.text.strip().startswith("Consolation"):
                        for row in sp_table.find_elements(By.TAG_NAME, "tr")[1:]:
                            for td in row.find_elements(By.TAG_NAME, "td"):
                                val = td.text.strip()
                                if val and val != "----":
                                    consolations.append(val)
            except Exception:
                pass
            if consolations:
                print("Consolation:", ", ".join(consolations))
            print("------------------------")

        print("==============================")
        driver.quit()
    except Exception as ex:
        print("❌ ERROR:", ex)
        driver.quit()

if __name__ == "__main__":
    main()
