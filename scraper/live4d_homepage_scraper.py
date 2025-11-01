import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os

# Setup Chrome options
options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-infobars")
options.add_argument("--start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)

# Launch browser
driver = webdriver.Chrome(options=options)
driver.get("https://www.live4d2u.net/")

try:
    print("🕒 Waiting for result boxes to load...")
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, "result-box")))
    
    # Scroll slowly to bottom to trigger all JS loads
    scroll_pause_time = 1.5
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(15):
        driver.execute_script("window.scrollBy(0, 500);")
        time.sleep(scroll_pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    time.sleep(5)  # Final wait for any dynamic blocks

    # Remove ads if present
    driver.execute_script("""
        const ads = document.querySelectorAll('iframe, .adsbygoogle, .ad, .advertisement');
        ads.forEach(ad => ad.remove());
    """)

    # Get page source after scrolling
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    result_boxes = soup.select(".result-box")
    print(f"✅ Found {len(result_boxes)} result blocks")

    results = []
    for box in result_boxes:
        provider = box.select_one(".provider-title")
        title = provider.get_text(strip=True) if provider else "Unknown Provider"
        
        date_elem = box.select_one(".drawdate")
        draw_date = date_elem.get_text(strip=True) if date_elem else "Unknown Date"

        prizes = box.select(".result-number")
        numbers = [p.get_text(strip=True) for p in prizes]

        results.append({
            "provider": title,
            "date": draw_date,
            "numbers": " | ".join(numbers)
        })

    # Ensure output folder exists
    os.makedirs("scraper", exist_ok=True)

    # Save to CSV
    csv_file = "scraper/homepage_live_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["provider", "date", "numbers"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Data saved to {csv_file}")

except Exception as e:
    print("❌ Error during scraping:", str(e))
    # Save debug page
    with open("scraper/page_debug_error.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    print("🪪 Saved error HTML to scraper/page_debug_error.html")

finally:
    driver.quit()
