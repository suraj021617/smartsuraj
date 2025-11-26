from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome()
driver.get("https://www.live4d2u.net/past-results")

try:
    # 1. Find date input
    date_box = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "datepicker"))
    )
    print("✅ Found date box")

    # 2. Set date
    target_date = "2015-07-01"
    driver.execute_script("arguments[0].value = arguments[1];", date_box, target_date)
    print(f"✅ Date set to {target_date}")

    # 3. List all input elements and their values
    input_elements = driver.find_elements(By.TAG_NAME, "input")
    for i, inp in enumerate(input_elements):
        print(f"{i}: type={inp.get_attribute('type')}, value={inp.get_attribute('value')}, id={inp.get_attribute('id')}, class={inp.get_attribute('class')}")

    input("Check browser. Tell me which is the Show Results button (what value/id/class). Then press Enter...")

except Exception as e:
    print(f"❌ Error: {e}")

driver.quit()
# ... (same code as before for date setting)

# Find all buttons on the page
button_elements = driver.find_elements(By.TAG_NAME, "button")
for i, btn in enumerate(button_elements):
    print(f"{i}: text={btn.text}, id={btn.get_attribute('id')}, class={btn.get_attribute('class')}")

input("Check browser. Tell me which is the Show Results button (what text/id/class). Then press Enter...")
# Set the date in the datepicker
driver.find_element(By.ID, "datepicker").clear()
driver.find_element(By.ID, "datepicker").send_keys("2015-07-01")

# Wait for any result table to appear/refreh
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '//div[contains(@class,"panel") and .//h4[contains(text(),"Magnum 4D")]]'))
)

# Now parse all required result sections as needed!
