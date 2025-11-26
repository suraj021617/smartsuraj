from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

print("Opening Chrome...")
print("Show me what to click and I'll record it.\n")

chrome_options = Options()

try:
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
except:
    driver = webdriver.Chrome(options=chrome_options)

driver.get("https://www.live4d2u.net/past-results")

print("Browser opened. Show me the steps...")
input("\nPress ENTER when done...")

driver.quit()
print("Done!")
