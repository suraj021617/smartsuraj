import requests
from bs4 import BeautifulSoup

try:
    response = requests.get('http://127.0.0.1:5000/', timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Content length: {len(response.text)} chars")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all provider names
    providers = []
    for text in soup.stripped_strings:
        if any(p in text for p in ['CashSweep', 'Magnum', 'Damacai', 'Harihari', 'Perdana', 'Sabah', 'Sandakan', 'Singapore', 'Dragon']):
            providers.append(text)
    
    print("\nProviders found in HTML:")
    for p in set(providers):
        print(f"  - {p}")
    
    # Find all 2nd Prize values
    second_prizes = soup.find_all(string=lambda text: '2nd Prize' in str(text) if text else False)
    print(f"\n2nd Prize mentions: {len(second_prizes)}")
    
    # Check for "2025" in 2nd prize context
    has_2025 = '2025' in response.text
    print(f"Contains '2025': {has_2025}")
    
    # Save HTML for inspection
    with open('live_response.html', 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("\nSaved response to: live_response.html")
    
except requests.exceptions.ConnectionError:
    print("ERROR: Flask is not running on http://127.0.0.1:5000/")
    print("Please start Flask with: python app.py")
except Exception as e:
    print(f"ERROR: {e}")
