import time
import requests

print("Waiting for Flask to start...")
for i in range(10):
    try:
        response = requests.get('http://127.0.0.1:5000/power-simple', timeout=2)
        if response.status_code == 200:
            print(f"✅ SUCCESS! Route works! Status: {response.status_code}")
            print(f"Response length: {len(response.text)} bytes")
            break
        else:
            print(f"❌ Route returned: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"Attempt {i+1}/10: Flask not ready yet...")
        time.sleep(2)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(2)
else:
    print("❌ Flask didn't start. Please run: python app.py")
