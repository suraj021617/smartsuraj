#!/usr/bin/env python3
"""Quick test to verify all routes work"""

import requests
import sys

# List of all routes from the dashboard
routes = [
    '/dashboard',
    '/match-checker', 
    '/decision-helper',
    '/quick-pick',
    '/ultimate-ai', 
    '/ultimate-predictor',
    '/best-predictions',
    '/prediction-history',
    '/auto-validator',
    '/past-results',
    '/positional-ocr',
    '/pattern-analyzer',
    '/accuracy-dashboard',
    '/statistics',
    '/lucky-generator',
    '/day-to-day-predictor',
    '/frequency-analyzer',
    '/missing-number-finder',
    '/empty-box-predictor',
    '/master-analyzer',
    '/smart-auto-weight',
    '/ml-predictor',
    '/consensus-predictor',
    '/learning-dashboard',
    '/smart-predictor',
    '/smart-history',
    '/hot-cold',
    '/theme-gallery'
]

base_url = 'http://localhost:5000'

def test_routes():
    working = []
    broken = []
    
    for route in routes:
        try:
            response = requests.get(f"{base_url}{route}", timeout=5)
            if response.status_code == 200:
                working.append(route)
                print(f"✅ {route}")
            else:
                broken.append(f"{route} - Status: {response.status_code}")
                print(f"❌ {route} - Status: {response.status_code}")
        except Exception as e:
            broken.append(f"{route} - Error: {str(e)}")
            print(f"❌ {route} - Error: {str(e)}")
    
    print(f"\n📊 Results:")
    print(f"✅ Working: {len(working)}")
    print(f"❌ Broken: {len(broken)}")
    
    if broken:
        print(f"\n🔧 Broken routes:")
        for b in broken:
            print(f"   {b}")
    
    return len(broken) == 0

if __name__ == '__main__':
    print("🧪 Testing all dashboard routes...")
    print("Make sure Flask app is running on localhost:5000\n")
    
    all_working = test_routes()
    sys.exit(0 if all_working else 1)