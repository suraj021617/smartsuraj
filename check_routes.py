#!/usr/bin/env python3
"""Check if all routes are defined in the Flask app"""

from app import app

# Get all routes
routes = []
for rule in app.url_map.iter_rules():
    routes.append(rule.rule)

# Key routes from dashboard
key_routes = [
    '/dashboard',
    '/match-checker', 
    '/decision-helper',
    '/quick-pick',
    '/ultimate-ai', 
    '/ultimate-predictor',
    '/best-predictions',
    '/prediction-history',
    '/auto-validator',
    '/pattern-analyzer',
    '/frequency-analyzer',
    '/ml-predictor',
    '/missing-number-finder',
    '/smart-predictor',
    '/hot-cold'
]

print("🔍 Checking key routes...")
missing = []
working = []

for route in key_routes:
    if route in routes:
        working.append(route)
        print(f"✅ {route}")
    else:
        missing.append(route)
        print(f"❌ {route}")

print(f"\n📊 Summary:")
print(f"✅ Working: {len(working)}")
print(f"❌ Missing: {len(missing)}")

if missing:
    print(f"\n🔧 Missing routes:")
    for m in missing:
        print(f"   {m}")

print(f"\n📋 All routes in app:")
for route in sorted(routes):
    print(f"   {route}")