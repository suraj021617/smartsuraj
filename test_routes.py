from app import app

print("Registered routes:")
for rule in app.url_map.iter_rules():
    if 'power' in str(rule) or 'adaptive' in str(rule):
        print(f"  {rule}")
