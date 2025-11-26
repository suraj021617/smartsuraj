import os
import re

templates_to_fix = [
    'lucky_generator.html',
    'frequency_analyzer.html', 
    'hot_cold.html',
    'quick_pick.html'
]

filter_html = '''
    <div class="filters" style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 5px;">
      <form method="get" style="display: flex; gap: 10px; align-items: center;">
        <label>Provider:</label>
        <select name="provider" style="padding: 8px;">
          {% for opt in provider_options %}
          <option value="{{ opt }}" {% if opt == provider %}selected{% endif %}>{{ opt|upper }}</option>
          {% endfor %}
        </select>
        
        <label>Month:</label>
        <select name="month" style="padding: 8px;">
          <option value="">All Months</option>
          {% for m in month_options %}
          <option value="{{ m }}" {% if m == selected_month %}selected{% endif %}>{{ m }}</option>
          {% endfor %}
        </select>
        
        <button type="submit" style="padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Apply Filters</button>
      </form>
    </div>
'''

for template in templates_to_fix:
    path = f'templates/{template}'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if filters already exist
        if 'class="filters"' not in content and 'provider_options' not in content:
            # Find first h1 or h2 tag and insert filters after it
            match = re.search(r'(<h[12][^>]*>.*?</h[12]>)', content, re.DOTALL)
            if match:
                insert_pos = match.end()
                new_content = content[:insert_pos] + filter_html + content[insert_pos:]
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Added filters to {template}")
            else:
                print(f"Could not find h1/h2 in {template}")
        else:
            print(f"{template} already has filters")
    else:
        print(f"{template} not found")

print("\nFilter addition complete!")
