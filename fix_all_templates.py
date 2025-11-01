"""
Fix all templates that display prize data - remove fallback to wrong columns
"""
import os
import re

templates_dir = 'templates'
templates_to_check = [
    'index.html',
    'past_results.html', 
    'results.html',
    'history.html',
    'smart_history.html',
    'results_viewer.html',
    'results_new.html',
    'index_new.html',
    'index_cards.html'
]

# Patterns to find and fix
patterns = [
    (r"card\.get\('1st_real',\s*card\.get\('1st',\s*''\)\)", "card.get('1st_real', '')"),
    (r"card\.get\('2nd_real',\s*card\.get\('2nd',\s*''\)\)", "card.get('2nd_real', '')"),
    (r"card\.get\('3rd_real',\s*card\.get\('3rd',\s*''\)\)", "card.get('3rd_real', '')"),
]

print("=" * 80)
print("FIXING ALL TEMPLATES - REMOVE FALLBACK TO WRONG DATA")
print("=" * 80)

fixed_count = 0
checked_count = 0

for template in templates_to_check:
    filepath = os.path.join(templates_dir, template)
    
    if not os.path.exists(filepath):
        print(f"\n[SKIP] {template} - File not found")
        continue
    
    checked_count += 1
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    for pattern, replacement in patterns:
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            changes_made.append(f"  - Fixed {len(matches)} occurrence(s) of {pattern[:30]}...")
    
    if content != original_content:
        # Backup original
        backup_path = filepath + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Write fixed version
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n[FIXED] {template}")
        for change in changes_made:
            print(change)
        print(f"  Backup: {backup_path}")
        fixed_count += 1
    else:
        print(f"\n[OK] {template} - No issues found")

print("\n" + "=" * 80)
print(f"SUMMARY: Checked {checked_count} templates, fixed {fixed_count} templates")
print("=" * 80)
