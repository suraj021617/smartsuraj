# 🎨 Theme System

## Features
✅ **3 Themes**: Dark, Light, Neon
✅ **Auto-Save**: Remembers your choice
✅ **Smooth Animations**: Fade, slide, pulse, glow
✅ **One-Click Switch**: Floating theme toggle (top-right)

## Usage

### Quick Start
1. Run app: `python app.py`
2. Visit: `http://127.0.0.1:5000/theme-demo`
3. Click theme buttons (🌙 ☀️ ⚡) in top-right corner

### Apply to Any Page
Add to your HTML template:
```html
<link href="{{ url_for('static', filename='themes.css') }}" rel="stylesheet"/>
<script src="{{ url_for('static', filename='theme-switcher.js') }}"></script>
```

### Use Theme Variables
```css
background: var(--bg-primary);
color: var(--text-primary);
border: 1px solid var(--border);
```

### Add Animations
```html
<div class="card animate-fade">Content</div>
<button class="btn animate-glow">Click Me</button>
```

## Theme Variables
- `--bg-primary`: Main background
- `--bg-secondary`: Secondary background
- `--bg-card`: Card background
- `--text-primary`: Main text color
- `--text-secondary`: Secondary text color
- `--accent`: Primary accent color
- `--success`: Success color (green)
- `--warning`: Warning color (yellow)
- `--danger`: Danger color (red)
- `--border`: Border color
- `--shadow`: Shadow color

## Animations
- `animate-fade`: Fade in from bottom
- `animate-slide`: Slide in from left
- `animate-pulse`: Subtle pulse effect
- `animate-glow`: Glowing border effect

## Files
- `static/themes.css` - Theme styles
- `static/theme-switcher.js` - Theme logic
- `templates/base_theme.html` - Base template
- `templates/theme_demo.html` - Demo page
