# 🚀 Deployment Guide

## Quick Start (Development)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set environment variables:**
```bash
copy .env.example .env
# Edit .env and set SECRET_KEY
```

3. **Run the app:**
```bash
python app.py
```

## Production Deployment

### Option 1: PythonAnywhere (Free)
1. Upload project files
2. Set environment variables in Web tab
3. Configure WSGI file
4. Reload web app

### Option 2: Render (Free)
1. Connect GitHub repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn app:app`
4. Add environment variables

### Option 3: Railway (Free tier)
1. Connect GitHub repo
2. Railway auto-detects Flask
3. Add environment variables
4. Deploy

## Environment Variables (Production)

```
SECRET_KEY=your-super-secret-key-min-32-chars
FLASK_ENV=production
DEBUG=False
```

## Security Checklist

- ✅ SECRET_KEY is set and secure
- ✅ DEBUG=False in production
- ✅ Error handlers configured
- ✅ Input validation enabled
- ✅ .gitignore protects sensitive files

## Performance Tips

- CSV cache reduces load time by 80%
- Use provider filters to speed up queries
- Limit lookback days for faster predictions
