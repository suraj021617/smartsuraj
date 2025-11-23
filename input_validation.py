"""
Input validation utilities
"""
import re
from datetime import datetime

def validate_provider(provider):
    """Validate provider input"""
    if not provider:
        return 'all'
    provider = str(provider).strip().lower()
    allowed = ['all', 'magnum', 'damacai', 'toto', 'singapore', 'sandakan', 'cashsweep', 'sabah88', 'gdlotto', 'perdana', 'harihari']
    return provider if provider in allowed else 'all'

def validate_date(date_str):
    """Validate date string (YYYY-MM-DD)"""
    if not date_str:
        return None
    try:
        datetime.strptime(str(date_str), '%Y-%m-%d')
        return str(date_str)
    except:
        return None

def validate_month(month_str):
    """Validate month string (YYYY-MM)"""
    if not month_str:
        return None
    try:
        datetime.strptime(str(month_str), '%Y-%m')
        return str(month_str)
    except:
        return None

def validate_number(number_str):
    """Validate 4D number"""
    if not number_str:
        return None
    number_str = str(number_str).strip()
    if re.match(r'^\d{4}$', number_str):
        return number_str
    return None

def validate_lookback(lookback):
    """Validate lookback days"""
    try:
        val = int(lookback)
        return max(10, min(val, 1000))
    except:
        return 100

def sanitize_string(text, max_length=100):
    """Sanitize string input"""
    if not text:
        return ''
    text = str(text).strip()[:max_length]
    return re.sub(r'[<>\"\'&]', '', text)
