"""
Configuration Management for 4D Prediction System
"""
import os
from datetime import datetime

class Config:
    """Base configuration"""
    
    # Application Settings
    DEBUG = True
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    SCRAPER_DIR = os.path.join(BASE_DIR, 'scraper')
    TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    
    # CSV Files
    RESULTS_CSV = '4d_results_history.csv'
    CLEANED_RESULTS_CSV = '4d_results_history_cleaned.csv'
    TRAINING_DATA_CSV = os.path.join(DATA_DIR, '4d_training_ready.csv')
    PREDICTION_TRACKING_CSV = 'prediction_tracking.csv'
    SMART_HISTORY_CSV = 'smart_history.csv'
    
    # Model Files
    XGBOOST_MODEL = os.path.join(MODELS_DIR, '4d_xgboost_model.joblib')
    LABEL_ENCODER = os.path.join(MODELS_DIR, 'label_encoder.pkl')
    XGB_MODEL = os.path.join(MODELS_DIR, 'xgb_model.pkl')
    
    # Scraper Settings
    SCRAPER_URL = 'https://www.live4d2u.net/past-results'
    SCRAPER_WAIT_TIME = 15  # seconds
    SCRAPER_RETRY_ATTEMPTS = 3
    SCRAPER_DELAY_BETWEEN_DATES = 2  # seconds
    
    # Date Range for Scraping
    DEFAULT_START_DATE = '2025-08-06'
    DEFAULT_END_DATE = '2025-09-17'
    
    # Cache Settings
    CSV_CACHE_TIMEOUT = 300  # 5 minutes in seconds
    
    # Prediction Settings
    LOOKBACK_DAYS = 200
    TOP_PREDICTIONS = 5
    
    # Providers
    PROVIDERS = {
        'Magnum 4D': 'magnum',
        'Da Ma Cai 1+3D': 'damacai',
        'SportsToto 4D': 'toto',
        'Sandakan 4D': 'sandakan',
        'Special CashSweep': 'cashsweep',
        'Grand Dragon 4D': 'granddragon',
        'Perdana Lottery 4D': 'perdana',
        'Lucky HariHari': 'harihari'
    }
    
    # Prediction Modes
    PREDICTION_MODES = {
        'pattern': 'Pattern-based prediction',
        'frequency': 'Frequency-based prediction',
        'combined': 'Combined multi-method prediction',
        'ml': 'Machine Learning prediction',
        'smart': 'Smart auto-weight prediction'
