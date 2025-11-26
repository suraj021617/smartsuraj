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
    CLEANED_RESULTS_CSV = '4d_results_history_cleaned.csv' # This will be in DATA_DIR
    TRAINING_DATA_CSV = '4d_training_ready.csv' # This will be in DATA_DIR
    PREDICTION_TRACKING_CSV = 'prediction_tracking.csv' # This will be in DATA_DIR
    SMART_HISTORY_CSV = 'smart_history.csv' # This will be in DATA_DIR
    
    # Model Files
    XGB_MODEL_JOBLIB = os.path.join(MODELS_DIR, '4d_xgboost_model.joblib')
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
    }
    
    # Grid Settings
    GRID_SIZE = 4  # 4x4 grid
    
    # ML Settings
    ML_LOOKBACK = 500
    SMART_LOOKBACK = 300
    ADVANCED_LOOKBACK = 200
    
    # Display Settings
    RECENT_DRAWS_LIMIT = 10
    HISTORY_PAGE_SIZE = 20
    
    # Feature Flags
    ENABLE_AUTO_SCRAPING = False
    ENABLE_ML_PREDICTIONS = True
    ENABLE_CACHING = True
    ENABLE_LOGGING = True
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'app.log'
    
    @classmethod
    def get_data_path(cls, filename):
        """Get full path for a file in the DATA_DIR"""
        return os.path.join(cls.DATA_DIR, filename)
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.SCRAPER_DIR,
            cls.TEMPLATES_DIR,
            cls.STATIC_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check if critical files exist
        if not os.path.exists(cls.get_csv_path(cls.RESULTS_CSV)):
        if not os.path.exists(cls.get_data_path(cls.RESULTS_CSV)):
            errors.append(f"Warning: {cls.RESULTS_CSV} not found")
        
        # Check if directories exist
        if not os.path.exists(cls.DATA_DIR):
            errors.append(f"Warning: {cls.DATA_DIR} directory not found")
        
        if not os.path.exists(cls.MODELS_DIR):
            errors.append(f"Warning: {cls.MODELS_DIR} directory not found")
        
        return errors


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', None)
    
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY must be set in production")


class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    TESTING = True
    CSV_CACHE_TIMEOUT = 0  # Disable caching in tests


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get configuration for specified environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config_map.get(env, config_map['default'])
