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
