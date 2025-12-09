"""
Machine Learning-based 4D Prediction Engine
Uses XGBoost and statistical analysis for better predictions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    def __init__(self, model_path='models/4d_xgboost_model.joblib', encoder_path='models/label_encoder.pkl'):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.encoder = None
        self.load_model()
