import pandas as pd
import numpy as np
import pickle
import os
import re
import requests
from io import BytesIO
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Import hackathon utilities
import sys
sys.path.append('../../student_resource/src')
from utils import download_images

def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100

class MLPipeline:
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.image_features_cache = {}
        
    def extract_text_features(self, text):
        """Extract comprehensive features from catalog content text"""
        if pd.isna(text):
            return {}
            
        text_str = str(text).lower()
        features = {}
        
        # Basic text features
        features['text_length'] = len(text_str)
        features['word_count'] = len(text_str.split())
        features['char_count'] = len(text_str)
        features['sentence_count'] = text_str.count('.') + text_str.count('!') + text_str.count('?')
        
        # Extract Item Pack Quantity (IPQ) - key feature for pricing
        ipq_patterns = [
            r'pack\s*of\s*(\d+)',
            r'(\d+)\s*(?:pack|pcs|pieces|count|items)',
            r'(\d+)\s*x\s*\d+',
            r'(\d+)\s*pack',
        ]
        
        ipq_found = False
        for pattern in ipq_patterns:
            match = re.search(pattern, text_str)
            if match:
                features['ipq'] = int(match.group(1))
                ipq_found = True
                break
        
        if not ipq_found:
            features['ipq'] = 1  # Default to single item
        
        # Extract weight/volume information
        weight_patterns = [
            r'(\d+\.?\d*)\s*(?:oz|ounce|ounces)',
            r'(\d+\.?\d*)\s*(?:lb|pound|pounds)',
            r'(\d+\.?\d*)\s*(?:kg|kilogram)',
            r'(\d+\.?\d*)\s*(?:g|gram)',
            r'(\d+\.?\d*)\s*(?:ml|milliliter)',
            r'(\d+\.?\d*)\s*(?:l|liter)',
        ]
        
        weight_found = False
        for pattern in weight_patterns:
            match = re.search(pattern, text_str)
            if match:
                features['weight_value'] = float(match.group(1))
                weight_found = True
                break
        
        if not weight_found:
            features['weight_value'] = 0
        
        # Extract numeric patterns
        numbers = re.findall(r'\d+\.?\d*', text_str)
        if numbers:
            numeric_values = [float(n) for n in numbers]
            features['avg_number'] = np.mean(numeric_values)
            features['max_number'] = max(numeric_values)
            features['min_number'] = min(numeric_values)
            features['number_count'] = len(numbers)
            features['number_std'] = np.std(numeric_values)
        else:
            features['avg_number'] = 0
            features['max_number'] = 0
            features['min_number'] = 0
            features['number_count'] = 0
            features['number_std'] = 0
            
        # Brand/quality indicators
        premium_keywords = ['premium', 'organic', 'natural', 'gourmet', 'artisan', 'handmade', 'authentic', 'original']
        features['premium_keywords'] = sum(1 for keyword in premium_keywords if keyword in text_str)
        
        # Product category indicators
        food_keywords = ['food', 'sauce', 'cookie', 'chutney', 'spice', 'organic', 'natural']
        features['food_category'] = sum(1 for keyword in food_keywords if keyword in text_str)
        
        # Size indicators
        size_keywords = ['large', 'small', 'medium', 'jumbo', 'mini', 'family', 'bulk']
        features['size_indicators'] = sum(1 for keyword in size_keywords if keyword in text_str)
        
        # Special characters and formatting
        features['has_bullet_points'] = text_str.count('bullet point') > 0
        features['has_emojis'] = len(re.findall(r'[^\x00-\x7F]', text_str)) > 0
        features['has_caps'] = sum(1 for c in str(text) if c.isupper()) / len(str(text)) if len(str(text)) > 0 else 0
        
        return features
    
    def extract_image_features(self, image_url):
        """Extract features from image URL and downloaded images"""
        features = {}
        
        if pd.isna(image_url) or not image_url:
            features['has_image'] = 0
            features['image_domain'] = 0
            features['image_filename_length'] = 0
        else:
            features['has_image'] = 1
            
            try:
                # Extract domain from URL
                if '://' in image_url:
                    domain = image_url.split('/')[2]
                    features['image_domain'] = hash(domain) % 1000  # Hash encoding
                else:
                    features['image_domain'] = 0
                
                # Extract filename features
                filename = os.path.basename(image_url)
                features['image_filename_length'] = len(filename)
                features['image_extension'] = hash(filename.split('.')[-1] if '.' in filename else 'none') % 100
                
                # Check if image is from Amazon (common in dataset)
                features['is_amazon_image'] = 1 if 'amazon.com' in image_url else 0
                
            except Exception as e:
                features['image_domain'] = 0
                features['image_filename_length'] = 0
                features['image_extension'] = 0
                features['is_amazon_image'] = 0
                
        return features
    
    def prepare_features(self, df):
        """Prepare all features for training/prediction"""
        features_list = []
        
        for idx, row in df.iterrows():
            features = {}
            
            # Text features
            text_features = self.extract_text_features(row.get('catalog_content', ''))
            features.update(text_features)
            
            # Image features
            image_features = self.extract_image_features(row.get('image_link', ''))
            features.update(image_features)
            
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        return features_df
    
    def train(self, df):
        """Train the ML model"""
        print("Starting model training...")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df['price'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Try multiple models and pick the best one
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'CatBoost': CatBoostRegressor(iterations=100, random_seed=42, verbose=False)
        }
        
        best_model = None
        best_score = float('inf')
        best_name = ''
        
        for name, model in models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                
                print(f"{name} MAE: {mae:.4f}")
                
                if mae < best_score:
                    best_score = mae
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if best_model is None:
            raise Exception("No model could be trained successfully")
        
        self.model = best_model
        self.is_trained = True
        
        # Calculate final metrics including SMAPE
        y_pred = best_model.predict(X_test_scaled)
        metrics = {
            'model_name': best_name,
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'smape': calculate_smape(y_test, y_pred),
            'best_score': best_score
        }
        
        print(f"Best model: {best_name} with MAE: {best_score:.4f}")
        
        # Save model
        self.save_model()
        
        return metrics
    
    def predict(self, df):
        """Make predictions on new data"""
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Ensure we have the same features as training
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        # Reorder columns to match training
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_hackathon_format(self, test_df):
        """Generate predictions in exact hackathon format"""
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions")
        
        # Make predictions
        predictions = self.predict(test_df)
        
        # Create output DataFrame in exact hackathon format
        output_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': predictions
        })
        
        # Ensure all prices are positive floats
        output_df['price'] = np.maximum(output_df['price'], 0.01)  # Minimum price of 1 cent
        
        return output_df
    
    def save_model(self):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open('../models/saved_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saved successfully")
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            with open('../models/saved_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
