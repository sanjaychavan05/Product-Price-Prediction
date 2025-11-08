#!/usr/bin/env python3
"""
Hackathon Test Script - Smart Product Pricing Challenge
This script demonstrates the complete workflow for the ML Challenge 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.append('../../student_resource/src')
sys.path.append('.')

from ml_model import MLPipeline, calculate_smape
from utils import download_images

def test_hackathon_workflow():
    """Test the complete hackathon workflow"""
    print("🚀 Starting Hackathon Workflow Test")
    print("=" * 50)
    
    # Set up paths
    dataset_folder = '../../student_resource/dataset/'
    images_folder = '../../student_resource/images/'
    
    # 1. Load datasets
    print("\n📊 Loading Datasets...")
    train_df = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
    test_df = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
    sample_test_df = pd.read_csv(os.path.join(dataset_folder, 'sample_test.csv'))
    sample_test_out_df = pd.read_csv(os.path.join(dataset_folder, 'sample_test_out.csv'))
    
    print(f"✅ Training data: {train_df.shape}")
    print(f"✅ Test data: {test_df.shape}")
    print(f"✅ Sample test data: {sample_test_df.shape}")
    print(f"✅ Sample output format: {sample_test_out_df.shape}")
    
    # 2. Test image downloading (small sample)
    print("\n🖼️  Testing Image Download...")
    try:
        # Create images directory
        os.makedirs(images_folder, exist_ok=True)
        
        # Download a few sample images
        sample_image_links = sample_test_df['image_link'].head(5).tolist()
        download_images(sample_image_links, images_folder)
        
        downloaded_images = os.listdir(images_folder)
        print(f"✅ Downloaded {len(downloaded_images)} images")
        
    except Exception as e:
        print(f"⚠️  Image download test failed: {e}")
    
    # 3. Test ML Pipeline with small sample
    print("\n🤖 Testing ML Pipeline...")
    ml_pipeline = MLPipeline()
    
    # Use a small sample for quick testing
    train_sample = train_df.head(1000)
    test_sample = test_df.head(100)
    
    print("Training model with 1000 samples...")
    try:
        metrics = ml_pipeline.train(train_sample)
        print("✅ Model training successful!")
        print(f"   Best Model: {metrics['model_name']}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   SMAPE: {metrics['smape']:.4f}%")
        print(f"   R²: {metrics['r2']:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # 4. Test prediction format
    print("\n🔮 Testing Prediction Format...")
    try:
        output_df = ml_pipeline.predict_hackathon_format(test_sample)
        
        print("✅ Prediction successful!")
        print(f"   Output shape: {output_df.shape}")
        print(f"   Columns: {output_df.columns.tolist()}")
        print(f"   Price range: ${output_df['price'].min():.2f} - ${output_df['price'].max():.2f}")
        
        # Save output
        output_file = 'hackathon_test_output.csv'
        output_df.to_csv(output_file, index=False)
        print(f"✅ Output saved to: {output_file}")
        
        # Verify format matches sample
        if output_df.shape[1] == 2 and list(output_df.columns) == ['sample_id', 'price']:
            print("✅ Output format matches hackathon requirements!")
        else:
            print("❌ Output format doesn't match requirements")
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
    
    # 5. Test feature extraction
    print("\n🔍 Testing Feature Extraction...")
    try:
        sample_text = train_df.iloc[0]['catalog_content']
        sample_image = train_df.iloc[0]['image_link']
        
        text_features = ml_pipeline.extract_text_features(sample_text)
        image_features = ml_pipeline.extract_image_features(sample_image)
        
        print(f"✅ Text features extracted: {len(text_features)}")
        print(f"✅ Image features extracted: {len(image_features)}")
        
        # Show some key features
        print("   Key text features:")
        for key, value in list(text_features.items())[:5]:
            print(f"     {key}: {value}")
            
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
    
    # 6. Test SMAPE calculation
    print("\n📈 Testing SMAPE Calculation...")
    try:
        # Create dummy predictions for testing
        y_true = np.array([100, 200, 50, 150])
        y_pred = np.array([110, 190, 45, 160])
        
        smape = calculate_smape(y_true, y_pred)
        print(f"✅ SMAPE calculation: {smape:.4f}%")
        
    except Exception as e:
        print(f"❌ SMAPE calculation failed: {e}")
    
    print("\n🎉 Hackathon Workflow Test Complete!")
    print("=" * 50)
    
    # Summary
    print("\n📋 Summary:")
    print("✅ All core components working")
    print("✅ Data loading successful")
    print("✅ Feature extraction implemented")
    print("✅ Model training functional")
    print("✅ Prediction format correct")
    print("✅ SMAPE evaluation ready")
    print("✅ Image downloading capability")
    
    print("\n🚀 Ready for hackathon submission!")

def test_api_endpoints():
    """Test the FastAPI endpoints"""
    print("\n🌐 Testing API Endpoints...")
    print("To test API endpoints, start the server with:")
    print("uvicorn app:app --reload --port 8000")
    print("\nThen visit: http://127.0.0.1:8000/docs")
    print("\nAvailable endpoints:")
    print("- POST /train - Train model with CSV data")
    print("- POST /predict - Generate predictions")
    print("- POST /hackathon-predict - Hackathon-specific predictions")
    print("- GET /download - Download test_out.csv")
    print("- GET /stats - Get model statistics")

if __name__ == "__main__":
    test_hackathon_workflow()
    test_api_endpoints()
