from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from ml_model import MLPipeline
import json, os, asyncio
from typing import List, Dict

app = FastAPI(title="Smart Pricing ML - Hackathon API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost/pricing_ml_db")
engine = create_engine(DATABASE_URL)
ml_pipeline = MLPipeline()

@app.get("/")
async def root():
    return {"message": "Smart Pricing ML API", "status": "running", "docs": "/docs"}

@app.post("/train")
async def train_model(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        metrics = await asyncio.to_thread(ml_pipeline.train, df)
        return {"message": "Model trained successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        output_df = await asyncio.to_thread(ml_pipeline.predict_hackathon_format, df)
        output_df.to_csv('test_out.csv', index=False)
        return {"predictions": output_df['price'].tolist()[:10], "total": len(output_df), "sample_ids": output_df['sample_id'].tolist()[:10]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/download")
async def download_predictions():
    return FileResponse('test_out.csv', filename='test_out.csv')

@app.get("/stats")
async def get_stats():
    # Return mock stats for now since we don't have a database set up
    return {
        "total": 0,
        "avg_price": 0,
        "message": "No predictions yet"
    }

@app.post("/upload-images")
async def upload_images(files: List[UploadFile]):
    saved = []
    for f in files:
        path = f"../data/images/{f.filename}"
        with open(path, "wb") as img:
            img.write(await f.read())
        saved.append(f.filename)
    return {"saved": saved}

@app.get("/leaderboard")
async def leaderboard():
    # Return mock leaderboard for now
    return [
        {
            "model_name": "CatBoost",
            "accuracy": 85.5,
            "created_at": "2025-10-12T23:00:00Z"
        }
    ]

@app.post("/hackathon-predict")
async def hackathon_predict(file: UploadFile):
    """Hackathon-specific prediction endpoint that returns exact format"""
    try:
        df = pd.read_csv(file.file)
        
        # Validate required columns
        required_cols = ['sample_id', 'catalog_content', 'image_link']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(status_code=400, detail=f"Missing required columns. Expected: {required_cols}")
        
        output_df = await asyncio.to_thread(ml_pipeline.predict_hackathon_format, df)
        output_df.to_csv('test_out.csv', index=False)
        
        return {
            "message": "Predictions generated successfully",
            "total_samples": len(output_df),
            "output_file": "test_out.csv",
            "sample_predictions": output_df.head(5).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hackathon prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
