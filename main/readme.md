## Smart Product Pricing Challenge - Project Guide

### Overview
End-to-end solution for ML Challenge 2025. Backend provides training and prediction APIs; frontend offers simple UI; Streamlit app for resume relevance is separate and optional.

### Compliance Checklist
- Output format: sample_id,price CSV (see /predict and test_out.csv).
- Positive prices enforced (min 0.01).
- No external price lookups; only provided data used.
- Model well under 8B params; libraries under permissive licenses.
- 1–2 page methodology provided in backend/HACKATHON_SOLUTION.md.

### Prerequisites
- Python 3.10+
- Node not required (static frontend).
- Optional: Docker & Docker Compose.

### Local (no Docker)
1) Backend API
cd main/backend
pip install -r requirements.txt
uvicorn fastapi_app:app --reload --port 8000
2) Frontend
python main/frontend/run_server.py
Open http://127.0.0.1:3000

### Docker Compose
cd main
docker compose up --build
- Backend: http://127.0.0.1:8000
- Frontend: http://127.0.0.1:3000

### Training and Prediction (Organizer format)
1) Train with full training set (75k)
- Via UI: upload main/data/train.csv in the frontend and click Train.
- Via API:
curl -F "file=@main/data/train.csv" http://127.0.0.1:8000/train
Check trained_samples and metrics.

2) Predict on test set (75k)
- Via UI: upload main/data/test.csv, click Generate Predictions, then Download Results.
- Via API:
curl -F "file=@main/data/test.csv" http://127.0.0.1:8000/predict
curl -O http://127.0.0.1:8000/download
The file test_out.csv matches sample_test_out.csv format.

3) Metrics and Status
curl http://127.0.0.1:8000/stats
Returns model_loaded, trained_samples, and last_metrics (includes SMAPE).

### Improving SMAPE (optional)
- Enrich IPQ/unit parsing; add brand/size/category features.
- Add simple image-derived features (if images downloaded locally).
- Tune hyperparameters (estimators, depth, learning rate) with CV.

### Files
- Backend API: main/backend/fastapi_app.py
- ML Pipeline: main/backend/ml_model.py
- Utilities: main/backend/utils.py
- Frontend: main/frontend/index.html
- Compose: main/docker-compose.yml
- Docs: main/backend/HACKATHON_SOLUTION.md 
