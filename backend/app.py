import os
import sys
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import csv

# Add src to sys path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from src.feature_engineer import count_syllables

app = FastAPI(title="Reading Difficulty Live Inference")

# Allow all origins for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessor on startup
ridge_model_path = ROOT_DIR / "models" / "ridge_model.pkl"
preprocessor_path = ROOT_DIR / "models" / "preprocessor.pkl"
live_data_csv = ROOT_DIR / "data" / "processed" / "live_webcam_data.csv"

ridge_model = None
preprocessor = None

feature_cols = [
    'first_fixation_ms_log',
    'first_pass_ms_log',
    'fixation_count',
    'regression_count',
    'skip_flag',
    'word_length',
    'syllable_count',
    'word_length_sq',
    'char_per_syllable',
    'is_long_word',
    'word_position',
    'is_sentence_initial'
]

@app.on_event("startup")
def load_models():
    global ridge_model, preprocessor
    try:
        if ridge_model_path.exists():
            ridge_model = joblib.load(ridge_model_path)
            print("Loaded Ridge Model")
        
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            print("Loaded preprocessor")
    except Exception as e:
        print(f"Error loading models: {e}")

class WordData(BaseModel):
    word: str
    word_id: int
    first_fixation_ms: float
    first_pass_ms: float
    fixation_count: int
    regression_count: int
    skip_flag: int

class PredictRequest(BaseModel):
    session_id: Optional[str] = None
    words: List[WordData]

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "EyeTrackit Inference API",
        "model_loaded": ridge_model is not None
    }

@app.post("/predict")
def predict_difficulty(req: PredictRequest):
    if ridge_model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
        
    session_id = req.session_id or str(uuid.uuid4())
    
    if not req.words:
        return {"session_id": session_id, "results": []}
    
    # 1. Prepare Dataframe with baseline imputation
    df_data = []
    for w in req.words:
        row = w.model_dump()
        
        # New Logic: Impute zeros with a realistic baseline (~200ms) 
        # to match the model's training distribution.
        if row['first_fixation_ms'] <= 0: row['first_fixation_ms'] = 200.0
        if row['first_pass_ms'] <= 0: row['first_pass_ms'] = 200.0
        
        row['text_id'] = session_id
        row['word_length'] = len(w.word)
        row['syllable_count'] = count_syllables(w.word)
        df_data.append(row)
        
    df = pd.DataFrame(df_data)
    save_live_data(df)
    
    # 2. Extract features without refitting
    df['first_fixation_ms_log'] = np.log1p(df['first_fixation_ms'])
    df['first_pass_ms_log'] = np.log1p(df['first_pass_ms'])
    df['word_length_sq'] = df['word_length'] ** 2
    df['char_per_syllable'] = df['word_length'] / (df['syllable_count'] + 1e-5)
    df['is_long_word'] = (df['word_length'] > 8).astype(int)
    
    # Contextual features
    df['word_position'] = df.groupby('text_id')['word_id'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-5))
    df['is_sentence_initial'] = (df['word_id'] == 1).astype(int)
    
    X = df[feature_cols].copy()
    
    # Important: use TRANSFORM instead of fit_transform
    X_scaled = preprocessor.transform(X)
        
    # 3. Predict mapping back
    y_pred_log = ridge_model.predict(X_scaled)
    y_pred_orig = np.expm1(y_pred_log) # Convert back from log1p space
    
    results = []
    for i, w in enumerate(req.words):
        # Initial prediction from the ML model
        predicted_ms = float(y_pred_orig[i])
        
        # New Heuristic Logic: If the word has zero fixation (skip or eye-track fail),
        # provide a linguistic-based fallback to ensure variance and realism.
        if req.words[i].first_fixation_ms <= 0:
            word_len = len(w.word)
            syl_count = count_syllables(w.word)
            # Base 180ms + word length weight + syllable weight
            heuristic_ms = 180.0 + (word_len * 12.0) + (syl_count * 5.0)
            
            # Blend: We favor the heuristic when gaze data is missing
            predicted_ms = max(predicted_ms, heuristic_ms)

        # Final human-range clipping (min 150ms)
        predicted_ms = max(150.0, predicted_ms)
        
        results.append({
            "word_id": w.word_id,
            "word": w.word,
            "predicted_difficulty_ms": predicted_ms
        })
    
    # Debug log for distribution
    if results:
        diffs = [r["predicted_difficulty_ms"] for r in results]
        print(f"Prediction Stats: Min={min(diffs):.1f}ms, Max={max(diffs):.1f}ms, Mean={sum(diffs)/len(diffs):.1f}ms")
        
    return {"session_id": session_id, "results": results}


def save_live_data(df: pd.DataFrame):
    """Appends live data to a separate CSV for later dataset expansion."""
    try:
        live_data_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = live_data_csv.exists()
        df.to_csv(live_data_csv, mode='a', header=not file_exists, index=False)
    except Exception as e:
        print(f"Failed to log live data: {e}")
