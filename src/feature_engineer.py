import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from utils import logger, get_project_root

import re

def count_syllables(word):
    word = str(word).lower()
    if not word:
        return 1
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and not word.endswith('le'):
        count -= 1
    return max(1, count)

def extract_features(df: pd.DataFrame):
    """Engineers features and returns X, y, and feature_names."""
    logger.info("Starting feature engineering...")
    df = df.copy()
    
    # Target variable (y)
    y = np.log1p(df['total_fixation_ms'].values)
    
    # Group A - Raw eye-movement
    df['first_fixation_ms_log'] = np.log1p(df['first_fixation_ms'])
    df['first_pass_ms_log'] = np.log1p(df['first_pass_ms'])
    
    # Group B - Word-level linguistic
    if 'syllable_count' not in df.columns:
        logger.info("syllable_count missing from data, calculating estimates...")
        df['syllable_count'] = df['word'].apply(count_syllables)
        
    df['word_length_sq'] = df['word_length'] ** 2
    df['char_per_syllable'] = df['word_length'] / (df['syllable_count'] + 1e-5)
    df['is_long_word'] = (df['word_length'] > 8).astype(int)
    
    # Group C - Contextual / positional
    # Normalize word_number per text_id as word_position
    if 'text_id' in df.columns and 'word_id' in df.columns:
        df['word_position'] = df.groupby('text_id')['word_id'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-5))
        df['is_sentence_initial'] = (df['word_id'] == 1).astype(int) # Using word_id as Word_Number
    else:
        logger.warning("text_id or word_id missing, using dummy values for position features")
        df['word_position'] = 0.0
        df['is_sentence_initial'] = 0
    
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
    
    X = df[feature_cols].copy()
    
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    logger.info("Fitting preprocessor...")
    X_scaled = preprocessor.fit_transform(X)
    
    root = get_project_root()
    features_dir = root / "features"
    models_dir = root / "models"
    
    # Save feature names
    features_dir.mkdir(exist_ok=True)
    with open(features_dir / "feature_names.txt", "w") as f:
        for col in feature_cols:
            f.write(col + "\n")
            
    # Save preprocessor
    models_dir.mkdir(exist_ok=True)
    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
    logger.info("Saved preprocessor to models/preprocessor.pkl")
    
    return X_scaled, y, feature_cols

if __name__ == "__main__":
    root = get_project_root()
    data_path = root / "data" / "processed" / "dataset.csv"
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}. Run data_loader.py first.")
    else:
        df = pd.read_csv(data_path)
        X_scaled, y, feature_names = extract_features(df)
        logger.info(f"Feature engineering complete. X shape: {X_scaled.shape}, y shape: {y.shape}")
