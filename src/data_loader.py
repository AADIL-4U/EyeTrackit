import os
import requests
import pandas as pd
from pathlib import Path
from utils import logger, get_project_root

def download_provo(save_path: str):
    """Downloads the Provo Corpus from OSF if not already present."""
    if os.path.exists(save_path):
        logger.info(f"Provo Corpus already exists at {save_path}")
        return
        
    logger.info("Fetching OSF file list for project sjefs...")
    api_url = "https://api.osf.io/v2/nodes/sjefs/files/osfstorage/"
    r = requests.get(api_url)
    r.raise_for_status()
    
    download_url = None
    for f in r.json().get('data', []):
        if f['attributes']['name'] == 'Provo_Corpus-Eyetracking_Data.csv':
            download_url = f['links']['download']
            break
            
    if not download_url:
        logger.error("Could not find Provo CSV file on OSF.")
        return
        
    logger.info(f"Downloading Provo Corpus from {download_url}...")
    response = requests.get(download_url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info("Download complete.")

def load_provo(path: str) -> pd.DataFrame:
    """Loads and cleans the Provo Corpus."""
    logger.info(f"Loading Provo data from {path}")
    df = pd.read_csv(path)
    
    keep_cols = [
        'Text_ID', 'Word_Number', 'Word', 'IA_FIRST_FIXATION_DURATION',
        'IA_DWELL_TIME', 'IA_REGRESSION_IN_COUNT', 'IA_SKIP',
        'IA_FIRST_RUN_DWELL_TIME', 'IA_FIXATION_COUNT', 'Syllable_Count',
        'Word_Length'
    ]
    if 'surprisal' in df.columns:
        keep_cols.append('surprisal')
        
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()
    
    rename_mapping = {
        'Text_ID': 'text_id',
        'Word_Number': 'word_id',
        'Word': 'word',
        'IA_FIRST_FIXATION_DURATION': 'first_fixation_ms',
        'IA_DWELL_TIME': 'total_fixation_ms',
        'IA_REGRESSION_IN_COUNT': 'regression_count',
        'IA_SKIP': 'skip_flag',
        'IA_FIRST_RUN_DWELL_TIME': 'first_pass_ms',
        'IA_FIXATION_COUNT': 'fixation_count',
        'Syllable_Count': 'syllable_count',
        'Word_Length': 'word_length'
    }
    df = df.rename(columns=rename_mapping)
    
    # Drop rows where first_fixation_ms is NaN
    before_len = len(df)
    df = df.dropna(subset=['first_fixation_ms'])
    after_len = len(df)
    logger.info(f"Dropped {before_len - after_len} rows due to NaN in first_fixation_ms.")
    
    df = df.reset_index(drop=True)
    
    logger.info(f"Final shape: {df.shape}")
    logger.info("Column dtypes:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")
        
    return df

if __name__ == "__main__":
    root = get_project_root()
    provo_path = root / "data" / "raw" / "provo" / "Provo_Corpus-Eyetracking_Data.csv"
    processed_path = root / "data" / "processed" / "dataset.csv"
    
    download_provo(str(provo_path))
    
    df_cleaned = load_provo(str(provo_path))
    
    # Ensure processed directory exists
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(processed_path, index=False)
    logger.info(f"Processed dataset saved to {processed_path}")
