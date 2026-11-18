import os
import json
import logging
import pathlib
import numpy as np
import torch

def get_project_root() -> pathlib.Path:
    """Returns project root folder."""
    return pathlib.Path(__file__).parent.parent

def setup_logger():
    """Sets up a logger to write to results/run.log and console."""
    log_dir = get_project_root() / "results"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "run.log"
    
    logger = logging.getLogger("reading_difficulty")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()

def set_seed(seed=42):
    """Sets seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def save_json(data, path):
    """Saves dictionary to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved data to {path}")

def load_json(path):
    """Loads dictionary from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded data from {path}")
    return data
