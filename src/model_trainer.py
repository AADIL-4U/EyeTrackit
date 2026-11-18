import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils import logger, get_project_root, set_seed

# ==========================================
# RIDGE REGRESSION
# ==========================================
def train_ridge(X, y, cv=5):
    logger.info("Training Ridge Regression model...")
    alphas = [0.01, 0.1, 1, 10, 100, 1000]
    model = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error')
    model.fit(X, y)
    
    best_alpha = model.alpha_
    logger.info(f"Ridge best alpha: {best_alpha}")
    
    # Calculate training RMSE in original space
    preds = model.predict(X)
    mse_orig = mean_squared_error(np.expm1(y), np.expm1(preds))
    rmse_orig = np.sqrt(mse_orig)
    logger.info(f"Ridge training RMSE (ms): {rmse_orig:.2f}")
    
    save_path = get_project_root() / "models" / "ridge_model.pkl"
    save_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, save_path)
    logger.info(f"Saved Ridge model to {save_path}")
    
    return model

# ==========================================
# LSTM MODEL
# ==========================================
class LSTMDifficultyNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, 
            hidden_size=64, 
            num_layers=2, 
            dropout=0.3, 
            batch_first=True
        )
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        out, (hn, cn) = self.lstm(x)
        # Use the output from the last time step
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # Output shape: (batch, 1)
        return out.squeeze()

def create_sequences(X, y, seq_len=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])  # Predict the target of the last word in seq
    return np.array(X_seq), np.array(y_seq)

def train_lstm(X, y, seq_len=10, epochs=30, lr=0.001):
    logger.info("Preparing data for LSTM...")
    X_seq, y_seq = create_sequences(X, y, seq_len)
    
    # 80/20 train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    
    # To PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = LSTMDifficultyNet(n_features=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / "lstm_best.pt"
    
    logger.info("Training LSTM...")
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_batch_losses.append(loss.item())
                
        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
    logger.info(f"Saved best LSTM model to {best_model_path}")
    
    results_dir = get_project_root() / "results"
    results_dir.mkdir(exist_ok=True)
    pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    }).to_csv(results_dir / "lstm_loss_history.csv", index=False)
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    from feature_engineer import extract_features
    # Load data
    data_path = get_project_root() / "data" / "processed" / "dataset.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        X_scaled, y, _ = extract_features(df)
        
        set_seed(42)
        train_ridge(X_scaled, y)
        train_lstm(X_scaled, y)
    else:
        logger.error("dataset.csv not found for training.")
