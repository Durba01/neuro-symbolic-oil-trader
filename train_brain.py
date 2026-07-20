import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import warnings

# Ignore pandas warnings
warnings.filterwarnings("ignore")

# Import the brain we built earlier!
from deep_learning_model import DeepOilLSTM

def train_neural_network():
    print("🧠 Welcome to Quant School. Initializing Training Sequence...\n")
    
    # 1. Load the Historical Data
    df = pd.read_csv("wti_historical_data.csv")
    
    # Sometimes yfinance adds multi-level headers. Let's clean it up safely.
    if isinstance(df.columns, pd.MultiIndex) or len(df.columns) > 7:
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    else:
        # Standard yfinance output
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Drop missing values
    df = df.dropna()

    # 2. Define Features (Inputs) and Target (Outputs)
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    
    # Target: Did the price go UP (1) or DOWN (0) the next day?
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    targets = df['Target'].dropna().values
    features = features[:-1] # Match lengths
    
    # 3. Scale the Data (Neural Networks need numbers between 0 and 1)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 4. Create sequences (Look back 5 days to predict tomorrow)
    seq_length = 5
    X, y = [], []
    for i in range(len(features_scaled) - seq_length):
        # Grab 5 days of price history
        price_seq = features_scaled[i:i+seq_length]
        
        # Add "Neutral" Dummy Sentiment [0.5, 0.5, 0.5] to match our 8-input architecture
        neutral_sentiment = np.array([[0.5, 0.5, 0.5]] * seq_length)
        combined_seq = np.hstack((price_seq, neutral_sentiment))
        
        X.append(combined_seq)
        y.append(targets[i + seq_length])
        
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # 5. Initialize the Brain and the Optimizer
    model = DeepOilLSTM()
    criterion = nn.BCELoss() # Binary Cross Entropy (Perfect for Up/Down prediction)
    optimizer = optim.Adam(model.parameters(), lr=0.005) # Learning Rate

    epochs = 50 # Let it read the 2-year history 50 times
    print("🚀 Training Started. Watch the 'Loss' go down as the AI gets smarter...\n")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Make predictions
        predictions = model(X)
        
        # Calculate how wrong it was
        loss = criterion(predictions, y)
        
        # Learn from mistakes (Backpropagation)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | AI Error (Loss): {loss.item():.4f}")

    # 6. Save the newly educated brain!
    torch.save(model.state_dict(), "oil_brain_weights.pth")
    print("\n🎓 Training Complete! The AI's knowledge has been saved to 'oil_brain_weights.pth'")

if __name__ == "__main__":
    train_neural_network()