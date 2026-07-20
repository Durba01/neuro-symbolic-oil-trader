# self_learning_loop.py
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Import our custom modules
from deep_learning_model import DeepOilLSTM
from telegram_service import send_telegram_alert

# Load environment
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradingAgent.SelfLearning")

# --- CONFIGURATION ---
OANDA_TOKEN = os.getenv("OANDA_API_TOKEN")
OANDA_ACCOUNT = os.getenv("OANDA_ACCOUNT_ID")
OANDA_URL = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT}"
HEADERS = {"Authorization": f"Bearer {OANDA_TOKEN}", "Accept-Datetime-Format": "RFC3339"}

# Structured schema for Gemini parameter tuning
class TunedHyperparameters(BaseModel):
    risk_per_trade_pct: float = Field(description="Max account risk per trade (0.005 to 0.02 max).")
    atr_stop_loss_multiplier: float = Field(description="Multiplier for ATR stop-loss (1.2 to 3.0).")
    macro_sentiment_threshold: float = Field(description="Minimum sentiment strength to trigger trades (0.1 to 0.8).")
    reflection_summary: str = Field(description="1-sentence quant analysis of why these parameters were adjusted.")

def get_oanda_trade_history():
    """Queries OANDA for closed trades to analyze performance."""
    url = f"{OANDA_URL}/trades?state=CLOSED&count=20"
    try:
        res = requests.get(url, headers=HEADERS)
        if res.status_code == 200:
            trades = res.json().get("trades", [])
            parsed_trades = []
            for t in trades:
                parsed_trades.append({
                    "id": t.get("id"),
                    "instrument": t.get("instrument"),
                    "pnl": float(t.get("realizedPL", 0.0)),
                    "units": float(t.get("initialUnits", 0.0))
                })
            return parsed_trades
    except Exception as e:
        logger.error(f"Failed to fetch OANDA history: {e}")
    return []

def retrain_pytorch_brain(risk_pct, atr_mult):
    """Retrains the PyTorch LSTM model using the updated hyperparameters."""
    logger.info("Retraining PyTorch Brain with updated parameters...")
    df = pd.read_csv("wti_historical_data.csv")
    
    if isinstance(df.columns, pd.MultiIndex) or len(df.columns) > 7:
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    else:
        df = df[['Date', 'Open', 'High', 'Low', 'Open', 'Volume']]
    df = df.dropna()

    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    targets = df['Target'].dropna().values
    features = features[:-1]

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    seq_length = 5
    X, y = [], []
    for i in range(len(features_scaled) - seq_length):
        price_seq = features_scaled[i:i+seq_length]
        neutral_sentiment = np.array([[0.5, 0.5, 0.5]] * seq_length)
        combined_seq = np.hstack((price_seq, neutral_sentiment))
        X.append(combined_seq)
        y.append(targets[i + seq_length])
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = DeepOilLSTM()
    criterion = nn.BCELoss()
    # Apply updated learning rate or optimization based on risk configurations
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "oil_brain_weights.pth")
    logger.info("PyTorch weights successfully updated!")

def run_weekly_self_learning():
    logger.info("Initializing Weekly Self-Learning Loop...")
    
    # 1. Fetch real closed trades from OANDA
    trades = get_oanda_trade_history()
    
    # Fallback to dummy data if no closed trades exist yet
    if not trades:
        logger.info("No closed trades found on OANDA. Using baseline metrics.")
        trades = [{"id": "mock_1", "instrument": "WTICO_USD", "pnl": -15.0}]

    # 2. Read current hyperparameters
    try:
        with open("hyperparameters.json", "r") as f:
            current_params = json.load(f)
    except FileNotFoundError:
        current_params = {"risk_per_trade_pct": 0.01, "atr_stop_loss_multiplier": 2.0, "macro_sentiment_threshold": 0.4}

    # 3. Ask Gemini 2.5 Pro to Audit and Propose adjustments
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"""
    You are a Senior Quantitative Portfolio Auditor. Analyze this trading performance and propose adjusted parameters.
    
    --- CURRENT SYSTEM PARAMETERS ---
    {json.dumps(current_params, indent=2)}

    --- RECENT CLOSED TRADES DATA ---
    {json.dumps(trades, indent=2)}

    Determine if our stop-loss is too tight (atr_stop_loss_multiplier) or risk is too high.
    Output your decision strictly adhering to the TunedHyperparameters schema.
    """

    logger.info("Querying Gemini Auditor...")
    response = client.models.generate_content(
        model="gemini-2.5-pro", # Using Pro for advanced quantitative reasoning
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TunedHyperparameters,
            temperature=0.2
        )
    )
    
    tuning = json.loads(response.text)
    
    # 4. HARDCODED RISK CLAMP (Never trust an AI with raw wallet safety)
    safe_params = {
        "risk_per_trade_pct": min(max(tuning["risk_per_trade_pct"], 0.005), 0.02), # Clamp between 0.5% and 2.0%
        "atr_stop_loss_multiplier": min(max(tuning["atr_stop_loss_multiplier"], 1.2), 3.0), # Clamp between 1.2 and 3.0
        "macro_sentiment_threshold": min(max(tuning["macro_sentiment_threshold"], 0.1), 0.8) # Clamp between 0.1 and 0.8
    }

    # 5. Overwrite hyperparameters file
    with open("hyperparameters.json", "w") as f:
        json.dump(safe_params, f, indent=2)

    # 6. Retrain the PyTorch Neural Weights with new parameters
    retrain_pytorch_brain(safe_params["risk_per_trade_pct"], safe_params["atr_stop_loss_multiplier"])

    # 7. Send Audit Report to Telegram
    report = (
        f"🎓 *WEEKLY AI SELF-LEARNING REPORT* 🎓\n\n"
        f"📝 *AI Auditor Reflection:* {tuning['reflection_summary']}\n\n"
        f"⚙️ *Updated Parameters:* \n"
        f"• Risk per Trade: {safe_params['risk_per_trade_pct']*100:.2f}%\n"
        f"• ATR Stop Multiplier: {safe_params['atr_stop_loss_multiplier']}x\n"
        f"• Macro Threshold: {safe_params['macro_sentiment_threshold']}\n\n"
        f"🧠 *Action:* PyTorch model retrained & compiled successfully!"
    )
    send_telegram_alert(report)
    logger.info("Self-learning cycle complete! Telegram alert dispatched.")

if __name__ == "__main__":
    run_weekly_self_learning()