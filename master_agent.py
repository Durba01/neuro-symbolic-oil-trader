# master_agent.py
import os
import json
import numpy as np
import torch
import requests
from dotenv import load_dotenv

# Import our custom modules
from gemini_brain import analyze_news_tensor
from deep_learning_model import DeepOilLSTM
from risk_manager import calculate_order_size
from telegram_service import send_telegram_alert

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
OANDA_TOKEN = os.getenv("OANDA_API_TOKEN")
OANDA_ACCOUNT = os.getenv("OANDA_ACCOUNT_ID")
OANDA_URL = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT}"
HEADERS = {"Authorization": f"Bearer {OANDA_TOKEN}", "Accept-Datetime-Format": "RFC3339"}
INSTRUMENT = "WTICO_USD" # OANDA's ticker for WTI Crude Oil

def get_latest_oanda_data():
    """Fetches account balance and the last 5 days of Oil prices from OANDA."""
    print("🌐 Connecting to OANDA...")
    
    # 1. Get Balance
    bal_res = requests.get(f"{OANDA_URL}/summary", headers=HEADERS)
    
    # Optional Debug Lines
    print(f"DEBUG - OANDA HTTP Status: {bal_res.status_code}")
    print(f"DEBUG - OANDA Raw Response: {bal_res.text}")
    
    if bal_res.status_code != 200:
        print("OANDA is under maintenance or keys are invalid.")
        return None, None, None
        
    balance = float(bal_res.json()['account']['balance'])
    
    # 2. Get last 5 days of prices (to feed the LSTM sequence)
    candle_url = f"https://api-fxpractice.oanda.com/v3/instruments/{INSTRUMENT}/candles?count=5&granularity=D"
    candle_res = requests.get(candle_url, headers=HEADERS)
    
    # Robust error check for the candle data
    if candle_res.status_code != 200:
        print(f"❌ OANDA Candle Fetch Failed: {candle_res.status_code} - {candle_res.text}")
        return None, None, None
        
    candles = candle_res.json().get('candles', [])
    
    price_data = []
    current_price = 0.0
    for c in candles:
        mid = c['mid']
        # [Open, High, Low, Close, Volume]
        price_data.append([float(mid['o']), float(mid['h']), float(mid['l']), float(mid['c']), float(c['volume'])])
        current_price = float(mid['c'])
        
    return balance, price_data, current_price

def execute_oanda_order(units: float, action: str):
    """Sends the actual trade order to OANDA's V20 execution API."""
    url = f"{OANDA_URL}/orders"
    
    # OANDA represents Short (SELL) trades using negative units
    trade_units = str(int(units)) if "BUY" in action else str(-int(units))
    
    payload = {
        "order": {
            "units": trade_units,
            "instrument": INSTRUMENT,
            "timeInForce": "FOK", # Fill-Or-Kill (execute immediately or cancel)
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    
    res = requests.post(url, json=payload, headers=HEADERS)
    return res.json()

def run_trading_cycle():
    print("=========================================")
    print("🚀 DEEP-OIL NEURO-SYMBOLIC AGENT AWAKE")
    print("=========================================")
    
    # STEP 1: The Senses (Fetch live news and Gemini sentiment)
    print("\n[STEP 1] Activating AI Senses...")
    sentiment = analyze_news_tensor()
    gemini_vector = [sentiment['bullish_score'], sentiment['bearish_score'], sentiment['supply_shock_risk']]
    
    # STEP 2: The Data (Fetch OANDA prices and balance)
    print("\n[STEP 2] Fetching Live Broker Data...")
    balance, price_history, current_price = get_latest_oanda_data()
    
    if balance is None:
        print("⚠️ Waiting for OANDA maintenance to end (7:00 PM BST). Agent resting.")
        return
        
    print(f"💰 Confirmed Balance: £{balance}")
    print(f"🛢️ Current WTI Price: ${current_price}")

    # STEP 3: The Brain (Load PyTorch and Predict)
    print("\n[STEP 3] Waking up Deep Learning Brain...")
    model = DeepOilLSTM()
    model.load_state_dict(torch.load("oil_brain_weights.pth"))
    model.eval() # Set to evaluation mode
    
    # Format data for the Neural Network (Scale prices roughly between 0-1)
    scaled_prices = np.array(price_history) / 100.0 
    
    # Combine 5 days of prices with today's Gemini news vector
    combined_seq = []
    for prices in scaled_prices:
        combined_seq.append(np.hstack((prices, gemini_vector)))
        
    # Optimized numpy conversion to silence the PyTorch UserWarning
    tensor_data = torch.tensor(np.array([combined_seq]), dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(tensor_data).item()
        
    print(f"🧠 Neural Network Prediction Score: {prediction:.4f} (0=Crash, 1=Moon)")

    # STEP 4: The Shield (Risk Management)
    print("\n[STEP 4] Passing through Risk Gate...")
    if prediction > 0.65:
        trade_direction = "BUY (Long)"
        units = calculate_order_size(balance, current_price)
    elif prediction < 0.35:
        trade_direction = "SELL (Short)"
        units = calculate_order_size(balance, current_price)
    else:
        print("⚖️ Market is neutral. AI chooses to HOLD.")
        send_telegram_alert(f"🤖 *Deep-Oil Agent*\nMarket neutral (Score: {prediction:.2f}). Holding positions.")
        return

    # STEP 5: Execution Intent
    print("\n[STEP 5] Routing Order to OANDA Exchange...")
    order_res = execute_oanda_order(units, trade_direction)
    
    # Parse the response from OANDA
    if "orderFillTransaction" in order_res:
        fill = order_res["orderFillTransaction"]
        price_filled = fill.get("price")
        order_id = fill.get("id")
        
        print(f"✅ Trade Successful! Order ID: {order_id} filled at ${price_filled}")
        
        # CLEANED MARKDOWN STRING (Safely formatted for Telegram)
        alert = (
            f"🚨 *DEEP-OIL TRADE EXECUTED* 🚨\n\n"
            f"*Action:* {trade_direction}\n"
            f"*Asset:* WTI Crude Oil (WTICO-USD)\n"  # Changed underscore to hyphen to prevent Telegram crash
            f"*Price Filled:* ${price_filled}\n"
            f"*Size:* {units} units\n"
            f"*Order ID:* {order_id}\n\n"
            f"*AI Confidence:* {prediction:.2f}\n"
            f"*Macro Rationale:* {sentiment['summary']}"
        )
        send_telegram_alert(alert)
        
    elif "orderCancelTransaction" in order_res:
        cancel = order_res["orderCancelTransaction"]
        reason = cancel.get("reason", "UNKNOWN_REASON")
        
        if reason == "MARKET_HALTED":
            friendly_reason = "OANDA Market is currently Halted (Weekend / Holiday Close)."
        else:
            friendly_reason = f"Order canceled by broker. Reason: {reason}"
            
        print(f"⚠️ Order Canceled: {friendly_reason}")
        send_telegram_alert(f"🤖 *Deep-Oil Agent*\nOrder proposed but canceled: {friendly_reason}")
        
    else:
        error_msg = order_res.get("errorMessage", "Unknown execution error.")
        print(f"❌ Order Placement Failed: {error_msg}")
        print(f"DEBUG - Full Broker Response: {json.dumps(order_res, indent=2)}")
        send_telegram_alert(f"⚠️ *Deep-Oil Agent Error*\nFailed to execute order: {error_msg}")
        
    print("✅ Cycle Complete!")

# These two lines are completely left-aligned (no spaces)
if __name__ == "__main__":
    run_trading_cycle()