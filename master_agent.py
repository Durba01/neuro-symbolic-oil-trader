# master_agent.py
import os
import json
import numpy as np
import pandas as pd
import torch
import requests
import yfinance as yf
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

# Institutional Risk Settings
MAX_ALLOWED_SPREAD = 0.08  # Max $0.08 bid-ask spread tolerance to prevent entry slippage

def get_latest_oanda_data():
    """Fetches balance, daily candles, 1-hour candles, and live spread from OANDA."""
    print("🌐 Connecting to OANDA V20 Engine...")
    
    # 1. Fetch Account Balance
    bal_res = requests.get(f"{OANDA_URL}/summary", headers=HEADERS)
    if bal_res.status_code != 200:
        print("OANDA is under maintenance or keys are invalid.")
        return None, None, None, None, None
        
    balance = float(bal_res.json()['account']['balance'])
    
    # 2. Fetch Live Price & Spread Check (Feature A: Slippage Shield)
    price_url = f"{OANDA_URL}/pricing?instruments={INSTRUMENT}"
    price_res = requests.get(price_url, headers=HEADERS)
    if price_res.status_code != 200:
        print("Failed to fetch live pricing.")
        return None, None, None, None, None
        
    prices = price_res.json().get("prices", [])[0]
    bid = float(prices["bids"][0]["price"])
    ask = float(prices["asks"][0]["price"])
    current_price = (bid + ask) / 2.0
    spread = ask - bid
    
    # 3. Fetch 5 Daily Candles (Macro Trend)
    d1_url = f"https://api-fxpractice.oanda.com/v3/instruments/{INSTRUMENT}/candles?count=5&granularity=D"
    d1_res = requests.get(d1_url, headers=HEADERS)
    d1_candles = d1_res.json().get('candles', [])
    
    d1_data = []
    for c in d1_candles:
        mid = c['mid']
        d1_data.append([float(mid['o']), float(mid['h']), float(mid['l']), float(mid['c']), float(c['volume'])])
        
    # 4. Fetch 1-Hour Candles (Feature B: Multi-Timeframe Confluence)
    h1_url = f"https://api-fxpractice.oanda.com/v3/instruments/{INSTRUMENT}/candles?count=6&granularity=H1"
    h1_res = requests.get(h1_url, headers=HEADERS)
    h1_candles = h1_res.json().get('candles', [])
    
    # Compute 1-Hour Trend Direction
    h1_open = float(h1_candles[0]['mid']['o'])
    h1_close = float(h1_candles[-1]['mid']['c'])
    h1_bullish = h1_close > h1_open

    return balance, d1_data, current_price, spread, h1_bullish

def get_dxy_macro_signal():
    """Feature D: Ingests the US Dollar Index (DXY) to measure currency correlation."""
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        hist = dxy.history(period="2d")
        if len(hist) >= 2:
            prev_close = hist['Close'].iloc[-2]
            latest_close = hist['Close'].iloc[-1]
            pct_change = ((latest_close - prev_close) / prev_close) * 100
            return round(pct_change, 2)
    except Exception as e:
        print(f"DXY fetch failed: {e}")
    return 0.0

def execute_oanda_order(units: float, action: str, current_price: float, atr_val: float):
    """Sends trade order to OANDA with automated SL & TP bracket parameters."""
    url = f"{OANDA_URL}/orders"
    trade_units = str(int(units)) if "BUY" in action else str(-int(units))
    
    sl_multiplier = 2.0  
    tp_multiplier = 3.0  
    
    if "BUY" in action:
        sl_price = current_price - (atr_val * sl_multiplier)
        tp_price = current_price + (atr_val * tp_multiplier)
    else:  
        sl_price = current_price + (atr_val * sl_multiplier)
        tp_price = current_price - (atr_val * tp_multiplier)
        
    payload = {
        "order": {
            "units": trade_units,
            "instrument": INSTRUMENT,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"timeInForce": "GTC", "price": f"{sl_price:.3f}"},
            "takeProfitOnFill": {"timeInForce": "GTC", "price": f"{tp_price:.3f}"}
        }
    }
    return requests.post(url, json=payload, headers=HEADERS).json()

def get_open_trades():
    """Queries OANDA for active trades."""
    url = f"{OANDA_URL}/openTrades"
    res = requests.get(url, headers=HEADERS)
    if res.status_code == 200:
        return [t for t in res.json().get("trades", []) if t.get("instrument") == INSTRUMENT]
    return []

def partial_close_trade(trade_id: str, units_to_close: int):
    """Feature C: Scale-Out Profit Engine (Closes a fraction of units)."""
    url = f"{OANDA_URL}/trades/{trade_id}/close"
    payload = {"units": str(units_to_close)}
    res = requests.put(url, json=payload, headers=HEADERS)
    return res.json()

def modify_trade_stop_loss(trade_id: str, price: float):
    """Updates stop loss price."""
    url = f"{OANDA_URL}/trades/{trade_id}/orders"
    payload = {"stopLoss": {"timeInForce": "GTC", "price": f"{price:.3f}"}}
    return requests.put(url, json=payload, headers=HEADERS).json()

def run_trading_cycle():
    print("=========================================")
    print("🚀 DEEP-OIL INSTITUTIONAL AGENT AWAKE")
    print("=========================================")
    
    # STEP 1: AI Senses
    print("\n[STEP 1] Activating AI Senses & Macro Scrapers...")
    try:
        sentiment = analyze_news_tensor()
        gemini_vector = [sentiment['bullish_score'], sentiment['bearish_score'], sentiment['supply_shock_risk']]
    except Exception as e:
        print(f"❌ AI Senses Failed due to network drop: {e}")
        send_telegram_alert(f"⚠️ *Deep-Oil Agent Warning*\nNetwork drop during news analysis: {e}")
        return

    # STEP 2: Ingest Broker & DXY Data
    print("\n[STEP 2] Fetching Live Market Data & Macro Metrics...")
    balance, d1_history, current_price, spread, h1_bullish = get_latest_oanda_data()
    
    if balance is None:
        print("⚠️ Waiting for OANDA maintenance to end. Agent resting.")
        return
        
    dxy_change = get_dxy_macro_signal()
    atr_val = np.mean([c[1] - c[2] for c in d1_history])
    
    print(f"💰 Confirmed Balance: £{balance:.2f}")
    print(f"🛢️ WTI Price: ${current_price:.3f} | Live Spread: ${spread:.3f}")
    print(f"💵 US Dollar Index (DXY) 24H Shift: {dxy_change}%")
    print(f"📊 Calculated Volatility (ATR Proxy): ${atr_val:.3f}")

    # FEATURE A: SLIPPAGE SHIELD CHECK
    if spread > MAX_ALLOWED_SPREAD:
        print(f"🛡️ SPREAD SHIELD ACTIVE: Live spread (${spread:.3f}) exceeds max allowed limit (${MAX_ALLOWED_SPREAD}). Halting entry.")
        send_telegram_alert(f"🛡️ *Risk Shield Active*\nExecution paused: Spread spiked to `${spread:.3f}` (exceeds $0.08 limit).")
        return

    # STEP 3: PyTorch Neural Prediction
    print("\n[STEP 3] Running PyTorch LSTM Inference...")
    model = DeepOilLSTM()
    model.load_state_dict(torch.load("oil_brain_weights.pth"))
    model.eval()
    
    scaled_prices = np.array(d1_history) / 100.0 
    combined_seq = [np.hstack((prices, gemini_vector)) for prices in scaled_prices]
    tensor_data = torch.tensor(np.array([combined_seq]), dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(tensor_data).item()
        
    print(f"🧠 Neural Prediction Score: {prediction:.4f} (0=Crash/Short, 1=Moon/Long)")

    # STEP 4: Risk Gate & Macro Overrides
    print("\n[STEP 4] Passing Through Risk Gate & Multi-Timeframe Check...")
    
    # Read EIA Fundamental Data
    eia_veto = False
    eia_reason = ""
    if os.path.exists("eia_fundamental.json"):
        try:
            with open("eia_fundamental.json", "r") as f:
                eia_data = json.load(f)
            if eia_data.get("is_new_release"):
                impact = eia_data.get("fundamental_impact")
                if prediction > 0.65 and impact == "BEARISH":
                    eia_veto = True
                    eia_reason = "Bearish EIA Inventory Report Build."
                elif prediction < 0.35 and impact == "BULLISH":
                    eia_veto = True
                    eia_reason = "Bullish EIA Inventory Report Draw."
        except Exception as e:
            print(f"EIA read error: {e}")

    # FEATURE B: MULTI-TIMEFRAME CONFLUENCE CHECK
    if eia_veto:
        print(f"🛡️ RISK OVERRIDE: Trade vetoed due to: {eia_reason}")
        send_telegram_alert(f"🛡️ *Risk Shield: Trade Vetoed*\nContradictory EIA Release: {eia_reason}")
        units = 0
    elif prediction > 0.65:
        if not h1_bullish:
            print("⏳ MULTI-TIMEFRAME HOLD: Daily trend is Bullish, but 1-Hour chart is pulling back. Waiting for entry alignment.")
            units = 0
        else:
            trade_direction = "BUY (Long)"
            units = calculate_order_size(balance, current_price)
    elif prediction < 0.35:
        if h1_bullish:
            print("⏳ MULTI-TIMEFRAME HOLD: Daily trend is Bearish, but 1-Hour chart is bouncing. Waiting for entry alignment.")
            units = 0
        else:
            trade_direction = "SELL (Short)"
            units = calculate_order_size(balance, current_price)
    else:
        print("⚖️ Market Signal Neutral. Agent Holding.")
        units = 0

    # STEP 5: Order Execution with Brackets
    if units > 0:
        print(f"\n[STEP 5] Dispatching {trade_direction} Order to OANDA...")
        order_res = execute_oanda_order(units, trade_direction, current_price, atr_val)
        
        if "orderFillTransaction" in order_res:
            fill = order_res["orderFillTransaction"]
            price_filled = float(fill.get("price"))
            order_id = fill.get("id")
            
            alert = (
                f"🚨 *INSTITUTIONAL TRADE EXECUTED* 🚨\n\n"
                f"*Action:* {trade_direction}\n"
                f"*Asset:* WTI Crude Oil (WTICO-USD)\n"
                f"*Price Filled:* ${price_filled:.3f}\n"
                f"*Size:* {units} units\n"
                f"*Order ID:* {order_id}\n\n"
                f"*Spread:* `${spread:.3f}` | *DXY Shift:* `{dxy_change}%`\n"
                f"*AI Confidence:* `{prediction:.2f}`"
            )
            send_telegram_alert(alert)

    # STEP 6: Active Risk Sweep & Feature C Scale-Out Engine
    print("\n[STEP 6] Running Active Risk Sweep & Scale-Out Engine...")
    open_trades = get_open_trades()
    print(f"Evaluating {len(open_trades)} active trades.")
    
    scale_out_threshold = atr_val * 1.5   # Target for 50% partial profit take
    break_even_threshold = atr_val * 1.5  # Move SL to entry
    trailing_multiplier = atr_val * 2.0   # Trail behind price
    
    for t in open_trades:
        trade_id = t["id"]
        entry_price = float(t["price"])
        initial_units = abs(float(t["initialUnits"]))
        current_units = float(t["currentUnits"])
        abs_current_units = abs(current_units)
        side = "LONG" if current_units > 0 else "SHORT"
        
        current_sl = float(t["stopLossOrder"].get("price")) if "stopLossOrder" in t else None
        
        if side == "LONG":
            profit_distance = current_price - entry_price
            
            # FEATURE C: SCALE-OUT HARVESTER (Close 50% of units at 1.5x ATR)
            if profit_distance >= scale_out_threshold and abs_current_units == initial_units:
                half_units = int(initial_units / 2)
                print(f"💰 SCALE-OUT HARVESTER: Closing 50% ({half_units} units) of Trade ID {trade_id} to lock in cash!")
                partial_close_trade(trade_id, half_units)
                send_telegram_alert(f"💰 *Profit Harvester Active*\nTrade ID {trade_id} hit 1.5x ATR profit! **Closed 50% ({half_units} units)** to secure realized cash.")
            
            # Break-Even & Trailing Stop Evaluation
            if profit_distance >= break_even_threshold:
                if current_sl is None or current_sl < entry_price:
                    modify_trade_stop_loss(trade_id, entry_price)
                    send_telegram_alert(f"🛡️ *Break-Even Shield:* Trade ID {trade_id} SL moved to entry price (${entry_price:.3f}).")
                    
            calc_sl = current_price - trailing_multiplier
            if calc_sl > entry_price and (current_sl is None or calc_sl > current_sl):
                modify_trade_stop_loss(trade_id, calc_sl)
                send_telegram_alert(f"📈 *Trailing Stop:* Trade ID {trade_id} SL slid up to `${calc_sl:.3f}`.")
                
    print("\n✅ Cycle Complete!")

if __name__ == "__main__":
    run_trading_cycle()