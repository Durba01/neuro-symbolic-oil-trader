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

def execute_oanda_order(units: float, action: str, current_price: float, atr_val: float):
    """Sends the actual trade order to OANDA's V20 API with automated, volatility-adjusted SL & TP."""
    url = f"{OANDA_URL}/orders"
    
    # OANDA represents Short (SELL) trades using negative units
    trade_units = str(int(units)) if "BUY" in action else str(-int(units))
    
    # Volatility Multipliers (Risk-to-Reward ratio of 1:1.5)
    sl_multiplier = 2.0  # Risk 2x ATR
    tp_multiplier = 3.0  # Target 3x ATR (Balanced, neither too conservative nor too greedy)
    
    if "BUY" in action:
        sl_price = current_price - (atr_val * sl_multiplier)
        tp_price = current_price + (atr_val * tp_multiplier)
    else:  # SELL / Short
        sl_price = current_price + (atr_val * sl_multiplier)
        tp_price = current_price - (atr_val * tp_multiplier)
        
    payload = {
        "order": {
            "units": trade_units,
            "instrument": INSTRUMENT,
            "timeInForce": "FOK", # Fill-Or-Kill
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {
                "timeInForce": "GTC",
                "price": f"{sl_price:.3f}"
            },
            "takeProfitOnFill": {
                "timeInForce": "GTC",
                "price": f"{tp_price:.3f}"
            }
        }
    }
    
    res = requests.post(url, json=payload, headers=HEADERS)
    return res.json()

def get_open_trades():
    """Queries OANDA for all active, open trades for our target instrument."""
    url = f"{OANDA_URL}/openTrades"
    res = requests.get(url, headers=HEADERS)
    if res.status_code == 200:
        trades = res.json().get("trades", [])
        return [t for t in trades if t.get("instrument") == INSTRUMENT]
    return []

def modify_trade_stop_loss(trade_id: str, price: float):
    """Updates or attaches a Stop Loss order to an active trade."""
    url = f"{OANDA_URL}/trades/{trade_id}/orders"
    payload = {
        "stopLoss": {
            "timeInForce": "GTC",
            "price": f"{price:.3f}"
        }
    }
    res = requests.put(url, json=payload, headers=HEADERS)
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

    # Calculate rolling daily volatility (ATR proxy) from the last 5 daily candles
    atr_val = np.mean([c[1] - c[2] for c in price_history])
    print(f"Calculated Daily Volatility (ATR Proxy): ${atr_val:.3f}")

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

    # STEP 4: The Shield (Risk Management with EIA Fundamental Overlay)
    print("\n[STEP 4] Passing through Risk Gate...")
    
    # --- EIA FUNDAMENTAL OVERLAY ---
    eia_veto = False
    eia_reason = ""
    
    if os.path.exists("eia_fundamental.json"):
        try:
            with open("eia_fundamental.json", "r") as f:
                eia_data = json.load(f)
            
            # If a fresh report (last 24-48 hours) is active
            if eia_data.get("is_new_release"):
                impact = eia_data.get("fundamental_impact")
                change = eia_data.get("inventory_change_millions")
                
                # Check for contradictory signals
                if prediction > 0.65 and impact == "BEARISH":
                    eia_veto = True
                    eia_reason = f"Highly Bearish Weekly EIA Inventory Report (Unexpected Build of {change}M barrels)."
                elif prediction < 0.35 and impact == "BULLISH":
                    eia_veto = True
                    eia_reason = f"Highly Bullish Weekly EIA Inventory Report (Unexpected Draw of {change}M barrels)."
        except Exception as e:
            print(f"Failed to parse EIA fundamental data: {e}")
    # -------------------------------

    if eia_veto:
        print(f"🛡️ RISK OVERRIDE: Vetoing trade signal due to: {eia_reason}")
        send_telegram_alert(f"🛡️ *Risk Shield: Trade Vetoed*\nAI suggested a Long trade, but the **Risk Gate vetoed it** due to a contradictory: {eia_reason}")
        units = 0
    elif prediction > 0.65:
        trade_direction = "BUY (Long)"
        units = calculate_order_size(balance, current_price)
    elif prediction < 0.35:
        trade_direction = "SELL (Short)"
        units = calculate_order_size(balance, current_price)
    else:
        print("⚖️ Market is neutral. AI chooses to HOLD.")
        send_telegram_alert(f"🤖 *Deep-Oil Agent*\nMarket neutral (Score: {prediction:.2f}). Holding positions.")
        units = 0

    # STEP 5: Execution Intent with Automated Bracket Orders
    if units > 0:
        print("\n[STEP 5] Routing Order with Bracket SL/TP to OANDA...")
        # Now passing current_price and atr_val to calculate automated targets on-fill
        order_res = execute_oanda_order(units, trade_direction, current_price, atr_val)
        
        # Parse the response from OANDA
        if "orderFillTransaction" in order_res:
            fill = order_res["orderFillTransaction"]
            price_filled = float(fill.get("price"))
            order_id = fill.get("id")
            
            # Extract actual placed TP/SL levels for logging
            trade_opened = fill.get("tradeOpened", {})
            sl_level = float(trade_opened.get("stopLossOrder", {}).get("price", 0.0))
            tp_level = float(trade_opened.get("takeProfitOrder", {}).get("price", 0.0))
            
            print(f"✅ Trade Successful! Order ID: {order_id} filled at ${price_filled:.3f}")
            print(f"🎯 Bracket Targets Attached -> SL: ${sl_level:.3f} | TP: ${tp_level:.3f}")
            
            alert = (
                f"🚨 *DEEP-OIL BRACKET TRADE EXECUTED* 🚨\n\n"
                f"*Action:* {trade_direction}\n"
                f"*Asset:* WTI Crude Oil (WTICO-USD)\n"
                f"*Price Filled:* ${price_filled:.3f}\n"
                f"*Size:* {units} units\n"
                f"*Order ID:* {order_id}\n\n"
                f"🎯 *Bracket Targets Locked On-Fill:*\n"
                f"• Stop Loss: `${sl_level:.3f}`\n"
                f"• Take Profit: `${tp_level:.3f}`\n\n"
                f"🧠 *AI Confidence:* {prediction:.2f}\n"
                f"📰 *Macro Rationale:* {sentiment['summary']}"
            )
            send_telegram_alert(alert)
        elif "orderCancelTransaction" in order_res:
            cancel = order_res["orderCancelTransaction"]
            reason = cancel.get("reason", "UNKNOWN_REASON")
            print(f"⚠️ Order Canceled: {reason}")
        else:
            error_msg = order_res.get("errorMessage", "Unknown execution error.")
            print(f"❌ Order Placement Failed: {error_msg}")

    # STEP 6: Active Risk Sweep (The Break-Even & Trailing Stop Guardrails)
    print("\n[STEP 6] Running Active Risk Sweep on Open Positions...")
    open_trades = get_open_trades()
    print(f"Found {len(open_trades)} active trades to evaluate.")
    
    # Sizing thresholds matching OANDA parameters
    break_even_threshold = atr_val * 1.5   
    trailing_stop_multiplier = atr_val * 2.0  
    
    for t in open_trades:
        trade_id = t["id"]
        entry_price = float(t["price"])
        current_units = float(t["currentUnits"])
        side = "LONG" if current_units > 0 else "SHORT"
        
        # Get the current active Stop Loss price if it exists
        current_sl = None
        if "stopLossOrder" in t:
            current_sl = float(t["stopLossOrder"].get("price"))
            
        print(f"\nEvaluating Trade ID {trade_id} ({side}) | Entry: ${entry_price:.3f} | Current SL: ${f'{current_sl:.3f}' if current_sl else 'None'}")
        
        if side == "LONG":
            # 1. Evaluate Break-Even Shield
            if (current_price - entry_price) >= break_even_threshold:
                if current_sl is None or current_sl < entry_price:
                    print(f"🛡️ BREAK-EVEN SHIELD ACTIVE: Moving SL to entry price (${entry_price:.3f})")
                    modify_trade_stop_loss(trade_id, entry_price)
                    send_telegram_alert(f"🛡️ *Risk Shield: Break-Even Activated*\nTrade ID {trade_id} (WTI Long) Stop Loss moved to Entry Price: `${entry_price:.3f}`. Trade is now risk-free!")
                    current_sl = entry_price 
            
            # 2. Evaluate Trailing Stop
            calculated_sl = current_price - trailing_stop_multiplier
            if calculated_sl > entry_price: 
                if current_sl is None or calculated_sl > current_sl:
                    print(f"📈 TRAILING STOP ACTIVE: Sliding SL up to ${calculated_sl:.3f}")
                    modify_trade_stop_loss(trade_id, calculated_sl)
                    send_telegram_alert(f"📈 *Risk Shield: Trailing Stop Activated*\nTrade ID {trade_id} (WTI Long) Stop Loss slid up to `${calculated_sl:.3f}` to lock in profits!")
                    
        elif side == "SHORT":
            # 1. Evaluate Break-Even Shield
            if (entry_price - current_price) >= break_even_threshold:
                if current_sl is None or current_sl > entry_price:
                    print(f"🛡️ BREAK-EVEN SHIELD ACTIVE: Moving SL to entry price (${entry_price:.3f})")
                    modify_trade_stop_loss(trade_id, entry_price)
                    send_telegram_alert(f"🛡️ *Risk Shield: Break-Even Activated*\nTrade ID {trade_id} (WTI Short) Stop Loss moved to Entry Price: `${entry_price:.3f}`. Trade is now risk-free!")
                    current_sl = entry_price
            
            # 2. Evaluate Trailing Stop
            calculated_sl = current_price + trailing_stop_multiplier
            if calculated_sl < entry_price: 
                if current_sl is None or calculated_sl < current_sl:
                    print(f"📈 TRAILING STOP ACTIVE: Sliding SL down to ${calculated_sl:.3f}")
                    modify_trade_stop_loss(trade_id, calculated_sl)
                    send_telegram_alert(f"📈 *Risk Shield: Trailing Stop Activated*\nTrade ID {trade_id} (WTI Short) Stop Loss slid down to `${calculated_sl:.3f}` to lock in profits!")
                    
    print("\n✅ Cycle Complete!")

# These two lines are completely left-aligned (no spaces)
if __name__ == "__main__":
    run_trading_cycle()