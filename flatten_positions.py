# flatten_positions.py
import os
import requests
import json
from dotenv import load_dotenv
from telegram_service import send_telegram_alert

# Load credentials
load_dotenv()

OANDA_TOKEN = os.getenv("OANDA_API_TOKEN")
OANDA_ACCOUNT = os.getenv("OANDA_ACCOUNT_ID")
OANDA_URL = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT}"
HEADERS = {"Authorization": f"Bearer {OANDA_TOKEN}", "Accept-Datetime-Format": "RFC3339"}
INSTRUMENT = "WTICO_USD"

def flatten_oil_portfolio():
    print("🚨 Initializing Emergency Portfolio Flattening Sequence...")
    
    # 1. Fetch all active trades for WTICO_USD
    trades_url = f"{OANDA_URL}/openTrades"
    res = requests.get(trades_url, headers=HEADERS)
    if res.status_code != 200:
        print(f"❌ Failed to fetch open trades: {res.text}")
        return
        
    trades = [t for t in res.json().get("trades", []) if t.get("instrument") == INSTRUMENT]
    
    if not trades:
        print("ℹ️ No open positions detected. Portfolio is already flat.")
        return
        
    print(f"Detected {len(trades)} open trades. Closing them individually to secure profit...")
    
    total_pnl = 0.0
    closed_count = 0
    
    # 2. Loop through and close each trade independently
    for t in trades:
        trade_id = t["id"]
        close_url = f"{OANDA_URL}/trades/{trade_id}/close"
        payload = {"units": "ALL"}
        
        close_res = requests.put(close_url, json=payload, headers=HEADERS)
        
        if close_res.status_code == 200:
            data = close_res.json()
            # Extract realized P&L from the transaction fill block
            fill = data.get("orderFillTransaction", {})
            pnl = float(fill.get("pl", 0.0))
            total_pnl += pnl
            closed_count += 1
            print(f"✅ Closed Trade ID {trade_id} | Realized P&L: £{pnl:.2f}")
        else:
            print(f"❌ Failed to close Trade ID {trade_id}: {close_res.text}")
            
    # 3. Send a consolidated Markdown alert to Telegram
    alert = (
        f"🚨 *PORTFOLIO FLATTENED VIA CODE* 🚨\n\n"
        f"*Asset:* WTI Crude Oil (WTICO-USD)\n"
        f"*Action:* Closed {closed_count} Active Trades\n"
        f"*Total Realized P&L:* `£{total_pnl:.2f}`\n\n"
        f"🛡️ *System Status:* Portfolio is flat. All risk has been successfully exited."
    )
    send_telegram_alert(alert)
    print(f"\n🎉 SUCCESS! Closed {closed_count} trades. Total Realized P&L: £{total_pnl:.2f}")

if __name__ == "__main__":
    flatten_oil_portfolio()