# dashboard.py
import os
import sys
import subprocess
import requests
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load settings
load_dotenv()

st.set_page_page_config = st.set_page_config(
    page_title="DEEP-OIL COMMAND CENTER",
    page_icon="🛢️",
    layout="wide"
)

# --- CONFIGURATION ---
OANDA_TOKEN = os.getenv("OANDA_API_TOKEN")
OANDA_ACCOUNT = os.getenv("OANDA_ACCOUNT_ID")
OANDA_URL = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT}"
HEADERS = {"Authorization": f"Bearer {OANDA_TOKEN}", "Accept-Datetime-Format": "RFC3339"}
INSTRUMENT = "WTICO_USD"

def get_dashboard_data():
    """Queries OANDA for live balance, positions, and chart data."""
    try:
        # 1. Fetch Account Summary
        bal_res = requests.get(f"{OANDA_URL}/summary", headers=HEADERS)
        if bal_res.status_code != 200:
            return None
        account = bal_res.json().get('account', {})
        
        # 2. Fetch last 15 daily candles for the chart
        candle_url = f"https://api-fxpractice.oanda.com/v3/instruments/{INSTRUMENT}/candles?count=15&granularity=D"
        candle_res = requests.get(candle_url, headers=HEADERS)
        candles = candle_res.json().get('candles', [])
        
        price_history = []
        for c in candles:
            mid = c['mid']
            price_history.append({
                "Date": c['time'][:10],
                "Open": float(mid['o']),
                "High": float(mid['h']),
                "Low": float(mid['l']),
                "Close": float(mid['c'])
            })
            
        return {
            "balance": float(account.get("balance", 0.0)),
            "unrealized_pnl": float(account.get("unrealizedPL", 0.0)),
            "margin_used": float(account.get("marginUsed", 0.0)),
            "open_trades": int(account.get("openTradeCount", 0)),
            "prices": pd.DataFrame(price_history)
        }
    except Exception as e:
        st.error(f"Failed to connect to OANDA: {e}")
        return None

# --- UI HEADER ---
st.title("🛢️ DEEP-OIL COMMAND CENTER")
st.markdown("### Real-Time Algorithmic Execution & AI Operations")
st.markdown("---")

# Query OANDA Data
data = get_dashboard_data()

if data:
    # --- METRICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("OANDA Demo Balance", f"£{data['balance']:.2f}")
    with col2:
        # Dynamic color indicator for P&L
        pnl = data['unrealized_pnl']
        st.metric("Unrealized P&L", f"£{pnl:.2f}", delta=f"{pnl:.2f}" if pnl != 0 else None)
    with col3:
        st.metric("Active Trades", data['open_trades'])
    with col4:
        st.metric("Margin Used", f"£{data['margin_used']:.2f}")

    # --- MAIN GRID ---
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("#### Live WTI Crude Oil Price Chart (OANDA Feed)")
        df = data['prices']
        
        # Render a beautiful interactive Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        )])
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.markdown("#### Manual Control Deck")
        
        # This interactive button lets you trigger the master bot directly from your web portal!
        if st.button("⚡ TRIGGER AGENT CYCLE NOW", use_container_width=True):
            with st.spinner("Executing live cycle... check terminal/Telegram"):
                try:
                    # Executes the master_agent.py script in the background
                    result = subprocess.run(
                        [sys.executable, "master_agent.py"], 
                        capture_output=True, 
                        text=True, 
                        timeout=30
                    )
                    st.success("Cycle completed!")
                    st.code(result.stdout)
                except Exception as e:
                    st.error(f"Execution failed: {e}")

        st.markdown("---")
        st.markdown("#### Background Runtime Logs (`master_agent.log`)")
        
        # Reads your background log file and displays the last 15 lines directly on your webpage!
        log_file_path = "master_agent.log"
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as f:
                log_lines = f.readlines()[-20:]
            st.text_area("Last 20 log events:", value="".join(log_lines), height=250)
        else:
            st.info("No log file detected yet. Run the master agent to start logging.")

else:
    st.warning("OANDA is currently offline or your API keys are misconfigured.")