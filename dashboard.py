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

st.set_page_config(
    page_title="UNIFIED PORTFOLIO COMMAND CENTER",
    page_icon="🖥️",
    layout="wide"
)

# --- SYSTEM ENVIRONMENT CONFIGURATION ---
OANDA_TOKEN = os.getenv("OANDA_API_TOKEN")
OANDA_ACCOUNT = os.getenv("OANDA_ACCOUNT_ID")
OANDA_URL = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT}"
OANDA_HEADERS = {"Authorization": f"Bearer {OANDA_TOKEN}", "Accept-Datetime-Format": "RFC3339"}
OANDA_INSTRUMENT = "WTICO_USD"

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_URL = "https://paper-api.alpaca.markets/v2"
ALPACA_HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# --- DATA RETRIEVAL SERVICES ---

def get_oanda_live_data():
    """Queries OANDA for live balance, open trades, margin, and candlestick price charts."""
    try:
        # 1. Fetch Account Summary
        bal_res = requests.get(f"{OANDA_URL}/summary", headers=OANDA_HEADERS)
        if bal_res.status_code != 200:
            return None
        account = bal_res.json().get('account', {})
        
        # 2. Fetch last 15 daily candles for the chart
        candle_url = f"https://api-fxpractice.oanda.com/v3/instruments/{OANDA_INSTRUMENT}/candles?count=15&granularity=D"
        candle_res = requests.get(candle_url, headers=OANDA_HEADERS)
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
            
        # 3. Fetch list of open trades to display SL levels
        open_trades_res = requests.get(f"{OANDA_URL}/openTrades", headers=OANDA_HEADERS)
        open_trades = open_trades_res.json().get("trades", [])
        
        return {
            "balance": float(account.get("balance", 0.0)),
            "unrealized_pnl": float(account.get("unrealizedPL", 0.0)),
            "margin_used": float(account.get("marginUsed", 0.0)),
            "open_trade_count": int(account.get("openTradeCount", 0)),
            "prices": pd.DataFrame(price_history),
            "open_trades_list": open_trades
        }
    except Exception as e:
        return None

def get_oanda_closed_metrics():
    """Queries OANDA for closed trades to calculate locked-in performance metrics."""
    url = f"{OANDA_URL}/trades?state=CLOSED&count=100"
    try:
        res = requests.get(url, headers=OANDA_HEADERS)
        if res.status_code == 200:
            trades = res.json().get("trades", [])
            total_trades = len(trades)
            wins = sum(1 for t in trades if float(t.get("realizedPL", 0.0)) > 0)
            losses = sum(1 for t in trades if float(t.get("realizedPL", 0.0)) <= 0)
            realized_pnl = sum(float(t.get("realizedPL", 0.0)) for t in trades)
            
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
            
            gross_wins = sum(float(t.get("realizedPL", 0.0)) for t in trades if float(t.get("realizedPL", 0.0)) > 0)
            gross_losses = abs(sum(float(t.get("realizedPL", 0.0)) for t in trades if float(t.get("realizedPL", 0.0)) < 0))
            profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (gross_wins if gross_wins > 0 else 1.0)
            
            return {
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "realized_pnl": realized_pnl,
                "profit_factor": profit_factor,
                "list": trades
            }
    except Exception as e:
        pass
    return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "realized_pnl": 0.0, "profit_factor": 1.0, "list": []}

def get_alpaca_live_data():
    """Queries Alpaca Paper Trading for account equity and open positions."""
    try:
        # 1. Fetch Account Details
        acc_res = requests.get(f"{ALPACA_URL}/account", headers=ALPACA_HEADERS)
        if acc_res.status_code != 200:
            return None
        account = acc_res.json()
        
        # 2. Fetch Open Positions
        pos_res = requests.get(f"{ALPACA_URL}/positions", headers=ALPACA_HEADERS)
        positions = pos_res.json() if pos_res.status_code == 200 else []
        
        return {
            "equity": float(account.get("equity", 0.0)),
            "cash": float(account.get("cash", 0.0)),
            "buying_power": float(account.get("buying_power", 0.0)),
            "unrealized_pl": float(account.get("profit_loss", 0.0)) if "profit_loss" in account else 0.0,
            "positions_list": positions
        }
    except Exception as e:
        return None

# --- UI MAIN ENGINE ---
st.title("🖥️ UNIFIED PORTFOLIO COMMAND CENTER")
st.markdown("### Real-Time Cross-Asset AI Trading Operations")
st.markdown("---")

# Setup Tabs for Milestone 2
tab1, tab2, tab3 = st.tabs([
    "🛢️ WTI Crude Oil Agent (OANDA)", 
    "🪙 Crypto & ETF Agent (Alpaca)", 
    "📊 System Health & Cron Logs"
])

# =====================================================================
# TAB 1: OANDA OIL BOT
# =====================================================================
with tab1:
    oanda_data = get_oanda_live_data()
    metrics = get_oanda_closed_metrics()
    
    if oanda_data:
        # Real-Time Open Position Metrics
        st.markdown("#### 🟢 Active Open Positions")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("OANDA Demo Balance", f"£{oanda_data['balance']:.2f}")
        with col2:
            pnl = oanda_data['unrealized_pnl']
            st.metric("Unrealized Open P&L", f"£{pnl:.2f}", delta=f"£{pnl:.2f}" if pnl != 0 else None)
        with col3:
            st.metric("Active Open Trades", oanda_data['open_trade_count'])
        with col4:
            st.metric("Margin Committed", f"£{oanda_data['margin_used']:.2f}")
            
        st.markdown("---")
        
        # LOCKED-IN HISTORICAL METRICS
        st.markdown("#### 🔒 Locked-In Historical Performance (Closed Trades)")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Total Closed Trades", metrics["total_trades"])
        with m_col2:
            st.metric("Wins / Losses", f"🟢 {metrics['wins']} / 🔴 {metrics['losses']}")
        with m_col3:
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        with m_col4:
            st.metric("Realized P&L (Locked-In)", f"£{metrics['realized_pnl']:.2f}")
        with m_col5:
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}x")
            
        st.markdown("---")

        # --- NEW SECTION: EIA WEEKLY FUNDAMENTAL CARD ---
        st.markdown("#### 📊 Weekly EIA Inventory Fundamental Status")
        if os.path.exists("eia_fundamental.json"):
            try:
                with open("eia_fundamental.json", "r") as f:
                    eia_data = json.load(f)
                
                e_col1, e_col2, e_col3 = st.columns(3)
                with e_col1:
                    change = eia_data.get("inventory_change_millions", 0.0)
                    # Green for positive draws (bullish), red for positive builds (bearish)
                    st.metric(
                        "EIA Inventory Change", 
                        f"{change:+.3f}M Barrels", 
                        delta=f"{change:+.3f}M" if change != 0 else None, 
                        delta_color="inverse"
                    )
                with e_col2:
                    impact = eia_data.get("fundamental_impact", "NEUTRAL")
                    st.metric("Fundamental Impact", impact)
                with e_col3:
                    is_new = "ACTIVE (Last 48H)" if eia_data.get("is_new_release") else "INACTIVE (Old Report)"
                    st.metric("Overlay Status", is_new)
                
                st.info(f"💡 *EIA AI Summary:* {eia_data.get('summary', 'No summary available.')}")
            except Exception as e:
                st.error(f"Failed to load EIA data: {e}")
        else:
            st.info("No EIA fundamental report parsed yet. Run eia_parser.py to initialize.")
            
        st.markdown("---")

        # Live Chart & Controls
        left_layout, right_layout = st.columns([2, 1])
        
        with left_layout:
            st.markdown("##### Live WTI Price Chart")
            df = oanda_data['prices']
            fig = go.Figure(data=[go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
            )])
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Displays list of active trades showing entry price, size, and current stop-losses
            st.markdown("##### Open Position Detail")
            if oanda_data["open_trades_list"]:
                trade_records = []
                for t in oanda_data["open_trades_list"]:
                    sl = t.get("stopLossOrder", {}).get("price", "None")
                    trade_records.append({
                        "Trade ID": t.get("id"),
                        "Side": "LONG" if float(t.get("initialUnits")) > 0 else "SHORT",
                        "Units": abs(float(t.get("initialUnits"))),
                        "Entry Price": f"${float(t.get('price')):.3f}",
                        "Current SL": f"${float(sl):.3f}" if sl != "None" else "None",
                        "Unrealized P&L": f"£{float(t.get('unrealizedPL')):.2f}"
                    })
                st.table(pd.DataFrame(trade_records))
            else:
                st.info("No open trades on WTICO_USD currently.")
            
        with right_layout:
            st.markdown("##### Manual Agent Override")
            if st.button("⚡ TRIGGER DEEP-OIL CYCLE NOW", use_container_width=True):
                with st.spinner("Processing deep-oil execution..."):
                    res = subprocess.run([sys.executable, "master_agent.py"], capture_output=True, text=True)
                    st.success("Execution complete!")
                    st.code(res.stdout)
                    
            st.markdown("---")
            st.markdown("##### Performance Log (`master_agent.log`)")
            if os.path.exists("master_agent.log"):
                with open("master_agent.log", "r") as f:
                    lines = f.readlines()[-15:]
                st.text_area("Live Oil Logs:", "".join(lines), height=200)
    else:
        st.warning("OANDA connection offline. Confirm OANDA keys in `.env`.")

# =====================================================================
# TAB 2: ALPACA BOT
# =====================================================================
with tab2:
    alpaca_data = get_alpaca_live_data()
    
    if alpaca_data:
        st.markdown("#### 🪙 Alpaca Portfolio Status")
        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            st.metric("Alpaca Paper Equity", f"${alpaca_data['equity']:.2f}")
        with a_col2:
            st.metric("Available Cash", f"${alpaca_data['cash']:.2f}")
        with a_col3:
            st.metric("Buying Power", f"${alpaca_data['buying_power']:.2f}")
            
        st.markdown("---")
        
        st.markdown("#### Active ETF & Crypto Holdings")
        if alpaca_data["positions_list"]:
            pos_records = []
            for p in alpaca_data["positions_list"]:
                pos_records.append({
                    "Symbol": p.get("symbol"),
                    "Market Value": f"${float(p.get('market_value', 0.0)):.2f}",
                    "Quantity": float(p.get("qty")),
                    "Avg Entry Price": f"${float(p.get('avg_entry_price', 0.0)):.2f}",
                    "Total Profit/Loss": f"${float(p.get('unrealized_pl', 0.0)):.2f}"
                })
            st.table(pd.DataFrame(pos_records))
        else:
            st.info("No active holdings on Alpaca paper account.")
    else:
        st.warning("Alpaca connection offline. Confirm Alpaca keys in `.env`.")

# =====================================================================
# TAB 3: SYSTEM HEALTH & CRON LOGS
# =====================================================================
with tab3:
    st.markdown("#### 🖥️ Active Background Scheduler (`crontab`)")
    try:
        cron_check = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if cron_check.return_code == 0:
            st.code(cron_check.stdout)
        else:
            st.info("No active user crontabs installed.")
    except Exception as e:
        st.error(f"Failed to check crontab: {e}")
        
    st.markdown("---")
    
    left_log, right_log = st.columns(2)
    with left_log:
        st.markdown("##### WTI Oil Bot Logs (`master_agent.log`)")
        if os.path.exists("master_agent.log"):
            with open("master_agent.log", "r") as f:
                lines = f.readlines()[-30:]
            st.text_area("Last 30 Oil Log Events:", "".join(lines), height=350)
        else:
            st.info("No master_agent.log file detected yet.")
            
    with right_log:
        st.markdown("##### Crypto & ETF Bot Logs (`fast.log`)")
        # Adjust path to match where your original bot saves logs
        original_log_path = "../Projects/trading-agent/logs/fast.log"
        if os.path.exists(original_log_path):
            with open(original_log_path, "r") as f:
                lines = f.readlines()[-30:]
            st.text_area("Last 30 original Log Events:", "".join(lines), height=350)
        else:
            # Check backup path
            alt_path = "logs/fast.log"
            if os.path.exists(alt_path):
                with open(alt_path, "r") as f:
                    lines = f.readlines()[-30:]
                st.text_area("Last 30 original Log Events:", "".join(lines), height=350)
            else:
                st.info("Original bot logs not found at standard path.")